/* pqv2_model.c — load IBF v6 PQv2 file into an inferbit_model.
 *
 * Architecture is hardcoded to TinyLlama-1.1B for now. Future:
 * extend IBF v6 header with a config blob and read those.
 */
#include "inferbit_internal.h"
#include "pqv2_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define IB_PQV2_MAGIC "IBFV6PQ2"

extern inferbit_model* ibf_load(const char* path, const inferbit_config* config);
/* Runtime-state helpers exposed from ibf_loader.c. */
int ib_alloc_kv_caches(inferbit_model* model, int context_length, int dynamic);
int ib_alloc_buffers(inferbit_model* model);
/* SIMD + thread pool — declared in inferbit_internal.h with the right
 * types (ib_simd_level, ib_thread_pool); already in scope. */

static void fill_llama_defaults(ib_ibf_header* h) {
    memset(h, 0, sizeof(*h));
    strncpy(h->architecture, "llama", sizeof(h->architecture) - 1);
    h->rope_theta          = 10000.0f;
    h->norm_epsilon        = 1e-5f;
    strncpy(h->norm_type,  "rmsnorm", sizeof(h->norm_type) - 1);
    strncpy(h->activation, "silu",    sizeof(h->activation) - 1);
    h->tie_word_embeddings = false;
    h->bos_token_id = 1;
    h->eos_token_id = 2;
    h->default_bits = 4;
    /* KV cache bits: 16 (fp32) by default — quality-safe. Override via
     * IB_PQV2_KV_BITS (4/8/16). INT4 only pays off at long context
     * (≥1K tokens) where K/V cache memory traffic dominates attention. */
    {
        const char *kvb = getenv("IB_PQV2_KV_BITS");
        h->kv_bits = (kvb && kvb[0]) ? atoi(kvb) : 16;
        if (h->kv_bits != 16 && h->kv_bits != 8 && h->kv_bits != 4) h->kv_bits = 16;
    }
    h->alignment = 64;
    h->max_context_length  = 2048;
}

/* Detect architecture from PQv2 tensor shapes. We hardcode known
 * LLaMA-family models by signature (q_proj M, hidden, n_layers).
 * TODO: replace with a config blob in the IBF v6 header. */
static int detect_arch_from_tensors(const ib_pqv2_file* f, ib_ibf_header* h) {
    /* Find L0.self_attn.q_proj to read q_proj_M and hidden. */
    int q_proj_M = 0, hidden = 0, n_layers = 0;
    int v_proj_M = 0;
    int gate_proj_M = 0;
    for (int i = 0; i < f->n_tensors; i++) {
        const ib_pqv2_named_tensor* nt = &f->tensors[i];
        if (nt->kind != IB_PQV2_KIND_PQV2) continue;
        int li;
        char parent[32], proj[32];
        if (sscanf(nt->name, "L%d.%31[^.].%31s", &li, parent, proj) != 3) continue;
        if (li + 1 > n_layers) n_layers = li + 1;
        if (li == 0 && strcmp(parent, "self_attn") == 0) {
            if (strcmp(proj, "q_proj") == 0) {
                q_proj_M = (int)nt->pq.M;
                hidden   = (int)nt->pq.N;
            } else if (strcmp(proj, "v_proj") == 0) {
                v_proj_M = (int)nt->pq.M;
            }
        } else if (li == 0 && strcmp(parent, "mlp") == 0 &&
                    strcmp(proj, "gate_proj") == 0) {
            gate_proj_M = (int)nt->pq.M;
        }
    }
    if (!q_proj_M || !hidden || !n_layers) return -1;

    fill_llama_defaults(h);
    h->hidden_size       = hidden;
    h->num_layers        = n_layers;
    /* head_dim heuristic: 128 when hidden > 2048 (Llama-3.1-8B, larger),
     * else 64. Covers TinyLlama (2048/64=32), Llama-3.2-1B (2048/64=32),
     * Llama-3.1-8B (4096/128=32). For exotic configs, override the
     * detection by storing the right header in a future IBF v7. */
    int head_dim = (hidden > 2048) ? 128 : 64;
    h->num_heads         = q_proj_M / head_dim;
    h->head_dim          = head_dim;
    h->num_kv_heads      = v_proj_M ? (v_proj_M / head_dim) : h->num_heads;
    h->intermediate_size = gate_proj_M ? gate_proj_M : (hidden * 4);

    /* Vocab size — read from token_embedding tensor (PQ-encoded or raw fp16). */
    h->vocab_size = 32000;
    for (int i = 0; i < f->n_tensors; i++) {
        const ib_pqv2_named_tensor* nt = &f->tensors[i];
        if (strcmp(nt->name, "token_embedding") != 0) continue;
        if (nt->kind == IB_PQV2_KIND_PQV2) {
            h->vocab_size = (int)nt->pq.M;
        } else {
            /* fp16 raw: bytes / (hidden * 2) = vocab_size */
            h->vocab_size = (int)(nt->raw_size / ((size_t)hidden * 2));
        }
        break;
    }

    /* Architecture-specific overrides keyed on vocab_size signature.
     * Llama-3 family uses vocab=128256 and rope_theta=500000 with a
     * much longer max context. */
    if (h->vocab_size >= 100000) {
        h->rope_theta          = 500000.0f;
        h->max_context_length  = 131072;
    }

    /* Pretty-print name for debugging. */
    snprintf(h->name, sizeof(h->name),
             "llama-%dL-%dH-%dheads-%dkv",
             n_layers, hidden, h->num_heads, h->num_kv_heads);
    return 0;
}

static void set_pq_meta(ib_tensor_meta* t, const pqv2_t* pq) {
    memset(t, 0, sizeof(*t));
    t->pq = pq;
    t->shape[0] = (int)pq->M;
    t->shape[1] = (int)pq->N;
    t->ndim = 2;
    t->bits = -1;     /* sentinel — pq path takes over */
}

/* For raw tensors: t->offset is relative to m->weight_data which is set
 * to the mmap'd IBF v6 buffer below. */
static void set_raw_meta(ib_tensor_meta* t,
                          const void* data, const void* base,
                          size_t bytes,
                          int M, int N) {
    memset(t, 0, sizeof(*t));
    t->pq = NULL;
    t->shape[0] = M;
    t->shape[1] = N;
    t->ndim = (N > 1) ? 2 : 1;
    t->bits = 16;
    t->offset = (size_t)((const uint8_t*)data - (const uint8_t*)base);
    t->size = bytes;
}

static inferbit_model* pqv2_load_internal(const char* path,
                                           const inferbit_config* config) {
    (void)config;
    ib_pqv2_file* f = calloc(1, sizeof(*f));
    if (!f) return NULL;
    if (ib_pqv2_file_load(path, f) != 0) {
        free(f);
        return NULL;
    }
    inferbit_model* m = calloc(1, sizeof(*m));
    if (!m) {
        ib_pqv2_file_free(f);
        free(f);
        return NULL;
    }
    if (detect_arch_from_tensors(f, &m->header) != 0) {
        fprintf(stderr, "pqv2_load: cannot detect architecture from %s\n", path);
        ib_pqv2_file_free(f);
        free(f);
        free(m);
        return NULL;
    }
    m->layers = calloc((size_t)m->header.num_layers, sizeof(ib_layer_meta));
    if (!m->layers) goto fail;

    /* m->weight_data points at the IBF v6 mmap. tensor_data() returns
     * weight_data + offset, so we encode raw-tensor file offsets
     * relative to f->_buffer. PQv2 tensors don't use tensor_data — they
     * use t->pq pointers (which already index into f->_buffer). */
    m->weight_data = f->_buffer;
    m->weight_data_size = f->_buffer_size;
    m->weight_data_mmap = (f->_is_mmap != 0);
    m->mmap_fd = f->_fd;
    m->pqv2_file_backing = f;

    for (int i = 0; i < f->n_tensors; i++) {
        const ib_pqv2_named_tensor* nt = &f->tensors[i];
        const char* n = nt->name;
        if (nt->kind == IB_PQV2_KIND_PQV2) {
            /* Layer projections: L<L>.<parent>.<proj> */
            int li;
            char parent[32], proj[32];
            if (sscanf(n, "L%d.%31[^.].%31s", &li, parent, proj) == 3 &&
                li >= 0 && li < m->header.num_layers) {
                ib_layer_meta* L = &m->layers[li];
                ib_tensor_meta* slot = NULL;
                if (strcmp(parent, "self_attn") == 0) {
                    if (strcmp(proj, "q_proj") == 0) slot = &L->q_proj;
                    else if (strcmp(proj, "k_proj") == 0) slot = &L->k_proj;
                    else if (strcmp(proj, "v_proj") == 0) slot = &L->v_proj;
                    else if (strcmp(proj, "o_proj") == 0) slot = &L->o_proj;
                } else if (strcmp(parent, "mlp") == 0) {
                    if (strcmp(proj, "gate_proj") == 0) slot = &L->gate_proj;
                    else if (strcmp(proj, "up_proj") == 0) slot = &L->up_proj;
                    else if (strcmp(proj, "down_proj") == 0) slot = &L->down_proj;
                }
                if (slot) set_pq_meta(slot, &nt->pq);
            } else if (strcmp(n, "token_embedding") == 0) {
                set_pq_meta(&m->token_embedding, &nt->pq);
            } else if (strcmp(n, "lm_head") == 0) {
                set_pq_meta(&m->output_head, &nt->pq);
            }
        } else {
            int li;
            char rest[64];
            if (sscanf(n, "L%d.%63s", &li, rest) == 2 &&
                li >= 0 && li < m->header.num_layers) {
                ib_layer_meta* L = &m->layers[li];
                if (strcmp(rest, "input_layernorm") == 0)
                    set_raw_meta(&L->input_norm, nt->raw_data, f->_buffer,
                                  nt->raw_size, m->header.hidden_size, 1);
                else if (strcmp(rest, "post_attention_layernorm") == 0)
                    set_raw_meta(&L->post_attn_norm, nt->raw_data, f->_buffer,
                                  nt->raw_size, m->header.hidden_size, 1);
            } else if (strcmp(n, "token_embedding") == 0) {
                set_raw_meta(&m->token_embedding, nt->raw_data, f->_buffer,
                              nt->raw_size, m->header.vocab_size,
                              m->header.hidden_size);
            } else if (strcmp(n, "output_norm") == 0) {
                set_raw_meta(&m->output_norm, nt->raw_data, f->_buffer,
                              nt->raw_size, m->header.hidden_size, 1);
            } else if (strcmp(n, "lm_head") == 0) {
                set_raw_meta(&m->output_head, nt->raw_data, f->_buffer,
                              nt->raw_size, m->header.vocab_size,
                              m->header.hidden_size);
            }
        }
    }

    /* Runtime state init: same as legacy ibf_load post-load path. */
    int ctx_len = m->header.max_context_length;
    int kv_dynamic = 0;
    int threads = 4;
    if (config) {
        ctx_len    = config->context_length > 0 ? config->context_length : ctx_len;
        kv_dynamic = config->kv_dynamic;
        threads    = config->threads > 0 ? config->threads : 4;
    }
    m->num_threads = threads;

    /* Path D residency mode (env-gated for now). "drive" => evict
     * indices pages after each matmul on CPU. Default 0 = RAM. */
    {
        const char *rm = getenv("IB_RESIDENCY_MODE");
        m->residency_mode = (rm && (!strcmp(rm, "drive") || !strcmp(rm, "1"))) ? 1 : 0;
    }

    if (ib_alloc_kv_caches(m, ctx_len, kv_dynamic) != 0) goto fail;
    if (ib_alloc_buffers(m) != 0) goto fail;

    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);
    m->thread_pool = ib_pool_create(threads);

    /* Pre-allocate threading scratch sized for the largest matvec
     * across all PQv2 tensors × n_threads. Reused per matvec to avoid
     * per-call aligned_alloc in the hot path. */
    uint32_t max_M = 0;
    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta *L = &m->layers[li];
        const ib_tensor_meta *slots[7] = {
            &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
            &L->gate_proj, &L->up_proj, &L->down_proj,
        };
        for (int si = 0; si < 7; si++) {
            if (slots[si]->pq && slots[si]->pq->M > max_M) max_M = slots[si]->pq->M;
        }
    }
    if (max_M > 0) {
        size_t n_threads_eff = (threads > 1) ? (size_t)threads : 1;
        size_t pool_floats = n_threads_eff * (size_t)max_M;
        size_t pool_bytes  = (pool_floats * sizeof(float) + 63) & ~(size_t)63;
        m->pqv2_thread_acc_pool = aligned_alloc(64, pool_bytes);
        m->pqv2_thread_acc_pool_floats = pool_floats;
    }

    /* Path D drive mode setup (Solution 5): allocate one shared indices
     * scratch buffer sized for the model's largest PQv2 matmul, walk
     * every PQv2 tensor and redirect pq->indices to the scratch. After
     * this, every matmul does pread(file_fd, scratch, len, offset)
     * before kernel dispatch — the kernel reads from scratch.
     *
     * Walks token_embedding + output_head + all layer projections. */
    if (m->residency_mode == 1) {
        const uint8_t *file_base = (const uint8_t *)f->_buffer;
        size_t max_idx_bytes = 0;
        /* First pass: max indices size + store original offsets. */
        const ib_tensor_meta *tslots[2 + 7 * 256];   /* head + per-layer 7 */
        int nslots = 0;
        /* token_embedding is intentionally NOT redirected: cpu_embed_lookup
         * reads emb->pq->indices directly (not via tensor_matmul) so its
         * pointer must keep pointing into the mmap region.
         * Its full indices region for a 32k-vocab model is ~32 MB; keeping
         * it RAM-resident is the right trade. output_head IS redirected
         * because it goes through tensor_matmul. */
        if (m->output_head.pq)     tslots[nslots++] = &m->output_head;
        for (int li = 0; li < m->header.num_layers; li++) {
            ib_layer_meta *L = &m->layers[li];
            const ib_tensor_meta *s7[7] = {
                &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
                &L->gate_proj, &L->up_proj, &L->down_proj,
            };
            for (int i = 0; i < 7; i++) {
                if (s7[i]->pq) tslots[nslots++] = s7[i];
            }
        }
        for (int i = 0; i < nslots; i++) {
            const pqv2_t *pq = tslots[i]->pq;
            size_t b = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
            if (b > max_idx_bytes) max_idx_bytes = b;
        }
        /* Page-align the scratch. */
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) ps = 4096;
        size_t scratch_size = (max_idx_bytes + (size_t)ps - 1) & ~((size_t)ps - 1);
        void *scratch = NULL;
        if (scratch_size > 0) scratch = aligned_alloc((size_t)ps, scratch_size);
        if (!scratch) {
            fprintf(stderr, "ib pqv2: drive mode scratch alloc failed (%zu B) — falling back to RAM mode\n",
                    scratch_size);
            m->residency_mode = 0;
        } else {
            m->drive_indices_scratch = scratch;
            m->drive_indices_scratch_size = scratch_size;
            m->drive_fd = f->_fd;
            /* Second pass: rewrite pq->indices to point at scratch.
             * The kernel always reads from this pointer; we refill via
             * pread before each matmul. */
            for (int i = 0; i < nslots; i++) {
                pqv2_t *mpq = (pqv2_t *)tslots[i]->pq;   /* cast away const */
                /* Record file offset of indices region, then redirect
                 * indices to the shared scratch. Per-matmul pread will
                 * refill scratch from the file at indices_file_offset. */
                size_t off = (const uint8_t *)mpq->indices - file_base;
                mpq->indices_file_offset = off;
                mpq->indices = (const uint8_t *)scratch;
            }
            fprintf(stderr, "ib pqv2: drive mode ON. scratch=%zu B, fd=%d, %d tensors redirected\n",
                    scratch_size, f->_fd, nslots);
        }
    }
    return m;

fail:
    if (f) { ib_pqv2_file_free(f); free(f); }
    free(m);
    return NULL;
}

inferbit_model* pqv2_or_legacy_load(const char* path,
                                      const inferbit_config* config) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;
    char magic[8];
    if (fread(magic, 1, 8, fp) != 8) { fclose(fp); return NULL; }
    fclose(fp);
    if (memcmp(magic, IB_PQV2_MAGIC, 8) == 0) {
        return pqv2_load_internal(path, config);
    }
    return ibf_load(path, config);
}
