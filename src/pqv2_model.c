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
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>

/* Forward decl for drive-mode pre-transposed sidecar builder. Defined
 * later in this file; called from the drive-mode setup. */
static int build_pretransposed_sidecar(inferbit_model *m,
                                         const uint8_t *file_base,
                                         pqv2_t **pq_list, int n_pq);

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
    m->drive_fd_pretransposed = -1;  /* set by build_pretransposed_sidecar if drive mode */
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
        m->kv_window = config->kv_window;
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
            m->drive_fd_pretransposed = -1;
            /* Second pass: rewrite pq->indices to point at scratch (CPU
             * path). Record original file offset for CPU drive_load_indices. */
            for (int i = 0; i < nslots; i++) {
                pqv2_t *mpq = (pqv2_t *)tslots[i]->pq;
                size_t off = (const uint8_t *)mpq->indices - file_base;
                mpq->indices_file_offset = off;
                mpq->indices = (const uint8_t *)scratch;
                mpq->indices_pretransposed_offset = 0;
            }
            /* Build pre-transposed sidecar for the GPU drive path. Writes
             * each tensor's indices in kernel-native [M][total] layout to
             * an unlinked tmpfile and sets indices_pretransposed_offset
             * on each pqv2_t. The GPU drive-mode preads pull bytes
             * directly to MTLBuffer scratch — NO per-matmul transpose,
             * which was the doc-35 CPU-bottleneck floor.
             *
             * CPU drive path (chunk-major in-file) unaffected: still
             * preads from drive_fd at indices_file_offset. */
            pqv2_t *mpq_list[nslots];
            for (int i = 0; i < nslots; i++) {
                mpq_list[i] = (pqv2_t *)tslots[i]->pq;
            }
            int rc = -1;
            const char *no_sidecar = getenv("IB_NO_SIDECAR");
            if (no_sidecar && no_sidecar[0] == '1') {
                fprintf(stderr, "ib pqv2: IB_NO_SIDECAR=1 — using legacy per-matmul transpose path\n");
            } else {
                /* Doc-35 feature 1: include token_embedding in the
                 * sidecar so embed_lookup can pread one row per token
                 * (1024 contiguous bytes) instead of mmap-reading 1024
                 * widely-strided bytes. Token embedding stays mmap-
                 * pointer (no redirect) — only the sidecar entry is
                 * added. embed_lookup checks sidecar_offset != 0 and
                 * preads if available. */
                pqv2_t *sidecar_list[nslots + 1];
                int sidecar_n = 0;
                for (int i = 0; i < nslots; i++) sidecar_list[sidecar_n++] = mpq_list[i];
                if (m->token_embedding.pq) {
                    sidecar_list[sidecar_n++] = (pqv2_t *)m->token_embedding.pq;
                    /* Also set indices_file_offset for the embed so the
                     * sidecar builder can pread the source bytes. */
                    pqv2_t *emb_pq = (pqv2_t *)m->token_embedding.pq;
                    if (emb_pq->indices_file_offset == 0) {
                        emb_pq->indices_file_offset =
                            (size_t)((const uint8_t *)emb_pq->indices - file_base);
                    }
                }
                rc = build_pretransposed_sidecar(m, file_base, sidecar_list, sidecar_n);
                if (rc != 0) {
                    fprintf(stderr, "ib pqv2: pre-transposed sidecar build failed (rc=%d); GPU drive will use per-matmul transpose fallback\n", rc);
                }
            }
            fprintf(stderr, "ib pqv2: drive mode ON. scratch=%zu B, fd=%d, sidecar_fd=%d, %d tensors\n",
                    scratch_size, m->drive_fd, m->drive_fd_pretransposed, nslots);
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

/* Pre-transposed sidecar for drive mode.
 *
 * Each PQv2 tensor's indices are stored on disk in [n_chunks][n_subchunks][M]
 * (chunk-major, the legacy format). Both GPU upload and drive-mode preads
 * have to transpose this to [M][total = n_chunks * n_subchunks] before the
 * kernel can consume it. Per-matmul transpose is the CPU-bottleneck floor
 * that caps drive-mode tok/s at ~2.3 tok/s (doc 35).
 *
 * This function does the transpose ONCE at load and writes results to an
 * unlinked tmpfile in [M][total] order. drive_load_pq_idx_to_slot then
 * preads bytes that are already in kernel-native layout — zero-copy into
 * the MTLBuffer scratch.
 *
 * Sets m->drive_fd to the sidecar fd and each pq->indices_file_offset to
 * its position in the sidecar.
 *
 * Cost: one-time read of full indices region + one-time write of same.
 * For TinyLlama: ~528 MB indices × 2 = ~1 GB I/O ≈ 350 ms one-time.
 *
 * Returns 0 on success. Caller falls back to in-file drive mode on
 * non-zero return. */
static int build_pretransposed_sidecar(inferbit_model *m,
                                         const uint8_t *file_base,
                                         pqv2_t **pq_list, int n_pq)
{
    if (!m || !file_base || !pq_list || n_pq <= 0) return -1;

    /* Open + unlink tmpfile so it auto-cleans on process exit. */
    char path[] = "/tmp/inferbit-pretposed-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) return -2;
    if (unlink(path) != 0) {
        /* benign — just means we leak the path until process exits */
    }
#ifdef F_NOCACHE
    /* macOS: bypass UBC for the sidecar — we only ever pread from it,
     * never want it accumulating in page cache. */
    (void)fcntl(fd, F_NOCACHE, 1);
#endif

    /* Pre-pad with one byte so all tensor offsets are ≥ 1. The
     * GPU upload code treats pq_drive_file_offset == 0 as "not
     * streamed"; the first tensor would otherwise land at offset 0
     * and be incorrectly skipped. */
    {
        const uint8_t pad = 0;
        if (write(fd, &pad, 1) != 1) {
            close(fd); return -3;
        }
    }
    /* Walk each tensor, transpose its indices to the sidecar.
     * Layout: source [c][s][m] → dest [m][c*ns+s]. */
    off_t sidecar_off = 1;
    /* Reuse one staging buffer sized for the largest tensor's indices. */
    size_t max_bytes = 0;
    for (int i = 0; i < n_pq; i++) {
        size_t b = (size_t)pq_list[i]->M * (pq_list[i]->N / pq_list[i]->G)
                   * pq_list[i]->n_subchunks;
        if (b > max_bytes) max_bytes = b;
    }
    uint8_t *staging = (uint8_t *)malloc(max_bytes);
    if (!staging) { close(fd); return -3; }

    /* We need a second staging buffer to hold the original chunk-major
     * bytes we pread from the source file. Mmap-read would touch every
     * page into UBC, defeating drive-mode RAM savings — pread keeps
     * the source-file pages out of cache (F_NOCACHE is set on the source
     * fd in drive mode). */
    uint8_t *read_buf = (uint8_t *)malloc(max_bytes);
    if (!read_buf) { free(staging); close(fd); return -5; }

    for (int i = 0; i < n_pq; i++) {
        pqv2_t *pq = pq_list[i];
        uint32_t M = pq->M;
        uint32_t nc = pq->N / pq->G;
        uint32_t ns = pq->n_subchunks;
        uint32_t total = nc * ns;
        size_t idx_bytes = (size_t)M * total;

        /* pread the original chunk-major bytes from the source IBF.
         * pq->indices_file_offset was set just above to point at the
         * indices region in the source file. */
        size_t got = 0;
        off_t src_off = (off_t)pq->indices_file_offset;
        while (got < idx_bytes) {
            ssize_t r = pread(m->drive_fd, read_buf + got,
                              idx_bytes - got, src_off + (off_t)got);
            if (r <= 0) {
                if (r == -1 && errno == EINTR) continue;
                free(read_buf); free(staging); close(fd); return -6;
            }
            got += (size_t)r;
        }
        const uint8_t *src = read_buf;
        /* Transpose into staging: dst[m * total + c*ns + s] = src[(c*ns+s)*M + m]. */
        for (uint32_t m_ = 0; m_ < M; m_++) {
            uint8_t *row = staging + (size_t)m_ * total;
            for (uint32_t c = 0; c < nc; c++) {
                for (uint32_t s = 0; s < ns; s++) {
                    row[c * ns + s] = src[((size_t)c * ns + s) * M + m_];
                }
            }
        }
        /* Write transposed bytes to sidecar at current offset. */
        size_t written = 0;
        while (written < idx_bytes) {
            ssize_t w = write(fd, staging + written, idx_bytes - written);
            if (w <= 0) {
                if (w == -1 && errno == EINTR) continue;
                free(staging); close(fd); return -4;
            }
            written += (size_t)w;
        }
        /* Record this tensor's sidecar offset for runtime GPU preads. */
        pq->indices_pretransposed_offset = (size_t)sidecar_off;
        sidecar_off += (off_t)idx_bytes;
    }

    free(staging);
    free(read_buf);
    m->drive_fd_pretransposed = fd;
    fprintf(stderr, "ib pqv2: pre-transposed sidecar built (%lld B, %d tensors) — GPU drive preads skip transpose\n",
            (long long)sidecar_off, n_pq);
    return 0;
}
