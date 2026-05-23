/* pqv2_model.c — load IBF v6 PQv2 file into an inferbit_model.
 *
 * Architecture is hardcoded to TinyLlama-1.1B for now. Future:
 * extend IBF v6 header with a config blob and read those.
 */
#include "inferbit_internal.h"
#include "pqv2_format.h"
#include "platform.h"   /* ib_close/ib_write/pread/mkstemp/sysconf shims */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

/* Drive-mode peak-RAM page cap default (MB), used when IB_DRIVE_PAGE_MB
 * is unset or invalid. 8 MB caps the in-focus index bytes per scratch
 * slot at 8 MB → worst-case in-focus index RAM = 2 × 8 = 16 MB (the
 * 2-slot prefetch ring), versus ~33 MB per slot for a 32k-vocab lm_head
 * (65 MB doubled) before the cap. */
#ifndef IB_DRIVE_PAGE_MB_DEFAULT
#define IB_DRIVE_PAGE_MB_DEFAULT 8
#endif

/* Forward decl for drive-mode pre-transposed sidecar builder. Defined
 * later in this file; called from the drive-mode setup. */
static int build_pretransposed_sidecar(inferbit_model *m,
                                         const uint8_t *file_base,
                                         pqv2_t **pq_list, int n_pq);

/* Strict parser for MoME expert tensor names of the form
 *   "L<li>.mlp.<proj>.expert<eidx>"
 * with NO trailing characters after <eidx>.
 *
 * Background — round-3 C1 precedent in set_pq_meta: a sscanf using
 * "L%d.mlp.router" returned 1 (number of converted specifiers) as long
 * as %d converted; it did NOT report whether the trailing literal
 * ".mlp.router" matched, so any "L<N>.…" silently hit the router branch.
 * The same pitfall applies to "L%d.mlp.%[^.].expert%d": %d at the tail
 * stops at the first non-digit, so "expert3.foo" or "expert3bar" still
 * yield 3 conversions with eidx=3 and trailing garbage silently dropped.
 *
 * Fix mirrors the round-3 pattern: capture the tail with %63s and
 * validate the entire suffix structure ourselves. proj_out must point
 * at a buffer of at least 32 bytes. Returns 1 on strict match, 0
 * otherwise. */
static int parse_mome_expert_name(const char *name,
                                   int *li_out,
                                   char proj_out[32],
                                   int *eidx_out) {
    int li = 0;
    char proj[32];
    char tail[64];
    /* "L%d.mlp.%31[^.].%63s" — last %63s captures "expert<N>" with any
     * (illegal) trailing characters. We then enforce structure on tail. */
    if (sscanf(name, "L%d.mlp.%31[^.].%63s", &li, proj, tail) != 3) return 0;
    /* tail must be exactly "expert" + 1+ digits, nothing else. */
    if (strncmp(tail, "expert", 6) != 0) return 0;
    const char *digits = tail + 6;
    if (*digits == '\0') return 0;
    int eidx = 0;
    for (const char *p = digits; *p; p++) {
        if (*p < '0' || *p > '9') return 0;   /* trailing non-digit → reject */
        eidx = eidx * 10 + (*p - '0');
        if (eidx < 0) return 0;               /* overflow guard */
    }
    *li_out = li;
    memcpy(proj_out, proj, sizeof(proj));
    *eidx_out = eidx;
    return 1;
}

#define IB_PQV2_MAGIC "IBFV6PQ2"

extern inferbit_model* ibf_load(const char* path, const inferbit_config* config);
/* Runtime-state helpers exposed from ibf_loader.c. */
int ib_alloc_kv_caches(inferbit_model* model, int context_length, int dynamic);
int ib_alloc_buffers(inferbit_model* model);
/* Stage 3b: kv_format → kv_bits bridge, shared with ibf_loader.c. */
void ib_apply_kv_format(inferbit_model* model, const inferbit_config* config);
/* Perf: pre-decode static fp16 scale/norm buffers, shared with ibf_loader.c. */
void ib_cache_model_static_fp32(inferbit_model* m);
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
    /* Find L0.self_attn.q_proj to read q_proj_M and hidden.
     * Tensors may arrive as either IB_PQV2_KIND_PQV2 (FFN, o_proj, etc.)
     * or IB_PQV2_KIND_RAW_FP16 (attention Q/K/V when the encoder kept
     * them uncompressed). Accept both so this code stays orthogonal to
     * the encoder's per-tensor format choices. */
    int q_proj_M = 0, hidden = 0, n_layers = 0;
    int v_proj_M = 0;
    int gate_proj_M = 0;
    int gate_proj_expert_M = 0;   /* MoME: per-expert M (= total M / K) */
    int gate_proj_expert_count = 0;
    for (int i = 0; i < f->n_tensors; i++) {
        const ib_pqv2_named_tensor* nt = &f->tensors[i];
        if (nt->kind != IB_PQV2_KIND_PQV2 &&
            nt->kind != IB_PQV2_KIND_RAW_FP16) continue;
        int li;
        char parent[32], proj[32];
        if (sscanf(nt->name, "L%d.%31[^.].%31s", &li, parent, proj) != 3) continue;
        if (li + 1 > n_layers) n_layers = li + 1;
        /* Read [M, N] from the appropriate field based on kind. */
        int t_M = (nt->kind == IB_PQV2_KIND_PQV2)
                  ? (int)nt->pq.M
                  : (nt->ndim >= 1 ? nt->shape[0] : 0);
        int t_N = (nt->kind == IB_PQV2_KIND_PQV2)
                  ? (int)nt->pq.N
                  : (nt->ndim >= 2 ? nt->shape[1] : 0);
        if (li == 0 && strcmp(parent, "self_attn") == 0) {
            if (strcmp(proj, "q_proj") == 0) {
                q_proj_M = t_M;
                hidden   = t_N;
            } else if (strcmp(proj, "v_proj") == 0) {
                v_proj_M = t_M;
            }
        } else if (li == 0 && strcmp(parent, "mlp") == 0 &&
                    strcmp(proj, "gate_proj") == 0) {
            gate_proj_M = t_M;
        }
        /* MoME variant: when the encoder splits gate_proj into K expert
         * sub-tensors, the legacy `L0.mlp.gate_proj` tensor doesn't
         * exist. Reconstruct intermediate_size by multiplying per-
         * expert M by the count of expertN tensors observed on layer
         * 0. We rely on this scan running over every tensor; the count
         * tracks how many `L0.mlp.gate_proj.expert{e}` we saw. */
        {
            /* Round-3 C1 precedent: sscanf("…expert%d") accepts trailing
             * garbage after %d (e.g. "expert3.foo"), silently corrupting
             * eidx. Use strict parser that rejects any non-digit tail. */
            int li2, eidx2;
            char mproj2[32];
            if (parse_mome_expert_name(nt->name, &li2, mproj2, &eidx2) &&
                li2 == 0 && strcmp(mproj2, "gate_proj") == 0) {
                if (gate_proj_expert_M == 0) gate_proj_expert_M = t_M;
                gate_proj_expert_count++;
                (void)t_N;
            }
        }
    }
    /* Promote MoME-split FFN to a flat intermediate_size for header
     * detection. The hidden/num_heads code below doesn't depend on
     * MoME being on, so we just synthesise gate_proj_M when only
     * expert sub-tensors are present. */
    if (gate_proj_M == 0 && gate_proj_expert_count > 0) {
        gate_proj_M = gate_proj_expert_M * gate_proj_expert_count;
    }
    if (!q_proj_M || !hidden || !n_layers) {
        fprintf(stderr,
                "[N15] detect_arch: FAILED — q_proj_M=%d hidden=%d n_layers=%d "
                "(L0.self_attn.q_proj not found or shape unreadable; "
                "gate_proj_M=%d expert_M=%d expert_count=%d)\n",
                q_proj_M, hidden, n_layers,
                gate_proj_M, gate_proj_expert_M, gate_proj_expert_count);
        return -1;
    }
    fprintf(stderr,
            "[N15] detect_arch: OK — hidden=%d n_layers=%d q_proj_M=%d "
            "v_proj_M=%d gate_proj_M=%d (expert_M=%d × count=%d)\n",
            hidden, n_layers, q_proj_M, v_proj_M, gate_proj_M,
            gate_proj_expert_M, gate_proj_expert_count);

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

static void set_pq_meta(ib_tensor_meta* t, const ib_pqv2_named_tensor* nt) {
    memset(t, 0, sizeof(*t));
    t->pq = &nt->pq;
    t->shape[0] = (int)nt->pq.M;
    t->shape[1] = (int)nt->pq.N;
    t->ndim = 2;
    t->bits = -1;     /* sentinel — pq path takes over */
    /* Stage 5c — propagate the on-disk residency hint to the loader's
     * per-tensor meta so other subsystems (drive-mode setup, future
     * mlock policy) can read it. */
    t->residency_hint = nt->residency_hint;
    /* Stage 5b — record the on-disk format choice. pq->l2_kind drives
     * the runtime kernel dispatch already (0 = flat, 2 = pyramid); this
     * field exists for diagnostics + tools that want to surface the
     * mixed-format inventory without reaching into the pq descriptor. */
    t->tensor_format = (nt->pq.l2_kind == 2)
                         ? (int)INFERBIT_CONVERT_PQV2_PYRAMID
                         : (int)INFERBIT_CONVERT_PQV2_FLAT;
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
    /* Raw tensors carry no on-disk hint; default to AUTO. They're norms
     * and the MoME router — small, hot-path tensors. The loader's
     * heuristic in pqv2_load_internal can promote norms to RAM after the
     * fact via the global IB_RESIDENCY_RAM_LAYERS override. */
    t->residency_hint = (int)INFERBIT_RESIDENCY_AUTO;
    t->tensor_format = (int)INFERBIT_CONVERT_INT4;  /* "raw fp16" sentinel */
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
    /* Safety: always-precompute fp32 codebooks so cb_fp32 / l2_cb_fp32
     * are NEVER NULL at matvec time, regardless of residency mode. NULL
     * forces per-matvec re-decode (slower kernel branch — one source of
     * pyramid RAM/drive PPL divergence). format.c is the usual source;
     * this re-walk is the belt-and-braces guarantee. */
    for (int i = 0; i < f->n_tensors; i++) {
        ib_pqv2_named_tensor *nt = &f->tensors[i];
        if (nt->kind != IB_PQV2_KIND_PQV2) continue;
        pqv2_t *pq = &nt->pq;
        size_t cb_n = (size_t)pq->n_subchunks * pq->K * pq->half;
        if (!pq->cb_fp32 && pq->cb_q && pq->cb_scale && cb_n) {
            float *cb = (float *)malloc(cb_n * sizeof(float));
            if (cb) { for (size_t j = 0; j < cb_n; j++)
                cb[j] = (float)pq->cb_q[j] * pqv2_h2f(pq->cb_scale[j / pq->half]);
                pq->cb_fp32 = cb; }
        }
        if (pq->l2_kind == 2 && !pq->l2_cb_fp32 && pq->l2_cb_q && pq->l2_cb_scale) {
            size_t l2_n = (size_t)pq->n_subchunks * pq->l2_K * pq->half;
            float *cb = (float *)malloc(l2_n * sizeof(float));
            if (cb) { for (size_t j = 0; j < l2_n; j++)
                cb[j] = (float)pq->l2_cb_q[j] * pqv2_h2f(pq->l2_cb_scale[j / pq->half]);
                pq->l2_cb_fp32 = cb; }
        }
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
    /* Default every layer to mome_experts = 1 (= no MoME). The MoME
     * scan below bumps this on layers that ship `.expert{e}` tensors. */
    for (int li = 0; li < m->header.num_layers; li++) {
        m->layers[li].mome_experts = 1;
        m->layers[li].gate_proj_experts = NULL;
        m->layers[li].up_proj_experts   = NULL;
        m->layers[li].down_proj_experts = NULL;
        memset(&m->layers[li].router, 0, sizeof(m->layers[li].router));
    }

    /* ── MoME detection pass (Stage 3a) ──────────────────────────────
     *
     * Scan tensor names twice: first to count per-layer max-expert-id
     * (so we know how many slots to allocate); then the main loop
     * populates the slots. Naming convention is documented in
     * pqv2_format.h. A layer that has any `Lk.mlp.<proj>.expert{e}`
     * tensor is treated as MoME; the highest e + 1 becomes
     * mome_experts. Layers without expert tensors stay
     * mome_experts == 1 and use the legacy gate/up/down slots. */
    for (int i = 0; i < f->n_tensors; i++) {
        const ib_pqv2_named_tensor* nt = &f->tensors[i];
        int li, eidx;
        char proj[32];
        /* Round-3 C1 precedent: %d at tail of sscanf accepts trailing
         * garbage, so "expert3.foo" silently parses eidx=3. Strict
         * parser rejects any non-digit tail. */
        if (parse_mome_expert_name(nt->name, &li, proj, &eidx) &&
            li >= 0 && li < m->header.num_layers && eidx >= 0) {
            ib_layer_meta* L = &m->layers[li];
            int want = eidx + 1;
            if (want > L->mome_experts) L->mome_experts = want;
        }
    }
    /* Allocate per-layer expert arrays once the count is known. */
    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta* L = &m->layers[li];
        if (L->mome_experts <= 1) continue;
        size_t bytes = (size_t)L->mome_experts * sizeof(ib_tensor_meta);
        L->gate_proj_experts = (ib_tensor_meta *)calloc(1, bytes);
        L->up_proj_experts   = (ib_tensor_meta *)calloc(1, bytes);
        L->down_proj_experts = (ib_tensor_meta *)calloc(1, bytes);
        if (!L->gate_proj_experts || !L->up_proj_experts ||
            !L->down_proj_experts) {
            fprintf(stderr,
                    "pqv2_load: oom allocating MoME expert slots for layer %d (K=%d)\n",
                    li, L->mome_experts);
            goto fail;
        }
    }

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
            /* MoME expert slot: L<L>.mlp.<proj>.expert<E>. Matched BEFORE
             * the generic parent.proj parser so the expert suffix wins.
             * Per-expert sub-tensor is a normal PQv2 tensor — we just
             * route it into the right slot in the K-stacked expert
             * array on the owning layer. */
            {
                /* Round-3 C1 precedent: strict tail validation. The
                 * previous sscanf("…expert%d") accepted trailing junk
                 * because %d stops at first non-digit and sscanf
                 * doesn't report unmatched literal tails — would route
                 * garbage tensor names into a real expert slot. */
                int li, eidx;
                char mproj[32];
                if (parse_mome_expert_name(n, &li, mproj, &eidx) &&
                    li >= 0 && li < m->header.num_layers && eidx >= 0) {
                    ib_layer_meta* L = &m->layers[li];
                    if (L->mome_experts > 1 && eidx < L->mome_experts) {
                        ib_tensor_meta* slot = NULL;
                        if (strcmp(mproj, "gate_proj") == 0)
                            slot = &L->gate_proj_experts[eidx];
                        else if (strcmp(mproj, "up_proj") == 0)
                            slot = &L->up_proj_experts[eidx];
                        else if (strcmp(mproj, "down_proj") == 0)
                            slot = &L->down_proj_experts[eidx];
                        if (slot) {
                            set_pq_meta(slot, nt);
                            continue;
                        }
                    }
                }
            }
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
                if (slot) set_pq_meta(slot, nt);
            } else if (strcmp(n, "token_embedding") == 0) {
                set_pq_meta(&m->token_embedding, nt);
            } else if (strcmp(n, "lm_head") == 0) {
                set_pq_meta(&m->output_head, nt);
            }
        } else {
            int li;
            char rest[64];
            /* MoME router tensor: L<L>.mlp.router, raw fp16 [K, hidden].
             * Matched first so the generic parent.proj parser below
             * doesn't grab it.
             *
             * NOTE: scanf("L%d.mlp.router") returns 1 (number of converted
             * specifiers) as long as %d converts — it does NOT report
             * whether the trailing literal ".mlp.router" matched. So
             * `L0.input_layernorm` and any other `L<N>.…` name would
             * spuriously hit this branch, clobbering router with norm
             * metadata. Verify the full suffix with strcmp instead. */
            {
                int li_r;
                char tail[64];
                if (sscanf(n, "L%d.%63s", &li_r, tail) == 2 &&
                    li_r >= 0 && li_r < m->header.num_layers &&
                    strcmp(tail, "mlp.router") == 0) {
                    ib_layer_meta* L = &m->layers[li_r];
                    if (L->mome_experts > 1) {
                        int rows = nt->ndim >= 1 ? nt->shape[0] : 0;
                        int cols = nt->ndim >= 2 ? nt->shape[1] : 1;
                        set_raw_meta(&L->router, nt->raw_data, f->_buffer,
                                      nt->raw_size, rows, cols);
                        continue;
                    }
                }
            }
            /* First try parent.proj form (e.g. self_attn.q_proj) — needed
             * when the encoder stores attention QKV (or other layer
             * projections) as raw fp16 rather than PQv2. The IBF v6 PQv2
             * format permits this for tensors whose PQv2 kernels aren't
             * implemented (Q/K/V on the current Metal/CPU dispatch). */
            {
                char parent[32], proj[32];
                if (sscanf(n, "L%d.%31[^.].%31s", &li, parent, proj) == 3 &&
                    li >= 0 && li < m->header.num_layers) {
                    ib_layer_meta* L = &m->layers[li];
                    ib_tensor_meta* slot = NULL;
                    int rows = nt->ndim >= 1 ? nt->shape[0] : 0;
                    int cols = nt->ndim >= 2 ? nt->shape[1] : 1;
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
                    if (slot) {
                        set_raw_meta(slot, nt->raw_data, f->_buffer,
                                      nt->raw_size, rows, cols);
                        continue;
                    }
                }
            }
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

    /* Tied-embedding fallback. Models like Llama-3.2-1B/3B don't emit a
     * separate lm_head tensor — config.json says tie_word_embeddings=true
     * and the runtime is expected to reuse token_embedding for the output
     * head. Without this, tensor_matmul(output_head) finds pq==NULL and
     * bits==0 and silently no-ops, leaving logits uninitialized.
     *
     * Mirror whatever kind token_embedding ended up as: PQv2 → reuse the
     * same pq descriptor; raw fp16 → reuse the raw bytes. */
    if (!m->output_head.pq && m->output_head.bits == 0 &&
        (m->token_embedding.pq || m->token_embedding.bits != 0)) {
        m->output_head = m->token_embedding;
        m->header.tie_word_embeddings = true;
    }

    /* Runtime state init: same as legacy ibf_load post-load path. */
    int ctx_len = m->header.max_context_length;
    int kv_dynamic = 0;
    int threads = ib_hardware_concurrency();
    if (config) {
        ctx_len    = config->context_length > 0 ? config->context_length : ctx_len;
        kv_dynamic = config->kv_dynamic;
        threads    = config->threads > 0 ? config->threads : ib_hardware_concurrency();
        m->kv_window = config->kv_window;
    }
    m->num_threads = threads;

    /* Path D residency mode (env-gated for now). "drive" => evict
     * indices pages after each matmul on CPU. Default 0 = RAM. */
    {
        const char *rm = getenv("IB_RESIDENCY_MODE");
        m->residency_mode = (rm && (!strcmp(rm, "drive") || !strcmp(rm, "1"))) ? 1 : 0;
    }

    /* Stage 5c — runtime promotion + diagnostic of per-tensor residency
     * hints. IB_RESIDENCY_RAM_LAYERS=N promotes the first N layers'
     * AUTO/DRIVE hints to RAM at load time (for the "always-hot first
     * layers" decode pattern). IB_PQV2_TRACE=1 dumps each tensor's hint
     * + on-disk format to stderr.
     *
     * v1 scope: this is diagnostic + state-propagation only; the
     * mlock-on-RAM / madvise-on-DRIVE enforcement is wired by the
     * existing drive_mode block below (which already handles the
     * indices-streaming policy). Honoring per-tensor RAM-pinning under
     * IB_RESIDENCY_MODE=drive is a follow-up.
     *
     * Iterates token_embedding + output_head + all layer projections,
     * including MoME expert slots when present. */
    int ram_layers_override = -1;
    {
        const char *e = getenv("IB_RESIDENCY_RAM_LAYERS");
        if (e && e[0]) {
            int v = atoi(e);
            if (v >= 0 && v < 100000) ram_layers_override = v;
        }
    }
    const int pq_trace =
        (getenv("IB_PQV2_TRACE") != NULL &&
         getenv("IB_PQV2_TRACE")[0] != '\0' &&
         getenv("IB_PQV2_TRACE")[0] != '0');
    {
        ib_tensor_meta *globals[2] = { &m->token_embedding, &m->output_head };
        const char *gnames[2] = { "token_embedding", "lm_head" };
        for (int gi = 0; gi < 2; gi++) {
            ib_tensor_meta *t = globals[gi];
            if (!t->pq && t->bits == 0) continue;
            /* Embedding / lm_head are always considered "early/hot":
             * promote AUTO → RAM regardless of ram_layers_override.
             * DRIVE-tagged tensors (explicit by the encoder) stay DRIVE. */
            if (t->residency_hint == (int)INFERBIT_RESIDENCY_AUTO) {
                t->residency_hint = (int)INFERBIT_RESIDENCY_RAM;
            }
            if (pq_trace) {
                fprintf(stderr,
                        "[pqv2] residency: %-20s fmt=%d hint=%d\n",
                        gnames[gi], t->tensor_format, t->residency_hint);
            }
        }
        for (int li = 0; li < m->header.num_layers; li++) {
            ib_layer_meta *L = &m->layers[li];
            int promote_to_ram =
                (ram_layers_override >= 0 && li < ram_layers_override);
            ib_tensor_meta *slots[7] = {
                &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
                &L->gate_proj, &L->up_proj, &L->down_proj,
            };
            const char *sn[7] = {
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj",
            };
            for (int si = 0; si < 7; si++) {
                ib_tensor_meta *t = slots[si];
                if (!t->pq && t->bits == 0) continue;
                if (promote_to_ram &&
                    t->residency_hint != (int)INFERBIT_RESIDENCY_DRIVE) {
                    t->residency_hint = (int)INFERBIT_RESIDENCY_RAM;
                }
                if (pq_trace) {
                    fprintf(stderr,
                            "[pqv2] residency: L%d.%-10s fmt=%d hint=%d\n",
                            li, sn[si], t->tensor_format, t->residency_hint);
                }
            }
            /* MoME experts: same promotion rule by layer index. */
            if (L->mome_experts > 1) {
                ib_tensor_meta *exps[3] = {
                    L->gate_proj_experts, L->up_proj_experts, L->down_proj_experts,
                };
                for (int xi = 0; xi < 3; xi++) {
                    if (!exps[xi]) continue;
                    for (int e = 0; e < L->mome_experts; e++) {
                        ib_tensor_meta *t = &exps[xi][e];
                        if (!t->pq && t->bits == 0) continue;
                        if (promote_to_ram &&
                            t->residency_hint != (int)INFERBIT_RESIDENCY_DRIVE) {
                            t->residency_hint = (int)INFERBIT_RESIDENCY_RAM;
                        }
                    }
                }
            }
        }
    }

    /* Stage 3b: resolve kv_format → kv_bits before KV alloc. PQ8
     * currently falls back to INT8 (see ib_apply_kv_format). */
    ib_apply_kv_format(m, config);

    if (ib_alloc_kv_caches(m, ctx_len, kv_dynamic) != 0) goto fail;
    if (ib_alloc_buffers(m) != 0) goto fail;

    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);
    m->thread_pool = ib_pool_create(threads);

    /* Pre-allocate threading scratch sized for the largest matvec
     * across all PQv2 tensors × n_threads. Reused per matvec to avoid
     * per-call aligned_alloc in the hot path. */
    uint32_t max_M = 0;
    int has_any_l2 = 0;
    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta *L = &m->layers[li];
        const ib_tensor_meta *slots[7] = {
            &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
            &L->gate_proj, &L->up_proj, &L->down_proj,
        };
        for (int si = 0; si < 7; si++) {
            if (slots[si]->pq) {
                if (slots[si]->pq->M > max_M) max_M = slots[si]->pq->M;
                if (slots[si]->pq->l2_kind == 2) has_any_l2 = 1;
            }
        }
    }
    /* output_head also goes through tensor_matmul → pqv2_threaded_matvec_k256,
     * so include it when sizing the scratch and probing for l2_kind. */
    if (m->output_head.pq) {
        if (m->output_head.pq->M > max_M) max_M = m->output_head.pq->M;
        if (m->output_head.pq->l2_kind == 2) has_any_l2 = 1;
    }
    if (max_M > 0) {
        size_t n_threads_eff = (threads > 1) ? (size_t)threads : 1;
        /* Size the acc pool for the LARGER of {n_threads, batched-paged B}
         * column-slabs of max_M. The batched paged matvec
         * (drive_paged_matvec with B>1) streams each lane-group ONCE and
         * accumulates all B positions into B contiguous M-float slabs — that
         * read-once/compute-B amortisation is what makes the speculative
         * verify cheap in drive mode. It caps B at 8 (see tensor_matmul_batch),
         * so reserve >= 8 slabs. The paged path is single-threaded, so it
         * never overlaps the n_threads use of the same pool. Extra cost is
         * tiny: (8 - n_threads) * max_M floats. */
        size_t cols = n_threads_eff < 8 ? 8 : n_threads_eff;
        size_t pool_floats = cols * (size_t)max_M;
        size_t pool_bytes  = (pool_floats * sizeof(float) + 63) & ~(size_t)63;
        m->pqv2_thread_acc_pool = aligned_alloc(64, pool_bytes);
        m->pqv2_thread_acc_pool_floats = pool_floats;
        /* Companion L2 scratch — same shape, only on pyramid models. Avoids
         * ~88 aligned_alloc/free per decode token on pyramid checkpoints. */
        if (has_any_l2) {
            m->pqv2_thread_acc_l2_pool = aligned_alloc(64, pool_bytes);
            m->pqv2_thread_acc_l2_pool_floats = pool_floats;
        }
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
        /* First pass: max indices size + store original offsets. The slot
         * list holds output_head + per-layer {q,k,v,o,gate,up,down} and,
         * for MoME layers (mome_experts > 1), each layer's gate/up/down
         * expert sub-tensors. Capacity is computed up-front so the heap
         * array never overflows (MoME multiplies the per-layer count by
         * 3*K, well past the old fixed 7-per-layer bound). */
        int slot_cap = 1;   /* output_head */
        for (int li = 0; li < m->header.num_layers; li++) {
            ib_layer_meta *L = &m->layers[li];
            slot_cap += 7;
            if (L->mome_experts > 1) slot_cap += 3 * L->mome_experts;
        }
        const ib_tensor_meta **tslots =
            (const ib_tensor_meta **)malloc((size_t)slot_cap * sizeof(*tslots));
        if (!tslots) {
            fprintf(stderr, "ib pqv2: drive mode tslots alloc failed — falling back to RAM mode\n");
            m->residency_mode = 0;
        } else {
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
            /* MoME experts: gate/up/down per expert. Add to the drive walk
             * so each gets indices_file_offset honored, contributes to
             * scratch sizing (under the cap), and enters drive_pq_order
             * for prefetch. The base gate/up/down_proj above are unused
             * when mome_experts > 1, but harmless to include (their pq is
             * a zeroed struct → pq == NULL → skipped). */
            if (L->mome_experts > 1 &&
                L->gate_proj_experts && L->up_proj_experts &&
                L->down_proj_experts) {
                for (int e = 0; e < L->mome_experts; e++) {
                    if (L->gate_proj_experts[e].pq)
                        tslots[nslots++] = &L->gate_proj_experts[e];
                    if (L->up_proj_experts[e].pq)
                        tslots[nslots++] = &L->up_proj_experts[e];
                    if (L->down_proj_experts[e].pq)
                        tslots[nslots++] = &L->down_proj_experts[e];
                }
            }
        }
        size_t max_l2_idx_bytes = 0;
        uint32_t max_drive_M = 0;
        for (int i = 0; i < nslots; i++) {
            const pqv2_t *pq = tslots[i]->pq;
            size_t b = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
            if (b > max_idx_bytes) max_idx_bytes = b;
            if (pq->M > max_drive_M) max_drive_M = pq->M;
            /* Goal C3 — compute L2 indices size for pyramid tensors.
             * Goal N36 fix: branch on l2_idx_bits (4/6/8) via the shared
             * kernel helper. The 4-bit packing is ceil(M/2) per row, not the
             * 8-bit M-per-row size that this previously fell through to —
             * the wrong (larger) size sized the scratch generously but the
             * matching pread over-read, causing the l2k16 drive slowdown. */
            if (pq->l2_kind == 2 && pq->l2_indices) {
                size_t l2b = pqv2_l2_total_index_bytes(pq);
                if (l2b > max_l2_idx_bytes) max_l2_idx_bytes = l2b;
            }
        }
        /* Page-align the scratch. */
        long ps = sysconf(_SC_PAGESIZE);
        if (ps <= 0) ps = 4096;
        /* ── Peak-RAM page cap ───────────────────────────────────────
         * IB_DRIVE_PAGE_MB sets the per-slot in-focus index cap (the
         * user's RAM dial: smaller cap → lower peak RAM, more preads).
         * Default IB_DRIVE_PAGE_MB_DEFAULT MB when unset/invalid. When
         * the largest tensor already fits under the cap, scratch keeps
         * the smaller whole-tensor size (no behavior change). Tensors
         * larger than the cap are paged in lane-groups by forward.c.
         *
         * The L1 cap must hold at least ONE full lane (M bytes) so a
         * single (c,s) lane always fits — clamp up to M bytes if a tiny
         * IB_DRIVE_PAGE_MB was given. The L2 cap is scaled by the same
         * lane-group count so L1 and L2 page in lockstep. */
        size_t page_cap = (size_t)IB_DRIVE_PAGE_MB_DEFAULT * 1024u * 1024u;
        {
            const char *e = getenv("IB_DRIVE_PAGE_MB");
            if (e && e[0]) {
                char *endp = NULL;
                long mb = strtol(e, &endp, 10);
                if (endp != e && mb > 0)
                    page_cap = (size_t)mb * 1024u * 1024u;
            }
        }
        /* Page-align the cap (round up to page size). */
        page_cap = (page_cap + (size_t)ps - 1) & ~((size_t)ps - 1);
        m->drive_page_bytes = page_cap;
        /* L1 cap floor: at least one full (c,s) lane = max_drive_M bytes,
         * page-aligned, so the smallest paged group is always one lane. */
        size_t l1_lane_floor = ((size_t)max_drive_M + (size_t)ps - 1)
                               & ~((size_t)ps - 1);
        size_t l1_cap_eff = page_cap;
        if (l1_cap_eff < l1_lane_floor) l1_cap_eff = l1_lane_floor;
        /* ── Paged-matmul accumulator availability check ─────────────
         * forward.c::drive_paged_matvec accumulates each tensor's paged
         * partials into the model-scope threaded-matmul acc pools
         * (pqv2_thread_acc_pool / _l2_pool, sized n_threads×max_M ≥ max_M
         * and freed in model.c) — REQUIRED for a correct paged result,
         * and the L2 pool is required to avoid dropping a pyramid
         * residual. Those pools are allocated above when max_M > 0 (and
         * the L2 pool when has_any_l2). If the required pool is missing
         * (alloc failed earlier), DISABLE the cap (l1_cap_eff =
         * whole-tensor) so every tensor fits a slot and the unchanged
         * whole-tensor path runs — never produce a partial/wrong result. */
        int need_paging = (max_idx_bytes > l1_cap_eff) && (max_drive_M > 0);
        if (need_paging) {
            int acc_ok = (m->pqv2_thread_acc_pool != NULL) &&
                         (m->pqv2_thread_acc_pool_floats >= (size_t)max_drive_M) &&
                         (!has_any_l2 ||
                          (m->pqv2_thread_acc_l2_pool != NULL &&
                           m->pqv2_thread_acc_l2_pool_floats >= (size_t)max_drive_M));
            if (!acc_ok) {
                fprintf(stderr, "ib pqv2: drive page-acc pool unavailable — disabling page cap (whole-tensor scratch)\n");
                /* No cap: scratch sized to the whole largest tensor. */
                l1_cap_eff = (max_idx_bytes + (size_t)ps - 1)
                             & ~((size_t)ps - 1);
                if (l1_cap_eff < l1_lane_floor) l1_cap_eff = l1_lane_floor;
            }
        }
        /* Number of L1 lanes that fit per group at the effective cap. */
        size_t lane_bytes_l1 = (max_drive_M > 0) ? (size_t)max_drive_M : 1;
        size_t lanes_per_group = l1_cap_eff / lane_bytes_l1;
        if (lanes_per_group == 0) lanes_per_group = 1;
        /* Scratch size = min(whole-tensor bytes, effective cap),
         * page-aligned. When the largest tensor already fits, keep the
         * smaller size (no paging needed for any tensor). */
        size_t scratch_full = (max_idx_bytes + (size_t)ps - 1) & ~((size_t)ps - 1);
        size_t scratch_size = scratch_full;
        if (l1_cap_eff < scratch_size) scratch_size = l1_cap_eff;
        scratch_size = (scratch_size + (size_t)ps - 1) & ~((size_t)ps - 1);
        /* L2 scratch: size to the WHOLE largest L2 residual, NOT a capped
         * lane-group. Rationale: the page cap exists to bound the L1 index
         * stream, whose dominant tensor (the un-tied/​tied lm_head, M=vocab)
         * is huge and carries NO L2. The L2 residual streams are small (a
         * 4-bit residual is ≈ half an L1 row, and only on FFN/attn tensors,
         * all far smaller than lm_head). Capping the L2 slot to a lane-group
         * would force every L2-bearing tensor whose whole L2 exceeds that
         * tiny slot to PAGE — even when its L1 fits the slot and it would
         * otherwise run the (more accurate, batched) non-paged kernel. That
         * needlessly diverged pyramid output (~+0.17% PPL) and slowed it.
         * Keeping the whole L2 resident costs only a few MB (it tracks the
         * largest FFN L2, ~3 MB on TinyLlama) and lets only the L1-oversized
         * lm_head page — so the default cap stays ~bit-exact for pyramid.
         * (A future very-large-L2 model could reintroduce an L2-specific
         * cap; for now L1 is the axis that matters.) max_l2_row is still
         * needed by forward.c to bound a paged tensor's L2 lane-group. */
        size_t max_l2_row = 0;
        for (int i = 0; i < nslots; i++) {
            const pqv2_t *pq = tslots[i]->pq;
            if (pq->l2_kind == 2 && pq->l2_indices) {
                size_t rb = pqv2_l2_row_bytes(pq->M, pq->l2_idx_bits);
                if (rb > max_l2_row) max_l2_row = rb;
            }
        }
        (void)max_l2_row;   /* informational; forward.c clamps per-tensor */
        size_t l2_scratch_full = (max_l2_idx_bytes + (size_t)ps - 1) & ~((size_t)ps - 1);
        size_t l2_scratch_size = l2_scratch_full;
        void *scratch = NULL;
        if (scratch_size > 0) scratch = aligned_alloc((size_t)ps, scratch_size);
        if (!scratch) {
            fprintf(stderr, "ib pqv2: drive mode scratch alloc failed (%zu B) — falling back to RAM mode\n",
                    scratch_size);
            m->residency_mode = 0;
        } else {
            m->drive_indices_scratch = scratch;
            m->drive_indices_scratch_size = scratch_size;
            /* Perf fix: allocate a second slot for the 2-slot prefetch
             * ring. If it fails we silently degrade to single-slot
             * synchronous load (same as before the perf fix). */
            m->drive_indices_scratch2 = aligned_alloc((size_t)ps, scratch_size);
            /* Goal C3 — allocate L2 scratch ring iff the model has any
             * L2 pyramid tensors. Failure is non-fatal: the legacy mmap
             * path for L2 still works (just slower under F_NOCACHE). */
            if (l2_scratch_size > 0) {
                m->drive_l2_indices_scratch  = aligned_alloc((size_t)ps, l2_scratch_size);
                m->drive_l2_indices_scratch2 = aligned_alloc((size_t)ps, l2_scratch_size);
                if (m->drive_l2_indices_scratch && m->drive_l2_indices_scratch2) {
                    m->drive_l2_indices_scratch_size = l2_scratch_size;
                } else {
                    if (m->drive_l2_indices_scratch)  { free(m->drive_l2_indices_scratch);  m->drive_l2_indices_scratch  = NULL; }
                    if (m->drive_l2_indices_scratch2) { free(m->drive_l2_indices_scratch2); m->drive_l2_indices_scratch2 = NULL; }
                    m->drive_l2_indices_scratch_size = 0;
                    fprintf(stderr, "ib pqv2: drive mode L2 scratch alloc failed (%zu B) — L2 stays mmap'd\n",
                            l2_scratch_size);
                }
            }
            m->drive_fd = f->_fd;
            m->drive_fd_pretransposed = -1;
            /* Second pass: rewrite pq->indices to point at scratch (CPU
             * path). Record original file offset for CPU drive_load_indices.
             * Goal C3: also record l2_indices_file_offset for pyramid
             * tensors and (when L2 scratch is available) repoint
             * pq->l2_indices to the L2 scratch slot 0 so a stale mmap
             * pointer is never dereferenced under F_NOCACHE. */
            for (int i = 0; i < nslots; i++) {
                pqv2_t *mpq = (pqv2_t *)tslots[i]->pq;
                size_t off = (const uint8_t *)mpq->indices - file_base;
                mpq->indices_file_offset = off;
                mpq->indices = (const uint8_t *)scratch;
                mpq->indices_pretransposed_offset = 0;
                if (mpq->l2_kind == 2 && mpq->l2_indices) {
                    size_t l2_off = (const uint8_t *)mpq->l2_indices - file_base;
                    mpq->l2_indices_file_offset = l2_off;
                    if (m->drive_l2_indices_scratch) {
                        mpq->l2_indices = (const uint8_t *)m->drive_l2_indices_scratch;
                    }
                }
            }
            /* Build the decode-order tensor list used by the prefetcher to
             * predict the next pread target. Order: per layer Q,K,V,O,
             * gate,up,down then (for MoME layers) each expert's gate/up/
             * down; output_head appended last. Same traversal order as the
             * tslots[] walk above and the forward.c layer loop. The
             * prefetch order need only be a reasonable next-tensor guess;
             * MoME experts dispatch e=0..K-1 within a layer so this order
             * tracks the batched-FFN expert loop. */
            int order_cap = slot_cap;   /* same upper bound as tslots[] */
            const ib_tensor_meta **order = (const ib_tensor_meta **)
                malloc((size_t)order_cap * sizeof(*order));
            int order_n = 0;
            if (order) {
                for (int li = 0; li < m->header.num_layers; li++) {
                    ib_layer_meta *L = &m->layers[li];
                    const ib_tensor_meta *s7[7] = {
                        &L->q_proj, &L->k_proj, &L->v_proj, &L->o_proj,
                        &L->gate_proj, &L->up_proj, &L->down_proj,
                    };
                    for (int i = 0; i < 7; i++) {
                        if (s7[i]->pq) order[order_n++] = s7[i];
                    }
                    if (L->mome_experts > 1 &&
                        L->gate_proj_experts && L->up_proj_experts &&
                        L->down_proj_experts) {
                        for (int e = 0; e < L->mome_experts; e++) {
                            if (L->gate_proj_experts[e].pq)
                                order[order_n++] = &L->gate_proj_experts[e];
                            if (L->up_proj_experts[e].pq)
                                order[order_n++] = &L->up_proj_experts[e];
                            if (L->down_proj_experts[e].pq)
                                order[order_n++] = &L->down_proj_experts[e];
                        }
                    }
                }
                if (m->output_head.pq) order[order_n++] = &m->output_head;
            }
            m->drive_pq_order = order;
            m->drive_pq_order_len = order_n;
            /* Build pre-transposed sidecar for the GPU drive path. Writes
             * each tensor's indices in kernel-native [M][total] layout to
             * an unlinked tmpfile and sets indices_pretransposed_offset
             * on each pqv2_t. The GPU drive-mode preads pull bytes
             * directly to MTLBuffer scratch — NO per-matmul transpose,
             * which was the doc-35 CPU-bottleneck floor.
             *
             * CPU drive path (chunk-major in-file) unaffected: still
             * preads from drive_fd at indices_file_offset. */
            /* Heap-allocated (not a VLA — MSVC has no C99 VLA support). */
            pqv2_t **mpq_list = (pqv2_t **)malloc((size_t)nslots * sizeof(pqv2_t *));
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
                /* Heap-allocated (not a VLA — MSVC compatibility). */
                pqv2_t **sidecar_list =
                    (pqv2_t **)malloc((size_t)(nslots + 1) * sizeof(pqv2_t *));
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
                free(sidecar_list);
            }
            free(mpq_list);
            fprintf(stderr, "ib pqv2: drive mode ON. scratch=%zu B (cap=%zu B, page=%zu B), fd=%d, sidecar_fd=%d, %d tensors\n",
                    scratch_size, l1_cap_eff, m->drive_page_bytes,
                    m->drive_fd, m->drive_fd_pretransposed, nslots);
        }
        /* tslots[] is consumed by both passes + the sidecar list build;
         * free once the drive setup (success or scratch-alloc failure) is
         * done. drive_pq_order keeps its own malloc'd copy. */
        free(tslots);
        }   /* close: if (!tslots) ... else { ... } */
    }
    /* Perf: pre-decode fp16 scale/norm buffers (covers raw-fp16 norm
     * tensors and any legacy quantized sub-tensors a hybrid PQv2 file
     * might carry). PQv2 tensors (bits == -1) are skipped automatically. */
    ib_cache_model_static_fp32(m);
    return m;

fail:
    if (f) { ib_pqv2_file_free(f); free(f); }
    if (m) {
        /* Free MoME expert arrays if the failure happened after they
         * were allocated. The inferbit_free path does the same but we
         * never reach that here. */
        if (m->layers) {
            for (int li = 0; li < m->header.num_layers; li++) {
                free(m->layers[li].gate_proj_experts);
                free(m->layers[li].up_proj_experts);
                free(m->layers[li].down_proj_experts);
            }
        }
        free(m->layers);
        free(m);
    }
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

/* Runtime drive-mode page-cap setter (M1 burst scaffolding).
 *
 * Stores the requested per-slot in-focus index cap (MB → bytes) into the
 * existing drive_page_bytes field. mb <= 0 means "no cap" (0 bytes =
 * legacy whole-tensor scratch).
 *
 * SAFE-RESIZE ANALYSIS — why the scratch ring is NOT re-allocated here
 * ===================================================================
 * forward.c::drive_paged_matvec derives its lane-group chunking from
 * `m->drive_indices_scratch_size` (slot_size / lane_bytes), NOT from
 * drive_page_bytes — drive_page_bytes is only consumed at load to SIZE
 * that scratch. So a runtime cap change cannot take effect on subsequent
 * matmuls unless the scratch ring itself is re-sized.
 *
 * Re-sizing the ring here is NOT safe from pqv2_model.c, because the
 * prefetch thread state (ib_drive_pf_state, owned by forward.c) caches
 * the scratch pointers AND size at first use (drive_pf_get: st->scratch[]
 * / st->scratch_size). Freeing/realloc'ing m->drive_indices_scratch from
 * here would leave that worker dereferencing stale (freed) pointers and
 * paging against a stale size — a use-after-free / overflow. Tearing the
 * prefetcher down and refreshing its cache requires editing forward.c,
 * which this agent must not touch.
 *
 * CORRECTNESS-FIRST clamp: we still store the requested cap, but never let
 * it EXCEED the already-allocated slot size. If forward.c is later changed
 * (M2) to read drive_page_bytes for chunking before the ring is re-sized,
 * an un-clamped larger cap could compute lanes_per_group that overflows
 * the existing smaller slot. Clamping to drive_indices_scratch_size makes
 * that impossible. A smaller cap is always safe (fewer lanes per group).
 *
 * TODO(burst-M2: forward.c re-page on cap change): to make a runtime cap
 * change actually alter paging, the integration agent must, in forward.c:
 *   1. Quiesce the prefetcher: ensure no matmul is in flight, then
 *      tear down ib_drive_pf_state (drive_pf_get's cached st->scratch* /
 *      st->scratch_size) so it re-reads the model's buffers.
 *   2. Re-allocate m->drive_indices_scratch{,2} (and the L2 ring) to the
 *      new size — replicating pqv2_load_internal's sizing math
 *      (l1_cap_eff floor = one full lane = max_drive_M bytes, page-align,
 *      acc-pool availability gate), then update drive_indices_scratch_size.
 *   3. Re-point every pq->indices to the new slot-0 buffer (or rely on
 *      drive_repoint_indices, which already re-points per matmul).
 * The model would need to retain max_drive_M / max_idx_bytes / page-size
 * to re-run that math; today those are load-local. */
void inferbit_set_page_cap_mb(inferbit_model* m, int mb) {
    if (!m) return;
    size_t want = (mb > 0) ? (size_t)mb * 1024u * 1024u : 0u;
    /* Clamp up: a non-zero requested cap may not exceed the slot already
     * allocated for the drive ring (see analysis above). A cap of 0 ("no
     * cap" / whole-tensor) is stored verbatim. When drive mode is off
     * (no scratch allocated) there is nothing to overflow, so store as-is. */
    if (want != 0 && m->drive_indices_scratch_size != 0 &&
        want > m->drive_indices_scratch_size) {
        want = m->drive_indices_scratch_size;
    }
    m->drive_page_bytes = want;
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
        if (ib_write(fd, &pad, 1) != 1) {
            ib_close(fd); return -3;
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
    if (!staging) { ib_close(fd); return -3; }

    /* We need a second staging buffer to hold the original chunk-major
     * bytes we pread from the source file. Mmap-read would touch every
     * page into UBC, defeating drive-mode RAM savings — pread keeps
     * the source-file pages out of cache (F_NOCACHE is set on the source
     * fd in drive mode). */
    uint8_t *read_buf = (uint8_t *)malloc(max_bytes);
    if (!read_buf) { free(staging); ib_close(fd); return -5; }

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
                free(read_buf); free(staging); ib_close(fd); return -6;
            }
            got += (size_t)r;
        }
        const uint8_t *src = read_buf;
        /* Transpose into staging: dst[m * total + c*ns + s]. The on-disk
         * L1 indices are always chunk-major (src[(c*ns+s)*M + m]). */
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
            ssize_t w = ib_write(fd, staging + written, idx_bytes - written);
            if (w <= 0) {
                if (w == -1 && errno == EINTR) continue;
                free(staging); ib_close(fd); return -4;
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
