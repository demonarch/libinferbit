/*
 * forward.c — Transformer forward pass
 *
 * Implements: embedding → [RMSNorm → Attention → Residual → RMSNorm → MLP → Residual] × N → RMSNorm → Output head
 */

#include "inferbit_internal.h"
#include "platform.h"   /* pread + POSIX I/O shims (drive mode) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#ifdef IB_HAS_METAL
#include "metal/metal_runtime.h"
#endif

#include "mome.h"
#include "sparse_gate.h"   /* training-free sparse-FFN cluster gate (BURST draft) */

/* ── Training-free sparse-FFN cluster dispatch (BURST draft only) ──────
 *
 * On a CLUSTERED layer (ffn_n_clusters > 1) running under the BURST
 * compute profile, the FFN intermediate dim was permuted at convert so
 * each cluster owns a CONTIGUOUS output-row range of gate/up and the
 * matching input-column range of down. sparse_gate_select() scores the
 * post-attn-norm hidden vector against the per-cluster centroids and
 * returns the active clusters; we then compute gate/up ONLY for those
 * clusters' row ranges, leave the rest of the intermediate at zero, run
 * silu_mul on the active rows, and run down_proj over the (mostly-zero)
 * intermediate (multiply-by-zero on inactive cols => correct draft).
 *
 * This is the DRAFT half of the self-speculative loop: it need not be
 * exact — the COOLDOWN verify (which runs EXACT, never sparse) corrects
 * the emitted tokens. The invariant: when a layer has no clusters
 * (ffn_n_clusters <= 1) OR the active profile is NOT BURST, this whole
 * block is bypassed and the FFN runs every row exactly as before
 * (byte-identical).
 *
 * Env knobs (read once, cached):
 *   IB_FFN_THRESH  float, |silu(dot(x,centroid))| threshold for keeping a
 *                  cluster. Default 0.0 (keep nothing on threshold alone;
 *                  selection then falls to the top_min floor).
 *   IB_FFN_TOPMIN  int, minimum clusters to keep. Default -1 => use
 *                  max(1, n_clusters/4) per layer.
 *   IB_FFN_LOG     when set, accumulate mean active-cluster fraction and
 *                  print a one-line summary at generation end.
 */
#define IB_FFN_MAXK 256   /* max clusters supported per layer (stack active[]) */

static float ffn_gate_thresh(void) {
    static float cached = -1e30f;
    if (cached < -1e29f) {
        const char* e = getenv("IB_FFN_THRESH");
        cached = (e && e[0]) ? (float)atof(e) : 0.0f;
    }
    return cached;
}
/* Returns the configured top_min, or -1 meaning "derive per-layer as
 * max(1, n_clusters/4)". */
static int ffn_top_min_cfg(void) {
    static int cached = -2;
    if (cached == -2) {
        const char* e = getenv("IB_FFN_TOPMIN");
        cached = (e && e[0]) ? atoi(e) : -1;
    }
    return cached;
}
/* Telemetry accumulators for IB_FFN_LOG (mean active-cluster fraction). */
static double  g_ffn_active_frac_sum = 0.0;
static uint64_t g_ffn_gate_calls     = 0;

/* Print the sparse-FFN active-cluster telemetry. Registered via atexit when
 * IB_FFN_LOG is set (so we need not edit generate.c to hook generation end).
 * No-op when the gate never ran. */
static void ffn_log_atexit(void) {
    if (g_ffn_gate_calls == 0) return;
    double mean = g_ffn_active_frac_sum / (double)g_ffn_gate_calls;
    fprintf(stderr,
            "[ib_ffn] sparse-FFN gate: %llu calls, mean active-cluster "
            "fraction = %.3f (%.1f%% of FFN clusters computed)\n",
            (unsigned long long)g_ffn_gate_calls, mean, mean * 100.0);
}

static int ffn_log_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("IB_FFN_LOG");
        cached = (e && e[0]) ? 1 : 0;
        if (cached) atexit(ffn_log_atexit);
    }
    return cached;
}

/* W4A8 path is on by default. Set IB_W4A8=0 in env to force the FP32
 * activation fallback (used for A/B comparison and debugging). */
static int w4a8_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("IB_W4A8");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* ── FFN activation-sparsity QUALITY PROBE (env-gated, off by default) ──
 *
 * Hypothesis: per token most FFN intermediate neurons have silu(gate)≈0, so
 * we can zero them with negligible quality loss. This probe measures the
 * PPL-vs-density curve to bound an eventual sparse kernel's speedup; it does
 * NOT skip compute (multiply-by-zero == skip for QUALITY purposes).
 *
 *   IB_FFN_DENSITY        float in (0,1]; keep top density·inter neurons by
 *                         magnitude. Unset / >=1 / <=0 => probe disabled.
 *   IB_FFN_SPARSITY_MODE  0 (default): threshold |hb| = |silu(gate)·up|
 *                         1: threshold |silu(gate)| (the "predict from gate"
 *                            variant — lets a real kernel skip up+down).
 */
static float ffn_density(void) {
    static float cached = -2.0f;
    if (cached < -1.0f) {
        const char* e = getenv("IB_FFN_DENSITY");
        float d = (e && e[0]) ? (float)atof(e) : 1.0f;
        if (!(d > 0.0f) || d >= 1.0f) d = 1.0f; /* off */
        cached = d;
    }
    return cached;
}
static int ffn_sparsity_mode(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("IB_FFN_SPARSITY_MODE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* In-place quickselect on a scratch copy of |vals[0..n)| to find the
 * threshold τ = the (n-k)-th smallest magnitude, where k = #neurons to
 * keep. Returns τ such that keeping |x|>=τ retains ~k entries. */
static float ffn_select_threshold(const float* vals, int n, int keep,
                                  float* scratch) {
    if (keep <= 0) return 1e30f;       /* zero everything */
    if (keep >= n) return -1.0f;       /* keep everything */
    for (int i = 0; i < n; i++) {
        float v = vals[i];
        scratch[i] = v < 0.0f ? -v : v;
    }
    /* We want the (n-keep)-th smallest => rank index target = n-keep. */
    int target = n - keep;
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = scratch[(lo + hi) >> 1];
        int i = lo, j = hi;
        while (i <= j) {
            while (scratch[i] < pivot) i++;
            while (scratch[j] > pivot) j--;
            if (i <= j) {
                float t = scratch[i]; scratch[i] = scratch[j]; scratch[j] = t;
                i++; j--;
            }
        }
        if (target <= j)      hi = j;
        else if (target >= i) lo = i;
        else break;
    }
    return scratch[target];
}

/* Apply the density probe to the post-silu_mul activation hb[inter].
 * gate_pre points at the pre-silu_mul gate output (= hb before silu_mul was
 * called over it) for mode-1 gating; in mode 0 it is ignored. scratch must
 * hold >= inter floats. */
static void ffn_apply_density(float* hb, const float* gate_pre, int inter,
                              float* scratch) {
    float density = ffn_density();
    if (density >= 1.0f) return;
    int keep = (int)(density * (float)inter + 0.5f);
    if (keep >= inter) return;

    if (ffn_sparsity_mode() == 1 && gate_pre) {
        /* Mode 1: rank by |silu(gate)|. Reuse scratch to hold silu(gate). */
        for (int i = 0; i < inter; i++) {
            float g = gate_pre[i];
            scratch[i] = g / (1.0f + expf(-g));   /* silu(gate) */
        }
        /* select_threshold copies |scratch| into a second region — but we
         * only have one scratch buffer, so compute τ over scratch directly
         * via a magnitude copy at the tail half is unsafe. Instead inline:
         * temporarily abs scratch in place, quickselect, then re-derive
         * keep-mask by comparing |silu(gate)| to τ. */
        /* abs in place */
        for (int i = 0; i < inter; i++) {
            if (scratch[i] < 0.0f) scratch[i] = -scratch[i];
        }
        int target = inter - keep;
        int lo = 0, hi = inter - 1;
        while (lo < hi) {
            float pivot = scratch[(lo + hi) >> 1];
            int i = lo, j = hi;
            while (i <= j) {
                while (scratch[i] < pivot) i++;
                while (scratch[j] > pivot) j--;
                if (i <= j) {
                    float t = scratch[i]; scratch[i] = scratch[j]; scratch[j] = t;
                    i++; j--;
                }
            }
            if (target <= j)      hi = j;
            else if (target >= i) lo = i;
            else break;
        }
        float tau = scratch[target];
        for (int i = 0; i < inter; i++) {
            float g = gate_pre[i];
            float s = g / (1.0f + expf(-g));
            if ((s < 0.0f ? -s : s) < tau) hb[i] = 0.0f;
        }
    } else {
        /* Mode 0: rank by |hb| = |silu(gate)*up|. */
        float tau = ffn_select_threshold(hb, inter, keep, scratch);
        for (int i = 0; i < inter; i++) {
            float v = hb[i];
            if ((v < 0.0f ? -v : v) < tau) hb[i] = 0.0f;
        }
    }
}

/* ── Goal H4 — hot-cache framework (scaffolding) ─────────────────────
 *
 * See inferbit_internal.h for the contract. v1 ships:
 *   • ib_hotset_enabled  — cached IB_TENSOR_HOTSET env check.
 *   • ib_hot_lookup      — always returns NULL (no entries promoted).
 *   • ib_hot_promote     — no-op; reserved for the adaptive policy.
 *   • ib_hotset_report   — top-10 most-accessed-tensors summary
 *                          printed to stderr (called from inferbit_free).
 *
 * The counter increment lives in tensor_matmul() so every CPU matmul
 * (PQv2, W4A8, INT8, FP16) contributes regardless of which dispatch
 * branch is taken. The bump is gated on ib_hotset_enabled() so the
 * default path pays exactly one cached branch. */
int ib_hotset_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_TENSOR_HOTSET");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

/* Real hot-pool bodies live in ibf_loader.c as *_impl (named to avoid a
 * duplicate-symbol clash with these canonical entry points). Redirect to them.
 * Both impls are strict no-ops when the pool is disabled (hot_pool == NULL /
 * cap == 0), so with IB_HOT_POOL_MB unset this is byte-identical to the old
 * stub: lookup → NULL, promote → 1 ("not promoted"). */
extern const void *ib_hot_lookup_impl(const inferbit_model *m, const ib_tensor_meta *t);
extern const void *ib_hot_lookup_key(const inferbit_model *m, size_t key);
extern int ib_hot_promote_impl(inferbit_model *m, const ib_tensor_meta *t);
extern int ib_hot_promote_bytes(inferbit_model *m, size_t key,
                                const void *src, size_t nbytes);
extern void *ib_hot_reserve(inferbit_model *m, size_t key, size_t nbytes);

/* Hot-pool keys are the on-disk file offsets of the index streams:
 * pq->indices_file_offset for L1, pq->l2_indices_file_offset for L2. These are
 * unique and nonzero per tensor in drive mode (t->offset is 0/unused there, so
 * it must NOT be used as a key — doing so collides every tensor onto key 0). */

/* Forward decls — defined later in this file but called by the drive-mode
 * resident fast-path in drive_paged_matvec (above their definitions). */
static void pqv2_matvec_dispatch(const pqv2_t *t, const float *x, float *y);
static void pqv2_threaded_matvec_k256(const inferbit_model *m,
                                      struct ib_thread_pool *tp, int n_threads,
                                      const pqv2_t *t, const float *x, float *y);
static void pqv2_threaded_matvec_k256_batch(const inferbit_model *m,
                                            struct ib_thread_pool *tp, int n_threads,
                                            const pqv2_t *t, const float *x_batch,
                                            int B, float *y_batch);

const void *ib_hot_lookup(const inferbit_model *m, const ib_tensor_meta *t) {
    return ib_hot_lookup_impl(m, t);
}

int ib_hot_promote(inferbit_model *m, const ib_tensor_meta *t) {
    return ib_hot_promote_impl(m, t);
}

/* Walk every ib_tensor_meta the model owns and visit it via `fn`.
 * Centralised here so ib_hotset_report doesn't need to know the model
 * layout, and so future per-tensor sweeps (e.g. promotion scoring) can
 * share the same enumeration. */
static void ib_for_each_tensor(const inferbit_model *m,
                               void (*fn)(const ib_tensor_meta *, const char *, int, void *),
                               void *udata) {
    fn(&m->token_embedding, "token_embedding", -1, udata);
    fn(&m->output_norm,     "output_norm",     -1, udata);
    fn(&m->output_head,     "output_head",     -1, udata);
    if (!m->layers) return;
    for (int L = 0; L < m->header.num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        fn(&lm->q_proj,         "q_proj",         L, udata);
        fn(&lm->k_proj,         "k_proj",         L, udata);
        fn(&lm->v_proj,         "v_proj",         L, udata);
        fn(&lm->o_proj,         "o_proj",         L, udata);
        fn(&lm->gate_proj,      "gate_proj",      L, udata);
        fn(&lm->up_proj,        "up_proj",        L, udata);
        fn(&lm->down_proj,      "down_proj",      L, udata);
        fn(&lm->input_norm,     "input_norm",     L, udata);
        fn(&lm->post_attn_norm, "post_attn_norm", L, udata);
    }
}

typedef struct {
    const ib_tensor_meta *t;
    const char *name;
    int layer;
} ib_hot_entry;

#define IB_HOT_REPORT_TOPN 10

static void ib_hot_collect(const ib_tensor_meta *t, const char *name,
                           int layer, void *udata) {
    ib_hot_entry (*top)[IB_HOT_REPORT_TOPN] = udata;
    /* Insertion-sort into top-N. O(N) per entry, fine for the few
     * hundred tensors a transformer has. */
    int slot = -1;
    for (int i = 0; i < IB_HOT_REPORT_TOPN; i++) {
        if (!(*top)[i].t || t->access_count > (*top)[i].t->access_count) {
            slot = i; break;
        }
    }
    if (slot < 0) return;
    for (int i = IB_HOT_REPORT_TOPN - 1; i > slot; i--) (*top)[i] = (*top)[i-1];
    (*top)[slot].t = t;
    (*top)[slot].name = name;
    (*top)[slot].layer = layer;
}

void ib_hotset_report(const inferbit_model *m) {
    if (!m || !ib_hotset_enabled()) return;
    ib_hot_entry top[IB_HOT_REPORT_TOPN];
    for (int i = 0; i < IB_HOT_REPORT_TOPN; i++) {
        top[i].t = NULL; top[i].name = NULL; top[i].layer = -1;
    }
    ib_for_each_tensor(m, ib_hot_collect, &top);
    fprintf(stderr, "[ib-hotset] top-%d most-accessed tensors "
                    "(hot_pool=%zu bytes, entries=%d):\n",
            IB_HOT_REPORT_TOPN, m->hot_pool_bytes, m->hot_pool_entries);
    for (int i = 0; i < IB_HOT_REPORT_TOPN; i++) {
        if (!top[i].t || top[i].t->access_count == 0) break;
        if (top[i].layer >= 0) {
            fprintf(stderr, "  %2d. L%-2d %-16s  access=%llu  size=%zu\n",
                    i + 1, top[i].layer, top[i].name,
                    (unsigned long long)top[i].t->access_count,
                    top[i].t->size);
        } else {
            fprintf(stderr, "  %2d.     %-16s  access=%llu  size=%zu\n",
                    i + 1, top[i].name,
                    (unsigned long long)top[i].t->access_count,
                    top[i].t->size);
        }
    }
}

/* ── Stage 5d — hybrid CPU/GPU dispatch (docs/v2/00_CORRECTION.md) ──
 *
 * v1 ships a single env-var knob:
 *   IB_HYBRID_FFN_GPU=1  → at first ib_forward call, tag every layer's
 *                          gate_proj / up_proj / down_proj as
 *                          INFERBIT_BACKEND_METAL. The CPU forward then
 *                          routes those matmuls through a one-shot
 *                          Metal dispatch while the rest of the layer
 *                          (norms, attention, residuals, embed, lm_head)
 *                          stays on CPU.
 *
 * Default (env unset) leaves every preferred_backend at AUTO (=0) which
 * keeps the existing CPU-or-Metal end-to-end routing bit-identical to
 * pre-Stage-5d behaviour.
 *
 * Lazy tag application + lazy ctx creation. */
static int hybrid_ffn_gpu_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_HYBRID_FFN_GPU");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static void hybrid_apply_tags(inferbit_model *m) {
    if (!m || m->hybrid_tags_applied) return;
    m->hybrid_tags_applied = 1;
    if (!hybrid_ffn_gpu_enabled()) return;
    /* Tag FFN matmuls only. Attention stays AUTO so it follows the
     * surrounding forward (CPU here). */
    for (int L = 0; L < m->header.num_layers; L++) {
        m->layers[L].gate_proj.preferred_backend = INFERBIT_BACKEND_METAL;
        m->layers[L].up_proj.preferred_backend   = INFERBIT_BACKEND_METAL;
        m->layers[L].down_proj.preferred_backend = INFERBIT_BACKEND_METAL;
    }
}

#ifdef IB_HAS_METAL
/* Forward decl from below: lazy Metal ctx/buf creation. Returns 1 if
 * upload succeeded, 0 if Metal is unavailable or upload failed. */
static int ib_metal_route(inferbit_model* m);

/* Stage 5d helper: lazily create the Metal ctx + upload the model
 * REGARDLESS of IB_BACKEND=cpu (which ib_metal_route honors). The
 * hybrid hook needs the GPU available even when the surrounding
 * forward runs on CPU. Returns 1 on success, 0 if Metal is unavailable
 * or upload fails (caller falls back to CPU dispatch). */
static int hybrid_metal_route(inferbit_model *m) {
    if (m->metal_route_failed) return 0;
    if (m->metal_bufs) return 1;
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { m->metal_route_failed = 1; return 0; }
    ib_metal_model_buffers *bufs = ib_metal_upload_model(ctx, m);
    if (!bufs) { ib_metal_destroy(ctx); m->metal_route_failed = 1; return 0; }
    m->metal_ctx  = ctx;
    m->metal_bufs = bufs;
    return 1;
}

/* Ensure model->hybrid_x_buf / hybrid_y_buf are Metal-shared and at
 * least n_in / n_out floats long. Grow (re-alloc) if too small. Returns
 * 0 on success; -1 on failure (caller should fall back to CPU dispatch). */
static int hybrid_ensure_buffers(inferbit_model *m, size_t n_in, size_t n_out) {
    ib_metal_ctx *ctx = (ib_metal_ctx*)m->metal_ctx;
    if (!ctx) return -1;
    if (!m->hybrid_x_buf || m->hybrid_x_buf_floats < n_in) {
        if (m->hybrid_x_buf) ib_metal_free(ctx, m->hybrid_x_buf);
        m->hybrid_x_buf = ib_metal_alloc(ctx, n_in * sizeof(float), NULL);
        if (!m->hybrid_x_buf) { m->hybrid_x_buf_floats = 0; return -1; }
        m->hybrid_x_buf_floats = n_in;
    }
    if (!m->hybrid_y_buf || m->hybrid_y_buf_floats < n_out) {
        if (m->hybrid_y_buf) ib_metal_free(ctx, m->hybrid_y_buf);
        m->hybrid_y_buf = ib_metal_alloc(ctx, n_out * sizeof(float), NULL);
        if (!m->hybrid_y_buf) { m->hybrid_y_buf_floats = 0; return -1; }
        m->hybrid_y_buf_floats = n_out;
    }
    return 0;
}

/* Selector lookup: which IB_METAL_TB_* index corresponds to this tensor
 * within a layer. Returns -1 if the pointer isn't one of the known
 * matmul slots of the given layer (in which case the caller falls back
 * to CPU). */
static int hybrid_tensor_which(const ib_layer_meta *L, const ib_tensor_meta *t,
                               int *out_layer_idx, int layer_idx) {
    *out_layer_idx = layer_idx;
    if (t == &L->q_proj)    return IB_METAL_TB_Q_PROJ;
    if (t == &L->k_proj)    return IB_METAL_TB_K_PROJ;
    if (t == &L->v_proj)    return IB_METAL_TB_V_PROJ;
    if (t == &L->o_proj)    return IB_METAL_TB_O_PROJ;
    if (t == &L->gate_proj) return IB_METAL_TB_GATE_PROJ;
    if (t == &L->up_proj)   return IB_METAL_TB_UP_PROJ;
    if (t == &L->down_proj) return IB_METAL_TB_DOWN_PROJ;
    return -1;
}
#endif /* IB_HAS_METAL */

/* Forward decl of the CPU matmul (defined below). */
static void tensor_matmul(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf
);

/* Hybrid-aware matmul dispatcher. If the tensor is METAL-tagged AND the
 * model has (or can lazily acquire) a Metal context AND the layer was
 * uploaded, dispatch this single matmul to the GPU; otherwise fall
 * through to the CPU `tensor_matmul`. layer_idx is the owning layer
 * index for selector resolution. Caller passes M=out_rows, N=in_cols.
 *
 * On any failure the implementation transparently falls back to CPU so
 * the forward pass never crashes — the worst case is a one-time perf
 * regression. */
static void tensor_matmul_hybrid(
    inferbit_model *m, int layer_idx, const ib_tensor_meta *t,
    float *out, const float *input, int M, int N, float *scale_buf
) {
#ifdef IB_HAS_METAL
    if (t->preferred_backend == INFERBIT_BACKEND_METAL) {
        /* Lazy Metal ctx + upload. Use hybrid_metal_route — it ignores
         * IB_BACKEND=cpu (the user explicitly opted into hybrid by
         * tagging this tensor METAL). If Metal genuinely isn't
         * available (no device, unsupported layout), fall back to CPU. */
        if (hybrid_metal_route(m)) {
            if (hybrid_ensure_buffers(m, (size_t)N, (size_t)M) == 0) {
                int li = 0;
                int which = hybrid_tensor_which(&m->layers[layer_idx], t, &li, layer_idx);
                if (which >= 0) {
                    memcpy(m->hybrid_x_buf, input, (size_t)N * sizeof(float));
                    int rc = ib_metal_run_single_matmul(
                        (ib_metal_ctx*)m->metal_ctx, m->metal_bufs,
                        li, which, m->hybrid_x_buf, m->hybrid_y_buf);
                    if (rc == 0) {
                        memcpy(out, m->hybrid_y_buf, (size_t)M * sizeof(float));
                        return;
                    }
                }
            }
        }
        /* fallthrough → CPU */
    }
#else
    (void)layer_idx;
#endif
    tensor_matmul(m, t, out, input, M, N, scale_buf);
}

/* ── Weight data access helpers ─────────────────────────────── */

/* Get pointer to weight data for a tensor */
static inline const void* tensor_data(const inferbit_model* m, const ib_tensor_meta* t) {
    return (const uint8_t*)m->weight_data + t->offset;
}

/* Get pointer to scale factors for a tensor (FP16 stored, we read as half→float) */
static inline const void* tensor_scales_raw(const inferbit_model* m, const ib_tensor_meta* t) {
    if (t->scale_offset == 0 && t->scale_size == 0) return NULL;
    return (const uint8_t*)m->weight_data + t->scale_offset;
}

/* ── Path D drive mode + 2-slot prefetch ring (perf fix) ─────────
 *
 * Background: every PQv2 matmul in drive mode pread()s its indices
 * from disk into a shared scratch buffer the kernel reads from.
 * Per token: 22 layers × 7 matmuls = 154 synchronous preads, each
 * blocking on storage. The matmul kernel and the I/O were strictly
 * serialised → effective decode throughput floored at ~7-10 t/s.
 *
 * Fix: two scratch slots + a single background pread() worker. Before
 * each matmul we kick a prefetch for the NEXT tensor (decode order is
 * static — Q,K,V,O,gate,up,down per layer; output_head at the end). By
 * the time the kernel needs slot N, the worker is already filling slot
 * (N+1 % 2). The current matmul therefore overlaps with the next
 * tensor's I/O, hiding most of the pread() latency behind the kernel
 * compute. The model.c free path stops the worker via the public
 * ib_drive_prefetch_shutdown shim.
 *
 * The kernel reads from `pq->indices`. We MUST repoint `pq->indices`
 * to the slot whose data corresponds to the tensor about to run.
 * Worker writes into the *other* slot, so the active matmul never
 * races with the prefetch. */

#ifdef _WIN32
/* Reuse the Windows pthread shim already defined by threading.c. Including
 * it here would double-define; we replicate the minimal subset we need. */
#include <windows.h>
typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
#define pthread_mutex_init(m, a)     (InitializeSRWLock(m), 0)
#define pthread_mutex_destroy(m)     ((void)0)
#define pthread_mutex_lock(m)        AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m)      ReleaseSRWLockExclusive(m)
#define pthread_cond_init(c, a)      (InitializeConditionVariable(c), 0)
#define pthread_cond_destroy(c)      ((void)0)
#define pthread_cond_wait(c, m)      SleepConditionVariableSRW(c, m, INFINITE, 0)
#define pthread_cond_signal(c)       WakeConditionVariable(c)
#define pthread_cond_broadcast(c)    WakeAllConditionVariable(c)
typedef DWORD (WINAPI *win_thread_fn_pf)(LPVOID);
static int pthread_create(pthread_t* t, void* attr, void* (*fn)(void*), void* arg) {
    (void)attr; *t = CreateThread(NULL, 0, (win_thread_fn_pf)fn, arg, 0, NULL);
    return (*t == NULL) ? -1 : 0;
}
static int pthread_join(pthread_t t, void** retval) {
    (void)retval; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0;
}
#else
#include <pthread.h>
#endif

typedef struct ib_drive_pf_state {
    /* N29 — two worker threads: thread_l1 pread's L1 indices, thread_l2
     * pread's L2 indices. They run in parallel so the per-tensor I/O
     * time is max(L1, L2) instead of L1 + L2. */
    pthread_t      thread;        /* L1 worker (kept name for diff hygiene) */
    pthread_t      thread_l2;     /* L2 worker */
    pthread_mutex_t mu;
    pthread_cond_t  req_cv;       /* main → L1 worker */
    pthread_cond_t  req_cv_l2;    /* main → L2 worker */
    pthread_cond_t  done_cv;      /* worker → main: request complete */
    /* Request state (protected by mu). The L1 and L2 workers share the
     * same req_tensor / req_slot — main thread sets them before waking
     * both and only re-issues once both have completed. */
    const ib_tensor_meta *req_tensor;   /* what to prefetch */
    int             req_slot;           /* which slot to fill (0 or 1) */
    int             req_pending_l1;     /* 1 when L1 worker should service */
    int             req_pending_l2;     /* 1 when L2 worker should service */
    int             req_in_flight_l1;   /* 1 between L1 dequeue and completion */
    int             req_in_flight_l2;   /* 1 between L2 dequeue and completion */
    int             req_ok_l1;          /* set by L1 worker on success */
    /* Raw byte-range request mode (drive_paged_matvec lane-group pipeline).
     * When req_raw != 0 the workers pread the explicit file ranges below
     * into scratch[req_slot] / l2_scratch[req_slot] instead of deriving the
     * range from req_tensor. req_raw_l2_len == 0 => skip L2 for this group.
     * Every issue site sets req_raw (raw issues =1, tensor issues =0). */
    int             req_raw;
    off_t           req_raw_l1_off;
    size_t          req_raw_l1_len;
    off_t           req_raw_l2_off;
    size_t          req_raw_l2_len;
    /* Result of the most recently completed request. */
    const ib_tensor_meta *done_tensor;
    int             done_slot;
    /* Cached pointers for the worker (set once at init). */
    int             fd;
    void           *scratch[2];
    size_t          scratch_size;
    /* Goal C3 — L2 indices scratch ring (parallel to L1 ring). NULL
     * when the model has no L2 pyramid tensors or alloc failed. */
    void           *l2_scratch[2];
    size_t          l2_scratch_size;
    /* Shutdown flag. */
    int             stop;
} ib_drive_pf_state;

/* Goal C3 — total L2 indices bytes for tensor t (or 0 if no L2).
 * Branches on pq->l2_idx_bits (4/6/8) via the shared kernel helper so the
 * pread streams exactly the on-disk packed size. Goal N36 fix: the 4-bit
 * packing (ceil(M/2)) was previously falling through to the 8-bit size,
 * over-reading ~2x per matmul under F_NOCACHE. */
static inline size_t drive_l2_indices_bytes(const pqv2_t *pq) {
    return pqv2_l2_total_index_bytes(pq);
}

/* Goal C3 — pread `bytes` from `fd` at `off` into `buf`. Returns 1 on
 * success, 0 on failure. Shared by the worker (L1+L2 fill) and the
 * synchronous fallback. */
static int drive_pread_full(int fd, void *buf, size_t bytes, off_t off) {
    uint8_t *p = (uint8_t *)buf;
    size_t done = 0;
    while (done < bytes) {
        ssize_t r = pread(fd, p + done, bytes - done, off + (off_t)done);
        if (r <= 0) {
            if (r == -1 && errno == EINTR) continue;
            return 0;
        }
        done += (size_t)r;
    }
    return 1;
}

/* N29 — L1 worker. Pread's only the L1 indices for the current request.
 * Runs in parallel with the L2 worker. Completion is signaled jointly:
 * the second worker to finish broadcasts done_cv with the merged result. */
static void *ib_drive_pf_worker(void *arg) {
    ib_drive_pf_state *st = (ib_drive_pf_state *)arg;
    pthread_mutex_lock(&st->mu);
    for (;;) {
        while (!st->stop && !st->req_pending_l1) {
            pthread_cond_wait(&st->req_cv, &st->mu);
        }
        if (st->stop) break;
        const ib_tensor_meta *t = st->req_tensor;
        int slot = st->req_slot;
        int raw = st->req_raw;
        off_t raw_off = st->req_raw_l1_off;
        size_t raw_len = st->req_raw_l1_len;
        st->req_pending_l1 = 0;
        st->req_in_flight_l1 = 1;
        pthread_mutex_unlock(&st->mu);

        int ok = 0;
        if (raw) {
            if (slot >= 0 && slot < 2 && st->scratch[slot] &&
                raw_len > 0 && raw_len <= st->scratch_size) {
                ok = drive_pread_full(st->fd, st->scratch[slot], raw_len, raw_off);
            }
        } else if (t && t->pq && slot >= 0 && slot < 2 && st->scratch[slot]) {
            const pqv2_t *pq = t->pq;
            size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
            off_t off = (off_t)pq->indices_file_offset;
            if (bytes > 0 && bytes <= st->scratch_size && off != 0) {
                ok = drive_pread_full(st->fd, st->scratch[slot], bytes, off);
            }
        }

        pthread_mutex_lock(&st->mu);
        st->req_ok_l1 = ok;
        st->req_in_flight_l1 = 0;
        /* Publish result only when BOTH workers have finished. The last
         * one to finish wins the publish; the other waits on its own cv
         * for the next request. Raw (lane-group) requests never publish a
         * done_tensor — the paged loop waits on the idle flags, and a stale
         * tensor here could be mis-read as a prefetched whole tensor. */
        if (!st->req_in_flight_l2 && !st->req_pending_l2) {
            st->done_tensor = (ok && !raw) ? t : NULL;
            st->done_slot = (ok && !raw) ? slot : -1;
            pthread_cond_broadcast(&st->done_cv);
        }
    }
    pthread_mutex_unlock(&st->mu);
    return NULL;
}

/* N29 — L2 worker. Pread's only the L2 indices (or no-ops when the
 * tensor has no L2). Runs in parallel with the L1 worker. The L2 read
 * is best-effort: failure does not invalidate the request, matching the
 * pre-N29 semantics (kernel falls back to the existing pq->l2_indices
 * pointer). */
static void *ib_drive_pf_worker_l2(void *arg) {
    ib_drive_pf_state *st = (ib_drive_pf_state *)arg;
    pthread_mutex_lock(&st->mu);
    for (;;) {
        while (!st->stop && !st->req_pending_l2) {
            pthread_cond_wait(&st->req_cv_l2, &st->mu);
        }
        if (st->stop) break;
        const ib_tensor_meta *t = st->req_tensor;
        int slot = st->req_slot;
        int raw = st->req_raw;
        off_t raw_off = st->req_raw_l2_off;
        size_t raw_len = st->req_raw_l2_len;
        st->req_pending_l2 = 0;
        st->req_in_flight_l2 = 1;
        pthread_mutex_unlock(&st->mu);

        if (raw) {
            if (raw_len > 0 && slot >= 0 && slot < 2 &&
                st->l2_scratch[slot] && raw_len <= st->l2_scratch_size) {
                (void)drive_pread_full(st->fd, st->l2_scratch[slot],
                                       raw_len, raw_off);
            }
        } else if (t && t->pq && slot >= 0 && slot < 2 &&
            st->l2_scratch[slot] && t->pq->l2_kind == 2 &&
            t->pq->l2_indices_file_offset != 0) {
            const pqv2_t *pq = t->pq;
            size_t l2_bytes = drive_l2_indices_bytes(pq);
            if (l2_bytes > 0 && l2_bytes <= st->l2_scratch_size) {
                (void)drive_pread_full(st->fd, st->l2_scratch[slot],
                                       l2_bytes,
                                       (off_t)pq->l2_indices_file_offset);
            }
        }

        pthread_mutex_lock(&st->mu);
        st->req_in_flight_l2 = 0;
        if (!st->req_in_flight_l1 && !st->req_pending_l1) {
            /* L1 worker already finished — publish the result it left in
             * req_ok_l1. Raw lane-group requests never publish a tensor. */
            st->done_tensor = (st->req_ok_l1 && !raw) ? t : NULL;
            st->done_slot = (st->req_ok_l1 && !raw) ? slot : -1;
            pthread_cond_broadcast(&st->done_cv);
        }
    }
    pthread_mutex_unlock(&st->mu);
    return NULL;
}

/* Wait for any in-flight prefetch to finish (called under mu) and clear
 * the result. N29 — must wait until BOTH workers are idle. */
static void pf_wait_idle_locked(ib_drive_pf_state *st) {
    while (st->req_pending_l1 || st->req_in_flight_l1 ||
           st->req_pending_l2 || st->req_in_flight_l2) {
        pthread_cond_wait(&st->done_cv, &st->mu);
    }
}

/* Lazily init the prefetcher on the first matmul. Falls back silently to
 * the legacy synchronous pread path on init failure. */
static ib_drive_pf_state *drive_pf_get(const inferbit_model *m) {
    /* We mutate the cached pointer through (inferbit_model*) — the
     * "const" on m is decorative inside this module; matmul callers pass
     * const for read-only weight access, not because m is genuinely
     * immutable (drive scratch buffer is also overwritten). */
    inferbit_model *mm = (inferbit_model *)m;
    if (mm->drive_pf_state) return (ib_drive_pf_state *)mm->drive_pf_state;
    /* Disable prefetcher when IB_DRIVE_PF_OFF=1 — used for A/B baseline
     * comparison. Falls back to legacy synchronous load. */
    {
        const char *off = getenv("IB_DRIVE_PF_OFF");
        if (off && off[0] == '1') return NULL;
    }
    if (!mm->drive_indices_scratch || !mm->drive_indices_scratch2 ||
        !mm->drive_pq_order || mm->drive_pq_order_len <= 0 ||
        mm->drive_fd < 0) {
        return NULL;
    }
    ib_drive_pf_state *st = (ib_drive_pf_state *)calloc(1, sizeof(*st));
    if (!st) return NULL;
    if (pthread_mutex_init(&st->mu, NULL) != 0) { free(st); return NULL; }
    if (pthread_cond_init(&st->req_cv, NULL) != 0) {
        pthread_mutex_destroy(&st->mu); free(st); return NULL;
    }
    if (pthread_cond_init(&st->req_cv_l2, NULL) != 0) {
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu); free(st); return NULL;
    }
    if (pthread_cond_init(&st->done_cv, NULL) != 0) {
        pthread_cond_destroy(&st->req_cv_l2);
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu); free(st); return NULL;
    }
    st->fd = mm->drive_fd;
    st->scratch[0] = mm->drive_indices_scratch;
    st->scratch[1] = mm->drive_indices_scratch2;
    st->scratch_size = mm->drive_indices_scratch_size;
    /* Goal C3 — L2 scratch ring. May be NULL when the model has no L2. */
    st->l2_scratch[0] = mm->drive_l2_indices_scratch;
    st->l2_scratch[1] = mm->drive_l2_indices_scratch2;
    st->l2_scratch_size = mm->drive_l2_indices_scratch_size;
    st->done_slot = -1;
    if (pthread_create(&st->thread, NULL, ib_drive_pf_worker, st) != 0) {
        pthread_cond_destroy(&st->done_cv);
        pthread_cond_destroy(&st->req_cv_l2);
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu);
        free(st);
        return NULL;
    }
    if (pthread_create(&st->thread_l2, NULL, ib_drive_pf_worker_l2, st) != 0) {
        /* Tear down the L1 worker we just created so we don't leak it. */
        pthread_mutex_lock(&st->mu);
        st->stop = 1;
        pthread_cond_broadcast(&st->req_cv);
        pthread_mutex_unlock(&st->mu);
        pthread_join(st->thread, NULL);
        pthread_cond_destroy(&st->done_cv);
        pthread_cond_destroy(&st->req_cv_l2);
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu);
        free(st);
        return NULL;
    }
    mm->drive_pf_state = st;
    if (getenv("IB_DRIVE_PF_DEBUG")) {
        fprintf(stderr, "[ib drive-pf] prefetcher init: scratch_size=%zu order_len=%d fd=%d\n",
                (size_t)st->scratch_size, mm->drive_pq_order_len, st->fd);
    }
    /* Warm-start: prefetch the very first tensor so the first matmul of
     * the first decode step doesn't have to sync-load (matters only for
     * short generations; negligible for long ones but ~free). */
    if (mm->drive_pq_order && mm->drive_pq_order_len > 0) {
        pthread_mutex_lock(&st->mu);
        st->req_tensor = mm->drive_pq_order[0];
        st->req_slot = 0;
        st->req_raw = 0;
        st->req_ok_l1 = 0;
        st->req_pending_l1 = 1;
        st->req_pending_l2 = 1;
        pthread_cond_signal(&st->req_cv);
        pthread_cond_signal(&st->req_cv_l2);
        pthread_mutex_unlock(&st->mu);
    }
    return st;
}

/* Public shim called from model.c during inferbit_free. Stops the worker
 * and tears down its sync primitives. Safe to call when no prefetcher
 * was ever initialised. */
void ib_drive_prefetch_shutdown(inferbit_model *m);
void ib_drive_prefetch_shutdown(inferbit_model *m) {
    if (!m || !m->drive_pf_state) return;
    ib_drive_pf_state *st = (ib_drive_pf_state *)m->drive_pf_state;
    pthread_mutex_lock(&st->mu);
    pf_wait_idle_locked(st);
    st->stop = 1;
    pthread_cond_broadcast(&st->req_cv);
    pthread_cond_broadcast(&st->req_cv_l2);
    pthread_mutex_unlock(&st->mu);
    pthread_join(st->thread, NULL);
    pthread_join(st->thread_l2, NULL);
    pthread_cond_destroy(&st->done_cv);
    pthread_cond_destroy(&st->req_cv_l2);
    pthread_cond_destroy(&st->req_cv);
    pthread_mutex_destroy(&st->mu);
    free(st);
    m->drive_pf_state = NULL;
}

/* Find the index of `t` in m->drive_pq_order[] (linear scan over ~150
 * pointers — single cache-line walk on average). Returns -1 if not in
 * the list (sparsity-masked / non-drive / etc.). */
static int drive_order_index(const inferbit_model *m, const ib_tensor_meta *t) {
    int n = m->drive_pq_order_len;
    const ib_tensor_meta **arr = m->drive_pq_order;
    for (int i = 0; i < n; i++) {
        if (arr[i] == t) return i;
    }
    return -1;
}

/* Synchronously pread tensor t's indices into `slot` (and L2 into the
 * matching L2 slot when present). Returns 0 on ok. */
static int drive_sync_load_to_slot(const inferbit_model *m,
                                   const ib_tensor_meta *t,
                                   int slot) {
    if (!t || !t->pq) return -1;
    void *dst = (slot == 1) ? m->drive_indices_scratch2 : m->drive_indices_scratch;
    if (!dst) return -1;
    const pqv2_t *pq = t->pq;
    size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
    if (bytes == 0 || bytes > m->drive_indices_scratch_size) return -1;
    off_t off = (off_t)pq->indices_file_offset;
    if (off == 0) return -1;
    if (!drive_pread_full(m->drive_fd, dst, bytes, off)) return -1;
    /* Goal C3 — load L2 alongside L1 when this tensor has L2 indices
     * and the L2 ring is allocated. Non-fatal on L2 read failure: the
     * kernel still has a valid pq->l2_indices pointer (worst case the
     * previous slot's contents — still cache-resident, no F_NOCACHE
     * fault). */
    void *l2_dst = (slot == 1) ? m->drive_l2_indices_scratch2
                                : m->drive_l2_indices_scratch;
    if (l2_dst && pq->l2_kind == 2 && pq->l2_indices_file_offset != 0) {
        size_t l2_bytes = drive_l2_indices_bytes(pq);
        if (l2_bytes > 0 && l2_bytes <= m->drive_l2_indices_scratch_size) {
            (void)drive_pread_full(m->drive_fd, l2_dst, l2_bytes,
                                   (off_t)pq->l2_indices_file_offset);
        }
    }
    return 0;
}

/* Repoint pq->indices (and pq->l2_indices when pyramid) for tensor t to
 * the buffers in `slot`. The kernel reads via these pointers; this is
 * the swap step of the ring. Safe because the kernel is single-threaded
 * per matmul and we only mutate before dispatching. */
static void drive_repoint_indices(const inferbit_model *m,
                                  const ib_tensor_meta *t, int slot) {
    pqv2_t *mpq = (pqv2_t *)t->pq;
    void *l1_buf = (slot == 1) ? m->drive_indices_scratch2
                                : m->drive_indices_scratch;
    mpq->indices = (const uint8_t *)l1_buf;
    if (mpq->l2_kind == 2 && mpq->l2_indices_file_offset != 0 &&
        m->drive_l2_indices_scratch) {
        void *l2_buf = (slot == 1) ? m->drive_l2_indices_scratch2
                                    : m->drive_l2_indices_scratch;
        if (l2_buf) mpq->l2_indices = (const uint8_t *)l2_buf;
    }
}

/* Replacement for the old drive_load_indices. Ensures the active scratch
 * slot has tensor t's indices loaded and pq->indices points at it. Then
 * kicks off a background prefetch for the next tensor in decode order
 * (so the next matmul's I/O overlaps with this matmul's compute). */
static int drive_load_indices(const inferbit_model* m, const ib_tensor_meta* t) {
    if (!m || m->residency_mode != 1) return 0;
    if (!t || !t->pq) return 0;
    if (!m->drive_indices_scratch || m->drive_fd < 0) return 0;
    const pqv2_t* pq = t->pq;
    size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
    if (bytes == 0 || bytes > m->drive_indices_scratch_size) return -1;
    off_t off = (off_t)pq->indices_file_offset;
    if (off == 0) return 0;     /* not redirected; mmap'd path */

    /* L2 (pyramid residual) presence + size for this tensor. */
    int t_has_l2 = (pq->l2_kind == 2 && pq->l2_indices_file_offset != 0 &&
                    m->drive_l2_indices_scratch != NULL);
    size_t l2_bytes = 0;
    if (t_has_l2) {
        l2_bytes = drive_l2_indices_bytes(pq);
        if (l2_bytes == 0 || l2_bytes > m->drive_l2_indices_scratch_size)
            t_has_l2 = 0;   /* can't cache L2 → fall through to normal stream */
    }

    /* ── RAM-residency throttle (the "burst"): serve from RAM when this tensor's
     * L1 indices — AND, for pyramid, its L2 residual indices — are resident in
     * the hot-pool, then skip the disk stream + prefetch entirely. Cached bytes
     * are identical to streamed bytes → BIT-EXACT; only residency/speed change,
     * never the output. The pool fills on misses (promote below); its size IS
     * the dial (IB_HOT_POOL_MB → more tensors served from RAM → fewer disk
     * reads → faster, at exact quality). The arena copies are stable for this
     * matmul (a later promote/compaction runs only on a future miss). NOTE:
     * both L1 and L2 must be resident before short-circuiting — otherwise
     * pq->l2_indices would be left pointing at a stale scratch slot. */
    {
        const void *hotL1 = ib_hot_lookup_key(m, (size_t)off);
        const void *hotL2 = t_has_l2
            ? ib_hot_lookup_key(m, (size_t)pq->l2_indices_file_offset)
            : (const void *)1;   /* flat: no L2 to serve */
        if (hotL1 && hotL2) {
            ((pqv2_t *)pq)->indices = (const uint8_t *)hotL1;
            if (t_has_l2) ((pqv2_t *)pq)->l2_indices = (const uint8_t *)hotL2;
            return 0;
        }
    }

    ib_drive_pf_state *st = drive_pf_get(m);
    if (!st) {
        /* Prefetcher unavailable → legacy synchronous path into slot 0. */
        int rc = drive_sync_load_to_slot(m, t, 0);
        if (rc == 0) {
            drive_repoint_indices(m, t, 0);
            /* Promote the just-streamed L1 (and L2) indices into the RAM
             * hot-pool so the next token serves them from RAM (bit-exact).
             * pq->indices / pq->l2_indices now point at scratch slot 0. */
            (void)ib_hot_promote_bytes((inferbit_model *)m, (size_t)off,
                                       (const void *)pq->indices, bytes);
            if (t_has_l2)
                (void)ib_hot_promote_bytes((inferbit_model *)m,
                                           (size_t)pq->l2_indices_file_offset,
                                           (const void *)pq->l2_indices, l2_bytes);
        }
#if !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
        (void)posix_fadvise(m->drive_fd, off, (off_t)bytes, POSIX_FADV_DONTNEED);
#endif
        return rc;
    }

    int ready_slot = -1;
    pthread_mutex_lock(&st->mu);
    /* Wait for any pending prefetch to land — it may or may not be for
     * us. We don't preemptively cancel because a partial pread of an
     * unrelated tensor is harmless (just wasted I/O for a single tensor;
     * the case is rare — only on the very first call). */
    pf_wait_idle_locked(st);
    if (st->done_tensor == t && st->done_slot >= 0) {
        ready_slot = st->done_slot;
        st->done_tensor = NULL;
        st->done_slot = -1;
    } else {
        st->done_tensor = NULL;
        st->done_slot = -1;
    }
    pthread_mutex_unlock(&st->mu);

    if (ready_slot < 0) {
        /* Prefetch missed (first matmul, sparsity-masked detour, etc.).
         * Sync-load into slot 0 — slot 1 is now free for the next
         * prefetch kick below. */
        if (drive_sync_load_to_slot(m, t, 0) != 0) return -1;
        ready_slot = 0;
    }
    drive_repoint_indices(m, t, ready_slot);

    /* Promote the just-streamed L1 (and L2) indices into the RAM hot-pool
     * (bit-exact) BEFORE kicking the next prefetch — pq->indices/l2_indices
     * point at scratch `ready_slot`, which the next prefetch (other slot) won't
     * touch, so the copy is stable. Next token's lookup hits and skips disk. */
    (void)ib_hot_promote_bytes((inferbit_model *)m, (size_t)off,
                               (const void *)pq->indices, bytes);
    if (t_has_l2)
        (void)ib_hot_promote_bytes((inferbit_model *)m,
                                   (size_t)pq->l2_indices_file_offset,
                                   (const void *)pq->l2_indices, l2_bytes);

    /* Kick the prefetch for the NEXT tensor in decode order, into the
     * OTHER slot. If t isn't in the order list (sparse / output_head
     * tail) we just skip — the next call will sync-load. */
    int idx = drive_order_index(m, t);
    if (idx >= 0) {
        int next = idx + 1;
        if (next >= m->drive_pq_order_len) next = 0;   /* wrap to next decode step */
        const ib_tensor_meta *t_next = m->drive_pq_order[next];
        int next_slot = ready_slot ^ 1;
        pthread_mutex_lock(&st->mu);
        st->req_tensor = t_next;
        st->req_slot = next_slot;
        st->req_raw = 0;
        st->req_ok_l1 = 0;
        st->req_pending_l1 = 1;
        st->req_pending_l2 = 1;
        pthread_cond_signal(&st->req_cv);
        pthread_cond_signal(&st->req_cv_l2);
        pthread_mutex_unlock(&st->mu);
    }

#if !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
    /* Solution 4: on Linux, drop the just-read region from the page
     * cache so subsequent matmuls aren't biased by it. No-op on Darwin. */
    (void)posix_fadvise(m->drive_fd, off, (off_t)bytes, POSIX_FADV_DONTNEED);
#endif
    return 0;
}

/* Issue a raw lane-group prefetch into `slot`: L1 [l1_off, l1_len) and,
 * when l2_len>0, L2 [l2_off, l2_len). Caller MUST hold st->mu and have
 * ensured the ring is idle (no pending/in-flight request). Wakes both
 * workers; they pread in parallel into scratch[slot] / l2_scratch[slot]. */
static void drive_pf_issue_raw(ib_drive_pf_state *st, int slot,
                               off_t l1_off, size_t l1_len,
                               off_t l2_off, size_t l2_len) {
    st->req_raw = 1;
    st->req_slot = slot;
    st->req_raw_l1_off = l1_off;
    st->req_raw_l1_len = l1_len;
    st->req_raw_l2_off = l2_off;
    st->req_raw_l2_len = l2_len;
    st->req_ok_l1 = 0;
    st->req_pending_l1 = 1;
    st->req_pending_l2 = 1;
    pthread_cond_signal(&st->req_cv);
    pthread_cond_signal(&st->req_cv_l2);
}

/* Row-split task for the threaded paged compute: each worker accumulates a
 * disjoint output-row tile [start,end) of one lane-group into acc[] (and
 * acc_l2[] for pyramid — pqv2_acc_csrange row-ranges the L2 unpack+gather,
 * so each worker unpacks only its [start,end) L2 slice into its own
 * scratch and writes disjoint acc_l2 rows). */
typedef struct {
    const pqv2_t *pq;
    const float  *x;
    const float  *cb;        /* pq->cb_fp32 */
    const float  *l2_cb;     /* pq->l2_cb_fp32, or NULL (flat) */
    float        *acc;
    float        *acc_l2;    /* or NULL (flat) */
    uint32_t      cs_start;  /* lane-group start (local lane 0) */
    uint32_t      cs_count;
    int           use_l2;    /* burst: 0 = L1-only (skip L2 unpack+gather) */
} paged_csrange_task;

static void paged_csrange_row_task(void *raw, int tid, int start, int end) {
    (void)tid;
    const paged_csrange_task *a = (const paged_csrange_task *)raw;
    pqv2_acc_csrange(a->pq, a->x, a->cb, a->l2_cb,
                     a->acc, a->acc_l2,
                     a->cs_start, a->cs_count,
                     (uint32_t)start, (uint32_t)end,
                     a->use_l2);   /* burst: profile-driven L1+L2 vs L1-only */
}

/* Accumulate one lane-group [lane0, lane0+gc) into acc (and acc_l2).
 * When thread_compute, row-split the output across the model thread pool
 * (32-aligned tiles → bit-identical, disjoint rows → no race); else run
 * the single-thread kernel. thread_compute is only set for L2-free
 * tensors, so the threaded path never touches L2. */
static inline void paged_compute_group(const inferbit_model *m, pqv2_t *pq,
                                       const float *x, float *acc, float *acc_l2,
                                       int has_l2, int use_l2,
                                       uint32_t lane0, uint32_t gc,
                                       uint32_t M, int thread_compute) {
    /* `has_l2` is the structural "this tensor has a pyramid L2 residual";
     * `use_l2` is the active profile's runtime gate (0 = L1-only burst). The
     * kernel only reads/computes L2 when both hold. The caller has already
     * skipped the L2 prefetch/stream when use_l2==0, so acc_l2 stays zeroed. */
    int do_l2 = has_l2 && use_l2;
    if (thread_compute) {
        paged_csrange_task ta = { pq, x, pq->cb_fp32,
                                  do_l2 ? pq->l2_cb_fp32 : NULL,
                                  acc, do_l2 ? acc_l2 : NULL,
                                  lane0, gc, use_l2 };
        int nt = m->num_threads;
        uint32_t chunk = (M + (uint32_t)nt - 1) / (uint32_t)nt;
        chunk = (chunk + 31u) & ~31u;          /* 32-align for the K256 stride */
        if (chunk < 32u) chunk = 32u;
        ib_pool_run(m->thread_pool, paged_csrange_row_task, &ta,
                    (int)M, (int)chunk);
    } else {
        pqv2_acc_csrange(pq, x, pq->cb_fp32, do_l2 ? pq->l2_cb_fp32 : NULL,
                         acc, do_l2 ? acc_l2 : NULL, lane0, gc, 0, M,
                         use_l2);   /* burst: profile-driven L1+L2 vs L1-only */
    }
}

/* ── Peak-RAM PAGED drive matmul ────────────────────────────────────
 *
 * For tensors whose total L1 index bytes exceed the (capped) scratch
 * slot, stream the indices in contiguous (chunk,subchunk) lane-groups —
 * each group ≤ the slot — and accumulate the partial dot-products into a
 * persistent acc[M] (and acc_l2[M] for pyramid) across groups. row_scale
 * is applied ONCE at the end:  y[m] = acc[m]*row_scale[m] + acc_l2[m].
 *
 * Bit-identity: the lanes are processed in strictly increasing
 *   cs = c*ns + s
 * order across all groups, and pqv2_acc_csrange uses the SAME per-lane
 * LUT-build + NEON gather body as the non-paged kernels, so the summed
 * acc is identical to a single whole-tensor matvec; applying row_scale
 * once at the end (not per-group) keeps the fp32 arithmetic identical.
 *
 * Layout: the on-disk L1 region is chunk-major [n_chunks][n_subchunks][M],
 * i.e. nc*ns contiguous M-byte "lanes". A lane-group [g0, g0+gc) is ONE
 * contiguous pread of gc*M bytes at indices_file_offset + g0*M. The
 * matching L2 region is [n_chunks][n_subchunks][row_bytes(M)] (row_bytes
 * branches on l2_idx_bits), so the L2 group is gc*row_bytes at
 * l2_indices_file_offset + g0*row_bytes.
 *
 * Returns 1 if it handled the matmul, 0 if not applicable (caller falls
 * back to the normal load+dispatch path).
 *
 * Pipelining: CURRENTLY SYNCHRONOUS — each lane-group is pread then
 * computed. The two scratch slots are used in alternation so a future
 * async extension (issue group g+1's pread while the kernel runs group g)
 * can drop in without restructuring. TODO(peak-ram): extend the 2-slot
 * prefetch ring's worker to take a (offset, length, slot) request so
 * lane-group I/O overlaps compute the way whole-tensor prefetch does. */
/* Paged drive matvec, batched over B input positions.
 *
 * x is [B][N] (position b at x + b*pq->N), y is [B][M] (b at y + b*pq->M).
 * THE AMORTISATION: each lane-group's indices are streamed from disk ONCE
 * and then accumulated for ALL B positions (B separate acc slabs), so the
 * expensive disk read is shared across the batch instead of repeated B times.
 * This is what makes the speculative/batched verify cheap in drive mode —
 * read the model once per round, apply to k positions. B=1 is byte-identical
 * to the original single-position path. Caller must ensure B<=8 (the acc pool
 * is sized for 8 column-slabs; larger B falls back to per-position). */
static int drive_paged_matvec(const inferbit_model *m,
                              const ib_tensor_meta *t,
                              const float *x, int B, float *y) {
    if (!m || m->residency_mode != 1) return 0;
    if (!t || !t->pq) return 0;
    if (m->drive_fd < 0 || !m->drive_indices_scratch) return 0;
    if (B < 1) B = 1;
    pqv2_t *pq = (pqv2_t *)t->pq;
    if (!pq->cb_fp32) return 0;                 /* csrange needs fp32 cb */
    off_t off = (off_t)pq->indices_file_offset;
    if (off == 0) return 0;                     /* not redirected */
    const size_t Nstride = (size_t)pq->N;       /* per-position x stride */

    uint32_t M = pq->M;
    uint32_t ns = pq->n_subchunks;
    uint32_t nc = pq->N / pq->G;
    size_t total_lanes = (size_t)nc * ns;
    size_t lane_bytes = (size_t)M;              /* one (c,s) lane = M bytes */
    if (lane_bytes == 0 || total_lanes == 0) return 0;
    size_t total_bytes = total_lanes * lane_bytes;

    size_t slot_size = m->drive_indices_scratch_size;
    int l1_fits = (total_bytes <= slot_size);

    /* ── L2 (pyramid) parameters + fit check ─────────────────────────
     * A tensor that fits the L1 slot can STILL overflow the (capped) L2
     * scratch: the non-paged drive_load_indices/prefetch path writes the
     * tensor's WHOLE L2 residual into the L2 slot, and that slot is sized
     * to a lane-GROUP (not the whole tensor) when the page cap is active.
     * So we must PAGE whenever EITHER L1 or L2 does not fit its slot —
     * otherwise the non-paged path corrupts memory (observed: pyramid
     * crash/garbage in the cap band where FFN L1 fits but FFN L2 does not). */
    int has_l2 = (pq->l2_kind == 2 && pq->l2_cb_fp32 && pq->l2_K <= 64);

    /* Burst L1-only: the active compute profile may select the coarse tier
     * (precision_tier==1), in which case the kernel skips the L2 unpack+
     * gather. To make L1-only actually save disk/RAM bytes we ALSO skip the
     * L2 index prefetch/stream/read below — not just the compute. This only
     * applies to genuine pyramid tensors (has_l2); flat tensors are
     * unaffected. With burst disabled (the default) ib_active_use_l2()==1, so
     * use_l2 stays 1 and every L2 path below is byte-identical to today. */
    int use_l2 = ib_active_use_l2((inferbit_model *)m);

    /* Compute the L2 geometry from the STRUCTURAL has_l2 (independent of the
     * burst tier) so the page-vs-no-page decision below is byte-identical to
     * today: a tensor that pages because its L2 overflows the slot must STILL
     * page in L1-only mode (the non-paged fallback would read a stale/garbage
     * L2 residual from the slot — see the corruption note above). The runtime
     * tier only gates the actual L2 prefetch/stream/read/compute via page_l2. */
    size_t l2_row = 0, l2_slot_size = 0, l2_total = 0;
    off_t l2_off = 0;
    if (has_l2) {
        l2_row       = pqv2_l2_row_bytes(M, pq->l2_idx_bits);
        l2_slot_size = m->drive_l2_indices_scratch_size;
        l2_total     = pqv2_l2_total_index_bytes(pq);
        l2_off       = (off_t)pq->l2_indices_file_offset;
        if (l2_row == 0) has_l2 = 0;   /* defensive: nothing to page */
    }
    int l2_fits = (!has_l2) || (l2_total <= l2_slot_size);

    /* page_l2 == "this matmul will actually read+compute L2": structural L2
     * present AND the active profile wants it (burst L1-only sets use_l2==0).
     * When 0 we skip every L2 prefetch/stream/read and zero the L2 residual,
     * but we DO NOT change the page-vs-no-page decision (l2_fits, above). */
    int page_l2 = has_l2 && use_l2;

    /* Both fit a slot → the unchanged non-paged path is safe; skip paging.
     * (Unchanged from today: uses the structural l1_fits/l2_fits. In L1-only
     * mode the non-paged path still reads L2 — it lacks a use_l2 gate — so an
     * L1-only step that takes this fall-through produces the EXACT L1+L2
     * result, i.e. correct but without the coarse byte savings. The savings
     * land on tensors that page, which is where L1-only matters for RAM.) */
    if (l1_fits && l2_fits) return 0;

    /* ── RAM-residency throttle for big (would-page) tensors ──────────────
     * Before streaming this tensor from disk in lane-groups, try to make it
     * RESIDENT in the hot-pool: serve a cached copy, or promote it from disk
     * once (fill-once). If resident, run the normal non-paged matmul on the
     * full resident indices and report "handled" (return 1) so the caller
     * skips the streaming path entirely. Bit-exact (same bytes) — only
     * residency/speed change. The hot-pool budget (IB_HOT_POOL_MB) is the
     * dial: more budget → more big tensors resident → fewer disk reads. Only
     * taken when L2 is read normally (page_l2 == has_l2) so we never cache an
     * L2 the burst tier won't read; burst L1-only keeps paging. */
    if (page_l2 == has_l2) {
        const void *hotL1 = ib_hot_lookup_key(m, (size_t)off);
        const void *hotL2 = has_l2 ? ib_hot_lookup_key(m, (size_t)l2_off)
                                   : (const void *)1;
        if (!hotL1) {
            void *dst = ib_hot_reserve((inferbit_model *)m, (size_t)off, total_bytes);
            if (dst && drive_pread_full(m->drive_fd, dst, total_bytes, off))
                hotL1 = dst;
        }
        if (has_l2 && hotL1 && !hotL2) {
            void *dst2 = ib_hot_reserve((inferbit_model *)m, (size_t)l2_off, l2_total);
            if (dst2 && drive_pread_full(m->drive_fd, dst2, l2_total, l2_off))
                hotL2 = dst2;
        }
        if (hotL1 && hotL2) {
            pq->indices = (const uint8_t *)hotL1;
            if (has_l2) pq->l2_indices = (const uint8_t *)hotL2;
            /* Resident: weights are in RAM; the batched kernel reads them once
             * and applies to all B positions (amortised). */
            if (B > 1 && pq->K == 256 && m->thread_pool && m->num_threads > 1) {
                pqv2_threaded_matvec_k256_batch(m, m->thread_pool, m->num_threads,
                                                pq, x, B, y);
            } else if (pq->K == 256 && m->thread_pool && m->num_threads > 1) {
                pqv2_threaded_matvec_k256(m, m->thread_pool, m->num_threads,
                                          pq, x, y);
            } else {
                for (int b = 0; b < B; b++)
                    pqv2_matvec_dispatch(pq, x + (size_t)b * Nstride,
                                         y + (size_t)b * (size_t)M);
            }
            return 1;   /* handled resident — caller skips the streaming path */
        }
    }

    /* From here we WILL page. Require the accumulator pools (and, for a
     * pyramid tensor, the L2 scratch ring + L2 acc pool). We REUSE the
     * model-scope threaded-matmul acc pools (sized n_threads × max_M ≥ M
     * floats, freed in model.c): they are idle between matmuls and the
     * paged path is single-threaded, so there is no overlap with the
     * threaded K=256 path. model.c disables the page cap when these pools
     * are unavailable (sizing every slot to the whole tensor), in which
     * case l1_fits && l2_fits held above and we already returned — so
     * reaching here without them is an unexpected config; fall back to the
     * non-paged path (whose own size guards prevent an overflow) rather
     * than page incorrectly. */
    if (!m->pqv2_thread_acc_pool ||
        m->pqv2_thread_acc_pool_floats < (size_t)B * (size_t)M) return 0;
    if (page_l2 &&
        (!m->drive_l2_indices_scratch ||
         pq->l2_indices_file_offset == 0 ||
         !m->pqv2_thread_acc_l2_pool ||
         m->pqv2_thread_acc_l2_pool_floats < (size_t)B * (size_t)M)) {
        return 0;   /* can't page L2 safely → let caller fall back */
    }

    /* Lanes per group: the L1 slot bounds it; if L2 is paged, the L2 slot
     * may bound it tighter. At least 1 lane per group (slots are sized
     * with a one-lane floor at setup, so this never starves). */
    size_t lanes_per_group = slot_size / lane_bytes;
    if (page_l2 && l2_row > 0) {
        size_t l2_lpg = l2_slot_size / l2_row;
        if (l2_lpg < lanes_per_group) lanes_per_group = l2_lpg;
    }
    if (lanes_per_group == 0) lanes_per_group = 1;

    float *acc = m->pqv2_thread_acc_pool;
    float *acc_l2 = page_l2 ? m->pqv2_thread_acc_l2_pool : NULL;
    memset(acc, 0, (size_t)B * (size_t)M * sizeof(float));
    if (acc_l2) memset(acc_l2, 0, (size_t)B * (size_t)M * sizeof(float));

    /* Thread the paged compute by row-splitting each lane-group across the
     * model thread pool (works for flat AND pyramid: pqv2_acc_csrange
     * row-ranges the L2 unpack+gather, so each worker handles a disjoint
     * acc/acc_l2 row tile). Gated on:
     *   - M >= 512 (amortise dispatch);
     *   - the in-focus slot is CACHE-RESIDENT (slot_size <= 4 MB) → the
     *     per-group gather is COMPUTE-bound and parallelises well. At large
     *     caps (e.g. the default 8 MB) only lm_head pages and it is
     *     I/O-bound — the prefetch pipeline already hides its compute, so
     *     threading there only adds dispatch overhead (measured -3%). The
     *     win is at aggressive caps (cap<=4) where the FFN pages into
     *     cache-sized slots. */
    int thread_compute = (m->thread_pool && m->num_threads > 1 &&
                          M >= 512 && slot_size <= (4u << 20));
    /* A/B + safety knob: IB_PAGE_NOTHREAD=1 forces single-thread paged
     * compute (isolates the threading win; also a fallback). */
    {
        const char *e = getenv("IB_PAGE_NOTHREAD");
        if (e && e[0] == '1') thread_compute = 0;
    }

    /* Drain any in-flight whole-tensor prefetch before reusing the slots
     * (avoids racing the prefetch worker on scratch[]). */
    ib_drive_pf_state *st = drive_pf_get(m);
    if (st) {
        pthread_mutex_lock(&st->mu);
        pf_wait_idle_locked(st);
        st->done_tensor = NULL;
        st->done_slot = -1;
        pthread_mutex_unlock(&st->mu);
    }

    void *slot_l1[2]  = { m->drive_indices_scratch,
                          m->drive_indices_scratch2 };
    void *slot_l2[2]  = { m->drive_l2_indices_scratch,
                          m->drive_l2_indices_scratch2 };
    int have_slot1 = (slot_l1[1] != NULL);
    /* Pipelining needs the 2-slot ring AND, for pyramid, both L2 slots so
     * the next group's L2 can land in the OTHER slot while we compute.
     * (page_l2, not has_l2: an L1-only burst step never touches L2 slots,
     * so a pyramid tensor can still pipeline on L1 alone.) */
    int can_pipeline = (st && have_slot1 &&
                        (!page_l2 || (slot_l2[0] && slot_l2[1])));

    if (can_pipeline) {
        /* ── Pipelined: prefetch group g+1 (into the other slot) while the
         * kernel accumulates group g. Only one request is ever in flight;
         * the 2-slot ping-pong guarantees the in-flight slot != the slot
         * being read, so there is no scratch race. L1 and L2 of a group
         * are pread in parallel by the two ring workers. */
        size_t gc0 = (lanes_per_group < total_lanes) ? lanes_per_group
                                                     : total_lanes;
        pthread_mutex_lock(&st->mu);
        drive_pf_issue_raw(st, /*slot=*/0,
                           off, gc0 * lane_bytes,
                           page_l2 ? l2_off : 0,
                           page_l2 ? gc0 * l2_row : 0);
        pthread_mutex_unlock(&st->mu);

        int gi = 0;
        for (size_t lane0 = 0; lane0 < total_lanes;
             lane0 += lanes_per_group, gi++) {
            size_t gc = lanes_per_group;
            if (lane0 + gc > total_lanes) gc = total_lanes - lane0;
            int slot = gi & 1;

            pthread_mutex_lock(&st->mu);
            pf_wait_idle_locked(st);          /* group gi ready in `slot` */
            /* Kick the NEXT group into the other slot, overlapping compute. */
            size_t lane0_n = lane0 + lanes_per_group;
            if (lane0_n < total_lanes) {
                size_t gc_n = lanes_per_group;
                if (lane0_n + gc_n > total_lanes) gc_n = total_lanes - lane0_n;
                drive_pf_issue_raw(st, /*slot=*/(gi + 1) & 1,
                                   off + (off_t)(lane0_n * lane_bytes),
                                   gc_n * lane_bytes,
                                   page_l2 ? l2_off + (off_t)(lane0_n * l2_row) : 0,
                                   page_l2 ? gc_n * l2_row : 0);
            }
            pthread_mutex_unlock(&st->mu);

            pq->indices = (const uint8_t *)slot_l1[slot];
            if (page_l2) pq->l2_indices = (const uint8_t *)slot_l2[slot];
            for (int b = 0; b < B; b++)
                paged_compute_group(m, pq, x + (size_t)b * Nstride,
                                    acc + (size_t)b * (size_t)M,
                                    acc_l2 ? acc_l2 + (size_t)b * (size_t)M : NULL,
                                    has_l2, use_l2,
                                    (uint32_t)lane0, (uint32_t)gc, M, thread_compute);
        }

        /* Restore clean ring state: clear the raw flag + stale done_tensor
         * so the next whole-tensor matmul's prefetch logic starts fresh. */
        pthread_mutex_lock(&st->mu);
        pf_wait_idle_locked(st);
        st->req_raw = 0;
        st->done_tensor = NULL;
        st->done_slot = -1;
        pthread_mutex_unlock(&st->mu);
    } else {
        /* ── Synchronous fallback (prefetcher unavailable / single slot) ──
         * Pread each lane-group then compute it. Uses slot 0 only. */
        for (size_t lane0 = 0; lane0 < total_lanes; lane0 += lanes_per_group) {
            size_t gc = lanes_per_group;
            if (lane0 + gc > total_lanes) gc = total_lanes - lane0;
            size_t l1_bytes = gc * lane_bytes;
            if (!drive_pread_full(m->drive_fd, slot_l1[0], l1_bytes,
                                  off + (off_t)(lane0 * lane_bytes))) {
                return 0;   /* I/O error → bail */
            }
            pq->indices = (const uint8_t *)slot_l1[0];
            if (page_l2) {
                void *l2dst = slot_l2[0];
                if (!l2dst) return 0;   /* never drop a pyramid residual */
                if (!drive_pread_full(m->drive_fd, l2dst, gc * l2_row,
                                      l2_off + (off_t)(lane0 * l2_row))) {
                    return 0;
                }
                pq->l2_indices = (const uint8_t *)l2dst;
            }
            for (int b = 0; b < B; b++)
                paged_compute_group(m, pq, x + (size_t)b * Nstride,
                                    acc + (size_t)b * (size_t)M,
                                    acc_l2 ? acc_l2 + (size_t)b * (size_t)M : NULL,
                                    has_l2, use_l2,
                                    (uint32_t)lane0, (uint32_t)gc, M, thread_compute);
#if !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
            (void)posix_fadvise(m->drive_fd, off + (off_t)(lane0 * lane_bytes),
                                (off_t)l1_bytes, POSIX_FADV_DONTNEED);
#endif
        }
    }

    /* Final reduction (per position): apply row_scale once + fold L2. */
    for (int b = 0; b < B; b++) {
        const float *acc_b   = acc + (size_t)b * (size_t)M;
        const float *accl2_b = acc_l2 ? acc_l2 + (size_t)b * (size_t)M : NULL;
        float *y_b = y + (size_t)b * (size_t)M;
        for (uint32_t mm = 0; mm < M; mm++) {
            float rs = pqv2_h2f(pq->row_scale[mm]);
            y_b[mm] = acc_b[mm] * rs + (accl2_b ? accl2_b[mm] : 0.0f);
        }
    }
    return 1;
}

/* ── FP16 conversion ────────────────────────────────────────── */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Subnormal */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        /* Inf/NaN */
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Convert FP16 scale array to FP32 (temporary buffer) */
static void scales_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* Read FP16 weight as FP32 (for norms) */
static void fp16_weights_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* ── Embedding lookup ───────────────────────────────────────── */

/* Non-static: also used by inferbit_forward_with_hiddens (forward_hiddens.c)
 * to decode token IDs into fp32 embeddings for the Metal prefill path. */
void ib_embedding_lookup(const inferbit_model* m, int token_id, float* out);
void ib_embedding_lookup(const inferbit_model* m, int token_id, float* out) {
    int hidden = m->header.hidden_size;
    const ib_tensor_meta* emb = &m->token_embedding;

    if (emb->pq) {
        /* PQv2 embedding: decode one row. */
        const pqv2_t* pq = emb->pq;
        uint32_t G = pq->G;
        uint32_t ns = pq->n_subchunks;
        uint32_t K = pq->K;
        uint32_t HALF = pq->half;
        uint32_t nc = pq->N / G;
        uint32_t total = nc * ns;
        const int8_t* cb_q = (const int8_t*)pq->cb_q;           /* [ns][K][HALF] */
        const uint16_t* cb_s = (const uint16_t*)pq->cb_scale;   /* [ns][K] */
        float rs = pq->row_scale ? fp16_to_fp32(((const uint16_t*)pq->row_scale)[token_id]) : 1.0f;

        /* Doc-35 feature 1: if the pre-transposed sidecar is built and
         * the embedding has a sidecar entry, pread one ROW (total bytes)
         * from the sidecar instead of mmap-reading `total` widely-strided
         * single bytes. Sidecar layout is [token][total] so a row is
         * one contiguous pread = ~1024 bytes. Keeps the source mmap
         * region cold (cache-eviction-friendly). */
        if (m->residency_mode == 1 && m->drive_fd_pretransposed >= 0
            && pq->indices_pretransposed_offset != 0) {
            uint8_t row_buf[2048];   /* nc*ns ≤ 2048 in practice */
            if (total > sizeof(row_buf)) goto embed_mmap_path;
            off_t off = (off_t)pq->indices_pretransposed_offset
                      + (off_t)token_id * (off_t)total;
            size_t done = 0;
            while (done < total) {
                ssize_t r = pread(m->drive_fd_pretransposed,
                                  row_buf + done, total - done,
                                  off + (off_t)done);
                if (r <= 0) { if (r == -1 && errno == EINTR) continue; goto embed_mmap_path; }
                done += (size_t)r;
            }
            for (uint32_t c = 0; c < nc; c++) {
                for (uint32_t s = 0; s < ns; s++) {
                    uint8_t k = row_buf[c * ns + s];
                    float scl = fp16_to_fp32(cb_s[s * K + k]) * rs;
                    for (uint32_t h = 0; h < HALF; h++) {
                        int8_t q = cb_q[(s * K + k) * HALF + h];
                        out[c * G + s * HALF + h] = (float)q * scl;
                    }
                }
            }
            return;
        }

embed_mmap_path:
        {
            const uint8_t* idx_base = (const uint8_t*)pq->indices;
            /* L1 indices are always chunk-major on disk:
             *   idx[(c*ns+s)*M + token_id] */
            for (uint32_t c = 0; c < nc; c++) {
                for (uint32_t s = 0; s < ns; s++) {
                    uint8_t k = idx_base[((size_t)c * ns + s) * pq->M + token_id];
                    float scl = fp16_to_fp32(cb_s[s * K + k]) * rs;
                    for (uint32_t h = 0; h < HALF; h++) {
                        int8_t q = cb_q[(s * K + k) * HALF + h];
                        out[c * G + s * HALF + h] = (float)q * scl;
                    }
                }
            }
        }
        return;
    }
    if (emb->bits == 8) {
        /* INT8 embedding: dequantize row */
        const int8_t* data = (const int8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        const int8_t* row = data + (size_t)token_id * hidden;

        if (scales_raw) {
            /* Per-row scale factor */
            const uint16_t* scales_fp16 = (const uint16_t*)scales_raw;
            float scale = fp16_to_fp32(scales_fp16[token_id]);
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i] * scale;
            }
        } else {
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i];
            }
        }
    } else if (emb->bits == 16) {
        /* FP16 embedding */
        const uint16_t* data = (const uint16_t*)tensor_data(m, emb);
        const uint16_t* row = data + (size_t)token_id * hidden;
        for (int i = 0; i < hidden; i++) {
            out[i] = fp16_to_fp32(row[i]);
        }
    } else if (emb->bits == 4) {
        /* INT4 embedding */
        const uint8_t* data = (const uint8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        size_t row_bytes = (size_t)hidden / 2;
        const uint8_t* row = data + (size_t)token_id * row_bytes;
        float scale = 1.0f;
        if (scales_raw) {
            scale = fp16_to_fp32(((const uint16_t*)scales_raw)[token_id]);
        }
        for (int i = 0; i < hidden; i += 2) {
            uint8_t byte = row[i / 2];
            out[i]     = (float)((int8_t)(byte & 0x0F) - 8) * scale;
            out[i + 1] = (float)((int8_t)((byte >> 4) & 0x0F) - 8) * scale;
        }
    }
}

/* ── Matmul dispatch ────────────────────────────────────────── */

/*
 * Run quantized matmul for a tensor: out[M] = weights[M,N] @ input[N]
 * Handles bit-width dispatch and scale conversion.
 * `scale_buf` is a caller-provided temporary buffer of at least M floats.
 */
static void pqv2_matvec_dispatch(const pqv2_t *t, const float *x, float *y) {
    if (t->K == 256)      pqv2_matvec_tbl_int8_k256(t, x, y);
    else if (t->K == 128) pqv2_matvec_tbl_int8_k128(t, x, y);
    else if (t->K <= 64)  pqv2_matvec_tbl_int8(t, x, y);
    else                  pqv2_matvec_lut(t, x, y);
}

/* Per-chunk threading: each worker processes a slice of chunks, accumulating
 * into its own thread-local acc[M]. Main thread reduces across workers and
 * applies row_scale + L2 contribution.
 *
 * Why per-chunk and not per-row: the kernel builds an LUT per (chunk, subchunk)
 * that's INDEPENDENT of M but DEPENDS on x. Per-row threading would force
 * each worker to redundantly rebuild every LUT (4× total LUT-build work).
 * Per-chunk threading distributes LUT-build evenly with no redundancy. */
typedef struct {
    const pqv2_t *t;
    const float  *x;
    float        *acc_pool;
    float        *acc_l2_pool;
    uint32_t      M;
    int           chunk_size;
    int           n_slots;
    float         skip_thresh;   /* 0 = no skip */
} ib_pqv2_chunks_arg;

static void ib_pqv2_chunks_task(void *arg, int tid, int start, int end) {
    (void)tid;
    const ib_pqv2_chunks_arg *a = (const ib_pqv2_chunks_arg*)arg;
    int slot = start / a->chunk_size;
    if (slot < 0) slot = 0;
    if (slot >= a->n_slots) slot = a->n_slots - 1;
    float *acc    = a->acc_pool    + (size_t)slot * a->M;
    float *acc_l2 = a->acc_l2_pool ? a->acc_l2_pool + (size_t)slot * a->M : NULL;
    if (a->skip_thresh > 0.0f) {
        pqv2_acc_tbl_int8_k256_chunks_skip(a->t, a->x,
                                              a->t->cb_fp32, a->t->l2_cb_fp32,
                                              acc, acc_l2,
                                              (uint32_t)start, (uint32_t)end,
                                              a->skip_thresh);
    } else {
        pqv2_acc_tbl_int8_k256_chunks(a->t, a->x,
                                        a->t->cb_fp32, a->t->l2_cb_fp32,
                                        acc, acc_l2,
                                        (uint32_t)start, (uint32_t)end);
    }
}

/* Forward decl for the single-position threaded variant (defined below). */
static void pqv2_threaded_matvec_k256(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x, float *y);

/* Batched-aware variant of the per-chunk threading. Same chunk-to-slot
 * mapping as the single-position threaded path so each output position's
 * fp32 summation order is bit-identical between single-token decode and
 * spec verify. acc pool layout: [n_slots, B, M]. acc_l2_pool (same
 * layout, NULL when the tensor has no L2 stage) carries the pyramid
 * residual contribution, folded into y at reduction time. */
typedef struct {
    const pqv2_t *t;
    const float  *x_batch;
    int           B;
    float        *acc_pool;
    float        *acc_l2_pool;
    uint32_t      M;
    int           chunk_size;
    int           n_slots;
} ib_pqv2_chunks_batch_arg;

static void ib_pqv2_chunks_batch_task(void *arg, int tid, int start, int end) {
    (void)tid;
    const ib_pqv2_chunks_batch_arg *a = (const ib_pqv2_chunks_batch_arg*)arg;
    int slot = start / a->chunk_size;
    if (slot < 0) slot = 0;
    if (slot >= a->n_slots) slot = a->n_slots - 1;
    /* Slot owns a [B, M] block in each pool. */
    float *acc    = a->acc_pool    + (size_t)slot * a->B * a->M;
    float *acc_l2 = a->acc_l2_pool ? a->acc_l2_pool + (size_t)slot * a->B * a->M
                                   : NULL;
    pqv2_acc_tbl_int8_k256_chunks_batch(a->t, a->x_batch, a->B,
                                          a->t->cb_fp32, a->t->l2_cb_fp32,
                                          acc, acc_l2,
                                          (uint32_t)start, (uint32_t)end);
}

static void pqv2_threaded_matvec_k256_batch(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x_batch, int B, float *y_batch)
{
    if (B <= 0) return;
    if (B == 1) {
        pqv2_threaded_matvec_k256(m, tp, n_threads, t, x_batch, y_batch);
        return;
    }
    uint32_t M = t->M;
    uint32_t n_chunks = t->N / t->G;
    /* Match the single-position threaded path: L2 is engaged iff l2_kind==2,
     * the fp32 L2 codebook is present, and l2_K <= 64 (kernel constraint).
     * Otherwise the per-position fallback pqv2_matvec_tbl_int8_k256_batch
     * routes through pqv2_matvec_tbl_int8_k256, which already handles L2
     * correctly, so spec verify on pyramid tensors stays bit-identical
     * to single-token decode for any tensor that bails out of threading. */
    if (!tp || n_threads <= 1 || n_chunks < (uint32_t)n_threads ||
        !t->cb_fp32 || t->K != 256 || B > 8) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
        return;
    }
    /* Burst L1-only gate (read once per matvec). When use_l2==0 we drop the
     * L2 stage for every position in the batch — no acc_l2 pool, no residual
     * fold — matching the single-position path and the drive use_l2==0 path.
     * Default (burst off) → 1, so this is byte-identical to today.
     * NB: the per-position fallback below (pqv2_matvec_tbl_int8_k256_batch)
     * still folds L2 unconditionally, but it is only reached when this
     * threaded path bails out on geometry/alloc, not on the L1-only gate. */
    int use_l2 = ib_active_use_l2((inferbit_model *)m);
    int has_l2 = (t->l2_kind == 2 && t->l2_cb_fp32 && t->l2_K <= 64) && use_l2;
    int chunks_per_task = ((int)n_chunks + n_threads - 1) / n_threads;
    int n_slots = ((int)n_chunks + chunks_per_task - 1) / chunks_per_task;
    size_t pool_floats = (size_t)n_slots * B * M;
    float *acc_pool = aligned_alloc(64,
        (pool_floats * sizeof(float) + 63) & ~(size_t)63);
    if (!acc_pool) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
        return;
    }
    memset(acc_pool, 0, pool_floats * sizeof(float));
    float *acc_l2_pool = NULL;
    int acc_l2_owned = 0;
    if (has_l2) {
        /* Reuse model-scope L2 scratch when it fits (typical for B=1
         * decode where pool_floats = n_slots × M ≤ n_threads × max_M).
         * Larger batches (B>1) fall through to a fresh aligned_alloc. */
        if (m && m->pqv2_thread_acc_l2_pool &&
            pool_floats <= m->pqv2_thread_acc_l2_pool_floats) {
            acc_l2_pool = m->pqv2_thread_acc_l2_pool;
        } else {
            acc_l2_pool = aligned_alloc(64,
                (pool_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!acc_l2_pool) {
                /* L2 alloc failed: fall back to the per-position path that
                 * routes through the single-position matvec (which handles
                 * L2 correctly), so we never silently drop the pyramid
                 * residual. */
                free(acc_pool);
                pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
                return;
            }
            acc_l2_owned = 1;
        }
        memset(acc_l2_pool, 0, pool_floats * sizeof(float));
    }
    ib_pqv2_chunks_batch_arg arg = {
        .t = t, .x_batch = x_batch, .B = B,
        .acc_pool = acc_pool, .acc_l2_pool = acc_l2_pool, .M = M,
        .chunk_size = chunks_per_task, .n_slots = n_slots,
    };
    ib_pool_run(tp, ib_pqv2_chunks_batch_task, &arg,
                 (int)n_chunks, chunks_per_task);

    /* Reduce per-position: y[b,m] = (sum_s acc[s,b,m]) * row_scale[m]
     *                              + sum_s acc_l2[s,b,m].
     * Slot order is fixed (s=0..n_slots-1) so this matches the
     * single-position threaded reduction at forward.c::pqv2_threaded_matvec_k256
     * exactly when B=1. */
    for (int b = 0; b < B; b++) {
        float *yb = y_batch + (size_t)b * M;
        for (uint32_t mm = 0; mm < M; mm++) {
            float a = 0.0f, al2 = 0.0f;
            for (int s = 0; s < n_slots; s++) {
                a += acc_pool[(size_t)s * B * M + (size_t)b * M + mm];
                if (acc_l2_pool) {
                    al2 += acc_l2_pool[(size_t)s * B * M + (size_t)b * M + mm];
                }
            }
            float rs = pqv2_h2f(t->row_scale[mm]);
            yb[mm] = a * rs + al2;
        }
    }
    free(acc_pool);
    if (acc_l2_owned) free(acc_l2_pool);
}

static void pqv2_threaded_matvec_k256(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x, float *y)
{
    uint32_t M = t->M;
    uint32_t n_chunks = t->N / t->G;
    /* Bail out to single-thread when threading wouldn't pay off. */
    if (!tp || n_threads <= 1 || n_chunks < (uint32_t)n_threads ||
        !t->cb_fp32 || t->K != 256) {
        pqv2_matvec_dispatch(t, x, y);
        return;
    }
    /* Burst L1-only gate: read ONCE per matvec (not per row/chunk). When the
     * active profile selects the L1-only coarse tier, ib_active_use_l2()==0
     * and we drop the entire L2 stage for this call — no acc_l2 scratch, no
     * L2 index read, no residual gather/fold — exactly as the drive-mode
     * paged kernel does with use_l2==0. With burst disabled (the default)
     * this is 1, so has_l2 is unchanged and every L2 path below is
     * byte-identical to today. The chunk kernel engages L2 iff acc_l2 is
     * non-NULL, so gating has_l2 here (→ acc_l2_pool stays NULL) is the
     * single point that produces the L1-only reconstruction. */
    int use_l2 = ib_active_use_l2((inferbit_model *)m);
    int has_l2 = (t->l2_kind == 2 && t->l2_cb_fp32 && t->l2_K <= 64) && use_l2;
    int chunks_per_task = ((int)n_chunks + n_threads - 1) / n_threads;
    int n_slots = ((int)n_chunks + chunks_per_task - 1) / chunks_per_task;
    size_t pool_floats = (size_t)n_slots * M;
    /* Use model-scope scratch to avoid per-call aligned_alloc. The
     * scratch is sized for n_threads × max_M; fall back to a fresh
     * malloc only if (somehow) the request exceeds that budget. */
    float *acc_pool;
    int acc_pool_owned = 0;
    if (m && m->pqv2_thread_acc_pool && pool_floats <= m->pqv2_thread_acc_pool_floats) {
        acc_pool = m->pqv2_thread_acc_pool;
    } else {
        acc_pool = aligned_alloc(64,
            (pool_floats * sizeof(float) + 63) & ~(size_t)63);
        if (!acc_pool) { pqv2_matvec_dispatch(t, x, y); return; }
        acc_pool_owned = 1;
    }
    memset(acc_pool, 0, pool_floats * sizeof(float));
    float *acc_l2_pool = NULL;
    int acc_l2_owned = 0;
    if (has_l2) {
        /* Prefer model-scope L2 scratch (sized n_threads × max_M, same as
         * acc_pool). Falls back to fresh aligned_alloc only if the request
         * somehow exceeds the budget. Eliminates ~88 aligned_alloc/free per
         * decode token on pyramid models. */
        if (m && m->pqv2_thread_acc_l2_pool &&
            pool_floats <= m->pqv2_thread_acc_l2_pool_floats) {
            acc_l2_pool = m->pqv2_thread_acc_l2_pool;
        } else {
            acc_l2_pool = aligned_alloc(64,
                (pool_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!acc_l2_pool) {
                /* L2 alloc failed: do NOT silently run the threaded path with
                 * acc_l2_pool=NULL — the kernel would skip the L2 path entirely
                 * (see ib_pqv2_chunks_task → pqv2_acc_tbl_int8_k256_chunks_inner;
                 * acc_l2==NULL means "no L2"), which DROPS the pyramid residual
                 * and degrades a pyramid (l2_kind==2) model to a flat one
                 * (60% PPL regression observed on tl-pyramid.ibf in RAM mode
                 * where the mmap'd weights leave less headroom for the
                 * per-matmul aligned_alloc; drive mode evicts those pages and
                 * the alloc succeeds, which is why drive PPL was BETTER than
                 * RAM PPL — the bug is RAM-mode-only). Fall back to the
                 * single-threaded matvec, which uses one acc/acc_l2 pair the
                 * size of a single matvec (M floats each) and is allocated
                 * fresh inside pqv2_matvec_tbl_int8_k256. */
                if (acc_pool_owned) free(acc_pool);
                pqv2_matvec_dispatch(t, x, y);
                return;
            }
            acc_l2_owned = 1;
        }
        memset(acc_l2_pool, 0, pool_floats * sizeof(float));
    }
    /* Activation-aware skip: skip (c,s) iters with max|x_slice| < ratio *
     * max|x|. 1% threshold is essentially lossless on transformer
     * activations. Skip rate naturally adapts: outlier-heavy early layers
     * skip a lot, diffuse later layers skip little.
     *
     * The ratio is the cached `m->active_skip_thresh_ratio` (M1 burst):
     * inferbit_burst_attach seeds it from IB_PQV2_SKIP at load, and a later
     * BURST profile overrides it per step. Reading the cached float here
     * (instead of getenv/atof per matmul) preserves today's behaviour
     * exactly for the default, burst-disabled run while removing the
     * per-call env parse from the hot path. */
    float skip_thresh = 0.0f;
    {
        float ratio = m ? m->active_skip_thresh_ratio : 0.0f;
        if (ratio > 0.0f && ratio < 1.0f) {
            float xmax = 0.0f;
            for (uint32_t i = 0; i < t->N; i++) {
                float v = x[i]; if (v < 0) v = -v;
                if (v > xmax) xmax = v;
            }
            skip_thresh = ratio * xmax;
        }
    }
    ib_pqv2_chunks_arg arg = {
        .t = t, .x = x,
        .acc_pool = acc_pool, .acc_l2_pool = acc_l2_pool,
        .M = M,
        .chunk_size = chunks_per_task,
        .n_slots = n_slots,
        .skip_thresh = skip_thresh,
    };
    ib_pool_run(tp, ib_pqv2_chunks_task, &arg, (int)n_chunks, chunks_per_task);

    /* Reduce: sum across deterministic slot order, then apply row_scale + L2 */
    for (uint32_t m = 0; m < M; m++) {
        float a = 0.0f, al2 = 0.0f;
        for (int s = 0; s < n_slots; s++) {
            a += acc_pool[(size_t)s * M + m];
            if (acc_l2_pool) al2 += acc_l2_pool[(size_t)s * M + m];
        }
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = a * rs + al2;
    }
    if (acc_pool_owned) free(acc_pool);
    if (acc_l2_owned) free(acc_l2_pool);
}

static void tensor_matmul(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf
) {
    /* Goal H4 — hot-cache instrumentation. Single-branch fast path:
     * ib_hotset_enabled() is a cached int. When disabled this is one
     * predictable branch with no memory write, so the cost when off is
     * effectively zero. The cast strips const because access_count is
     * book-keeping, not part of the on-disk tensor identity. */
    if (ib_hotset_enabled()) {
        ((ib_tensor_meta *)t)->access_count++;
    }

    /* PQv2 dispatch — takes precedence when present. Per-chunk threading
     * for K=256; falls back to single-thread for other K or no pool. */
    if (t->pq) {
        const pqv2_t* pq = t->pq;
        /* Path D drive mode (Solution 5): pread the indices from disk
         * into the model's scratch buffer (which pq->indices was
         * redirected to at load). Kernel then reads from scratch.
         * Peak-RAM cap: tensors larger than the scratch slot are streamed
         * in lane-groups by drive_paged_matvec (which writes `out` and
         * returns 1). Tensors that fit a slot take the unchanged
         * load+dispatch path below. */
        if (m->residency_mode == 1) {
            if (drive_paged_matvec(m, t, input, 1, out)) return;
            (void)drive_load_indices(m, t);
        }
        if (pq->K == 256 && m->thread_pool && m->num_threads > 1) {
            pqv2_threaded_matvec_k256(m, m->thread_pool, m->num_threads,
                                        pq, input, out);
        } else {
            pqv2_matvec_dispatch(pq, input, out);
        }
        return;
    }

    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Detect per-block-32 INT4 scaling: scale_size > rows*2 ⇒ N/32 fp16
     * scales per row instead of one. Triggered by IB_INT4_BLK32 at convert
     * time. The new kernel handles a flat fp32 buffer of M*(N/32) scales.
     *
     * Perf (doc 36): the fp16 scale buffer is a STATIC property of the
     * weight tensor — pre-decoded into t->scales_fp32 / t->blk32_scales_fp32
     * by ib_cache_model_static_fp32() at load time. We pick those up here
     * and skip the per-call fp16→fp32 conversion. The local fallback path
     * stays in place for the (rare) case where the cache wasn't built. */
    int is_blk32_int4 = (t->bits == 4 && t->scale_size > (size_t)M * 2);
    const float* scales_eff = NULL;       /* per-row scales (M) */
    const float* blk32_scales = NULL;     /* M * (N/32) scales */
    float* blk32_owned = NULL;            /* malloc'd fallback only */
    if (is_blk32_int4) {
        if (t->blk32_scales_fp32) {
            blk32_scales = t->blk32_scales_fp32;
        } else {
            size_t total = (size_t)M * (size_t)(N / 32);
            blk32_owned = (float*)malloc(total * sizeof(float));
            if (blk32_owned) {
                scales_to_fp32(blk32_owned, scales_raw, (int)total);
                blk32_scales = blk32_owned;
            } else {
                is_blk32_int4 = 0;   /* fall back if alloc failed */
            }
        }
    }
    /* Per-row scales used by all non-blk32 paths AND as a safety fallback
     * if is_blk32_int4 is set but matmul_w4a8_blk32 is unavailable
     * (matches the original code's fall-through). */
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    if (is_blk32_int4 && ib_kern.matmul_w4a8_blk32) {
        /* Per-block-32 INT4 path: quantize input as usual, dispatch to the
         * blk32-aware kernel. No batched/parallel wrapper for now — the
         * scalar kernel is single-threaded.
         *
         * Hot-path scratch: prefer model-lifetime bb_qscratch/bb_sa (sized
         * for IB_BATCH_MAX*n_max, which always covers a single-position
         * matmul) over per-call malloc. Stack fallback retained for the
         * case where the model wasn't built with batch scratch. */
        int8_t stack_q[4096];
        float  stack_s[4096 / IB_W4A8_GROUP + 1];
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        int8_t* q_buf;
        float*  s_buf;
        int q_buf_owned = 0, s_buf_owned = 0;
        if (m->bb_qscratch && m->bb_sa) {
            q_buf = m->bb_qscratch;
            s_buf = m->bb_sa;
        } else {
            q_buf = (N <= 4096) ? stack_q : (int8_t*)malloc((size_t)N);
            s_buf = (n_groups <= (int)(sizeof stack_s / sizeof *stack_s))
                        ? stack_s
                        : (float*)malloc((size_t)n_groups * sizeof(float));
            q_buf_owned = (q_buf != stack_q);
            s_buf_owned = (s_buf != stack_s);
        }
        ib_quantize_input_int8_g128(input, q_buf, s_buf, N);
        ib_kern.matmul_w4a8_blk32(out, weights, blk32_scales,
                                  q_buf, s_buf, M, N);
        if (q_buf_owned) free(q_buf);
        if (s_buf_owned) free(s_buf);
        if (blk32_owned) free(blk32_owned);
        return;
    }
    if (blk32_owned) free(blk32_owned);

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8) {
        /* Quantize input to INT8 per-group (IB_W4A8_GROUP elements per
         * scale). Prefer model-lifetime scratch (bb_qscratch/bb_sa) so the
         * hot decode loop does no malloc for N > 4096 (e.g. MLP up/gate
         * with N=intermediate). Stack fallback retained for legacy paths
         * where the model isn't initialised with batch scratch. */
        int8_t stack_q[4096];
        float  stack_s[4096 / IB_W4A8_GROUP + 1];
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        int8_t* q_buf;
        float*  s_buf;
        int q_buf_owned = 0, s_buf_owned = 0;
        if (m->bb_qscratch && m->bb_sa) {
            q_buf = m->bb_qscratch;
            s_buf = m->bb_sa;
        } else {
            q_buf = (N <= 4096) ? stack_q : (int8_t*)malloc((size_t)N);
            s_buf = (n_groups <= (int)(sizeof stack_s / sizeof *stack_s))
                        ? stack_s
                        : (float*)malloc((size_t)n_groups * sizeof(float));
            q_buf_owned = (q_buf != stack_q);
            s_buf_owned = (s_buf != stack_s);
        }
        ib_quantize_input_int8_g128(input, q_buf, s_buf, N);
        ib_parallel_matmul_w4a8(m->thread_pool, out, weights, scales_eff,
                                q_buf, s_buf, M, N);
        if (q_buf_owned) free(q_buf);
        if (s_buf_owned) free(s_buf);
    } else if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
        ib_parallel_matmul(m->thread_pool, out, weights, scales_eff, input, M, N, t->bits);
    } else if (t->bits == 16) {
        const uint16_t* w = (const uint16_t*)weights;
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += fp16_to_fp32(w[i * N + j]) * input[j];
            }
            out[i] = sum;
        }
    }
}

/* Non-static thin wrapper exposing tensor_matmul to other TUs.
 * Declared in inferbit_internal.h. Used by src/mome.c so the MoME
 * dispatcher can run a per-expert matmul without forward.c growing
 * a public PQv2/W4A8/etc. dispatch surface. */
void ib_tensor_matmul_cpu(const inferbit_model *m, const ib_tensor_meta *t,
                          float *out, const float *input, int M, int N,
                          float *scale_buf) {
    tensor_matmul(m, t, out, input, M, N, scale_buf);
}

/* MoME expert-parallel matmul (Goal B2).
 *
 * Same dispatch as tensor_matmul but with explicit override of the
 * thread pool and per-call private chunk-accumulator scratch. Used by
 * src/mome.c when K expert pthreads each need their OWN thread pool +
 * private acc scratch so the inner PQv2 threaded path is race-free
 * across the concurrent expert dispatches.
 *
 *   pool         : private thread pool (or NULL for single-threaded).
 *   n_threads    : number of workers in `pool`.
 *   acc_pool     : caller-provided PQv2 chunk-acc scratch
 *                  (n_slots × M floats; NULL → falls back to fresh alloc).
 *   acc_pool_floats : capacity of acc_pool in floats.
 *   acc_l2_pool  : companion L2 scratch (same size; NULL on flat models).
 *   acc_l2_pool_floats : capacity of acc_l2_pool in floats.
 *
 * For non-PQv2 tensors falls through to the single-threaded matmul
 * (MoME experts ship as PQv2 in the v6 encoder, so this path is
 * the common case; non-PQv2 fallback exists only for safety). */
void ib_tensor_matmul_cpu_isolated(const inferbit_model *m,
                                   const ib_tensor_meta *t,
                                   float *out, const float *input,
                                   int M, int N, float *scale_buf,
                                   struct ib_thread_pool *pool,
                                   int n_threads,
                                   float *acc_pool, size_t acc_pool_floats,
                                   float *acc_l2_pool, size_t acc_l2_pool_floats)
{
    if (t->pq) {
        const pqv2_t *pq = t->pq;
        if (m->residency_mode == 1) {
            (void)drive_load_indices(m, t);
        }
        if (pq->K == 256 && pool && n_threads > 1) {
            /* Inline a private-scratch variant of pqv2_threaded_matvec_k256.
             * Mirrors that function's body but uses caller-supplied
             * acc_pool / acc_l2_pool instead of m->pqv2_thread_acc_pool*. */
            uint32_t Mt = pq->M;
            uint32_t n_chunks = pq->N / pq->G;
            if (n_chunks < (uint32_t)n_threads || !pq->cb_fp32) {
                pqv2_matvec_dispatch(pq, input, out);
                return;
            }
            /* Burst L1-only gate (read once). Same semantics as
             * pqv2_threaded_matvec_k256: use_l2==0 drops the L2 stage by
             * leaving al2 NULL, so the chunk kernel produces the L1-only
             * result. Default (burst off) → 1 → byte-identical to today. */
            int use_l2 = ib_active_use_l2((inferbit_model *)m);
            int has_l2 = (pq->l2_kind == 2 && pq->l2_cb_fp32 && pq->l2_K <= 64)
                         && use_l2;
            int chunks_per_task = ((int)n_chunks + n_threads - 1) / n_threads;
            int n_slots = ((int)n_chunks + chunks_per_task - 1) / chunks_per_task;
            size_t need = (size_t)n_slots * Mt;
            float *ap = NULL;
            int ap_owned = 0;
            if (acc_pool && need <= acc_pool_floats) {
                ap = acc_pool;
            } else {
                ap = aligned_alloc(64,
                    (need * sizeof(float) + 63) & ~(size_t)63);
                if (!ap) { pqv2_matvec_dispatch(pq, input, out); return; }
                ap_owned = 1;
            }
            memset(ap, 0, need * sizeof(float));
            float *al2 = NULL;
            int al2_owned = 0;
            if (has_l2) {
                if (acc_l2_pool && need <= acc_l2_pool_floats) {
                    al2 = acc_l2_pool;
                } else {
                    al2 = aligned_alloc(64,
                        (need * sizeof(float) + 63) & ~(size_t)63);
                    if (!al2) {
                        if (ap_owned) free(ap);
                        pqv2_matvec_dispatch(pq, input, out);
                        return;
                    }
                    al2_owned = 1;
                }
                memset(al2, 0, need * sizeof(float));
            }
            ib_pqv2_chunks_arg arg = {
                .t = pq, .x = input,
                .acc_pool = ap, .acc_l2_pool = al2,
                .M = Mt,
                .chunk_size = chunks_per_task,
                .n_slots = n_slots,
                .skip_thresh = 0.0f,
            };
            ib_pool_run(pool, ib_pqv2_chunks_task, &arg,
                         (int)n_chunks, chunks_per_task);
            for (uint32_t mm = 0; mm < Mt; mm++) {
                float a = 0.0f, l2 = 0.0f;
                for (int s = 0; s < n_slots; s++) {
                    a += ap[(size_t)s * Mt + mm];
                    if (al2) l2 += al2[(size_t)s * Mt + mm];
                }
                float rs = pqv2_h2f(pq->row_scale[mm]);
                out[mm] = a * rs + l2;
            }
            if (ap_owned) free(ap);
            if (al2_owned) free(al2);
        } else {
            pqv2_matvec_dispatch(pq, input, out);
        }
        return;
    }
    /* Non-PQv2 fallback: route through the regular tensor_matmul. NB:
     * this uses m->thread_pool, which is shared across the K expert
     * pthreads — non-PQv2 MoME tensors are not the design target. */
    tensor_matmul(m, t, out, input, M, N, scale_buf);
}

/* Batched variant of tensor_matmul.
 *
 *   out    [B * M]  row-major, out[b*M + i]
 *   input  [B * N]  row-major, input[b*N + j]
 *   scale_buf: caller-provided, at least M floats (weight scales).
 *   q_scratch: only used for INT4+W4A8 path — B * N int8 bytes for quantized
 *              activations, plus B * ceil(N/IB_W4A8_GROUP) floats for scales.
 *              Caller supplies both to avoid malloc in the hot loop. Pass
 *              NULL for paths that don't need them (INT8, FP16).
 *
 * Same dispatch policy as tensor_matmul: INT4 routes through W4A8 batched
 * kernel when enabled; INT8 uses matmul_int8_batch; FP16 falls back to the
 * sequential FP16 path because we don't have a batched FP16 kernel. */
static void tensor_matmul_batch(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N, int B,
    float* scale_buf, int8_t* q_scratch, float* sa_scratch
) {
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Perf: prefer load-time-cached fp32 scales (see ib_cache_model_static_fp32). */
    const float* scales_eff;
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    /* PQv2 batched path: per-chunk threading shared across B positions.
     * Each chunk slot contributes to ALL B output positions, so weight
     * reads are amortised across B. Same chunk-to-slot partition as
     * the single-position threaded path → identical fp32 sum order. */
    if (t->pq) {
        const pqv2_t* pq = t->pq;
        /* Drive mode: pread indices ONCE for this tensor; the batched
         * kernel below reuses the same scratch for all B positions.
         * Peak-RAM cap: when this tensor is too large for a scratch slot
         * (e.g. lm_head, or an oversized MoME expert), stream it in
         * lane-groups PER POSITION — each pass holds only one lane-group,
         * never the whole tensor (and the per-expert `for e` loop in the
         * batched MoME dispatcher keeps only one expert in focus). The
         * shared model-scope acc pool is reused single-threaded per
         * position, so we loop B explicitly rather than batch. drive_paged_matvec
         * returns 0 when the tensor fits a slot, so the normal batched
         * path below runs unchanged for the common (fits) case. */
        if (m->residency_mode == 1) {
            /* Batched paged matvec: stream each lane-group ONCE and accumulate
             * all B positions (read-once / compute-B amortisation). B<=8 fits
             * the acc pool; for larger B fall back to per-position paging.
             * Returns 1 if it paged (handled all B), 0 if the tensor fits a
             * slot (the batched kernel below handles it). */
            if (B <= 8) {
                if (drive_paged_matvec(m, t, input, B, out)) return;
            } else {
                if (drive_paged_matvec(m, t, input, 1, out)) {
                    for (int b = 1; b < B; b++)
                        (void)drive_paged_matvec(m, t,
                                                 input + (size_t)b * N, 1,
                                                 out + (size_t)b * M);
                    return;
                }
            }
            (void)drive_load_indices(m, t);
        }
        if (pq->K == 256 && B >= 1 && B <= 8 &&
            m->thread_pool && m->num_threads > 1) {
            pqv2_threaded_matvec_k256_batch(m, m->thread_pool, m->num_threads,
                                              pq, input, B, out);
            return;
        }
        if (pq->K == 256 && B > 1 && B <= 8) {
            pqv2_matvec_tbl_int8_k256_batch(pq, input, B, out);
            return;
        }
        for (int b = 0; b < B; b++) {
            /* Recursive call will re-pread; could optimize later by
             * not re-loading scratch within the same tensor. */
            tensor_matmul(m, t, out + (size_t)b * M, input + (size_t)b * N,
                          M, N, scale_buf);
        }
        return;
    }

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8_batch && q_scratch && sa_scratch) {
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        for (int b = 0; b < B; b++) {
            ib_quantize_input_int8_g128(input + (size_t)b * N,
                                        q_scratch + (size_t)b * N,
                                        sa_scratch + (size_t)b * n_groups, N);
        }
        ib_parallel_matmul_w4a8_batch(m->thread_pool, out, weights, scales_eff,
                                      q_scratch, sa_scratch, M, N, B);
    } else if (t->bits == 8 && ib_kern.matmul_int8_batch) {
        ib_parallel_matmul_int8_batch(m->thread_pool, out, weights, scales_eff,
                                      input, M, N, B);
    } else {
        /* Fallback: per-position sequential. */
        for (int b = 0; b < B; b++) {
            float* out_b = out + (size_t)b * M;
            const float* in_b = input + (size_t)b * N;
            if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
                ib_parallel_matmul(m->thread_pool, out_b, weights, scales_eff,
                                   in_b, M, N, t->bits);
            } else if (t->bits == 16) {
                const uint16_t* w = (const uint16_t*)weights;
                for (int i = 0; i < M; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        sum += fp16_to_fp32(w[(size_t)i * N + j]) * in_b[j];
                    }
                    out_b[i] = sum;
                }
            }
        }
    }
}

/*
 * Sparse matmul: same as tensor_matmul but skips rows where mask[row] == 0.
 * Outputs zero for skipped rows. mask is a byte array of length M.
 */
static void tensor_matmul_sparse(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf, const uint8_t* mask
) {
    if (!mask) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        return;
    }

    /* Count active rows and build index */
    int active = 0;
    for (int i = 0; i < M; i++) {
        if (mask[i]) active++;
    }

    /* If most rows are active (>80%), just run dense — overhead of sparse indexing isn't worth it */
    if (active > M * 4 / 5) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        /* Zero out masked rows */
        for (int i = 0; i < M; i++) {
            if (!mask[i]) out[i] = 0.0f;
        }
        return;
    }

    /* Sparse path: only compute active rows */
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Perf: prefer the load-time-cached fp32 scales. */
    const float* scales_eff;
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    /* Zero entire output first */
    memset(out, 0, M * sizeof(float));

    /* Compute only active rows */
    for (int i = 0; i < M; i++) {
        if (!mask[i]) continue;

        float sum = 0.0f;
        if (t->bits == 8) {
            const int8_t* w = (const int8_t*)weights + (size_t)i * N;
            for (int j = 0; j < N; j++) sum += (float)w[j] * input[j];
        } else if (t->bits == 4) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 2);
            for (int j = 0; j < N; j += 2) {
                uint8_t byte = w[j / 2];
                sum += (float)((int8_t)(byte & 0x0F) - 8) * input[j];
                if (j + 1 < N) sum += (float)((int8_t)((byte >> 4) & 0x0F) - 8) * input[j + 1];
            }
        } else if (t->bits == 2) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 4);
            for (int j = 0; j < N; j += 4) {
                uint8_t byte = w[j / 4];
                sum += (float)((byte & 0x03) - 1) * input[j];
                if (j+1 < N) sum += (float)(((byte >> 2) & 0x03) - 1) * input[j+1];
                if (j+2 < N) sum += (float)(((byte >> 4) & 0x03) - 1) * input[j+2];
                if (j+3 < N) sum += (float)(((byte >> 6) & 0x03) - 1) * input[j+3];
            }
        }
        out[i] = sum * scales_eff[i];
    }
}

/* ── Sparse-FFN: row-windowed matvec (BURST draft path) ─────────────────
 *
 * Compute ONLY output rows [m0, m1) of W[M,N] @ input[N] into out[m0:m1).
 * Rows outside [m0, m1) are left UNTOUCHED — the caller pre-zeroes the
 * full output buffer (so inactive intermediate rows stay 0, which the
 * subsequent down_proj treats as multiply-by-zero). This is the DRAFT
 * path: it never runs on EXACT/COOLDOWN, so it does not need to be
 * bit-identical to the full matvec.
 *
 * PQv2 (the v6 FFN format): accumulate the full lane range with the
 * kernel's [m0,m1) output window via pqv2_acc_csrange, then fold
 * row_scale + the L2 (pyramid) residual for the windowed rows. L2 use is
 * gated by the active profile (ib_active_use_l2), matching the dense
 * threaded path. Non-PQ tensors fall back to a per-row scalar window over
 * INT8/INT4/INT2/FP16 weights (same row layout as tensor_matmul_sparse).
 */
static void ffn_matvec_rows(const inferbit_model* m, const ib_tensor_meta* t,
                            float* out, const float* input,
                            int M, int N, int m0, int m1, float* scale_buf) {
    if (m0 < 0)  m0 = 0;
    if (m1 > M)  m1 = M;
    if (m0 >= m1) return;

    if (t->pq) {
        const pqv2_t* pq = t->pq;
        /* Drive mode: ensure indices are resident in scratch (whole-tensor
         * load). The paged streaming path does not support an output-row
         * window, so for the sparse draft we use the resident-load path. */
        if (m->residency_mode == 1) {
            (void)drive_load_indices(m, t);
        }
        uint32_t n_chunks = pq->N / pq->G;
        uint32_t cs_total = n_chunks * pq->n_subchunks;
        int use_l2 = ib_active_use_l2((inferbit_model*)m);
        int has_l2 = (pq->l2_kind == 2 && pq->l2_cb_fp32 && pq->l2_K <= 64) && use_l2;
        uint32_t rows = (uint32_t)(m1 - m0);
        /* pqv2_acc_csrange indexes acc/acc_l2 with ABSOLUTE row m in [m0,m1),
         * so the buffers must be addressable up to index m1-1. Allocate full
         * M floats (intermediate dim is bounded; this is a small per-call
         * malloc, not in the dense hot path). Only the [m0,m1) slice is
         * written + read back. */
        float* acc = (float*)calloc((size_t)M, sizeof(float));
        float* acc_l2 = has_l2 ? (float*)calloc((size_t)M, sizeof(float)) : NULL;
        if (!acc || (has_l2 && !acc_l2)) {
            free(acc); free(acc_l2);
            float* full = (float*)malloc((size_t)M * sizeof(float));
            if (full) {
                tensor_matmul(m, t, full, input, M, N, scale_buf);
                memcpy(out + m0, full + m0, (size_t)rows * sizeof(float));
                free(full);
            }
            return;
        }
        pqv2_acc_csrange(pq, input, pq->cb_fp32,
                         has_l2 ? pq->l2_cb_fp32 : NULL,
                         acc, has_l2 ? acc_l2 : NULL,
                         0, cs_total, (uint32_t)m0, (uint32_t)m1, use_l2);
        for (int r = m0; r < m1; r++) {
            float rs = pqv2_h2f(pq->row_scale[r]);
            out[r] = acc[r] * rs + (acc_l2 ? acc_l2[r] : 0.0f);
        }
        free(acc);
        free(acc_l2);
        return;
    }

    /* Non-PQ fallback: scalar per-row window. Same weight row layouts as
     * tensor_matmul_sparse. */
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);
    const float* scales_eff;
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }
    for (int i = m0; i < m1; i++) {
        float sum = 0.0f;
        if (t->bits == 8) {
            const int8_t* w = (const int8_t*)weights + (size_t)i * N;
            for (int j = 0; j < N; j++) sum += (float)w[j] * input[j];
            out[i] = sum * scales_eff[i];
        } else if (t->bits == 4) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 2);
            for (int j = 0; j < N; j += 2) {
                uint8_t byte = w[j / 2];
                sum += (float)((int8_t)(byte & 0x0F) - 8) * input[j];
                if (j + 1 < N) sum += (float)((int8_t)((byte >> 4) & 0x0F) - 8) * input[j + 1];
            }
            out[i] = sum * scales_eff[i];
        } else if (t->bits == 2) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 4);
            for (int j = 0; j < N; j += 4) {
                uint8_t byte = w[j / 4];
                sum += (float)((byte & 0x03) - 1) * input[j];
                if (j+1 < N) sum += (float)(((byte >> 2) & 0x03) - 1) * input[j+1];
                if (j+2 < N) sum += (float)(((byte >> 4) & 0x03) - 1) * input[j+2];
                if (j+3 < N) sum += (float)(((byte >> 6) & 0x03) - 1) * input[j+3];
            }
            out[i] = sum * scales_eff[i];
        } else if (t->bits == 16) {
            const uint16_t* w = (const uint16_t*)weights + (size_t)i * N;
            for (int j = 0; j < N; j++) sum += fp16_to_fp32(w[j]) * input[j];
            out[i] = sum;
        }
    }
}

/* ── Sparse-FFN dispatch (BURST draft on a clustered layer) ─────────────
 *
 * Pre-conditions (checked by the caller): layer->ffn_n_clusters > 1 AND
 * the active profile is IB_PROFILE_BURST. Computes the FFN result into
 * `out_xb` (the post-norm input `xb` is overwritten, same role as the
 * dense down_proj output). `hb`/`hb2` are the [inter] MLP scratch buffers.
 *
 * Steps:
 *   1. Gate: sparse_gate_select() on the post-norm hidden `xb`.
 *   2. Zero hb/hb2, then compute gate_proj & up_proj ONLY for active
 *      clusters' row ranges [offsets[c], offsets[c+1]).
 *   3. silu_mul on active rows only (inactive stay 0).
 *   4. down_proj over the full (mostly-zero) intermediate: inactive
 *      columns are multiply-by-zero, so this is correct. (We DO NOT skip
 *      inactive down columns — a clean PQv2 lane-range that maps a
 *      contiguous intermediate-COLUMN range is not bit-clean, and the
 *      draft tolerance does not justify the complexity. The gate/up
 *      read+compute saving is the win; down still reads full but on a
 *      mostly-zero input.)
 */
static void ffn_sparse_dispatch(inferbit_model* m, ib_layer_meta* layer,
                                float* out_xb, float* xb, float* hb, float* hb2,
                                int inter, int hidden, float* scale_buf) {
    int n_clusters = (int)layer->ffn_n_clusters;
    if (n_clusters > IB_FFN_MAXK) n_clusters = IB_FFN_MAXK;

    int top_min = ffn_top_min_cfg();
    if (top_min < 0) {
        top_min = n_clusters / 4;
        if (top_min < 1) top_min = 1;
    }
    if (top_min > n_clusters) top_min = n_clusters;

    int active[IB_FFN_MAXK];
    int k = sparse_gate_select(layer->ffn_centroids_fp16, n_clusters,
                               hidden, xb, ffn_gate_thresh(), top_min, active);
    if (k <= 0) {
        /* Defensive: gate returned nothing (shouldn't happen with top_min>=1).
         * Fall back to the dense FFN so the draft is still produced. */
        tensor_matmul_hybrid(m, (int)(layer - m->layers), &layer->gate_proj,
                             hb, xb, inter, hidden, scale_buf);
        tensor_matmul_hybrid(m, (int)(layer - m->layers), &layer->up_proj,
                             hb2, xb, inter, hidden, scale_buf);
        ib_kern.silu_mul(hb, hb, hb2, inter);
        tensor_matmul_hybrid(m, (int)(layer - m->layers), &layer->down_proj,
                             out_xb, hb, hidden, inter, scale_buf);
        return;
    }

    if (ffn_log_enabled()) {
        g_ffn_active_frac_sum += (double)k / (double)n_clusters;
        g_ffn_gate_calls++;
    }

    /* Inactive intermediate rows must be exactly 0 so the full down_proj
     * treats them as multiply-by-zero. */
    memset(hb,  0, (size_t)inter * sizeof(float));
    memset(hb2, 0, (size_t)inter * sizeof(float));

    /* gate_proj + up_proj only for active clusters' contiguous row ranges. */
    for (int ai = 0; ai < k; ai++) {
        int c = active[ai];
        int r0 = (int)layer->ffn_cluster_offsets[c];
        int r1 = (int)layer->ffn_cluster_offsets[c + 1];
        if (r0 < 0) r0 = 0;
        if (r1 > inter) r1 = inter;
        if (r0 >= r1) continue;
        ffn_matvec_rows(m, &layer->gate_proj, hb,  xb, inter, hidden, r0, r1, scale_buf);
        ffn_matvec_rows(m, &layer->up_proj,   hb2, xb, inter, hidden, r0, r1, scale_buf);
        /* silu_mul on this active row range only (inactive rows stay 0). */
        ib_kern.silu_mul(hb + r0, hb + r0, hb2 + r0, r1 - r0);
    }

    /* down_proj over the full (mostly-zero) intermediate. Inactive columns
     * are multiply-by-zero => correct draft. */
    tensor_matmul_hybrid(m, (int)(layer - m->layers), &layer->down_proj,
                         out_xb, hb, hidden, inter, scale_buf);
}

/* ── Stochastic importance-sampled FFN draft probe (IB_FFN_STOCH) ────
 * Tests the "Monte-Carlo matmul as draft" invention: estimate the FFN
 * intermediate h by importance-sampling only a fraction of the input lanes
 * (each lane = one (chunk,subchunk) = `half` input dims), p ∝ ||W_lane||
 * (data-free) · ||x_lane|| (runtime), unbiased reweighting. Reads only
 * s/n_lanes of the index bytes. Compares, against exact h:
 *   - pyramid draft (all lanes, L1)          — reads 100% of L1
 *   - stochastic at {12.5,25,50}% lanes       — reads that fraction
 *   - oracle stochastic (p ∝ ||lane contrib||) — cheap-proxy ceiling
 * Metrics: rel h-error ||ĥ-h||/||h|| + top-5% active-set recall.
 * MEASUREMENT-ONLY (never alters output). Enable IB_FFN_STOCH=1. */
static inline float ffn_st_siluf(float g){ return g/(1.0f+expf(-g)); }
static int ffn_st_cmpdesc(const void*a,const void*b){ float x=*(const float*)a,y=*(const float*)b; return (x<y)-(x>y); }

#define IB_FFN_ST_NBUD 3
static int    g_ffn_st_on = -1;
static const float g_ffn_st_bud[IB_FFN_ST_NBUD] = {0.125f, 0.25f, 0.5f};
static unsigned g_ffn_st_seed = 0x12345678u;
static int    g_ffn_st_M=0, g_ffn_st_K=0, g_ffn_st_nlanes=0, g_ffn_st_L=0;
static float **g_ffn_st_laneWg=NULL, **g_ffn_st_laneWu=NULL;   /* [L][nlanes] data-free */
static float *g_ffn_st_cg=NULL, *g_ffn_st_cu=NULL;            /* [nlanes*M] lane contribs */
static float *g_ffn_st_lut=NULL, *g_ffn_st_p=NULL, *g_ffn_st_cdf=NULL, *g_ffn_st_xln=NULL;
static int   *g_ffn_st_cnt=NULL;
static float *g_ffn_st_g=NULL, *g_ffn_st_u=NULL, *g_ffn_st_hh=NULL, *g_ffn_st_sort=NULL;
static double g_ffn_st_err_pr=0, g_ffn_st_rec_pr=0; static long long g_ffn_st_n=0;
static double g_ffn_st_err[IB_FFN_ST_NBUD]={0}, g_ffn_st_rec[IB_FFN_ST_NBUD]={0}, g_ffn_st_err_or[IB_FFN_ST_NBUD]={0};

static unsigned ffn_st_rng(void){ g_ffn_st_seed=g_ffn_st_seed*1664525u+1013904223u; return g_ffn_st_seed; }
static double ffn_st_recall(const float*sc,const float*h,int M,float frac,float*sb){
    int k=(int)(frac*M+0.5f); if(k<1)k=1; if(k>M)k=M;
    for(int i=0;i<M;i++)sb[i]=fabsf(h[i]);  qsort(sb,M,sizeof(float),ffn_st_cmpdesc); float tt=sb[k-1];
    for(int i=0;i<M;i++)sb[i]=fabsf(sc[i]); qsort(sb,M,sizeof(float),ffn_st_cmpdesc); float ts=sb[k-1];
    long long ov=0,tc=0; for(int i=0;i<M;i++){ if(fabsf(h[i])>=tt){tc++; if(fabsf(sc[i])>=ts)ov++; } }
    return tc?(double)ov/(double)tc:0.0;
}
static double ffn_st_relerr(const float*a,const float*b,int M){
    double dd=0,nn=0; for(int i=0;i<M;i++){ double d=(double)a[i]-b[i]; dd+=d*d; nn+=(double)b[i]*b[i]; }
    return nn>1e-20?sqrt(dd/nn):0.0;
}
static void ffn_st_contribs(const pqv2_t*pq,const float*x,int M,int K,int ns,int G,int half,int n_chunks,float*lut,float*contrib){
    for(int c=0;c<n_chunks;c++)for(int s=0;s<ns;s++){
        const float*xs=x+(size_t)c*G+(size_t)s*half;
        for(int k=0;k<K;k++){ const float*cw=pq->cb_fp32+((size_t)s*K+k)*half; float d=0; for(int hh=0;hh<half;hh++)d+=cw[hh]*xs[hh]; lut[k]=d; }
        const uint8_t*idx=pq->indices+((size_t)c*ns+s)*M; float*dst=contrib+(size_t)(c*ns+s)*M;
        for(int mm=0;mm<M;mm++)dst[mm]=lut[idx[mm]];
    }
}
static void ffn_st_laneW(const pqv2_t*pq,int M,int K,int ns,int G,int half,int n_chunks,float*out){
    (void)G;
    for(int c=0;c<n_chunks;c++)for(int s=0;s<ns;s++){
        double w=0; const uint8_t*idx=pq->indices+((size_t)c*ns+s)*M;
        for(int mm=0;mm<M;mm++){ float rs=pqv2_h2f(pq->row_scale[mm]); const float*cw=pq->cb_fp32+((size_t)s*K+idx[mm])*half;
            for(int d=0;d<half;d++){ float v=cw[d]*rs; w+=(double)v*v; } }
        out[c*ns+s]=(float)sqrt(w);
    }
}
static void ffn_st_estimate(const float*contrib,const float*prob,int nlanes,int M,const uint16_t*row_scale,int s,float*cdf,int*cnt,float*out){
    double tot=0; for(int i=0;i<nlanes;i++)tot+=prob[i];
    if(tot<=0){ for(int mm=0;mm<M;mm++)out[mm]=0; return; }
    double acc=0; for(int i=0;i<nlanes;i++){ acc+=prob[i]; cdf[i]=(float)acc; }
    for(int i=0;i<nlanes;i++)cnt[i]=0;
    for(int t=0;t<s;t++){ double u=((double)ffn_st_rng()/4294967296.0)*tot; int lo=0,hi=nlanes-1;
        while(lo<hi){ int mid=(lo+hi)>>1; if(cdf[mid]<u)lo=mid+1; else hi=mid; } cnt[lo]++; }
    for(int mm=0;mm<M;mm++)out[mm]=0;
    for(int i=0;i<nlanes;i++){ if(!cnt[i])continue; double w=(double)cnt[i]*tot/((double)s*(double)prob[i]);
        const float*src=contrib+(size_t)i*M; for(int mm=0;mm<M;mm++)out[mm]+=(float)(w*src[mm]); }
    for(int mm=0;mm<M;mm++)out[mm]*=pqv2_h2f(row_scale[mm]);
}
static void ffn_stoch_summary(void){
    if(g_ffn_st_n<=0)return; double n=(double)g_ffn_st_n;
    fprintf(stderr,"[ib-ffn-stoch] importance-sampled FFN draft vs pyramid, over %lld layer-tokens (n_lanes=%d):\n",g_ffn_st_n,g_ffn_st_nlanes);
    fprintf(stderr,"  %-26s | h rel-err | recall@5%% | ~reads\n","draft");
    fprintf(stderr,"  %-26s |  %6.3f   |  %5.1f%%   | 100%% of L1\n","pyramid (all-lane L1)",g_ffn_st_err_pr/n,100.0*g_ffn_st_rec_pr/n);
    for(int b=0;b<IB_FFN_ST_NBUD;b++)
        fprintf(stderr,"  stochastic %4.1f%% lanes      |  %6.3f   |  %5.1f%%   | %.1f%% of L1\n",100.0*g_ffn_st_bud[b],g_ffn_st_err[b]/n,100.0*g_ffn_st_rec[b]/n,100.0*g_ffn_st_bud[b]);
    for(int b=0;b<IB_FFN_ST_NBUD;b++)
        fprintf(stderr,"  oracle     %4.1f%% lanes      |  %6.3f   |    --     | (ceiling)\n",100.0*g_ffn_st_bud[b],g_ffn_st_err_or[b]/n);
}
static void ffn_stoch_probe(const inferbit_model*m,int layer_idx,int pos,const float*h,int inter,const float*x,int hidden,const pqv2_t*pg,const pqv2_t*pu){
    (void)pos;(void)hidden;
    if(g_ffn_st_on<0){ const char*e=getenv("IB_FFN_STOCH"); g_ffn_st_on=(e&&e[0]&&e[0]!='0')?1:0; if(g_ffn_st_on)atexit(ffn_stoch_summary); }
    if(!g_ffn_st_on||!m||!h||!x||!pg||!pg->cb_fp32||!pu||!pu->cb_fp32)return;
    int M=(int)pg->M,K=(int)pg->K,ns=(int)pg->n_subchunks,G=(int)pg->G,half=(int)pg->half;
    int n_chunks=(int)(pg->N/pg->G),nlanes=n_chunks*ns,L=m->header.num_layers;
    if(M!=inter||nlanes<=0||(int)pu->M!=M)return;
    if(g_ffn_st_M!=M||g_ffn_st_K!=K||g_ffn_st_nlanes!=nlanes){
        free(g_ffn_st_cg);g_ffn_st_cg=malloc((size_t)nlanes*M*sizeof(float));
        free(g_ffn_st_cu);g_ffn_st_cu=malloc((size_t)nlanes*M*sizeof(float));
        free(g_ffn_st_lut);g_ffn_st_lut=malloc((size_t)K*sizeof(float));
        free(g_ffn_st_p);g_ffn_st_p=malloc((size_t)nlanes*sizeof(float));
        free(g_ffn_st_cdf);g_ffn_st_cdf=malloc((size_t)nlanes*sizeof(float));
        free(g_ffn_st_xln);g_ffn_st_xln=malloc((size_t)nlanes*sizeof(float));
        free(g_ffn_st_cnt);g_ffn_st_cnt=malloc((size_t)nlanes*sizeof(int));
        free(g_ffn_st_g);g_ffn_st_g=malloc((size_t)M*sizeof(float));
        free(g_ffn_st_u);g_ffn_st_u=malloc((size_t)M*sizeof(float));
        free(g_ffn_st_hh);g_ffn_st_hh=malloc((size_t)M*sizeof(float));
        free(g_ffn_st_sort);g_ffn_st_sort=malloc((size_t)M*sizeof(float));
        free(g_ffn_st_laneWg);g_ffn_st_laneWg=(float**)calloc(L,sizeof(float*));
        free(g_ffn_st_laneWu);g_ffn_st_laneWu=(float**)calloc(L,sizeof(float*));
        g_ffn_st_M=M;g_ffn_st_K=K;g_ffn_st_nlanes=nlanes;g_ffn_st_L=L;
        g_ffn_st_err_pr=0;g_ffn_st_rec_pr=0;g_ffn_st_n=0;
        for(int b=0;b<IB_FFN_ST_NBUD;b++){g_ffn_st_err[b]=0;g_ffn_st_rec[b]=0;g_ffn_st_err_or[b]=0;}
    }
    if(!g_ffn_st_cg||!g_ffn_st_cu||!g_ffn_st_laneWg||!g_ffn_st_laneWu||layer_idx<0||layer_idx>=L)return;
    if(!g_ffn_st_laneWg[layer_idx]){ g_ffn_st_laneWg[layer_idx]=(float*)malloc((size_t)nlanes*sizeof(float)); if(g_ffn_st_laneWg[layer_idx])ffn_st_laneW(pg,M,K,ns,G,half,n_chunks,g_ffn_st_laneWg[layer_idx]); }
    if(!g_ffn_st_laneWu[layer_idx]){ g_ffn_st_laneWu[layer_idx]=(float*)malloc((size_t)nlanes*sizeof(float)); if(g_ffn_st_laneWu[layer_idx])ffn_st_laneW(pu,M,K,ns,G,half,n_chunks,g_ffn_st_laneWu[layer_idx]); }
    const float*lWg=g_ffn_st_laneWg[layer_idx],*lWu=g_ffn_st_laneWu[layer_idx]; if(!lWg||!lWu)return;

    ffn_st_contribs(pg,x,M,K,ns,G,half,n_chunks,g_ffn_st_lut,g_ffn_st_cg);
    ffn_st_contribs(pu,x,M,K,ns,G,half,n_chunks,g_ffn_st_lut,g_ffn_st_cu);
    for(int c=0;c<n_chunks;c++)for(int s=0;s<ns;s++){ const float*xs=x+(size_t)c*G+(size_t)s*half; double w=0; for(int d=0;d<half;d++)w+=(double)xs[d]*xs[d]; g_ffn_st_xln[c*ns+s]=(float)sqrt(w); }

    /* pyramid (all-lane L1) */
    for(int mm=0;mm<M;mm++){ double ag=0,au=0; for(int i=0;i<nlanes;i++){ ag+=g_ffn_st_cg[(size_t)i*M+mm]; au+=g_ffn_st_cu[(size_t)i*M+mm]; }
        g_ffn_st_g[mm]=(float)ag*pqv2_h2f(pg->row_scale[mm]); g_ffn_st_u[mm]=(float)au*pqv2_h2f(pu->row_scale[mm]); }
    for(int mm=0;mm<M;mm++)g_ffn_st_hh[mm]=ffn_st_siluf(g_ffn_st_g[mm])*g_ffn_st_u[mm];
    g_ffn_st_err_pr+=ffn_st_relerr(g_ffn_st_hh,h,M);
    g_ffn_st_rec_pr+=ffn_st_recall(g_ffn_st_hh,h,M,0.05f,g_ffn_st_sort);

    for(int b=0;b<IB_FFN_ST_NBUD;b++){
        int s=(int)(g_ffn_st_bud[b]*nlanes+0.5f); if(s<1)s=1; if(s>nlanes)s=nlanes;
        for(int i=0;i<nlanes;i++)g_ffn_st_p[i]=lWg[i]*g_ffn_st_xln[i];
        ffn_st_estimate(g_ffn_st_cg,g_ffn_st_p,nlanes,M,pg->row_scale,s,g_ffn_st_cdf,g_ffn_st_cnt,g_ffn_st_g);
        for(int i=0;i<nlanes;i++)g_ffn_st_p[i]=lWu[i]*g_ffn_st_xln[i];
        ffn_st_estimate(g_ffn_st_cu,g_ffn_st_p,nlanes,M,pu->row_scale,s,g_ffn_st_cdf,g_ffn_st_cnt,g_ffn_st_u);
        for(int mm=0;mm<M;mm++)g_ffn_st_hh[mm]=ffn_st_siluf(g_ffn_st_g[mm])*g_ffn_st_u[mm];
        g_ffn_st_err[b]+=ffn_st_relerr(g_ffn_st_hh,h,M);
        g_ffn_st_rec[b]+=ffn_st_recall(g_ffn_st_hh,h,M,0.05f,g_ffn_st_sort);
        for(int i=0;i<nlanes;i++){ double w=0; const float*sg=g_ffn_st_cg+(size_t)i*M; for(int mm=0;mm<M;mm++)w+=(double)sg[mm]*sg[mm]; g_ffn_st_p[i]=(float)sqrt(w); }
        ffn_st_estimate(g_ffn_st_cg,g_ffn_st_p,nlanes,M,pg->row_scale,s,g_ffn_st_cdf,g_ffn_st_cnt,g_ffn_st_g);
        for(int i=0;i<nlanes;i++){ double w=0; const float*su=g_ffn_st_cu+(size_t)i*M; for(int mm=0;mm<M;mm++)w+=(double)su[mm]*su[mm]; g_ffn_st_p[i]=(float)sqrt(w); }
        ffn_st_estimate(g_ffn_st_cu,g_ffn_st_p,nlanes,M,pu->row_scale,s,g_ffn_st_cdf,g_ffn_st_cnt,g_ffn_st_u);
        for(int mm=0;mm<M;mm++)g_ffn_st_hh[mm]=ffn_st_siluf(g_ffn_st_g[mm])*g_ffn_st_u[mm];
        g_ffn_st_err_or[b]+=ffn_st_relerr(g_ffn_st_hh,h,M);
    }
    g_ffn_st_n++;
}

/* ── RMSNorm with FP16 weights ──────────────────────────────── */

static void rmsnorm_fp16(float* out, const float* input,
                         const void* weight_fp16, float eps, int N,
                         float* weight_buf) {
    fp16_weights_to_fp32(weight_buf, weight_fp16, N);
    ib_kern.rmsnorm(out, input, weight_buf, eps, N);
}

/* Tensor-aware RMSNorm: prefer the load-time-cached fp32 norm weight
 * (t->norm_fp32) and skip the per-call fp16→fp32 conversion. Falls back
 * to the legacy decode path when no cache is present. */
static inline void rmsnorm_fp16_t(float* out, const float* input,
                                  const inferbit_model* m,
                                  const ib_tensor_meta* t,
                                  float eps, int N, float* weight_buf) {
    if (t->norm_fp32) {
        ib_kern.rmsnorm(out, input, t->norm_fp32, eps, N);
    } else {
        rmsnorm_fp16(out, input, tensor_data(m, t), eps, N, weight_buf);
    }
}

/* ── KV cache quantization helpers ──────────────────────────── */

static inline void kv_write_int4_row(uint8_t* dst, const float* src, float scale, int n) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i += 2) {
        int q0 = (int)roundf(src[i] * inv);
        int q1 = (i + 1 < n) ? (int)roundf(src[i + 1] * inv) : 0;
        if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
        if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;
        uint8_t lo = (uint8_t)(q0 + 8) & 0x0F;
        uint8_t hi = (uint8_t)(q1 + 8) & 0x0F;
        dst[i / 2] = lo | (hi << 4);
    }
}

static inline void kv_read_int4_row(float* out, const uint8_t* src, float scale, int n) {
    for (int i = 0; i < n; i += 2) {
        uint8_t b = src[i / 2];
        out[i] = ((float)((int)(b & 0x0F) - 8)) * scale;
        if (i + 1 < n) out[i + 1] = ((float)((int)((b >> 4) & 0x0F) - 8)) * scale;
    }
}

static void kv_cache_write(ib_kv_cache* kv, int pos,
                           const float* key, const float* value,
                           int kv_dim, int n_kv_heads, int head_dim, int kv_bits) {
    /* Rotating KV window (doc 36 phase 2.2): logical position `pos` lands
     * in physical slot pos % capacity. When not windowed, capacity is the
     * full context so pos < capacity and this is the identity map. */
    int phys = (kv->capacity > 0) ? (pos % kv->capacity) : pos;
    if (kv_bits >= 16) {
        float* k_store = (float*)kv->key_data;
        float* v_store = (float*)kv->value_data;
        memcpy(k_store + (size_t)phys * kv_dim, key, kv_dim * sizeof(float));
        memcpy(v_store + (size_t)phys * kv_dim, value, kv_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        int8_t* k_store = (int8_t*)kv->key_data + (size_t)phys * kv_dim;
        int8_t* v_store = (int8_t*)kv->value_data + (size_t)phys * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 127.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 127.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)phys * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)phys * n_kv_heads + h] = v_scale;
            float k_inv = 1.0f / k_scale;
            float v_inv = 1.0f / v_scale;
            for (int d = 0; d < head_dim; d++) {
                int kq = (int)roundf(k_h[d] * k_inv);
                int vq = (int)roundf(v_h[d] * v_inv);
                if (kq < -127) kq = -127; if (kq > 127) kq = 127;
                if (vq < -127) vq = -127; if (vq > 127) vq = 127;
                k_store[h * head_dim + d] = (int8_t)kq;
                v_store[h * head_dim + d] = (int8_t)vq;
            }
        }
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        uint8_t* k_store = (uint8_t*)kv->key_data + (size_t)phys * row_bytes;
        uint8_t* v_store = (uint8_t*)kv->value_data + (size_t)phys * row_bytes;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 7.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 7.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)phys * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)phys * n_kv_heads + h] = v_scale;
            kv_write_int4_row(k_store + (size_t)h * ((head_dim + 1) / 2), k_h, k_scale, head_dim);
            kv_write_int4_row(v_store + (size_t)h * ((head_dim + 1) / 2), v_h, v_scale, head_dim);
        }
        return;
    }
}

static void kv_cache_read_head(const ib_kv_cache* kv, int is_key, int pos, int kv_head,
                               int kv_dim, int n_kv_heads, int head_dim, int kv_bits,
                               float* out_head) {
    /* Rotating KV window: logical position -> physical slot pos % capacity
     * (identity map when not windowed). */
    int phys = (kv->capacity > 0) ? (pos % kv->capacity) : pos;
    if (kv_bits >= 16) {
        const float* src = is_key ? (const float*)kv->key_data : (const float*)kv->value_data;
        const float* row = src + (size_t)phys * kv_dim + kv_head * head_dim;
        memcpy(out_head, row, head_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        const int8_t* src = is_key ? (const int8_t*)kv->key_data : (const int8_t*)kv->value_data;
        const int8_t* row = src + (size_t)phys * kv_dim + kv_head * head_dim;
        float scale = is_key
            ? kv->key_scales[(size_t)phys * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)phys * n_kv_heads + kv_head];
        for (int d = 0; d < head_dim; d++) out_head[d] = (float)row[d] * scale;
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        const uint8_t* src = is_key ? (const uint8_t*)kv->key_data : (const uint8_t*)kv->value_data;
        const uint8_t* row = src + (size_t)phys * row_bytes + (size_t)kv_head * ((head_dim + 1) / 2);
        float scale = is_key
            ? kv->key_scales[(size_t)phys * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)phys * n_kv_heads + kv_head];
        kv_read_int4_row(out_head, row, scale, head_dim);
        return;
    }
}

/* ── Parallel attention task ─────────────────────────────────── */

typedef struct {
    float* q;
    float* att;
    float* xb2;
    ib_kv_cache* kv;
    int head_dim;
    int kv_dim;
    int n_kv_heads;
    int heads_per_kv;
    int pos;
    int kv_bits;
    float scale;
} ib_attn_ctx;

static void ib_attn_head_task(void* arg, int tid, int start, int end) {
    (void)tid;
    ib_attn_ctx* c = (ib_attn_ctx*)arg;
    float k_tmp[256];
    float v_tmp[256];

    /* Rotating KV window (doc 36 phase 2.2): only the most recent
     * `capacity` positions are physically live; older ones were evicted.
     * When not windowed, capacity is the full context so t_lo is 0 and
     * this attends to everything (unchanged behaviour). The att row is
     * compacted into [0, n_valid) so softmax + weighted-V operate on the
     * live window only; logical position t reads physical slot t % cap. */
    int cap = c->kv->capacity;
    int t_lo = (cap > 0 && c->pos + 1 > cap) ? (c->pos + 1 - cap) : 0;
    int n_valid = c->pos + 1 - t_lo;

    for (int h = start; h < end; h++) {
        float* q_h = c->q + h * c->head_dim;
        int kv_h = h / c->heads_per_kv;
        float* att_h = c->att + h * (c->pos + 1);

        for (int t = t_lo; t <= c->pos; t++) {
            int phys = (cap > 0) ? (t % cap) : t;
            float score = 0.0f;
            if (c->kv_bits >= 16) {
                float* k_cache = (float*)c->kv->key_data;
                float* k_t = k_cache + (size_t)phys * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_t[d];
            } else {
                kv_cache_read_head(c->kv, 1, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, k_tmp);
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_tmp[d];
            }
            att_h[t - t_lo] = score * c->scale;
        }

        ib_kern.softmax(att_h, n_valid);

        float* out_h = c->xb2 + h * c->head_dim;
        memset(out_h, 0, c->head_dim * sizeof(float));
        for (int t = t_lo; t <= c->pos; t++) {
            int phys = (cap > 0) ? (t % cap) : t;
            float a = att_h[t - t_lo];
            if (c->kv_bits >= 16) {
                float* v_cache = (float*)c->kv->value_data;
                float* v_t = v_cache + (size_t)phys * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_t[d];
            } else {
                kv_cache_read_head(c->kv, 0, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, v_tmp);
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_tmp[d];
            }
        }
    }
}

/* ── Single-token forward pass ──────────────────────────────── */

/* Extended single-token forward.
 *
 * compute_logits:
 *   0 — skip the final RMSNorm and LM head (prefill path, advances KV only)
 *   1 — compute logits via the LM head (default decode path)
 *
 * hidden_out: if non-NULL, writes the post-final-RMSNorm hidden state into
 *   hidden_out[hidden_size]. Used by the batched verify path to stack B
 *   positions' hidden states before a single batched LM head matmul. When
 *   hidden_out is supplied the final RMSNorm runs regardless of compute_logits. */
/* Forward decl: Goal N37 unified MoME FFN dispatch. Definition lives
 * further down (after mome_dispatch_ffn_batch) but both forward_single_ex
 * and forward_batch route their MoME branch through it. */
static void mome_ffn_dispatch(inferbit_model *m, const ib_layer_meta *layer,
                              float *xb_in_batch, float *hb_batch,
                              float *hb2_batch, float *xb_out_batch,
                              int B,
                              float *scale_buf, int8_t *q_scratch,
                              float *sa_scratch);

static int forward_single_ex_trunc(inferbit_model* m, int token_id, int pos,
                                   float* logits, int compute_logits,
                                   float* hidden_out, int max_layer) {
    int hidden   = m->header.hidden_size;
    int n_layers = m->header.num_layers;
    /* Burst early-exit / self-speculation: run only layers [0, max_layer).
     * max_layer < 0 or >= depth ⇒ full depth (the default, byte-identical to
     * the legacy forward). The finalize (output_norm + LM head) still runs on
     * whatever hidden state the truncated stack produced. NOTE: callers using
     * a partial depth leave KV holes in layers >= max_layer for this position
     * — only safe when a later full/verify pass backfills them (M3). The
     * default decode path passes -1, so no holes are created today. */
    if (max_layer >= 0 && max_layer < n_layers) n_layers = max_layer;
    int n_heads  = m->header.num_heads;
    int n_kv     = m->header.num_kv_heads;
    int head_dim = m->header.head_dim;
    int inter    = m->header.intermediate_size;
    int vocab    = m->header.vocab_size;
    float eps    = m->header.norm_epsilon;
    float theta  = m->header.rope_theta;

    int kv_dim   = n_kv * head_dim;
    int heads_per_kv = n_heads / n_kv;  /* For GQA */

    /* Activation buffers */
    float* x       = m->buf_residual;   /* [hidden] — residual stream */
    float* xb      = m->buf_hidden;     /* [hidden] — after norm */
    float* xb2     = m->buf_attn;       /* [hidden] — scratch */
    float* hb      = m->buf_mlp;        /* [inter]  — MLP scratch */
    float* hb2     = m->buf_mlp2;       /* [inter]  — MLP scratch 2 */
    float* qkv_buf = m->buf_qkv;        /* Scratch for projections + attention scores */

    /* Partition qkv_buf:
     * q:     [hidden]
     * k:     [kv_dim]
     * v:     [kv_dim]
     * att:   [n_heads * (pos+1)] — attention scores
     * scale: [max(hidden, inter, vocab)] — scale factor temp buffer
     */
    float* q     = qkv_buf;
    float* k     = q + hidden;
    float* v     = k + kv_dim;
    float* att   = v + kv_dim;
    int scale_sz = hidden > inter ? hidden : inter;
    if (vocab > scale_sz) scale_sz = vocab;
    float* scale_buf = att + (size_t)n_heads * (pos + 1);

    /* Embedding lookup */
    ib_embedding_lookup(m, token_id, x);

    /* Transformer layers */
    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm before attention */
        rmsnorm_fp16_t(xb, x, m, &layer->input_norm,
                       eps, hidden, scale_buf);

        /* Q/K/V projections */
        tensor_matmul(m, &layer->q_proj, q, xb, hidden, hidden, scale_buf);
        tensor_matmul(m, &layer->k_proj, k, xb, kv_dim, hidden, scale_buf);
        tensor_matmul(m, &layer->v_proj, v, xb, kv_dim, hidden, scale_buf);

        /* RoPE: apply to each Q head paired with its corresponding K head.
         * For GQA, multiple Q heads share one K head. Apply RoPE to each
         * K head only once (on the first Q head that maps to it). */
        /* Precomputed RoPE tables (NULL = kernel falls back to live sinf/cosf). */
        const float* rope_cos_tab = (m->rope_cos && pos < m->rope_table_ctx) ? m->rope_cos : NULL;
        const float* rope_sin_tab = (m->rope_sin && pos < m->rope_table_ctx) ? m->rope_sin : NULL;
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;
            int is_first = (h % heads_per_kv == 0);
            if (is_first) {
                ib_kern.rope(q + h * head_dim, k + kv_h * head_dim,
                             head_dim, pos, theta, rope_cos_tab, rope_sin_tab);
            } else {
                /* Apply RoPE to Q only — use a scratch buffer for K */
                float k_scratch[256];
                memcpy(k_scratch, k + kv_h * head_dim, head_dim * sizeof(float));
                ib_kern.rope(q + h * head_dim, k_scratch, head_dim, pos, theta,
                             rope_cos_tab, rope_sin_tab);
                /* Discard k_scratch — K was already rotated */
            }
        }

        /* Write K, V to cache */
        kv_cache_write(kv, pos, k, v, kv_dim, n_kv, head_dim, m->header.kv_bits);
        kv->length = pos + 1;

        /* Multi-head attention (parallelized across heads) */
        float attn_scale = 1.0f / sqrtf((float)head_dim);

        ib_attn_ctx attn_ctx = {
            .q = q, .att = att, .xb2 = xb2,
            .kv = kv,
            .head_dim = head_dim, .kv_dim = kv_dim,
            .n_kv_heads = n_kv,
            .heads_per_kv = heads_per_kv,
            .pos = pos,
            .kv_bits = m->header.kv_bits,
            .scale = attn_scale,
        };

        if (m->thread_pool && n_heads >= 4) {
            ib_pool_run(m->thread_pool, ib_attn_head_task, &attn_ctx, n_heads, 0);
        } else {
            ib_attn_head_task(&attn_ctx, 0, 0, n_heads);
        }

        /* Output projection: xb = O_proj @ xb2 */
        tensor_matmul(m, &layer->o_proj, xb, xb2, hidden, hidden, scale_buf);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }

        /* RMSNorm before MLP */
        rmsnorm_fp16_t(xb, x, m, &layer->post_attn_norm,
                       eps, hidden, scale_buf);

        /* MLP: gate + up + silu_mul + down
         * With sparsity: skip masked intermediate neurons entirely.
         *
         * Stage 3a — MoME router hook (docs/v2/00_CORRECTION.md). When
         * the layer carries mome_experts > 1 (= the file shipped K
         * expert sub-tensors), branch BEFORE the normal FFN dispatch:
         *
         *   - Router non-zero (calibrated): compute router_logits =
         *     xb @ router_weight, pick top-N indices via
         *     softmax-weighted top-N selection, and run only those
         *     experts through mome_dispatch_ffn. Output goes straight
         *     into xb (overwriting the post-norm input — same role as
         *     the legacy down_proj output).
         *
         *   - Router zero (v1 default, no calibration): run ALL K
         *     experts with weight 1.0. On the trivial row-split that
         *     produces the exact same result as the un-split FFN
         *     matmul — preserving the v1 correctness invariant
         *     ("MoME-enabled file = non-MoME file bit-for-bit, until a
         *     real router lands").
         *
         * Stage 5d hybrid hook: when gate/up/down carry a METAL
         * preferred_backend tag (set via IB_HYBRID_FFN_GPU=1) AND no
         * sparsity mask is active for this layer, route each FFN
         * matmul through tensor_matmul_hybrid which dispatches one
         * GPU matmul and copies the fp32 result back. Falls back
         * transparently to CPU when Metal is unavailable. The sparsity
         * path stays CPU-only — the Metal recorder has no sparse mask
         * variant yet, so sparsity wins when both are configured. */
        {
            const uint8_t* sp_mask = NULL;
            if (layer->sparsity_mask_size > 0) {
                sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            }

            int mome_handled = 0;
            if (!sp_mask && layer->mome_experts > 1 &&
                layer->gate_proj_experts && layer->up_proj_experts &&
                layer->down_proj_experts) {
                /* Goal N37: single entry point shared with forward_batch.
                 * For B==1, q_scratch / sa_scratch are unused — pass NULL. */
                mome_ffn_dispatch(m, layer, xb, hb, hb2, xb, /*B=*/1,
                                  scale_buf, /*q_scratch=*/NULL,
                                  /*sa_scratch=*/NULL);
                mome_handled = 1;
            }

            /* Training-free sparse-FFN cluster dispatch (BURST draft only).
             * Active iff: not a MoME layer, no activation-sparsity mask, the
             * layer is CLUSTERED (ffn_n_clusters > 1), AND the active compute
             * profile is BURST. The BURST draft is approximate — the COOLDOWN
             * verify (which runs EXACT, never reaches this branch) corrects
             * the emitted tokens (the spec-verify firewall). On EXACT /
             * COOLDOWN, or a non-clustered layer, this branch is skipped and
             * the dense FFN below runs every row exactly as before
             * (byte-identical). */
            int sparse = (!mome_handled && !sp_mask &&
                          layer->ffn_n_clusters > 1 &&
                          layer->ffn_cluster_offsets && layer->ffn_centroids_fp16 &&
                          m->burst.cur == IB_PROFILE_BURST);
            if (sparse) {
                ffn_sparse_dispatch(m, layer, xb, xb, hb, hb2, inter, hidden, scale_buf);
                mome_handled = 1;   /* reuse the "FFN already produced" flag */
            }

            if (!mome_handled) {
                if (sp_mask) {
                    tensor_matmul_sparse(m, &layer->gate_proj, hb, xb, inter, hidden, scale_buf, sp_mask);
                    tensor_matmul_sparse(m, &layer->up_proj, hb2, xb, inter, hidden, scale_buf, sp_mask);
                } else {
                    tensor_matmul_hybrid(m, l, &layer->gate_proj, hb, xb, inter, hidden, scale_buf);
                    tensor_matmul_hybrid(m, l, &layer->up_proj,   hb2, xb, inter, hidden, scale_buf);
                }
                /* FFN activation-sparsity QUALITY PROBE (env-gated, off by
                 * default). When IB_FFN_DENSITY<1: snapshot gate (mode 1
                 * needs pre-silu gate, since silu_mul overwrites hb), run
                 * silu_mul, then zero the bottom-(1-density) neurons. */
                if (ffn_density() < 1.0f) {
                    float* gate_snap = NULL;
                    float* dscratch  = (float*)malloc((size_t)inter * sizeof(float));
                    if (ffn_sparsity_mode() == 1) {
                        gate_snap = (float*)malloc((size_t)inter * sizeof(float));
                        if (gate_snap) memcpy(gate_snap, hb, (size_t)inter * sizeof(float));
                    }
                    ib_kern.silu_mul(hb, hb, hb2, inter);
                    if (dscratch) {
                        ffn_apply_density(hb, gate_snap, inter, dscratch);
                    }
                    free(dscratch);
                    free(gate_snap);
                } else {
                    ib_kern.silu_mul(hb, hb, hb2, inter);
                }
                /* Stochastic importance-sampled FFN draft probe (IB_FFN_STOCH):
                 * xb still holds the FFN input, hb holds exact h. */
                ffn_stoch_probe(m, l, pos, hb, inter, xb, hidden,
                                layer->gate_proj.pq, layer->up_proj.pq);
                /* down_proj reads from hb which already has zeros for masked rows —
                 * the multiply by zero propagates naturally, no sparse path needed */
                tensor_matmul_hybrid(m, l, &layer->down_proj, xb, hb, hidden, inter, scale_buf);
            }
        }

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }

        /* DFlash early-exit capture hook (Phase 4 / dflash_orchestrator.c).
         * Normally NULL — see struct inferbit_model. When a DFlash config
         * is attached we snapshot the post-residual hidden state at the
         * configured early-exit layer, so the orchestrator can read its
         * L2-norm as a "confidence" signal and (optionally on later steps)
         * project it through the LM head to skip layers > l. The cost when
         * inactive is one cmp+branch per layer. */
        if (m->dflash_capture_buf && m->dflash_cfg
            && l == m->dflash_cfg->early_exit_layer) {
            memcpy(m->dflash_capture_buf, x, (size_t)hidden * sizeof(float));
        }
    }

    if (compute_logits || hidden_out) {
        if (hidden_out) {
            /* Final RMSNorm into x (in place), then snapshot before LM head. */
            rmsnorm_fp16_t(x, x, m, &m->output_norm,
                           eps, hidden, scale_buf);
            memcpy(hidden_out, x, (size_t)hidden * sizeof(float));
            if (compute_logits) {
                tensor_matmul(m, &m->output_head, logits, x, vocab, hidden, scale_buf);
            }
        } else if (compute_logits) {
            /* Common path: factored helper. Behaviorally equivalent to the
             * previous inline RMSNorm + LM-head matmul. */
            ib_apply_lm_head_finalize(m, x, logits, scale_buf);
        }
    }

    return INFERBIT_OK;
}

/* Full-depth single-token forward (the legacy entry point). Thin wrapper over
 * forward_single_ex_trunc with max_layer=-1 → byte-identical to the previous
 * forward_single_ex. */
static int forward_single_ex(inferbit_model* m, int token_id, int pos,
                             float* logits, int compute_logits,
                             float* hidden_out) {
    return forward_single_ex_trunc(m, token_id, pos, logits, compute_logits,
                                   hidden_out, /*max_layer=*/-1);
}

static int forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    return forward_single_ex(m, token_id, pos, logits, 1, NULL);
}

/* Non-static wrapper. Lets dflash_orchestrator.c dispatch a full-forward
 * decode step without going through ib_forward() (whose routing called the
 * orchestrator in the first place). */
int ib_forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    return forward_single(m, token_id, pos, logits);
}

/* Truncated forward — run only layers [0, max_layer), then finalize logits
 * through the existing output_norm + LM-head path. max_layer < 0 or >= depth
 * runs the full stack (byte-identical to forward_single). Available for future
 * self-speculation; NOT wired into the default burst profile because skipped-
 * layer positions leave KV holes in layers >= max_layer that need a later
 * full/verify pass to backfill (= M3). */
int inferbit_forward_truncated(inferbit_model* model, int32_t token, int pos,
                               int max_layer, float* out_logits) {
    if (!model || !out_logits) {
        ib_set_error("NULL argument to inferbit_forward_truncated");
        return INFERBIT_ERROR_PARAM;
    }
    return forward_single_ex_trunc(model, (int)token, pos, out_logits,
                                   /*compute_logits=*/1, /*hidden_out=*/NULL,
                                   max_layer);
}

/* Factored: final-RMSNorm + LM-head matmul over a single hidden vector.
 * Lifted verbatim from forward_single_ex; behavior-preserving. Used both
 * by the standard decode path and by the DFlash orchestrator's early-exit
 * projection (which feeds an early-layer hidden state through the same
 * final-norm + output-head kernels). */
void ib_apply_lm_head_finalize(const inferbit_model* model,
                               float* hidden_io,
                               float* logits_out,
                               float* scale_buf) {
    int hidden = model->header.hidden_size;
    int vocab  = model->header.vocab_size;
    float eps  = model->header.norm_epsilon;
    /* Cast away const: the helper writes into model-owned scratch via the
     * matmul dispatch path. The model identity itself is unchanged. */
    inferbit_model* m = (inferbit_model*)model;
    rmsnorm_fp16_t(hidden_io, hidden_io, m, &m->output_norm,
                   eps, hidden, scale_buf);
    tensor_matmul(m, &m->output_head, logits_out, hidden_io, vocab, hidden, scale_buf);
}

/* ── Batched MoME FFN dispatch ───────────────────────────────────
 *
 * Per-prefill performance bug: the previous batched path looped over B and
 * called mome_dispatch_ffn per position, paying K * 3 matmul-dispatch setup
 * per token × B tokens × n_layers. For B=64, K=8, 22 layers that is
 * 22 × 64 × 8 × 3 = 33 792 individual matmul calls — TTFT dominated by
 * dispatch overhead, not compute.
 *
 * This helper runs each expert's gate/up/down ONCE for all B positions via
 * tensor_matmul_batch (weight read amortised across B), then does the
 * per-position SiLU * up and the weighted accumulate in plain C.
 *
 * Routing-policy choice (documented per task):
 *   - B == 1   : not used (forward_single_ex still owns the single-token
 *                path, which keeps the calibrated-router top-N behaviour
 *                bit-identical).
 *   - B  > 1   : we run ALL K experts and gate each position with a
 *                weight derived from its own router logits when the
 *                router is calibrated, or the uniform zero-router scale
 *                otherwise. Running the union (= all K) keeps the
 *                weight read fully shared across B; selective per-
 *                position dispatch would defeat the entire optimisation.
 *                On Llama-family MoME files K ∈ {2,4,8}, so "all K
 *                batched once" is ~K× cheaper than "top-N per position"
 *                only when K ≤ top_n * (something) — but in practice the
 *                top-N union over B=64 positions is virtually always the
 *                full set anyway, so this loses nothing measurable.
 *
 * Scratch usage:
 *   x_in_snap : [B * hidden] — written from xb_in BEFORE we zero xb_out.
 *               Reuses bb_xb2 (free at this stage of the layer; see
 *               forward_batch where xb2 last carries pre-O-proj data).
 *   hb_e      : [B * rows_per_expert] slice of bb_hb (rows_per_expert =
 *               inter / K ≤ inter, so it fits B × inter exactly).
 *   hb2_e     : [B * rows_per_expert] slice of bb_hb2 (same shape).
 *   hb_out_e  : [B * hidden] — reuses bb_hb (only after we are done with
 *               hb_e for this expert). We re-slot bb_hb for the down
 *               output because nothing else aliases it once SiLU*up has
 *               been folded in.
 */
static void mome_dispatch_ffn_batch(
    inferbit_model* m, const ib_layer_meta* layer,
    const float* xb_in, float* hb, float* hb2,
    float* xb_out, int B,
    float* x_in_snap,
    float* scale_buf, int8_t* q_scratch, float* sa_scratch)
{
    if (!m || !layer || B <= 0) return;
    if (layer->mome_experts <= 1 ||
        !layer->gate_proj_experts || !layer->up_proj_experts ||
        !layer->down_proj_experts) return;

    const int K       = layer->mome_experts;
    const int hidden  = m->header.hidden_size;
    const int inter   = m->header.intermediate_size;
    const int rows_per_expert = inter / K;
    if (rows_per_expert <= 0) return;
    if (K > IB_MOME_MAX_EXPERTS) return;

    /* Snapshot input — xb_in may alias xb_out (callers in forward_batch
     * pass xb for both). Aliasing-safe snapshot, just like the per-
     * position helper does on the stack. */
    memcpy(x_in_snap, xb_in, (size_t)B * hidden * sizeof(float));

    /* Per-position weights[b][e]. K is bounded, B is bounded by
     * IB_BATCH_MAX (32); stack allocation is fine. */
    float weights[IB_BATCH_MAX][IB_MOME_MAX_EXPERTS];

    /* Weighting policy for trivial row-split MoME (v1):
     *   - Router (if non-zero) is used for SELECTION only — picks the
     *     top_n experts per position.
     *   - WEIGHTS are uniform K/n_active over the selected experts,
     *     regardless of router magnitudes. This is the only weighting
     *     that reconstructs the un-split FFN exactly when n_active=K
     *     (and gives the correct K/n_active scaling for top_n<K).
     *     Softmax weights would sum to 1 and scale FFN by 1/K, which
     *     cascade-damages PPL through the model (R9 I4 regression).
     */
    const int top_n = mome_get_top_n(K);
    int calibrated = mome_router_is_nonzero(m, &layer->router);
    float* router_out = NULL;
    if (calibrated && top_n < K) {
        /* Only run the router matmul if we actually need it for selection
         * (top_n < K). Full-K activation makes the router a no-op. */
        router_out = hb;
        tensor_matmul_batch(m, &layer->router, router_out, x_in_snap,
                            K, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
    }
    const float w_active = (float)K / (float)top_n;
    for (int b = 0; b < B; b++) {
        int active_b[IB_MOME_MAX_TOP_N];
        if (top_n >= K) {
            /* All experts: natural order keeps fp32 accumulation bit-
             * identical to the zero-router path. */
            for (int i = 0; i < K; i++) active_b[i] = i;
        } else if (router_out) {
            float* logits_b = router_out + (size_t)b * K;
            mome_top_n(logits_b, K, top_n, active_b);
        } else {
            for (int i = 0; i < top_n; i++) active_b[i] = i;
        }
        for (int e = 0; e < K; e++) weights[b][e] = 0.0f;
        for (int i = 0; i < top_n; i++) weights[b][active_b[i]] = w_active;
    }

    /* Zero the accumulator. */
    memset(xb_out, 0, (size_t)B * hidden * sizeof(float));

    /* For each expert: 3 batched matmuls + per-position SiLU/accumulate. */
    for (int e = 0; e < K; e++) {
        const ib_tensor_meta* gate_e = &layer->gate_proj_experts[e];
        const ib_tensor_meta* up_e   = &layer->up_proj_experts[e];
        const ib_tensor_meta* down_e = &layer->down_proj_experts[e];

        if (gate_e->shape[0] != rows_per_expert ||
            up_e->shape[0]   != rows_per_expert ||
            down_e->shape[1] != rows_per_expert) {
            continue;
        }

        /* gate_e @ x → hb [B * rows_per_expert]  (hb capacity is B*inter
         * = B*rows_per_expert*K, so the B*rows_per_expert slice fits at
         * the head of hb). */
        tensor_matmul_batch(m, gate_e, hb,  x_in_snap,
                            rows_per_expert, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, up_e,   hb2, x_in_snap,
                            rows_per_expert, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* SiLU(gate) * up — per position, on contiguous B*rows_per_expert. */
        const size_t silu_n = (size_t)B * rows_per_expert;
        for (size_t i = 0; i < silu_n; i++) {
            float g = hb[i];
            hb[i] = (g / (1.0f + expf(-g))) * hb2[i];
        }

        /* down_e @ ffn_e → hb2 [B * hidden]. hb2 is B*inter ≥ B*hidden
         * on any Llama-family config (gated 4/3× MLP), so this fits. */
        tensor_matmul_batch(m, down_e, hb2, hb,
                            hidden, rows_per_expert, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Accumulate with per-position weight. */
        for (int b = 0; b < B; b++) {
            const float w = weights[b][e];
            if (w == 0.0f) continue;
            float* dst = xb_out + (size_t)b * hidden;
            const float* src = hb2 + (size_t)b * hidden;
            for (int h = 0; h < hidden; h++) dst[h] += w * src[h];
        }
    }
}

/* ── Unified MoME FFN dispatch (Goal N37) ────────────────────────
 *
 * Single entry point for the MoME branch shared by forward_single_ex
 * (B==1, single-token decode) and forward_batch (B>1, prefill). Both
 * call sites previously open-coded the same calibrated-vs-zero-router
 * decision and then dispatched to either mome_dispatch_ffn (per-position)
 * or mome_dispatch_ffn_batch (batched). That duplication had already
 * started to drift between the two paths (round-6 wired batched dispatch
 * to a new helper while the single-position branch kept its own copy).
 *
 * Contract:
 *   - Caller has already verified `layer->mome_experts > 1` and that the
 *     expert sub-tensor pointers are non-NULL. The helper re-checks and
 *     returns silently on mismatch (matches the defensive style of the
 *     two donor sites).
 *   - For B == 1, q_scratch / sa_scratch may be NULL — the per-position
 *     mome_dispatch_ffn does not consume them.
 *   - For B  > 1, the caller must guarantee m->bb_xb2 is free at the
 *     FFN stage of the layer (true in forward_batch; the model-lifetime
 *     scratch is sized for IB_BATCH_MAX × hidden floats).
 *   - xb_in_batch and xb_out_batch may alias (both donor call sites pass
 *     `xb` for both). mome_dispatch_ffn snapshots on the stack for B==1,
 *     and mome_dispatch_ffn_batch snapshots into m->bb_xb2 for B>1.
 *
 * Bit-identical to the prior open-coded paths: B==1 reproduces the
 * router-logit matmul + top-N selection + mome_dispatch_ffn call from
 * the old forward_single_ex block; B>1 just forwards into
 * mome_dispatch_ffn_batch, which already owns the union-of-K routing
 * documented in its header. */
/* Gate-energy MoME router probe toggle (IB_MOME_GATE_ENERGY=1). */
static int mome_gate_energy_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_MOME_GATE_ENERGY");
        cached = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return cached;
}
static void mome_ffn_dispatch(inferbit_model *m, const ib_layer_meta *layer,
                              float *xb_in_batch, float *hb_batch,
                              float *hb2_batch, float *xb_out_batch,
                              int B,
                              float *scale_buf, int8_t *q_scratch,
                              float *sa_scratch)
{
    if (!m || !layer || B <= 0) return;
    if (layer->mome_experts <= 1 ||
        !layer->gate_proj_experts || !layer->up_proj_experts ||
        !layer->down_proj_experts) return;

    if (B == 1) {
        const int K_ex   = layer->mome_experts;
        const int hidden = m->header.hidden_size;
        int top_n = mome_get_top_n(K_ex);
        int active[IB_MOME_MAX_TOP_N];

        /* ── BURST gate-energy expert sparsity (M2) ──────────────────
         * When the active compute profile selects a top-n < K (burst), pick
         * the n highest-energy experts by THIS step's gate activation and run
         * only those. ib_active_profile()->mome_top_n is -1 on EXACT/COOLDOWN
         * (and with burst disabled, which is the default), in which case
         * mome_select_experts_burst returns all-K → byte-identical to the
         * legacy path below; we only divert when it actually narrows.
         *
         * We must hand mome_select_experts_burst the FULL gate activation,
         * expert-contiguous, length K*(inter/K). The experts are separate
         * sub-tensors, so we materialise it with one gate matmul per expert
         * into hb_batch ([inter]) — the same K-matmul cost the IB_MOME_GATE_
         * ENERGY probe pays. The dispatch below recomputes gate internally,
         * so reusing hb_batch as the selection scratch is safe (selection
         * finishes before dispatch overwrites it). active_buf is int[K]
         * (K ≤ IB_MOME_MAX_EXPERTS), per the helper's contract. */
        {
            const ib_compute_profile *prof = ib_active_profile(m);
            int prof_top_n = prof ? prof->mome_top_n : -1;
            if (prof_top_n >= 0 && prof_top_n < K_ex && K_ex <= IB_MOME_MAX_EXPERTS &&
                layer->gate_proj_experts) {
                const int inter = m->header.intermediate_size;
                const int rpe   = inter / K_ex;
                if (rpe > 0 && rpe * K_ex <= inter) {
                    int ok = 1;
                    for (int e = 0; e < K_ex; e++) {
                        const ib_tensor_meta *gate_e = &layer->gate_proj_experts[e];
                        if (gate_e->shape[0] != rpe) { ok = 0; break; }
                        ib_tensor_matmul_cpu(m, gate_e, hb_batch + (size_t)e * rpe,
                                             xb_in_batch, rpe, hidden, scale_buf);
                    }
                    int active_buf[IB_MOME_MAX_EXPERTS];
                    int n_active = ok ? mome_select_experts_burst(
                                            m, layer, hb_batch, K_ex, active_buf)
                                      : 0;
                    if (ok && n_active > 0 && n_active < K_ex) {
                        mome_dispatch_ffn(m, layer, xb_in_batch, hb_batch,
                                          hb2_batch, xb_out_batch,
                                          /*router_logits=*/NULL, active_buf,
                                          /*n_active=*/n_active, scale_buf);
                        return;
                    }
                    /* n_active == K_ex (all-K) or selection failed → fall
                     * through to the legacy exact path (bit-identical). */
                }
            }
        }

        /* ── Gate-energy router (QUALITY PROBE, IB_MOME_GATE_ENERGY) ──
         * Training-free runtime router for MoME. Computes gate(x) for ALL
         * experts, then energy_e = Σ_{i∈e} |silu(gate_e[i])| and selects
         * the top_n highest-energy experts. Dispatch then runs only those
         * with uniform K/n_active weighting (selection-only semantics,
         * router_logits=NULL — same as the zero-router path). For the
         * probe we recompute gate inside mome_dispatch_ffn too (measure
         * quality first; skipping the redundant gate matmul for unselected
         * experts is the later speed optimisation). */
        if (top_n < K_ex && mome_gate_energy_enabled() &&
            layer->gate_proj_experts) {
            const int inter = m->header.intermediate_size;
            const int rows_per_expert = inter / K_ex;
            if (rows_per_expert > 0 && rows_per_expert <= inter) {
                float energy[IB_MOME_MAX_EXPERTS];
                int ok = 1;
                for (int e = 0; e < K_ex; e++) {
                    const ib_tensor_meta *gate_e = &layer->gate_proj_experts[e];
                    if (gate_e->shape[0] != rows_per_expert) { ok = 0; break; }
                    /* hb_batch ([inter]) reused as per-expert gate scratch. */
                    ib_tensor_matmul_cpu(m, gate_e, hb_batch, xb_in_batch,
                                         rows_per_expert, hidden, scale_buf);
                    float acc = 0.0f;
                    for (int r = 0; r < rows_per_expert; r++) {
                        float g = hb_batch[r];
                        float s = g / (1.0f + expf(-g));   /* silu(gate) */
                        acc += (s < 0.0f) ? -s : s;
                    }
                    energy[e] = acc;
                }
                if (ok) {
                    mome_top_n(energy, K_ex, top_n, active);
                    mome_dispatch_ffn(m, layer, xb_in_batch, hb_batch,
                                      hb2_batch, xb_out_batch,
                                      /*router_logits=*/NULL, active,
                                      /*n_active=*/top_n, scale_buf);
                    return;
                }
            }
        }

        if (top_n >= K_ex) {
            /* All experts selected: natural ascending order keeps fp32
             * accumulation bit-identical to the zero-router path. The
             * router (if any) has no actual selection work to do. */
            for (int i = 0; i < K_ex; i++) active[i] = i;
        } else if (mome_router_is_nonzero(m, &layer->router)) {
            /* Calibrated/heuristic router: use it for SELECTION ONLY
             * (which top_n experts), not for weighting. The trivial
             * row-split (v1) requires uniform K/n_active weights to
             * reconstruct the un-split FFN — softmax weights that sum
             * to 1 would scale FFN output by 1/K and cascade-damage
             * PPL over the depth of the model. */
            float router_logits[IB_MOME_MAX_EXPERTS];
            ib_tensor_matmul_cpu(m, &layer->router, router_logits,
                                 xb_in_batch, K_ex, hidden, scale_buf);
            mome_top_n(router_logits, K_ex, top_n, active);
        } else {
            /* Zero router: pick the first top_n experts (no preference). */
            for (int i = 0; i < top_n; i++) active[i] = i;
        }
        mome_dispatch_ffn(m, layer, xb_in_batch, hb_batch, hb2_batch,
                          xb_out_batch,
                          /*router_logits=*/NULL, active,
                          /*n_active=*/top_n, scale_buf);
        return;
    }

    /* B > 1: delegate to the batched helper, which owns its own router
     * decision (per-position softmax across all K when calibrated,
     * uniform 1.0 across all K otherwise — see mome_dispatch_ffn_batch
     * header). Snapshot scratch lives in m->bb_xb2, which is free at the
     * FFN stage of forward_batch (last carried pre-O-proj data). */
    mome_dispatch_ffn_batch(m, layer,
                            xb_in_batch, hb_batch, hb2_batch,
                            xb_out_batch, B,
                            m->bb_xb2,
                            scale_buf, q_scratch, sa_scratch);
}

/* ── Batched forward pass ───────────────────────────────────── */

/* Process B tokens (at contiguous positions positions[0..B-1]) through the
 * transformer, using batched matmul for projections, MLP, and LM head. Each
 * position's attention runs sequentially (each attends to its own prefix of
 * the KV cache, so there's no matmul-shape win from batching attention).
 *
 * If out_logits is non-NULL, writes logits there:
 *   last_logits_only == 0 — per-position logits [B * vocab] row-major.
 *   last_logits_only == 1 — only position B-1's logits, written to
 *                           out_logits[0 .. vocab) (a [vocab]-sized buffer).
 * In both cases the per-layer batched matmuls + KV writes for all B
 * positions still run; last_logits_only only skips the output-head matmul
 * for positions 0..B-2.
 *
 * Invariant: positions[b] = inferbit_kv_length(m) + b on entry (each position
 * gets appended to the KV cache as processed). Caller is responsible for
 * ensuring that's true. */
static int forward_batch(inferbit_model* m, const int32_t* tokens,
                         const int* positions, int B, float* out_logits,
                         int last_logits_only) {
    int hidden   = m->header.hidden_size;
    int n_layers = m->header.num_layers;
    int n_heads  = m->header.num_heads;
    int n_kv     = m->header.num_kv_heads;
    int head_dim = m->header.head_dim;
    int inter    = m->header.intermediate_size;
    int vocab    = m->header.vocab_size;
    float eps    = m->header.norm_epsilon;
    float theta  = m->header.rope_theta;
    int kv_dim   = n_kv * head_dim;
    int heads_per_kv = n_heads / n_kv;

    if (B > IB_BATCH_MAX) {
        ib_set_error("forward_batch B=%d exceeds IB_BATCH_MAX=%d", B, IB_BATCH_MAX);
        return INFERBIT_ERROR_PARAM;
    }

    /* Reuse model-lifetime preallocated scratch. Avoids ~1 MB malloc/free
     * per call in the spec-verify hot loop. Buffers are sized for
     * IB_BATCH_MAX positions; we only touch the first B slots. */
    float*  x          = m->bb_x;
    float*  xb         = m->bb_xb;
    float*  xb2        = m->bb_xb2;
    float*  q          = m->bb_q;
    float*  k          = m->bb_k;
    float*  v          = m->bb_v;
    float*  hb         = m->bb_hb;
    float*  hb2        = m->bb_hb2;
    float*  scale_buf  = m->bb_scale;
    float*  att        = m->bb_att;
    int8_t* q_scratch  = m->bb_qscratch;
    float*  sa_scratch = m->bb_sa;

    size_t x_sz = (size_t)B * hidden;

    /* Embed each token (cheap, per-position). */
    for (int b = 0; b < B; b++) {
        ib_embedding_lookup(m, tokens[b], x + (size_t)b * hidden);
    }

    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm per position. Uses rmsnorm_fp16_t to pick up cached fp32 norm weights. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16_t(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                           m, &layer->input_norm,
                           eps, hidden, scale_buf);
        }

        /* Q/K/V projections — batched. */
        tensor_matmul_batch(m, &layer->q_proj, q, xb, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->k_proj, k, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->v_proj, v, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Per-position: RoPE, KV-cache write, attention, O-proj-input
         * (accumulated per-position into xb2[b]). */
        for (int b = 0; b < B; b++) {
            int pos = positions[b];
            float* qb = q + (size_t)b * hidden;
            float* kb = k + (size_t)b * kv_dim;
            float* vb = v + (size_t)b * kv_dim;

            /* RoPE: same pattern as forward_single_ex. */
            const float* rope_cos_tab = (m->rope_cos && pos < m->rope_table_ctx) ? m->rope_cos : NULL;
            const float* rope_sin_tab = (m->rope_sin && pos < m->rope_table_ctx) ? m->rope_sin : NULL;
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / heads_per_kv;
                int is_first = (h % heads_per_kv == 0);
                if (is_first) {
                    ib_kern.rope(qb + h * head_dim, kb + kv_h * head_dim,
                                 head_dim, pos, theta, rope_cos_tab, rope_sin_tab);
                } else {
                    float k_scratch[256];
                    memcpy(k_scratch, kb + kv_h * head_dim, head_dim * sizeof(float));
                    ib_kern.rope(qb + h * head_dim, k_scratch, head_dim, pos, theta,
                                 rope_cos_tab, rope_sin_tab);
                }
            }

            kv_cache_write(kv, pos, kb, vb, kv_dim, n_kv, head_dim, m->header.kv_bits);
            kv->length = pos + 1;

            float attn_scale = 1.0f / sqrtf((float)head_dim);
            ib_attn_ctx ctx = {
                .q = qb, .att = att, .xb2 = xb2 + (size_t)b * hidden,
                .kv = kv,
                .head_dim = head_dim, .kv_dim = kv_dim,
                .n_kv_heads = n_kv,
                .heads_per_kv = heads_per_kv,
                .pos = pos,
                .kv_bits = m->header.kv_bits,
                .scale = attn_scale,
            };
            if (m->thread_pool && n_heads >= 4) {
                ib_pool_run(m->thread_pool, ib_attn_head_task, &ctx, n_heads, 0);
            } else {
                ib_attn_head_task(&ctx, 0, 0, n_heads);
            }
        }

        /* O projection — batched. */
        tensor_matmul_batch(m, &layer->o_proj, xb, xb2, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];

        /* RMSNorm before MLP, per position. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16_t(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                           m, &layer->post_attn_norm,
                           eps, hidden, scale_buf);
        }

        /* MLP — batched gate/up/down. Sparsity is not applied here; if a
         * layer has sparsity we fall back to the single-position path per
         * batch via tensor_matmul_sparse (rare for Llama, so OK).
         *
         * MoME (mome_experts>1) routes through mome_dispatch_ffn_batch,
         * which runs each expert's gate/up/down ONCE across all B
         * positions (weight read amortised across B). The legacy
         * gate_proj/up_proj/down_proj slots are EMPTY on MoME files —
         * the MoME branch MUST come before the legacy fast-path below. */
        if (layer->sparsity_mask_size > 0) {
            const uint8_t* sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            for (int b = 0; b < B; b++) {
                float* xb_b = xb + (size_t)b * hidden;
                float* hb_b = hb + (size_t)b * inter;
                float* hb2_b = hb2 + (size_t)b * inter;
                float* xb_out = xb + (size_t)b * hidden;
                tensor_matmul_sparse(m, &layer->gate_proj, hb_b, xb_b, inter, hidden, scale_buf, sp_mask);
                tensor_matmul_sparse(m, &layer->up_proj, hb2_b, xb_b, inter, hidden, scale_buf, sp_mask);
                ib_kern.silu_mul(hb_b, hb_b, hb2_b, inter);
                tensor_matmul(m, &layer->down_proj, xb_out, hb_b, hidden, inter, scale_buf);
            }
        } else if (layer->mome_experts > 1 &&
                   layer->gate_proj_experts && layer->up_proj_experts &&
                   layer->down_proj_experts) {
            /* Goal N37: single entry point shared with forward_single_ex.
             * For B>1 the helper delegates into mome_dispatch_ffn_batch,
             * which uses m->bb_xb2 as its aliasing-safe input snapshot
             * (free at this stage of the layer — last carried pre-O-proj
             * data, see batched-helper header). The full optimisation
             * notes — ~64× dispatch reduction for B=64/K=8/22 layers —
             * still apply; we just hide the call-site choice. */
            mome_ffn_dispatch(m, layer, xb, hb, hb2, xb, B,
                              scale_buf, q_scratch, sa_scratch);
        } else {
            tensor_matmul_batch(m, &layer->gate_proj, hb, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            tensor_matmul_batch(m, &layer->up_proj, hb2, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            for (int b = 0; b < B; b++) {
                ib_kern.silu_mul(hb + (size_t)b * inter,
                                 hb + (size_t)b * inter,
                                 hb2 + (size_t)b * inter, inter);
            }
            tensor_matmul_batch(m, &layer->down_proj, xb, hb, hidden, inter, B,
                                scale_buf, q_scratch, sa_scratch);
        }

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];
    }

    /* Final RMSNorm + LM head. */
    if (out_logits) {
        if (last_logits_only) {
            /* Only position B-1's logits are needed — RMSNorm + a single
             * output-head matmul for that one position. Skips B-1 vocab-
             * sized matmuls (the most expensive op) vs the full path. */
            float* x_last = x + (size_t)(B - 1) * hidden;
            rmsnorm_fp16_t(x_last, x_last, m, &m->output_norm,
                           eps, hidden, scale_buf);
            tensor_matmul(m, &m->output_head, out_logits, x_last,
                          vocab, hidden, scale_buf);
        } else {
            for (int b = 0; b < B; b++) {
                rmsnorm_fp16_t(x + (size_t)b * hidden, x + (size_t)b * hidden,
                               m, &m->output_norm,
                               eps, hidden, scale_buf);
            }
            tensor_matmul_batch(m, &m->output_head, out_logits, x, vocab, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
        }
    }

    /* Scratch buffers are model-lifetime; no free here. */
    return INFERBIT_OK;
}

/* ── Metal backend routing ──────────────────────────────────── */

#ifdef IB_HAS_METAL
/* Decide once whether this model runs on the Metal backend, lazily
 * creating + caching the GPU context/buffers on first use. Returns 1 if
 * Metal-routed, 0 for the CPU path. IB_BACKEND=cpu forces CPU. Once a
 * model is Metal-routed it stays Metal-routed for its whole life — we
 * must never silently CPU-fall-back mid-stream, because KV state then
 * lives in metal_bufs and the CPU kv_caches arrays are empty. */
static int ib_metal_route(inferbit_model* m) {
    fprintf(stderr, "[N15] ib_metal_route: enter (model=%p, name=%s)\n",
            (void*)m, m ? m->header.name : "(null)");
    static int forced_cpu = -1;
    if (forced_cpu < 0) {
        const char* e = getenv("IB_BACKEND");
        forced_cpu = (e && strcmp(e, "cpu") == 0) ? 1 : 0;
    }
    if (forced_cpu) {
        fprintf(stderr, "[N15] ib_metal_route: bail — IB_BACKEND=cpu forces CPU path\n");
        return 0;
    }
    if (m->metal_route_failed) {
        fprintf(stderr, "[N15] ib_metal_route: bail — metal_route_failed already set (sticky CPU after prior failure)\n");
        return 0;
    }
    if (m->metal_bufs) {
        fprintf(stderr, "[N15] ib_metal_route: already routed, returning metal_bufs=%p\n", m->metal_bufs);
        return 1;
    }
    fprintf(stderr, "[N15] ib_metal_route: calling ib_metal_create()\n");
    ib_metal_ctx* ctx = ib_metal_create();
    if (!ctx) {
        fprintf(stderr, "[N15] ib_metal_route: bail — ib_metal_create() returned NULL (no Metal device / ctx alloc failed)\n");
        m->metal_route_failed = 1;
        return 0;
    }
    fprintf(stderr, "[N15] ib_metal_route: ctx=%p, calling ib_metal_upload_model()\n", (void*)ctx);
    ib_metal_model_buffers* bufs = ib_metal_upload_model(ctx, m);
    if (!bufs) {
        fprintf(stderr, "[N15] ib_metal_route: bail — ib_metal_upload_model() returned NULL (model unsupported / upload OOM / arch mismatch)\n");
        ib_metal_destroy(ctx);
        m->metal_route_failed = 1;
        return 0;
    }
    fprintf(stderr, "[N15] ib_metal_route: success — bufs=%p, model routed to Metal\n", (void*)bufs);
    m->metal_ctx  = ctx;
    m->metal_bufs = bufs;
    return 1;
}

/* Metal-backed ib_forward: prefill (n_tokens>1, last-token logits) or
 * single-token decode. KV is written into metal_bufs at [kv_pos,
 * kv_pos+n_tokens); we then advance the logical kv_caches[].length
 * counter so inferbit_kv_length stays correct. */
static int ib_forward_metal(inferbit_model* m, const int32_t* tokens,
                            int num_tokens, int kv_pos, float* out_logits) {
    int hidden = m->header.hidden_size;
    float* embeds = (float*)malloc((size_t)num_tokens * hidden * sizeof(float));
    if (!embeds) { ib_set_error("oom: metal embed buffer"); return INFERBIT_ERROR_MEMORY; }
    for (int i = 0; i < num_tokens; i++)
        ib_embedding_lookup(m, tokens[i], embeds + (size_t)i * hidden);

    ib_metal_ctx* ctx = (ib_metal_ctx*)m->metal_ctx;
    ib_metal_model_buffers* bufs = (ib_metal_model_buffers*)m->metal_bufs;
    int rc;
    if (num_tokens == 1) {
        rc = ib_metal_forward_token(ctx, bufs, embeds, kv_pos, out_logits);
    } else {
        rc = ib_metal_forward_prefill(ctx, bufs, embeds, num_tokens, kv_pos, out_logits);
        if (rc == -2) {
            /* Batched prefill layout-incompatible — per-token GPU loop.
             * forward_token writes out_logits each call, so after the
             * loop out_logits holds the LAST token's logits (what prefill
             * callers consume). */
            rc = 0;
            for (int i = 0; i < num_tokens && rc == 0; i++)
                rc = ib_metal_forward_token(ctx, bufs, embeds + (size_t)i * hidden,
                                            kv_pos + i, out_logits);
        }
    }
    free(embeds);
    if (rc != 0) { ib_set_error("metal forward failed (rc=%d)", rc); return INFERBIT_ERROR_INTERNAL; }
    for (int L = 0; L < m->header.num_layers; L++)
        m->kv_caches[L].length = kv_pos + num_tokens;
    return INFERBIT_OK;
}

/* Metal-backed ib_forward_positions: per-position logits for num_tokens
 * tokens at [kv_pos, kv_pos+num_tokens). out_logits is [num_tokens][vocab]. */
static int ib_forward_positions_metal(inferbit_model* m, const int32_t* tokens,
                                      int num_tokens, int kv_pos, float* out_logits) {
    int hidden = m->header.hidden_size;
    int vocab  = m->header.vocab_size;
    float* embeds = (float*)malloc((size_t)num_tokens * hidden * sizeof(float));
    if (!embeds) { ib_set_error("oom: metal embed buffer"); return INFERBIT_ERROR_MEMORY; }
    for (int i = 0; i < num_tokens; i++)
        ib_embedding_lookup(m, tokens[i], embeds + (size_t)i * hidden);

    ib_metal_ctx* ctx = (ib_metal_ctx*)m->metal_ctx;
    ib_metal_model_buffers* bufs = (ib_metal_model_buffers*)m->metal_bufs;
    int rc = ib_metal_forward_prefill_logits_all(ctx, bufs, embeds, num_tokens,
                                                 kv_pos, out_logits);
    if (rc == -2) {
        /* Layout-incompatible — per-token GPU loop, capturing each
         * position's logits into its own out_logits slab. */
        rc = 0;
        for (int i = 0; i < num_tokens && rc == 0; i++)
            rc = ib_metal_forward_token(ctx, bufs, embeds + (size_t)i * hidden,
                                        kv_pos + i, out_logits + (size_t)i * vocab);
    }
    free(embeds);
    if (rc != 0) { ib_set_error("metal forward_positions failed (rc=%d)", rc); return INFERBIT_ERROR_INTERNAL; }
    for (int L = 0; L < m->header.num_layers; L++)
        m->kv_caches[L].length = kv_pos + num_tokens;
    return INFERBIT_OK;
}
#endif /* IB_HAS_METAL */

/* ── Public: backend warmup + introspection ─────────────────── */

#ifdef IB_HAS_METAL
int inferbit_model_warmup(inferbit_model* model) {
    if (!model) return 0;
    /* Resolve routing now — this triggers the (otherwise lazy) GPU
     * upload, moving the TTFT spike here instead of the first forward. */
    ib_metal_route(model);
    return 0;
}

const char* inferbit_model_backend(inferbit_model* model) {
    if (!model) return "cpu";
    return ib_metal_route(model) ? "metal" : "cpu";
}
#else
int inferbit_model_warmup(inferbit_model* model) {
    (void)model;
    return 0;
}

const char* inferbit_model_backend(inferbit_model* model) {
    (void)model;
    return "cpu";
}
#endif /* IB_HAS_METAL */

/* ── Public: forward pass ───────────────────────────────────── */

int ib_forward(inferbit_model* model, const int32_t* tokens, int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward");
        return INFERBIT_ERROR_PARAM;
    }

    /* Stage 5d: lazy-seed per-tensor preferred_backend from env vars
     * (currently just IB_HYBRID_FFN_GPU). One-shot per model. Done here
     * to avoid touching the loader files (pqv2_model.c / ibf_loader.c
     * are in the "do not modify" list for this stage). */
    hybrid_apply_tags(model);

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d", kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }

    /* Validate all tokens first */
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= model->header.vocab_size) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)", tokens[i], model->header.vocab_size);
            return INFERBIT_ERROR_PARAM;
        }
    }

#ifdef IB_HAS_METAL
    if (ib_metal_route(model))
        return ib_forward_metal(model, tokens, num_tokens, kv_pos, out_logits);
#endif

    /* DFlash hybrid orchestrator (Phase 4). CPU-only in v1, single-token
     * decode only. Placed AFTER the Metal-route check so the orchestrator
     * never sees Metal-routed calls. The orchestrator declines (handled=0)
     * for prefill or when no DFlash config is attached; we then fall
     * through to the existing CPU routing. The orchestrator's own
     * dispatch goes via ib_forward_single (non-static wrapper) so there
     * is no recursion through ib_forward. */
    if (model->dflash_cfg) {
        int handled = 0;
        int rc = ib_dflash_try_route(model, tokens, num_tokens, out_logits, &handled);
        if (handled) return rc;
    }

    if (num_tokens == 1) {
        /* Single token — standard decode path */
        return forward_single(model, tokens[0], kv_pos, out_logits);
    }

    /*
     * Batch prefill (CPU fallback path): process the prompt in chunks of
     * IB_BATCH_MAX tokens through the batched forward engine. forward_batch
     * advances kv_caches[].length itself; ib_forward only needs the LAST
     * position's logits, so we pass last_logits_only=1 — each chunk writes
     * just [vocab] into out_logits[0..vocab), and since chunks overwrite,
     * the final chunk's last-position logits are what remain (correct).
     */
    int offset = 0;
    while (offset < num_tokens) {
        int remaining = num_tokens - offset;
        int B = remaining < IB_BATCH_MAX ? remaining : IB_BATCH_MAX;
        /* forward_batch advanced kv_caches[].length on the prior chunk;
         * recompute the base position fresh each iteration. */
        int base = inferbit_kv_length(model);
        int positions[IB_BATCH_MAX];
        for (int j = 0; j < B; j++) positions[j] = base + j;
        int rc = forward_batch(model, tokens + offset, positions, B,
                               out_logits, 1);
        if (rc != INFERBIT_OK) {
            return rc;
        }
        offset += B;
    }
    return INFERBIT_OK;
}

int ib_forward_positions(inferbit_model* model, const int32_t* tokens,
                         int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward_positions");
        return INFERBIT_ERROR_PARAM;
    }

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;
    int vocab   = model->header.vocab_size;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d",
                     kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= vocab) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)",
                         tokens[i], vocab);
            return INFERBIT_ERROR_PARAM;
        }
    }

    if (num_tokens > IB_BATCH_MAX) {
        ib_set_error("forward_positions num_tokens=%d exceeds IB_BATCH_MAX=%d",
                     num_tokens, IB_BATCH_MAX);
        return INFERBIT_ERROR_PARAM;
    }

#ifdef IB_HAS_METAL
    if (ib_metal_route(model))
        return ib_forward_positions_metal(model, tokens, num_tokens, kv_pos, out_logits);
#endif

    /* Fill absolute positions in the preallocated scratch buffer. */
    int* positions = model->bb_positions;
    for (int i = 0; i < num_tokens; i++) positions[i] = kv_pos + i;

    return forward_batch(model, tokens, positions, num_tokens, out_logits, 0);
}
