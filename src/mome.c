/*
 * mome.c — Mixture-of-Mini-Experts runtime helpers (Stage 3a v1).
 *
 * See mome.h for the design rationale + correctness invariants. This
 * file holds:
 *   - mome_get_top_n         : env-driven top-N picker
 *   - mome_router_is_nonzero : zero-init router detector (correctness
 *                              fallback to "run all experts")
 *   - mome_top_n             : partial selection over router logits
 *   - mome_dispatch_ffn      : per-expert FFN matmul chain
 *
 * v1 trivial row-split semantics:
 *   - gate_proj_experts[e].pq->M = M / K, N = hidden.
 *   - up_proj_experts[e]   same shape.
 *   - down_proj_experts[e].pq->M = hidden, N = inter / K.
 *
 * The matmul itself is the existing PQv2 path — mome.c never touches
 * codebooks or indices, only routing and partial-sum accumulation.
 *
 * Perf (Goal B2): the K active experts are dispatched across K pthreads
 * so each gets a slice of the available decode threads for its own
 * chunk-parallel PQv2 matmul. The per-expert thread pools + private
 * acc/L2 scratch are cached at process scope (lazy init on first
 * dispatch, never destroyed) so the per-call overhead is just K-1
 * pthread_create + pthread_join cycles. To stay race-free across
 * concurrent experts, each pthread calls
 * `ib_tensor_matmul_cpu_isolated` (forward.c) with its OWN pool +
 * private acc/L2 buffer — the model-scope shared scratch is never
 * touched by the worker pthreads.
 */

#include "mome.h"
#include "inferbit_internal.h"
#include "pqv2_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Goal B2 expert-parallel dispatch uses pthread. On Windows the
 * existing pthread shim lives only in forward.c / threading.c — to
 * avoid coupling those translation units, the parallel path here is
 * POSIX-only. On Windows the file falls through to the sequential v1
 * path (same behaviour as before this patch). */
#ifndef _WIN32
#  include <pthread.h>
#  define MOME_HAS_PTHREAD 1
#else
#  define MOME_HAS_PTHREAD 0
#endif

/* Forward declarations from forward.c — kept here because the public
 * inferbit_internal.h surface is intentionally untouched by this
 * patch. The function signatures below MUST match forward.c. */
extern void ib_tensor_matmul_cpu(const inferbit_model *m, const ib_tensor_meta *t,
                                 float *out, const float *input, int M, int N,
                                 float *scale_buf);
extern void ib_tensor_matmul_cpu_isolated(const inferbit_model *m,
                                          const ib_tensor_meta *t,
                                          float *out, const float *input,
                                          int M, int N, float *scale_buf,
                                          struct ib_thread_pool *pool,
                                          int n_threads,
                                          float *acc_pool, size_t acc_pool_floats,
                                          float *acc_l2_pool, size_t acc_l2_pool_floats);

/* ── public helpers ──────────────────────────────────────────────── */

/* Forward decl — the definition lives further down (activation helpers).
 * mome_select_experts_burst (below) needs it before that point. */
static inline float mome_silu(float x);

/* fp16 → fp32 conversion used by the router matmul follow-up path
 * (calibrated router weights, future stage). v1's zero-router path
 * never touches this; the router matmul itself is a future patch.
 *
 * Kept out of this file for now — the router-matmul wrapper that will
 * live here (see TODO at the bottom) will pull the existing
 * fp16_to_fp32 helper from forward.c via a small internal API. The
 * v1 routine `mome_router_is_nonzero` works directly on the fp16
 * bit pattern so no conversion is needed. */

int mome_get_top_n(int K) {
    if (K <= 0) return 0;
    /* Default ALL experts (top_n = K) = exact un-split FFN reconstruction.
     * MoME is a quality/size/RAM-pagination tool, NOT a routing/compute-skip
     * tool: training-free top_n<K routing of a post-hoc row-split is proven
     * dead (see docs/v2/01_MOME_FINDINGS.md) and catastrophically degrades
     * PPL (e.g. K=4 top_n=2 prefill -> PPL 95+ vs 6.9 all-K on TinyLlama).
     * The previous default of min(2,K) silently routed and corrupted output.
     * Opt into routing experiments explicitly with IB_MOME_TOP_N. */
    int n = K;
    const char *env = getenv("IB_MOME_TOP_N");
    if (env && *env) {
        int v = atoi(env);
        if (v > 0) n = v;
    }
    if (n < 1) n = 1;
    if (n > K) n = K;
    if (n > IB_MOME_MAX_TOP_N) n = IB_MOME_MAX_TOP_N;
    return n;
}

/* BURST gate-energy expert selection (M2 — data-free).
 *
 * BURST runs the top-n highest-energy experts; COOL-DOWN / EXACT runs all
 * K. The selection is purely data-free: it ranks experts by the magnitude
 * of THIS step's gate activations — no router weights, no calibration, no
 * training. An expert whose gated rows are all ~0 contributes ~nothing to
 * the FFN output (down_proj @ (silu(gate)*up) ≈ 0 for that block), so
 * dropping the lowest-energy experts is the data-free sparsity dial.
 *
 * The active profile's mome_top_n (-1 = all-K) drives the regime:
 *   - n < 0 or n >= K  → COOL-DOWN / EXACT: fill 0..K-1, return K. This
 *     is byte-identical to the old M1 stub (and to the non-burst all-K
 *     dispatch the FFN runs by default).
 *   - 0 < n < K        → BURST: score each expert and return its top-n.
 *
 * Scoring: score[e] = sum over the expert's rows of |silu(gate[i])|, where
 * silu(x) = x / (1+exp(-x)). silu is the actual gating nonlinearity the
 * FFN applies (ffn = silu(gate) * up), so |silu(gate)| summed over a
 * block is the genuine firing energy of that expert — strictly truer to
 * the contribution than raw |gate| or gate^2 (silu saturates large
 * negatives toward 0, exactly the rows that should NOT count). It is one
 * pass over `gate` (length M = K * rows_per_expert), no allocation.
 *
 * `gate` is the full FFN-intermediate gate activation, length
 * M = K * rows_per_expert, laid out expert-contiguous: expert e owns
 * gate[e*rpe .. (e+1)*rpe). `layer` is unused — K and `gate` suffice.
 *
 * active_out must be a caller-allocated int[K]; on return its first
 * (return value) entries hold the selected expert indices (descending
 * energy, deterministic; ties resolve to the lower index via mome_top_n). */
int mome_select_experts_burst(inferbit_model *m, const void *layer,
                              const float *gate, int K, int *active_out) {
    (void)layer;   /* K + gate suffice; the layer meta is not needed here. */
    if (!active_out || K <= 0) return 0;

    /* Resolve the requested burst width. n < 0 (the -1 sentinel) or
     * n >= K means "run them all" — the COOL-DOWN / EXACT path. Fill the
     * identity order and return K, byte-identical to the M1 stub. */
    int n = -1;
    const ib_compute_profile *prof = ib_active_profile(m);
    if (prof) n = prof->mome_top_n;
    if (n < 0 || n >= K) {
        for (int e = 0; e < K; e++) active_out[e] = e;
        return K;
    }
    if (n == 0) return 0;

    /* Without gate energy we cannot rank — degrade to all-K (exact) rather
     * than silently corrupt the output by picking arbitrary experts. */
    if (!gate) {
        for (int e = 0; e < K; e++) active_out[e] = e;
        return K;
    }

    /* Guard the scratch array; K is capped by the loader but be defensive. */
    if (K > IB_MOME_MAX_EXPERTS) {
        for (int e = 0; e < K; e++) active_out[e] = e;
        return K;
    }

    /* Per-expert firing energy, one pass over the expert-contiguous gate.
     * `gate` length is M = K * rows_per_expert; M is not passed in, so
     * derive rows_per_expert = intermediate_size / K from the model header
     * — the SAME formula mome_dispatch_ffn uses, so the block boundaries
     * line up exactly with the experts the dispatcher will run. */
    float score[IB_MOME_MAX_EXPERTS];
    for (int e = 0; e < K; e++) score[e] = 0.0f;

    int rows_per_expert = 0;
    if (m) {
        int inter = m->header.intermediate_size;
        if (inter > 0) rows_per_expert = inter / K;
    }
    if (rows_per_expert <= 0) {
        /* Cannot determine block size → cannot rank safely; run all-K. */
        for (int e = 0; e < K; e++) active_out[e] = e;
        return K;
    }

    for (int e = 0; e < K; e++) {
        const float *blk = gate + (size_t)e * (size_t)rows_per_expert;
        float acc = 0.0f;
        for (int r = 0; r < rows_per_expert; r++) {
            acc += fabsf(mome_silu(blk[r]));
        }
        score[e] = acc;
    }

    /* Reuse the existing partial-selection helper: it returns the indices
     * of the top-n highest values from a length-K array — exactly a
     * top-n-by-energy ranking. No signature change, deterministic, O(K*n). */
    mome_top_n(score, K, n, active_out);
    return n;
}

int mome_router_is_nonzero(const inferbit_model *m,
                           const ib_tensor_meta *router) {
    if (!m || !router) return 0;
    /* Router is stored as raw fp16; bits == 16, size > 0. The encoder
     * fills it with zero-init in v1 — every byte is 0, so a memcmp
     * style scan against zero is both cheap and definitive. We scan
     * fp16 words rather than bytes so a future calibration that
     * happens to write fp16 0x8000 (signed zero) is still treated as
     * "zero" — both 0x0000 and 0x8000 decode to ±0.0f and convey no
     * routing information.
     *
     * Guard against uninit router noise — only treat as calibrated
     * when at least one logit weight is materially non-zero (i.e.
     * |v| > 1e-6f after fp16 decode). Denormal/tiny garbage from
     * uninitialised memory must NOT be treated as a calibrated
     * router; softmax on garbage produces NaN/wild weights. */
    if (router->bits != 16 || router->size == 0) return 0;
    const uint8_t *base = (const uint8_t *)m->weight_data;
    if (!base) return 0;
    const uint16_t *w = (const uint16_t *)(base + router->offset);
    size_t count = router->size / 2;
    for (size_t i = 0; i < count; i++) {
        uint16_t h = w[i];
        /* Inline fp16 → fp32 decode (IEEE half-precision). */
        uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        uint32_t exp  = (h >> 10) & 0x1Fu;
        uint32_t mant = h & 0x3FFu;
        uint32_t f;
        if (exp == 0) {
            if (mant == 0) {
                f = sign;            /* ±0 */
            } else {
                /* subnormal — normalise */
                while ((mant & 0x400u) == 0) { mant <<= 1; exp -= 1; }
                exp += 1;
                mant &= 0x3FFu;
                f = sign | ((exp + 112) << 23) | (mant << 13);
            }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000u | (mant << 13); /* inf/nan */
        } else {
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
        float v;
        memcpy(&v, &f, sizeof(v));
        if (fabsf(v) > 1e-6f) return 1;
    }
    return 0;
}

void mome_top_n(const float *logits, int K, int top_n, int *out_indices) {
    if (top_n <= 0 || K <= 0 || !logits || !out_indices) return;
    if (top_n > K) top_n = K;

    /* Tiny partial selection — maintain a length-top_n list sorted by
     * descending logit. Each new entry insertion-sorted into the list.
     * O(K * top_n); fine for K ≤ 32 and top_n ≤ 8. */
    int   idx_buf[IB_MOME_MAX_TOP_N];
    float val_buf[IB_MOME_MAX_TOP_N];
    int filled = 0;

    for (int k = 0; k < K; k++) {
        float v = logits[k];
        if (filled < top_n) {
            /* Insert at the right place. */
            int j = filled;
            while (j > 0 && val_buf[j - 1] < v) {
                val_buf[j] = val_buf[j - 1];
                idx_buf[j] = idx_buf[j - 1];
                j--;
            }
            val_buf[j] = v;
            idx_buf[j] = k;
            filled++;
        } else if (v > val_buf[top_n - 1]) {
            int j = top_n - 1;
            while (j > 0 && val_buf[j - 1] < v) {
                val_buf[j] = val_buf[j - 1];
                idx_buf[j] = idx_buf[j - 1];
                j--;
            }
            val_buf[j] = v;
            idx_buf[j] = k;
        }
    }

    for (int i = 0; i < filled; i++) out_indices[i] = idx_buf[i];
}

/* ── activation helpers ──────────────────────────────────────────── */

static inline float mome_silu(float x) {
    /* x * sigmoid(x). Matches kernels/scalar.c::silu_mul element op. */
    return x / (1.0f + expf(-x));
}

/* Softmax over a small length-N logits slice. In-place. */
static void mome_softmax_inplace(float *v, int n) {
    if (n <= 0) return;
    float mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); s += v[i]; }
    if (s > 0.0f) { float inv = 1.0f / s; for (int i = 0; i < n; i++) v[i] *= inv; }
    else          { for (int i = 0; i < n; i++) v[i] = 1.0f / (float)n; }
}

/* ── expert-parallel infrastructure (Goal B2) ──────────────────────
 *
 * Per-expert "slot": owns one private thread pool of
 * `threads_per_expert` workers and the matching PQv2 acc + L2 scratch
 * buffers. Allocated lazily on first dispatch, reused for the process
 * lifetime — Mac decode runs thousands of layer dispatches per token;
 * the alternative (build pools every call) would burn far more time
 * in pthread_create than it saves.
 *
 * Sizing is determined by the (n_threads_total, K, max_M) tuple
 * captured on the first call. If a later call observes a larger
 * max_M, the slot's scratch buffers are grown in place. K and the
 * thread-fanout are fixed by the first call — re-loading the model
 * with a different K (rare) would leak the previous slots; that's
 * acceptable since the slots are tiny (a few hundred KB total).
 *
 * Disabled when IB_MOME_PARALLEL=0 (env override for A/B). When K==1
 * or n_threads_total<=1 the fast path short-circuits to the existing
 * sequential `ib_tensor_matmul_cpu` calls. */

typedef struct mome_expert_slot {
    struct ib_thread_pool *pool;       /* sized threads_per_expert */
    int    threads_per_expert;
    float *acc_pool;
    size_t acc_pool_floats;
    float *acc_l2_pool;
    size_t acc_l2_pool_floats;
    /* Per-expert private activation scratch (eliminates the per-call
     * aligned_alloc that would otherwise run K × 32 layers × t tok/s
     * times per second of decode). Sizes are upper-bounded by
     * max(rows_per_expert, hidden); grown in mome_slots_prepare. */
    float *hb;            /* size rows_per_expert (≤ scratch_floats) */
    float *hb2;           /* size max(rows_per_expert, hidden) */
    float *xb_partial;    /* size hidden */
    float *scale_buf;     /* size hidden (unused on PQv2) */
    size_t scratch_floats;   /* current capacity of the four scratch buffers */
} mome_expert_slot;

#define MOME_MAX_SLOTS  IB_MOME_MAX_EXPERTS

#if MOME_HAS_PTHREAD
static mome_expert_slot g_slots[MOME_MAX_SLOTS];
static int              g_n_slots = 0;
static pthread_mutex_t  g_slots_mu = PTHREAD_MUTEX_INITIALIZER;

static int mome_parallel_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_MOME_PARALLEL");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* Lazily create/grow K slots so each has a pool of threads_per_expert
 * workers and per-slot acc/L2 + activation scratch. Activation scratch
 * uses one shared `scratch_floats` budget covering hb / hb2 /
 * xb_partial / scale_buf — sized for max(rows_per_expert, hidden) ×
 * the four buffers per expert. Returns 0 on success, non-zero on
 * failure (caller falls back to sequential). */
static int mome_slots_prepare(int n_active, int threads_per_expert,
                              size_t needed_acc_floats,
                              size_t needed_scratch_floats,
                              int need_l2)
{
    if (n_active <= 0 || n_active > MOME_MAX_SLOTS) return -1;
    pthread_mutex_lock(&g_slots_mu);
    /* Grow the static slot count if K rises. */
    if (n_active > g_n_slots) g_n_slots = n_active;
    for (int i = 0; i < n_active; i++) {
        mome_expert_slot *s = &g_slots[i];
        if (!s->pool || s->threads_per_expert != threads_per_expert) {
            if (s->pool) { ib_pool_destroy(s->pool); s->pool = NULL; }
            if (threads_per_expert > 1) {
                s->pool = ib_pool_create(threads_per_expert);
                if (!s->pool) { pthread_mutex_unlock(&g_slots_mu); return -2; }
            }
            s->threads_per_expert = threads_per_expert;
        }
        if (s->acc_pool_floats < needed_acc_floats) {
            free(s->acc_pool);
            s->acc_pool = (float *)aligned_alloc(64,
                (needed_acc_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!s->acc_pool) { s->acc_pool_floats = 0;
                pthread_mutex_unlock(&g_slots_mu); return -3; }
            s->acc_pool_floats = needed_acc_floats;
        }
        if (need_l2) {
            if (s->acc_l2_pool_floats < needed_acc_floats) {
                free(s->acc_l2_pool);
                s->acc_l2_pool = (float *)aligned_alloc(64,
                    (needed_acc_floats * sizeof(float) + 63) & ~(size_t)63);
                if (!s->acc_l2_pool) { s->acc_l2_pool_floats = 0;
                    pthread_mutex_unlock(&g_slots_mu); return -4; }
                s->acc_l2_pool_floats = needed_acc_floats;
            }
        }
        if (s->scratch_floats < needed_scratch_floats) {
            free(s->hb);   s->hb = NULL;
            free(s->hb2);  s->hb2 = NULL;
            free(s->xb_partial); s->xb_partial = NULL;
            free(s->scale_buf);  s->scale_buf = NULL;
            /* One slab carved into four sub-buffers — keeps each slot
             * within a single 64B-aligned allocation. */
            float *slab = (float *)aligned_alloc(64,
                (needed_scratch_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!slab) { s->scratch_floats = 0;
                pthread_mutex_unlock(&g_slots_mu); return -5; }
            /* Layout: each buffer gets the same `max_per` slot to
             * keep the math trivial. needed_scratch_floats == 4 *
             * max_per by construction in the caller, so the four
             * pointers are evenly spaced and each can safely hold up
             * to max_per floats. Wastes ≤ 3 × (max_per - hidden)
             * floats per slot — a few hundred KB total at K=2,
             * negligible. */
            size_t per = needed_scratch_floats / 4;
            s->hb         = slab + 0 * per;
            s->hb2        = slab + 1 * per;
            s->xb_partial = slab + 2 * per;
            s->scale_buf  = slab + 3 * per;
            s->scratch_floats = needed_scratch_floats;
        }
    }
    pthread_mutex_unlock(&g_slots_mu);
    return 0;
}

/* Per-expert pthread arg + worker. */
typedef struct {
    inferbit_model       *m;
    const ib_layer_meta  *layer;
    const float          *x_in;
    int                   expert;          /* 0..K-1 */
    int                   rows_per_expert;
    int                   hidden;
    float                *hb;              /* private, size rows_per_expert */
    float                *hb2;             /* private, size max(rows_per_expert, hidden) */
    float                *xb_partial;      /* private, size hidden */
    float                *scale_buf_priv;  /* private, size hidden (unused for PQv2) */
    mome_expert_slot     *slot;
} mome_expert_arg;

static void mome_expert_run(mome_expert_arg *a) {
    const ib_tensor_meta *gate_e = &a->layer->gate_proj_experts[a->expert];
    const ib_tensor_meta *up_e   = &a->layer->up_proj_experts[a->expert];
    const ib_tensor_meta *down_e = &a->layer->down_proj_experts[a->expert];

    if (gate_e->shape[0] != a->rows_per_expert ||
        up_e->shape[0]   != a->rows_per_expert ||
        down_e->shape[1] != a->rows_per_expert) {
        /* Malformed expert slot — produce zero partial, master will
         * silently drop. */
        memset(a->xb_partial, 0, (size_t)a->hidden * sizeof(float));
        return;
    }

    /* gate_e: [rows_per_expert, hidden] @ x_in[hidden] → hb */
    ib_tensor_matmul_cpu_isolated(
        a->m, gate_e, a->hb, a->x_in, a->rows_per_expert, a->hidden,
        a->scale_buf_priv,
        a->slot->pool, a->slot->threads_per_expert,
        a->slot->acc_pool, a->slot->acc_pool_floats,
        a->slot->acc_l2_pool, a->slot->acc_l2_pool_floats);
    /* up_e: same shape */
    ib_tensor_matmul_cpu_isolated(
        a->m, up_e, a->hb2, a->x_in, a->rows_per_expert, a->hidden,
        a->scale_buf_priv,
        a->slot->pool, a->slot->threads_per_expert,
        a->slot->acc_pool, a->slot->acc_pool_floats,
        a->slot->acc_l2_pool, a->slot->acc_l2_pool_floats);

    /* SiLU(gate) * up — element-wise on the per-expert slice. */
    for (int r = 0; r < a->rows_per_expert; r++) {
        a->hb[r] = mome_silu(a->hb[r]) * a->hb2[r];
    }

    /* down_e: [hidden, rows_per_expert] @ hb → xb_partial[hidden]. */
    ib_tensor_matmul_cpu_isolated(
        a->m, down_e, a->xb_partial, a->hb, a->hidden, a->rows_per_expert,
        a->scale_buf_priv,
        a->slot->pool, a->slot->threads_per_expert,
        a->slot->acc_pool, a->slot->acc_pool_floats,
        a->slot->acc_l2_pool, a->slot->acc_l2_pool_floats);
}

static void *mome_expert_thread(void *raw) {
    mome_expert_run((mome_expert_arg *)raw);
    return NULL;
}
#endif /* MOME_HAS_PTHREAD */

/* ── Goal N28: fused gate+up fast path (FLAT, shared codebook) ───────
 *
 * Runs the new pqv2_matvec_mome_gateup_k256 kernel ONCE for all K
 * active experts (instead of K * 2 ib_tensor_matmul_cpu calls). The
 * fused kernel re-uses the per-(c, s) INT8 LUT across every (expert,
 * gate/up) output, saving the bulk of the LUT-build cost.
 *
 * Eligibility:
 *   - every active expert's gate/up tensor is PQv2 (t->pq != NULL)
 *   - K == 256 with cb_fp32 pre-decoded
 *   - flat (l2_kind == 0 for both gate and up)
 *   - all experts share the same cb_fp32 pointer (shared-codebook MoME
 *     invariant — round-5 fix). We compare pointers rather than bytes
 *     because the loader hands out exactly one decoded codebook buffer
 *     per source codebook on the load path.
 *   - shapes match (gate_e->shape[0] == rows_per_expert, etc.).
 *
 * On success: per-expert hb / hb2 buffers are written and the per-expert
 * silu(hb) * hb2 + down + weighted-accumulate is performed sequentially.
 * Returns 1 on success (caller returns immediately), 0 if ineligible.
 *
 * Env knob IB_MOME_FUSED_GATEUP=0 disables this path for A/B. */
static int mome_fused_gateup_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_MOME_FUSED_GATEUP");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* Goal H1 — v2 fused MoME kernel A/B knob.
 *
 *   IB_MOME_FUSED_V2 unset / != "0"  → use the interleaved v2 kernel
 *                                       (default; falls back to v1 if
 *                                       the v2 kernel returns -1).
 *   IB_MOME_FUSED_V2 == "0"           → skip v2, go straight to v1.
 *
 * v2 reuses the LUT in NEON registers across all K experts at each
 * 32-row m-block, which keeps acc[m..m+32] cache-hot across slots and
 * eliminates the 2K-slot acc memory sweep that v1 paid per (c, s). */
static int mome_fused_v2_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_MOME_FUSED_V2");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

static int mome_try_fused_gateup(inferbit_model *m,
                                  const ib_layer_meta *layer,
                                  const float *x_in,
                                  const float *weights,
                                  const int *active, int n_active,
                                  int rows_per_expert, int hidden,
                                  float *xb_out, float *scale_buf)
{
    if (!mome_fused_gateup_enabled()) return 0;
    if (n_active <= 0 || n_active > IB_MOME_FUSED_MAX_K) return 0;

    /* Resolve the K active experts' gate/up pqv2_t* and check eligibility. */
    const pqv2_t *gate_pq[IB_MOME_FUSED_MAX_K];
    const pqv2_t *up_pq  [IB_MOME_FUSED_MAX_K];
    const ib_tensor_meta *down_t[IB_MOME_FUSED_MAX_K];
    const pqv2_t *anchor = NULL;
    for (int i = 0; i < n_active; i++) {
        const int e = active ? active[i] : i;
        if (e < 0 || e >= layer->mome_experts) return 0;
        const ib_tensor_meta *gt = &layer->gate_proj_experts[e];
        const ib_tensor_meta *ut = &layer->up_proj_experts[e];
        const ib_tensor_meta *dt = &layer->down_proj_experts[e];
        if (!gt->pq || !ut->pq) return 0;
        if (gt->shape[0] != rows_per_expert ||
            ut->shape[0] != rows_per_expert ||
            dt->shape[1] != rows_per_expert) return 0;
        const pqv2_t *g = gt->pq;
        const pqv2_t *u = ut->pq;
        if (g->K != 256 || u->K != 256) return 0;
        if (g->l2_kind != 0 || u->l2_kind != 0) return 0;
        if (!g->cb_fp32 || !u->cb_fp32) return 0;
        if (i == 0) anchor = g;
        if (g->cb_fp32 != anchor->cb_fp32) return 0;
        if (u->cb_fp32 != anchor->cb_fp32) return 0;
        gate_pq[i] = g;
        up_pq[i]   = u;
        down_t[i]  = dt;
    }

    /* Fused gate+up output slabs: 2 × n_active × rows_per_expert floats. */
    const size_t per = (size_t)rows_per_expert;
    float *hb_all  = (float *)aligned_alloc(64,
        ((size_t)n_active * per * sizeof(float) + 63) & ~(size_t)63);
    float *hb2_all = (float *)aligned_alloc(64,
        ((size_t)n_active * per * sizeof(float) + 63) & ~(size_t)63);
    if (!hb_all || !hb2_all) {
        free(hb_all); free(hb2_all);
        return 0;
    }

    /* Goal H1: prefer the v2 interleaved kernel when enabled. v2 has the
     * same eligibility envelope as v1, so a v2 success replaces the v1
     * call entirely. On v2 returning -1 (invariant mismatch or non-NEON
     * build) we transparently fall back to v1; on v1 still returning -1
     * we punt to the per-expert path.
     *
     * Goal I3: v2's slot-inner / m-outer reorder pays off only when M_per
     * is large enough that the per-slot acc array stays cache-resident
     * across the 2K-slot sweep AND the SIMD register pressure of holding
     * 2K accumulator vectors per m-block is amortised. On TinyLlama
     * (M_per=2816) the win turns negative; on Llama-3-8B (M_per=7168
     * for K=2) it should be net positive. Threshold picked empirically —
     * TinyLlama's M_per=2816 falls just below, Llama-3 at K=2 lands
     * above. Override via IB_MOME_V2_M_MIN env. */
    const int IB_MOME_V2_M_THRESHOLD = 4096;
    static int m_thresh_cached = -1;
    if (m_thresh_cached < 0) {
        const char *thresh_env = getenv("IB_MOME_V2_M_MIN");
        m_thresh_cached = thresh_env ? atoi(thresh_env)
                                     : IB_MOME_V2_M_THRESHOLD;
        if (m_thresh_cached < 0) m_thresh_cached = IB_MOME_V2_M_THRESHOLD;
    }
    static int profile_cached = -1;
    if (profile_cached < 0) {
        const char *pe = getenv("IB_MOME_PROFILE");
        profile_cached = (pe && pe[0] && pe[0] != '0') ? 1 : 0;
    }

    int rc = -1;
    const char *kernel_used = "none";
    if (rows_per_expert >= m_thresh_cached && mome_fused_v2_enabled()) {
        rc = pqv2_matvec_mome_gateup_k256_v2(
            gate_pq, up_pq, x_in, hb_all, hb2_all, n_active);
        if (rc == 0) kernel_used = "v2";
    }
    if (rc != 0) {
        rc = pqv2_matvec_mome_gateup_k256(
            gate_pq, up_pq, x_in, hb_all, hb2_all, n_active);
        if (rc == 0) kernel_used = "v1";
    }
    if (profile_cached) {
        fprintf(stderr,
                "[mome_profile] kernel=%s M_per=%d K=%d hidden=%d\n",
                kernel_used, rows_per_expert, n_active, hidden);
    }
    if (rc != 0) {
        free(hb_all); free(hb2_all);
        return 0;
    }

    /* Per-expert silu(hb) * hb2 → hb, then down_proj, then weighted
     * accumulate into xb_out. The down_proj still uses the full
     * per-expert path (no shared codebook there in general). */
    float *down_out = (float *)aligned_alloc(64,
        ((size_t)hidden * sizeof(float) + 63) & ~(size_t)63);
    if (!down_out) {
        free(hb_all); free(hb2_all);
        return 0;
    }
    for (int i = 0; i < n_active; i++) {
        const float w_e = weights[i];
        float *hb_e  = hb_all  + (size_t)i * per;
        float *hb2_e = hb2_all + (size_t)i * per;
        for (int r = 0; r < rows_per_expert; r++) {
            hb_e[r] = mome_silu(hb_e[r]) * hb2_e[r];
        }
        ib_tensor_matmul_cpu(m, down_t[i], down_out, hb_e,
                              hidden, rows_per_expert, scale_buf);
        for (int h = 0; h < hidden; h++) {
            xb_out[h] += w_e * down_out[h];
        }
    }
    free(down_out);
    free(hb_all); free(hb2_all);
    return 1;
}

/* ── drive-mode sequential dispatch (peak-RAM optimisation) ─────────
 *
 * In residency_mode == 1 (DRIVE) the engine streams every PQv2 tensor's
 * indices through ONE shared page-aligned scratch buffer (see
 * forward.c::drive_load_indices + the 2-slot prefetch ring). The
 * design goal is "one expert in focus" — only a single expert's working
 * set is resident at any instant, so peak RAM is independent of K.
 *
 * The parallel path (mome_expert_thread) and the fused gate+up path
 * (mome_try_fused_gateup) both violate that: the parallel path holds K
 * experts' scratch slots simultaneously AND has K pthreads racing on
 * the single shared drive scratch + global prefetch state; the fused
 * path reads ALL 2K experts' pq->indices in one sweep, which the
 * one-slot drive scratch cannot satisfy. So drive mode takes this
 * dedicated sequential path instead.
 *
 * Semantics in drive mode are FIXED to exact reconstruction: run ALL K
 * experts (e = 0..K-1) in ASCENDING index order with uniform weight
 * 1.0 — identical math to the zero-router fallback at n_active == K.
 * router_logits / a top_n < K passed by the caller are intentionally
 * IGNORED here (the trivial row-split only reconstructs the un-split
 * FFN when every expert runs; a top_n subset would be a lossy
 * approximation that drive mode must never silently take). Ascending
 * order matches the drive prefetcher's tensor-registration walk so the
 * next expert's I/O overlaps the current expert's compute.
 *
 * This path performs NO per-token heap allocation and never holds more
 * than one expert's working set: it reuses the caller-supplied hb / hb2
 * scratch exactly as the RAM-mode sequential fallback does. Output is
 * bit-identical to that fallback when it runs all K experts with weight
 * 1.0 (same matmul kernel, same fp32 accumulation order). */
static void mome_dispatch_ffn_drive_seq(inferbit_model *m,
                                        const ib_layer_meta *layer,
                                        const float *x_in,
                                        float *hb, float *hb2,
                                        float *xb_out,
                                        int K, int rows_per_expert,
                                        int hidden, float *scale_buf)
{
    for (int e = 0; e < K; e++) {
        const ib_tensor_meta *gate_e = &layer->gate_proj_experts[e];
        const ib_tensor_meta *up_e   = &layer->up_proj_experts[e];
        const ib_tensor_meta *down_e = &layer->down_proj_experts[e];

        if (gate_e->shape[0] != rows_per_expert ||
            up_e->shape[0]   != rows_per_expert ||
            down_e->shape[1] != rows_per_expert) {
            continue;
        }

        ib_tensor_matmul_cpu(m, gate_e, hb,  x_in, rows_per_expert, hidden, scale_buf);
        ib_tensor_matmul_cpu(m, up_e,   hb2, x_in, rows_per_expert, hidden, scale_buf);

        for (int r = 0; r < rows_per_expert; r++) {
            hb[r] = mome_silu(hb[r]) * hb2[r];
        }

        ib_tensor_matmul_cpu(m, down_e, hb2, hb, hidden, rows_per_expert,
                             scale_buf);
        /* Uniform weight 1.0 — exact un-split FFN reconstruction. */
        for (int h = 0; h < hidden; h++) {
            xb_out[h] += hb2[h];
        }
    }
}

/* ── dispatch ────────────────────────────────────────────────────── */

void mome_dispatch_ffn(inferbit_model *m,
                       const ib_layer_meta *layer,
                       const float *x,
                       float *hb, float *hb2, float *xb_out,
                       const float *router_logits,
                       const int *active, int n_active,
                       float *scale_buf) {
    if (!m || !layer || layer->mome_experts <= 1) return;
    if (!layer->gate_proj_experts || !layer->up_proj_experts ||
        !layer->down_proj_experts) return;

    const int K       = layer->mome_experts;
    const int hidden  = m->header.hidden_size;
    const int inter   = m->header.intermediate_size;
    /* Trivial row-split: each expert covers M/K rows of gate/up. */
    const int rows_per_expert = inter / K;
    if (rows_per_expert <= 0) return;

    /* Snapshot input before zeroing xb_out — callers may pass the same
     * buffer for `x` and `xb_out` (aliasing), and the memset below would
     * otherwise destroy the input. */
    float x_in[hidden];
    memcpy(x_in, x, (size_t)hidden * sizeof(float));

    /* ── Drive mode: one-expert-in-focus sequential dispatch ─────────
     *
     * residency_mode == 1 streams every tensor's PQ indices through a
     * single shared scratch buffer. Both the pthread-parallel path and
     * the fused gate+up path require multiple experts resident at once
     * (and the parallel path races on the shared drive scratch), so
     * neither is safe here. Take the dedicated sequential path that
     * runs ALL K experts in ascending order with uniform weight 1.0
     * (exact reconstruction). The caller's router_logits / active /
     * n_active are intentionally overridden to all-K-exact in drive
     * mode — never a top_n < K subset. mome_slots_prepare is NOT called
     * (no resident per-expert slots are allocated). */
    if (m->residency_mode == 1) {
        memset(xb_out, 0, (size_t)hidden * sizeof(float));
        mome_dispatch_ffn_drive_seq(m, layer, x_in, hb, hb2, xb_out,
                                    K, rows_per_expert, hidden, scale_buf);
        return;
    }

    /* Compute weights. Two regimes:
     *   - router_logits == NULL  → zero-router fallback, n_active = K,
     *     weights are 1.0 for every expert (NOT 1/K). This is the
     *     correctness invariant: summing all K experts on the trivial
     *     row-split reconstructs the un-split FFN matmul exactly.
     *   - router_logits != NULL  → softmax over the selected active
     *     subset; weights sum to 1. */
    float weights[IB_MOME_MAX_EXPERTS];
    if (router_logits) {
        if (n_active <= 0 || n_active > IB_MOME_MAX_EXPERTS) return;
        for (int i = 0; i < n_active; i++) weights[i] = router_logits[active[i]];
        mome_softmax_inplace(weights, n_active);
    } else {
        /* Zero-router fallback: caller may request fewer than K experts via
         * IB_MOME_TOP_N. Scale each weight by K/n_active so the K-sum still
         * approximates the full un-split FFN. n_active == K → weights = 1.0
         * (exact reconstruction); n_active < K → top-n approximation, K×
         * cheaper at the cost of some PPL. */
        if (n_active <= 0 || n_active > IB_MOME_MAX_EXPERTS) return;
        const float w = (float)K / (float)n_active;
        for (int i = 0; i < n_active; i++) weights[i] = w;
    }

    /* Zero the output — we accumulate per-expert contributions. */
    memset(xb_out, 0, (size_t)hidden * sizeof(float));

    /* ── Goal N28: try the fused gate+up fast path FIRST ──────────
     * When all K experts share a codebook (post round-5 fix) and the
     * tensors are FLAT K=256, the LUT-build cost can be amortised
     * across all 2K gate+up outputs in one (c, s) sweep — saving the
     * 4× LUT-build overhead the per-expert path currently pays. */
    if (mome_try_fused_gateup(m, layer, x_in, weights, active, n_active,
                                rows_per_expert, hidden, xb_out,
                                scale_buf)) {
        return;
    }

    /* ── Decide path: parallel vs sequential ───────────────────────
     *
     * Parallel path: split the n_active experts across pthreads, each
     * with a private (threads_per_expert)-worker pool. Falls back to
     * sequential when:
     *   - IB_MOME_PARALLEL=0
     *   - n_active < 2 (nothing to parallelise)
     *   - m->num_threads <= 1 (no concurrency budget)
     *   - slot infrastructure failed to initialise
     * Sequential path matches the original v1 behaviour exactly. */
    int do_parallel = 0;
#if MOME_HAS_PTHREAD
    do_parallel = (mome_parallel_enabled() && n_active >= 2 &&
                   m->num_threads > 1);
#endif
    int threads_per_expert = 0;
    if (do_parallel) {
        threads_per_expert = m->num_threads / n_active;
        if (threads_per_expert < 1) threads_per_expert = 1;
    }

    /* Detect whether any of this layer's expert tensors uses L2
     * (pyramid). Used to size the slot's L2 scratch. The cheapest
     * proxy: peek at expert 0's gate_proj_experts; all experts in a
     * layer share the same encoding kind. */
    int need_l2 = 0;
    if (do_parallel && layer->gate_proj_experts[0].pq) {
        const pqv2_t *pq0 = layer->gate_proj_experts[0].pq;
        if (pq0->l2_kind == 2 && pq0->l2_cb_fp32 && pq0->l2_K <= 64) {
            need_l2 = 1;
        }
    }

    /* Size the per-slot scratch. acc/L2 use threads_per_expert ×
     * max_M floats (matches PQv2 threaded path's n_slots × M layout).
     * Activation scratch uses 4 × max_M floats per slot (hb, hb2,
     * xb_partial, scale_buf — each over-sized to max_M for trivial
     * math; wastes a few hundred KB total, negligible). */
    size_t max_M = (size_t)(rows_per_expert > hidden ? rows_per_expert : hidden);
    size_t needed_acc_floats = (size_t)threads_per_expert * max_M;
    size_t needed_scratch_floats = 4 * max_M;

#if MOME_HAS_PTHREAD
    if (do_parallel) {
        if (mome_slots_prepare(n_active, threads_per_expert,
                               needed_acc_floats, needed_scratch_floats,
                               need_l2) != 0) {
            do_parallel = 0;   /* fall back to sequential */
        }
    }
#else
    (void)threads_per_expert;
    (void)max_M;
    (void)needed_acc_floats;
    (void)needed_scratch_floats;
    (void)need_l2;
#endif

#if MOME_HAS_PTHREAD
    if (do_parallel) {
        /* Per-expert args live on stack (small — K ≤ 32, ~64 bytes
         * each). Scratch buffers (hb, hb2, xb_partial, scale_buf) are
         * static-lifetime per slot — eliminates the per-call
         * aligned_alloc that would otherwise run ~64 times per
         * decoded token (K experts × 32 layers). */
        mome_expert_arg args[IB_MOME_MAX_EXPERTS];
        pthread_t tids[IB_MOME_MAX_EXPERTS];
        for (int i = 0; i < n_active; i++) {
            const int e = active ? active[i] : i;
            if (e < 0 || e >= K) {
                args[i].expert = -1;
                continue;
            }
            mome_expert_slot *slot = &g_slots[i];
            args[i].m              = m;
            args[i].layer          = layer;
            args[i].x_in           = x_in;
            args[i].expert         = e;
            args[i].rows_per_expert= rows_per_expert;
            args[i].hidden         = hidden;
            args[i].hb             = slot->hb;
            args[i].hb2            = slot->hb2;
            args[i].xb_partial     = slot->xb_partial;
            args[i].scale_buf_priv = slot->scale_buf;
            args[i].slot           = slot;
        }

        /* Spawn workers for experts 1..n_active-1; the calling
         * thread runs expert 0 itself to avoid the cost of a
         * pthread_create+join cycle for the easy case. */
        for (int i = 1; i < n_active; i++) {
            if (args[i].expert < 0) continue;
            if (pthread_create(&tids[i], NULL, mome_expert_thread,
                               &args[i]) != 0) {
                /* On thread-create failure, run the rest inline. */
                for (int j = i; j < n_active; j++) {
                    if (args[j].expert >= 0) mome_expert_run(&args[j]);
                }
                /* Join previously-spawned 1..i-1. */
                for (int j = 1; j < i; j++) {
                    if (args[j].expert >= 0) pthread_join(tids[j], NULL);
                }
                goto reduce;
            }
        }
        if (args[0].expert >= 0) mome_expert_run(&args[0]);
        for (int i = 1; i < n_active; i++) {
            if (args[i].expert >= 0) pthread_join(tids[i], NULL);
        }

reduce:
        /* Weighted reduce K partials → xb_out. */
        for (int i = 0; i < n_active; i++) {
            if (args[i].expert < 0) continue;
            const float w_e = weights[i];
            const float *p  = args[i].xb_partial;
            for (int h = 0; h < hidden; h++) xb_out[h] += w_e * p[h];
        }
        return;
    }
#endif /* MOME_HAS_PTHREAD */

    /* ── Sequential fallback (original v1 path) ──────────────────── */
    for (int i = 0; i < n_active; i++) {
        const int e = active ? active[i] : i;
        if (e < 0 || e >= K) continue;
        const float w_e = weights[i];

        const ib_tensor_meta *gate_e = &layer->gate_proj_experts[e];
        const ib_tensor_meta *up_e   = &layer->up_proj_experts[e];
        const ib_tensor_meta *down_e = &layer->down_proj_experts[e];

        if (gate_e->shape[0] != rows_per_expert ||
            up_e->shape[0]   != rows_per_expert ||
            down_e->shape[1] != rows_per_expert) {
            continue;
        }

        ib_tensor_matmul_cpu(m, gate_e, hb,  x_in, rows_per_expert, hidden, scale_buf);
        ib_tensor_matmul_cpu(m, up_e,   hb2, x_in, rows_per_expert, hidden, scale_buf);

        for (int r = 0; r < rows_per_expert; r++) {
            hb[r] = mome_silu(hb[r]) * hb2[r];
        }

        ib_tensor_matmul_cpu(m, down_e, hb2, hb, hidden, rows_per_expert,
                             scale_buf);
        for (int h = 0; h < hidden; h++) {
            xb_out[h] += w_e * hb2[h];
        }
    }
}

/* ── Training-free sparse-FFN clustering (CONVERT-TIME, data-free) ───
 *
 * See mome.h::ffn_compute_cluster_perm for the public contract. This is
 * the convert-time half of the sparse-FFN feature: it clusters a layer's
 * FFN intermediate dimension into N contiguous groups by cosine
 * similarity of the gate_proj rows and emits a permutation + cluster
 * offsets + per-cluster centroids. The matching on-disk record layout
 * lives in pqv2_format.h; the writer that emits it lives in
 * pqv2_encode.c (read_and_push_pqv2 / the FFN clustering pass).
 *
 * Algorithm: the SAME high-D spherical (cosine) k-means as the dormant
 * MoME cosine probe (pqv2_encode.c::mome_compute_cosine_perm). Cosine
 * similarity = dot product on L2-normalised rows, so "nearest center" =
 * "largest dot" and the centroid update is mean-then-renormalise. The
 * key difference from the MoME probe — and the FIX for its documented
 * perm/boundary inconsistency — is that we do NOT chop the row order
 * into forced-equal inter/K blocks. The MoME probe's balanced-overflow
 * step let a block boundary fall in the MIDDLE of a k-means cluster, so
 * the per-expert "cluster" boundaries (inter/K) did not match the
 * cluster labels — that mismatch is exactly what the dormant probe's
 * comment flags as the residual +1.9% PPL inconsistency. Here we instead
 * lay rows out cluster-by-cluster and record the NATURAL cluster sizes
 * in cluster_offsets, so the permutation and the cluster boundaries are
 * consistent by construction (each cluster_offsets block contains
 * exactly the rows of one k-means cluster). */

/* fp32 → fp16 (IEEE half) bit pattern. Self-contained to keep mome.c
 * free of the pq_decode.h include (mirrors the inline fp16→fp32 decode
 * already used in mome_router_is_nonzero). Round-to-nearest-even is not
 * required for these centroid signatures (a coarse runtime predictor),
 * so this uses simple truncation with the standard exponent rebias and
 * overflow/underflow clamping. */
static uint16_t ffn_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {
        /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {
        /* overflow → inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {
        /* subnormal or underflow to zero */
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;             /* restore implicit 1 */
        int shift = 14 - exp;          /* 14 = 23 - 10 + (1 - exp)... */
        uint32_t sub = mant >> shift;
        return (uint16_t)(sign | (sub & 0x3FFu));
    }
    return (uint16_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
}

int ffn_compute_cluster_perm(const float *W_gate, int inter, int hidden,
                             int n_clusters, uint32_t seed,
                             int *perm_out, uint32_t *offsets_out,
                             uint16_t *centroids_fp16_out)
{
    if (!W_gate || !perm_out || !offsets_out || !centroids_fp16_out) return -1;
    if (inter <= 0 || hidden <= 0 || n_clusters <= 1 || n_clusters > inter)
        return -1;

    /* L2-normalise each gate row into a scratch copy (cosine k-means =
     * Euclidean k-means on unit vectors). Keep the row norm so we can
     * later average the UN-normalised rows for the input-space centroid. */
    float *Wn = (float *)malloc((size_t)inter * (size_t)hidden * sizeof(float));
    if (!Wn) return -1;
    for (int r = 0; r < inter; r++) {
        const float *src = W_gate + (size_t)r * hidden;
        float *dst = Wn + (size_t)r * hidden;
        double ss = 0.0;
        for (int i = 0; i < hidden; i++) ss += (double)src[i] * (double)src[i];
        float invn = (ss > 1e-20) ? (float)(1.0 / sqrt(ss)) : 0.0f;
        for (int i = 0; i < hidden; i++) dst[i] = src[i] * invn;
    }

    float   *centers = (float *)malloc((size_t)n_clusters * hidden * sizeof(float));
    int32_t *labels  = (int32_t *)malloc((size_t)inter * sizeof(int32_t));
    if (!centers || !labels) { free(Wn); free(centers); free(labels); return -1; }
    for (int r = 0; r < inter; r++) labels[r] = 0;

    /* k-means++ / farthest-point seeding on cosine distance, then Lloyd
     * iterations with renormalised (spherical) centroid updates. */
    {
        uint32_t rng = seed ? seed : 1234u;
        rng = rng * 1664525u + 1013904223u;
        int seed0 = (int)(rng % (uint32_t)inter);
        memcpy(centers, Wn + (size_t)seed0 * hidden,
               (size_t)hidden * sizeof(float));
        float *mindist = (float *)malloc((size_t)inter * sizeof(float));
        if (!mindist) { free(Wn); free(centers); free(labels); return -1; }
        for (int r = 0; r < inter; r++) {
            const float *x = Wn + (size_t)r * hidden;
            float dot = 0.0f;
            for (int i = 0; i < hidden; i++) dot += x[i] * centers[i];
            mindist[r] = 1.0f - dot;
        }
        for (int kk = 1; kk < n_clusters; kk++) {
            int best = 0; float bestd = -1.0f;
            for (int r = 0; r < inter; r++)
                if (mindist[r] > bestd) { bestd = mindist[r]; best = r; }
            memcpy(centers + (size_t)kk * hidden, Wn + (size_t)best * hidden,
                   (size_t)hidden * sizeof(float));
            const float *c = centers + (size_t)kk * hidden;
            for (int r = 0; r < inter; r++) {
                const float *x = Wn + (size_t)r * hidden;
                float dot = 0.0f;
                for (int i = 0; i < hidden; i++) dot += x[i] * c[i];
                float d = 1.0f - dot;
                if (d < mindist[r]) mindist[r] = d;
            }
        }
        free(mindist);

        double *csum = (double *)malloc((size_t)n_clusters * hidden * sizeof(double));
        if (!csum) { free(Wn); free(centers); free(labels); return -1; }
        int max_iter = 25;
        {
            const char *e = getenv("IB_FFN_CLUSTER_ITERS");
            if (e && *e) { int v = atoi(e); if (v > 0) max_iter = v; }
        }
        for (int it = 0; it < max_iter; it++) {
            int changed = 0;
            for (int r = 0; r < inter; r++) {
                const float *x = Wn + (size_t)r * hidden;
                int bestk = 0; float bestdot = -2.0f;
                for (int k = 0; k < n_clusters; k++) {
                    const float *c = centers + (size_t)k * hidden;
                    float dot = 0.0f;
                    for (int i = 0; i < hidden; i++) dot += x[i] * c[i];
                    if (dot > bestdot) { bestdot = dot; bestk = k; }
                }
                if (labels[r] != bestk) changed = 1;
                labels[r] = bestk;
            }
            memset(csum, 0, (size_t)n_clusters * hidden * sizeof(double));
            int *cnt = (int *)calloc((size_t)n_clusters, sizeof(int));
            if (!cnt) { free(csum); free(Wn); free(centers); free(labels); return -1; }
            for (int r = 0; r < inter; r++) {
                int k = labels[r];
                const float *x = Wn + (size_t)r * hidden;
                double *acc = csum + (size_t)k * hidden;
                for (int i = 0; i < hidden; i++) acc[i] += x[i];
                cnt[k]++;
            }
            for (int k = 0; k < n_clusters; k++) {
                float *c = centers + (size_t)k * hidden;
                if (cnt[k] == 0) continue;
                const double *acc = csum + (size_t)k * hidden;
                double ss = 0.0;
                for (int i = 0; i < hidden; i++) ss += acc[i] * acc[i];
                float invn = (ss > 1e-20) ? (float)(1.0 / sqrt(ss)) : 0.0f;
                for (int i = 0; i < hidden; i++) c[i] = (float)(acc[i] * invn);
            }
            free(cnt);
            if (!changed && it > 0) break;
        }
        free(csum);
    }
    free(centers);   /* unit centers no longer needed; we recompute the
                        input-space centroid from the raw rows below. */
    free(Wn);

    /* Cluster sizes (natural k-means partition — NO forced balancing). */
    int *csize = (int *)calloc((size_t)n_clusters, sizeof(int));
    if (!csize) { free(labels); return -1; }
    for (int r = 0; r < inter; r++) {
        int c = labels[r];
        if (c < 0 || c >= n_clusters) c = 0;
        csize[c]++;
    }
    if (getenv("IB_FFN_CLUSTER_DEBUG")) {
        fprintf(stderr, "[ffn_cluster] sizes:");
        for (int c = 0; c < n_clusters; c++) fprintf(stderr, " %d", csize[c]);
        fprintf(stderr, "\n");
    }

    /* Contiguous cluster offsets in PERMUTED space. Clusters are laid out
     * in label order 0..n_clusters-1; cluster c occupies
     * [offsets[c], offsets[c+1]). offsets[0] == 0, offsets[N] == inter. */
    offsets_out[0] = 0u;
    for (int c = 0; c < n_clusters; c++)
        offsets_out[c + 1] = offsets_out[c] + (uint32_t)csize[c];
    /* offsets_out[n_clusters] now equals inter by construction. */

    /* Build the permutation: emit rows cluster-by-cluster in label order
     * so each cluster's rows are contiguous, matching offsets above.
     * perm_out[new_row] = old_row. */
    int pos = 0;
    for (int c = 0; c < n_clusters; c++) {
        for (int r = 0; r < inter && pos < inter; r++)
            if (labels[r] == c) perm_out[pos++] = r;
    }
    /* Safety: append any out-of-range-labelled rows (shouldn't happen). */
    if (pos < inter) {
        char *seen = (char *)calloc((size_t)inter, 1);
        if (seen) {
            for (int i = 0; i < pos; i++) seen[perm_out[i]] = 1;
            for (int r = 0; r < inter && pos < inter; r++)
                if (!seen[r]) perm_out[pos++] = r;
            free(seen);
        }
    }
    if (pos != inter) { free(labels); free(csize); return -1; }

    /* Per-cluster centroid in INPUT (hidden) space = arithmetic mean of
     * that cluster's UN-normalised gate rows. This is the input-space
     * "signature" the runtime compares the activation against — the mean
     * raw row, not the unit-normalised k-means center. */
    double *acc = (double *)malloc((size_t)hidden * sizeof(double));
    if (!acc) { free(labels); free(csize); return -1; }
    for (int c = 0; c < n_clusters; c++) {
        for (int i = 0; i < hidden; i++) acc[i] = 0.0;
        int cnt = 0;
        for (int r = 0; r < inter; r++) {
            if (labels[r] != c) continue;
            const float *row = W_gate + (size_t)r * hidden;
            for (int i = 0; i < hidden; i++) acc[i] += (double)row[i];
            cnt++;
        }
        double inv = (cnt > 0) ? 1.0 / (double)cnt : 0.0;
        uint16_t *dst = centroids_fp16_out + (size_t)c * (size_t)hidden;
        for (int i = 0; i < hidden; i++)
            dst[i] = ffn_f32_to_f16((float)(acc[i] * inv));
    }
    free(acc);

    free(labels);
    free(csize);
    return 0;
}

/* ── TODOs (v1 deliberately deferred) ────────────────────────────────
 *
 *  1. Real co-activation clustering. v1 uses a trivial row-range
 *     split (each expert covers a contiguous block of M/K rows).
 *     Real clustering is offline: a Python helper forward-passes a
 *     calibration set, captures FFN activation magnitudes per row,
 *     then runs agglomerative clustering (sklearn) on co-occurrence
 *     distance — output a per-tensor row-permutation that this
 *     encoder applies before slicing. See Stage 3a.1 / 3a.2.
 *
 *  2. Real router training. v1 emits a zero-init router so the
 *     runtime falls back to "all experts uniformly weighted = full
 *     FFN matmul". The follow-up calibration writes a [hidden, K]
 *     fp16 router whose logits direct top-N at decode time. Source:
 *     same calibration pass as #1, with logistic-regression / per-
 *     row mean-activation as the supervisory signal.
 *
 *  3. Router matmul integration. Once #2 lands, this file gains a
 *     `mome_router_matmul(model, layer, x, out)` that decodes the
 *     fp16 router weight and produces the [K] logits. Forward.c then
 *     calls top_n + dispatch_ffn instead of the zero-router fallback.
 *
 *  4. Metal MoME. The runtime path here is CPU-only. The Metal
 *     recorder will need a per-expert dispatch (or one larger
 *     packed-matmul that masks inactive experts). For v1, layers
 *     with mome_experts > 1 fall back to CPU at the layer level —
 *     see forward.c::forward_single_ex.
 *
 *  5. CLI `--mome` flag. Out of scope here (Wave 4); the public
 *     `inferbit_convert_config::mome_experts` field is exposed and
 *     wired through the C dispatcher in pqv2_convert(). The CLI
 *     binding lives in inferbit-py / inferbit-node and will follow
 *     in a future commit.
 */
