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
    int n = (K < 2) ? K : 2;   /* default min(2, K) */
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
     * we punt to the per-expert path. */
    int rc = -1;
    if (mome_fused_v2_enabled()) {
        rc = pqv2_matvec_mome_gateup_k256_v2(
            gate_pq, up_pq, x_in, hb_all, hb2_all, n_active);
    }
    if (rc != 0) {
        rc = pqv2_matvec_mome_gateup_k256(
            gate_pq, up_pq, x_in, hb_all, hb2_all, n_active);
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
