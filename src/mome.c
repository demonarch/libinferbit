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
 */

#include "mome.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

    for (int i = 0; i < n_active; i++) {
        const int e = active ? active[i] : i;
        if (e < 0 || e >= K) continue;
        const float w_e = weights[i];

        const ib_tensor_meta *gate_e = &layer->gate_proj_experts[e];
        const ib_tensor_meta *up_e   = &layer->up_proj_experts[e];
        const ib_tensor_meta *down_e = &layer->down_proj_experts[e];

        /* gate_e: [rows_per_expert, hidden] @ x[hidden] → hb[rows_per_expert]
         * up_e:   [rows_per_expert, hidden] @ x[hidden] → hb2[rows_per_expert]
         * Sub-tensor sizes are validated by the loader; if a slot is
         * malformed (M mismatch), fall back to skipping that expert. */
        if (gate_e->shape[0] != rows_per_expert ||
            up_e->shape[0]   != rows_per_expert ||
            down_e->shape[1] != rows_per_expert) {
            continue;
        }

        ib_tensor_matmul_cpu(m, gate_e, hb,  x_in, rows_per_expert, hidden, scale_buf);
        ib_tensor_matmul_cpu(m, up_e,   hb2, x_in, rows_per_expert, hidden, scale_buf);

        /* SiLU(gate) * up — element-wise on the per-expert slice. */
        for (int r = 0; r < rows_per_expert; r++) {
            hb[r] = mome_silu(hb[r]) * hb2[r];
        }

        /* down_e: [hidden, rows_per_expert] @ hb[rows_per_expert]
         *       → xb_tmp[hidden]. Accumulate into xb_out with w_e.
         *
         * Reuse hb2 as the per-expert down output scratch — it is at
         * least `inter` floats long which is >= hidden on any
         * reasonable Llama-family config; we only need `hidden`. */
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
