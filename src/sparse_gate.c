/*
 * sparse_gate.c — training-free sparse-FFN cluster gate (runtime side).
 *
 * Pure, allocation-free, deterministic. No model / forward / loader
 * dependencies beyond the function arguments and libm. See sparse_gate.h
 * for the API contract and the on-disk record layout.
 */

#include "sparse_gate.h"

#include <math.h>

/* Local IEEE-754 half -> single decode. Self-contained so the gate has
 * zero link dependencies (unit-testable in isolation). Bit-exact with
 * pq_decode.c::ib_fp16_to_fp32 for all inputs incl. subnormals/inf/NaN.
 * If a future build wants to share that symbol instead, swap the call
 * site — the numeric result is identical. */
static inline float sg_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0u) {
        if (mant == 0u) {
            bits = sign;                       /* +/- zero */
        } else {
            /* Subnormal: normalize. */
            exp = 1u;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FFu;
            bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        /* Inf / NaN. */
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }

    float out;
    /* Type-pun via memcpy-free union-equivalent: use a compound copy. */
    union { uint32_t u; float f; } cvt;
    cvt.u = bits;
    out = cvt.f;
    return out;
}

static inline float sg_silu(float v) {
    return v / (1.0f + expf(-v));
}

/* dot(x, centroid_c) over `hidden`, centroid in fp16. */
static inline float sg_dot_fp16(const float *x, const uint16_t *c, int hidden) {
    float acc = 0.0f;
    for (int i = 0; i < hidden; i++) {
        acc += x[i] * sg_fp16_to_fp32(c[i]);
    }
    return acc;
}

void sparse_gate_scores(const uint16_t *centroids_fp16, int n_clusters,
                        int hidden, const float *x, float *out_scores) {
    if (!centroids_fp16 || !x || !out_scores || n_clusters <= 0 || hidden <= 0)
        return;
    for (int c = 0; c < n_clusters; c++) {
        const uint16_t *cen = centroids_fp16 + (size_t)c * (size_t)hidden;
        out_scores[c] = sg_silu(sg_dot_fp16(x, cen, hidden));
    }
}

int sparse_gate_select(const uint16_t *centroids_fp16, int n_clusters,
                       int hidden, const float *x,
                       float thresh, int top_min, int *active_out) {
    if (!centroids_fp16 || !x || !active_out || n_clusters <= 0 || hidden <= 0)
        return 0;

    /* keep[c] flags whether cluster c is selected. Bounded stack scratch:
     * n_clusters is small (FFN cluster counts are tens, not thousands), so
     * a fixed cap keeps this allocation-free. Clusters beyond the cap are
     * never gated out (treated as kept) — a safe, behavior-preserving
     * fallback rather than a wrong skip. */
    enum { SG_MAX_CLUSTERS = 1024 };
    if (n_clusters > SG_MAX_CLUSTERS) {
        for (int c = 0; c < n_clusters; c++) active_out[c] = c;
        return n_clusters;
    }

    float scores_abs[SG_MAX_CLUSTERS];
    unsigned char keep[SG_MAX_CLUSTERS];

    int kept = 0;
    for (int c = 0; c < n_clusters; c++) {
        const uint16_t *cen = centroids_fp16 + (size_t)c * (size_t)hidden;
        float s = sg_silu(sg_dot_fp16(x, cen, hidden));
        float a = s < 0.0f ? -s : s;
        scores_abs[c] = a;
        if (a >= thresh) {
            keep[c] = 1;
            kept++;
        } else {
            keep[c] = 0;
        }
    }

    /* Clamp the minimum to the available clusters. */
    int need = top_min;
    if (need > n_clusters) need = n_clusters;
    if (need < 0) need = 0;

    /* Top-up: while fewer than `need` are kept, add the highest-|score|
     * not-yet-kept cluster. Deterministic tie-break: equal |score| prefers
     * the lower index (strict '>' on the running best). O(need*n) — fine
     * for small n_clusters and avoids any allocation/sort. */
    while (kept < need) {
        int best = -1;
        float best_a = -1.0f;
        for (int c = 0; c < n_clusters; c++) {
            if (keep[c]) continue;
            if (scores_abs[c] > best_a) {
                best_a = scores_abs[c];
                best = c;
            }
        }
        if (best < 0) break;   /* nothing left to add */
        keep[best] = 1;
        kept++;
    }

    /* Emit selected indices in ascending cluster-index order. */
    int k = 0;
    for (int c = 0; c < n_clusters; c++) {
        if (keep[c]) active_out[k++] = c;
    }
    return k;
}
