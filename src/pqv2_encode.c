/*
 * pqv2_encode.c — Production PQv2 / pyramid encoder + IBF v6 writer.
 *
 * Wired into convert.c via the format-flag dispatcher. Algorithm port
 * of scripts/poc/pqv2_encode.py to C; reuses ib_kmeans_fit from
 * pq_kmeans.c so we don't ship two k-means.
 *
 * See pqv2_encode.h for the public surface and design notes.
 */

#include "pqv2_encode.h"
#include "pqv2_format.h"
#include "pqv2_kernel.h"
#include "pq_decode.h"      /* ib_fp32_to_fp16, ib_fp16_to_fp32 */
#include "pq_kmeans.h"
#include "inferbit_internal.h"
#include "platform.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

/* fp16 helpers — keep this file self-contained beyond pq_decode's
 * already-exported converters. */
static inline uint16_t enc_f2h(float f) { return ib_fp32_to_fp16(f); }
static inline float    enc_h2f(uint16_t h) { return ib_fp16_to_fp32(h); }

/* fp16's smallest positive normal: 2^-14. Matches the value used by
 * scripts/poc/pqv2_encode.py:179 and the read-side row-scale invariant
 * documented in metal_model.mm:441-466. */
#define IB_PQV2_FP16_MIN_NORMAL  6.103515625e-5f

/* ── Stage 5k: fp8 (E4M3) row_scale + cb_scale codecs ────────────────
 *
 * E4M3 layout (per the OCP "FP8 Formats for Deep Learning" spec, see
 * https://arxiv.org/abs/2209.05433):
 *   sign : 1 bit
 *   exp  : 4 bits (bias = 7)
 *   mant : 3 bits
 *
 *   exp == 0       : subnormal value = sign * 2^-6 * (mant/8)
 *   1 <= exp <= 14 : normal    value = sign * 2^(exp-7) * (1 + mant/8)
 *   exp == 15 + mant == 7 (binary 1111 111) : sentinel NaN (no Inf in E4M3)
 *
 * Max representable magnitude ≈ 448 (1.75 × 2^8) — covers the full
 * dynamic range of PQv2 row_scale (multi-decade, ~10 decades) and
 * codebook scales.
 *
 * H2 sp2 redesign (Agent 3 round 1 fix): row_scale previously used a
 * linear int8[M] + fp16 row_max codec, which collapsed the multi-decade
 * row-scale range into a single linear quantum (~|row_max|/127 ≈ 3
 * decades useful). Llama-3 / TinyLlama row_scales span 6-10 decades
 * across rows of a single tensor, so the int8 path floored small rows
 * to zero and produced PPL 34-50. Switching row_scale to fp8 E4M3
 * (logarithmic, ~6-12% per-row relative error) restores ~10 decades
 * of dynamic range; cb_scale already uses E4M3 since it has the same
 * dynamic-range pathology.
 *
 * Decode is one read + a few shifts/multiplies; encode is one log2
 * + bias + clamp. Both stay inside a single SIMD register-equivalent —
 * matches the "inline-cost transform" rule from docs/v2 §5h "file-size
 * rule". The encoder is called once per codeword at conversion time
 * (cold path); the decoder is called once per codeword at load time
 * (cold path) — the hot inner kernel never sees fp8 because the loader
 * expands E4M3 back into fp16 before any matmul runs. */

/* Encoder-side decode is the unit-test / round-trip counterpart to
 * enc_f32_to_e4m3 — kept here for completeness so future agents can
 * verify the codec without re-deriving the bit layout. The production
 * loader uses pqv2_e4m3_to_f32 in pqv2_format.c. */
static inline float __attribute__((unused)) enc_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (uint32_t)(b >> 7) & 0x1u;
    uint32_t exp  = (uint32_t)(b >> 3) & 0xFu;
    uint32_t mant = (uint32_t)b & 0x7u;
    /* Sentinel NaN. */
    if (exp == 0xFu && mant == 0x7u) {
        uint32_t nan_bits = (sign << 31) | 0x7FC00000u;
        float f; memcpy(&f, &nan_bits, 4); return f;
    }
    float val;
    if (exp == 0u) {
        /* Subnormal: 2^-6 * (mant / 8). */
        val = (float)mant * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        /* Normal: 2^(exp-7) * (1 + mant/8). */
        int e = (int)exp - 7;
        float mantissa = 1.0f + (float)mant * (1.0f / 8.0f);
        val = ldexpf(mantissa, e);
    }
    return sign ? -val : val;
}

static inline uint8_t enc_f32_to_e4m3(float f) {
    if (f != f) return 0xFFu;          /* NaN → S=1 sentinel */
    uint32_t sign = (f < 0.0f) ? 1u : 0u;
    float a = sign ? -f : f;
    /* Round to the nearest representable E4M3 magnitude. Max normal
     * magnitude = 2^8 * (1 + 7/8) = 448; clamp at the saturation
     * level (= 0x7E = +max if positive, 0xFE = -max if negative). */
    const float E4M3_MAX = 448.0f;
    if (a >= E4M3_MAX) {
        return (uint8_t)((sign << 7) | 0x7Eu);
    }
    /* Compute target exponent (unbiased). Below 2^-6 we hit subnormal. */
    if (a < (1.0f / 64.0f)) {
        /* Subnormal region: value = (mant / 8) * 2^-6. */
        float scaled = a * 64.0f * 8.0f;  /* a / (2^-6 / 8) */
        int m = (int)lrintf(scaled);
        if (m <= 0) return (uint8_t)(sign << 7);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | (uint32_t)m);
    }
    int e;
    float mantissa = frexpf(a, &e);     /* a = mantissa * 2^e, 0.5 ≤ mantissa < 1 */
    /* frexpf returns mantissa ∈ [0.5, 1); E4M3 uses 1.xxx so shift. */
    mantissa *= 2.0f; e -= 1;            /* now mantissa ∈ [1.0, 2.0) and e is unbiased */
    int biased = e + 7;
    if (biased <= 0) {
        /* Falls into subnormal range when scaled. */
        float scaled = a * 64.0f * 8.0f;
        int m = (int)lrintf(scaled);
        if (m <= 0) return (uint8_t)(sign << 7);
        if (m > 7) m = 7;
        return (uint8_t)((sign << 7) | (uint32_t)m);
    }
    if (biased > 14) biased = 14;        /* will clamp mant below */
    /* mantissa ∈ [1, 2); store the fractional part as 3-bit field
     * rounded to nearest. */
    int m = (int)lrintf((mantissa - 1.0f) * 8.0f);
    if (m == 8) { m = 0; biased += 1; }
    if (biased > 15) biased = 15;        /* belt-and-suspenders */
    /* Reject the (15, 7) NaN sentinel — bump down to the max-finite
     * encoding 0x7E if rounding pushed us up to the reserved slot. */
    if (biased == 15 && m == 7) {
        biased = 14; m = 7;
    }
    return (uint8_t)((sign << 7) | ((uint32_t)biased << 3) | (uint32_t)m);
}

/* DEPRECATED (H2 sp2 redesign): the original Stage 5k row_scale codec
 * packed fp16[M] into int8[M] + fp16 row_max, recovering each row scale
 * as `(int8[m] / 127) * row_max`. That codec is fundamentally wrong for
 * LLM row_scales, which span 6-10 decades within a single tensor —
 * linear quantization with a per-tensor anchor floored the small-magnitude
 * rows to zero and inflated PPL to 34-50. Kept here (compile-time-unused)
 * so the codec history is recoverable; no caller uses it now. */
static void __attribute__((unused))
enc_pack_row_scale_int8(const uint16_t *src_fp16, int M,
                          int8_t *dst_int8, uint16_t *out_row_max)
{
    float ax = 0.0f;
    for (int m = 0; m < M; m++) {
        float v = enc_h2f(src_fp16[m]);
        float a = v < 0 ? -v : v;
        if (a > ax) ax = a;
    }
    if (ax < IB_PQV2_FP16_MIN_NORMAL) ax = IB_PQV2_FP16_MIN_NORMAL;
    uint16_t max_h = enc_f2h(ax);
    float max_f = enc_h2f(max_h);
    *out_row_max = max_h;
    float inv = 127.0f / max_f;
    for (int m = 0; m < M; m++) {
        float v = enc_h2f(src_fp16[m]);
        int q = (int)lrintf(v * inv);
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        dst_int8[m] = (int8_t)q;
    }
}

/* Encode an fp16-valued row_scale array as fp8 E4M3 bytes — one byte
 * per row scale. Same codec as cb_scale.
 *
 * Decode invariant: `recovered_fp16[m] ≈ pqv2_e4m3_to_f32(fp8[m])`.
 * Per-row relative error is bounded by E4M3's mantissa step
 * (~1/8 ≈ 6-12%), independent of the per-tensor dynamic range. Covers
 * approximately 10 decades end-to-end (2^-9 subnormal floor to 448
 * saturation), which is enough to encode every Llama-3 / TinyLlama
 * row_scale we have measured without anchor loss. */
static void enc_pack_row_scale_e4m3(const uint16_t *src_fp16, int M,
                                      uint8_t *dst_e4m3)
{
    for (int m = 0; m < M; m++) {
        dst_e4m3[m] = enc_f32_to_e4m3(enc_h2f(src_fp16[m]));
    }
}

/* Encode an fp16-valued codebook-scale array as fp8 E4M3 bytes. */
static void enc_pack_cb_scale_e4m3(const uint16_t *src_fp16, size_t n,
                                     uint8_t *dst_e4m3)
{
    for (size_t i = 0; i < n; i++) {
        dst_e4m3[i] = enc_f32_to_e4m3(enc_h2f(src_fp16[i]));
    }
}

/* ── source-row → fp32 (mirror quantize.c::read_row_fp32 inline) ──── */

static float pqv2_fp16_to_f32_local(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float f; uint32_t b = sign; memcpy(&f, &b, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t b = sign | 0x7F800000u | (mant << 13);
        float f; memcpy(&f, &b, 4); return f;
    }
    uint32_t b = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &b, 4); return f;
}

static float pqv2_bf16_to_f32_local(uint16_t h) {
    uint32_t b = (uint32_t)h << 16;
    float f; memcpy(&f, &b, 4); return f;
}

/* Read [rows][cols] tensor into row-major fp32 buffer. */
static int pqv2_read_matrix_fp32(float *out,
                                  const void *src, const char *dtype,
                                  int rows, int cols)
{
    size_t total = (size_t)rows * cols;
    if (strcmp(dtype, "F32") == 0) {
        memcpy(out, src, total * sizeof(float));
        return 0;
    }
    if (strcmp(dtype, "F16") == 0) {
        const uint8_t *p = (const uint8_t *)src;
        for (size_t i = 0; i < total; i++) {
            uint16_t v; memcpy(&v, p + i * 2, 2);
            out[i] = pqv2_fp16_to_f32_local(v);
        }
        return 0;
    }
    if (strcmp(dtype, "BF16") == 0) {
        const uint8_t *p = (const uint8_t *)src;
        for (size_t i = 0; i < total; i++) {
            uint16_t v; memcpy(&v, p + i * 2, 2);
            out[i] = pqv2_bf16_to_f32_local(v);
        }
        return 0;
    }
    return -1;
}

/* ── parallel slot worker (used by pqv2_encode_flat_impl) ─────────── */

typedef struct {
    int M, N, G, K, half;
    int n_chunks, n_sub, n_points, fit_sample_cap;
    uint32_t seed;
    const float *Wn;                    /* row-normalized W */
    int8_t   *cb_int8_out;              /* [n_sub][K][half] */
    uint16_t *cb_scale_fp16_out;        /* [n_sub][K] */
    /* On-disk u8 index output buffer. Logical extent is M*n_chunks*n_sub
     * bytes regardless of layout; what changes is the scatter pattern in
     * the slot worker, gated by `idx_layout_rowmajor`:
     *   0 → write at [(ch * n_sub + s) * M + m]   (chunk-major; legacy
     *                                              NEON-friendly).
     *   1 → write at [(m * n_chunks + ch) * n_sub + s] (row-major; the
     *                                              Metal upload becomes
     *                                              zero-copy because the
     *                                              GPU kernel already
     *                                              reads in this layout).
     * NULL = no on-disk u8 emission for this call (the pyramid L1 pass
     * uses indices_rowmajor_out below for the residual reconstruction
     * and re-encodes via push_pqv2_tensor with its own buffer). */
    uint8_t  *indices_chunkmajor_out;
    int       idx_layout_rowmajor;      /* 0 = chunk-major (legacy), 1 = row-major */
    int32_t  *indices_rowmajor_out;     /* [M][n_chunks][n_sub] int32 — caller may
                                           request this to drive a pyramid L1 recon
                                           independent of the u8 disk layout */
    float   **Xs_t;                     /* per-thread, [n_threads] of [n_points][half] */
    float   **centers_t;                /* per-thread, [n_threads] of [K][half] */
    int32_t **labels_t;                 /* per-thread, [n_threads] of [n_points] */
    int      *errors_t;                 /* per-thread error flags */
    int       is_l2;                    /* 1 ⇒ this is the pyramid L2 residual
                                           fit (apply_row_scale==0): use the
                                           L2-tuned k-means schedule (more
                                           iters + restarts) since K is tiny
                                           (≤16) and a better local minimum on
                                           the residual recovers PPL with no
                                           on-disk format change. */
} pqv2_slot_ctx;

static void pqv2_encode_slot_worker(void *arg, int thread_id,
                                     int start, int end)
{
    pqv2_slot_ctx *c = (pqv2_slot_ctx *)arg;
    if (c->errors_t[thread_id]) return;
    float   *Xs      = c->Xs_t[thread_id];
    float   *centers = c->centers_t[thread_id];
    int32_t *labels  = c->labels_t[thread_id];
    const int M = c->M, N = c->N, G = c->G, K = c->K, half = c->half;
    const int n_chunks = c->n_chunks, n_sub = c->n_sub;
    const int n_points = c->n_points, fit_sample_cap = c->fit_sample_cap;
    const float *Wn = c->Wn;

    for (int s = start; s < end; s++) {
        /* Pack Xs[i] = sub-chunk s of (row m, chunk ch) flattened. */
        int idx = 0;
        for (int m = 0; m < M; m++) {
            for (int ch = 0; ch < n_chunks; ch++) {
                const float *base = Wn + (size_t)m * N + (size_t)ch * G + s * half;
                float *dst = Xs + (size_t)idx * half;
                for (int h = 0; h < half; h++) dst[h] = base[h];
                idx++;
            }
        }

        ib_kmeans_config kcfg = {0};
        kcfg.K = K;
        kcfg.D = half;
        /* max_iter — K=256 on 2D converges by ~5-10 iters in practice;
         * 10 keeps the safety margin without paying for ~15 micro-shuffle
         * iters past convergence. Override via IB_PQV2_KMEANS_ITERS. */
        int max_iter = 10;
        {
            const char *env = getenv("IB_PQV2_KMEANS_ITERS");
            if (env && *env) {
                int v = atoi(env);
                if (v > 0) max_iter = v;
            }
        }
        int n_init = 1;
        if (c->is_l2) {
            /* Pyramid L2 residual fit. K ≤ 16 here, so a *better* codebook
             * is cheap: several k-means++ restarts (best by inertia wins)
             * tighten the partition the 16 codewords impose on the residual.
             * The decisive lever is the empty-cluster reseed in pq_kmeans.c
             * (a dead centroid is ~6% of a 16-entry codebook); restarts then
             * pick the luckiest reseeded run. Measured on TinyLlama wikitext
             * (warmup 64 / score 256 / ctx 1024): baseline l2k16 PPL 6.604;
             * n_init=8 → 6.576; n_init=16 + reseed → 6.497 — beating the full
             * l2_K=64 pyramid (6.520) at the SAME 712 MB file. More Lloyd
             * iters alone gave nothing (k-means already converges < 10 iters),
             * so max_iter just needs convergence headroom. This changes only
             * WHICH 16 codewords are picked — the on-disk layout
             * (l2_idx_bits=4, header) is untouched, so the existing decoder
             * reads the file verbatim. Tunable via env for ablation. */
            max_iter = 25;
            n_init   = 16;
            {
                const char *e = getenv("IB_L2_KMEANS_ITERS");
                if (e && *e) { int v = atoi(e); if (v > 0) max_iter = v; }
            }
            {
                const char *e = getenv("IB_L2_KMEANS_NINIT");
                if (e && *e) { int v = atoi(e); if (v > 0) n_init = v; }
            }
        }
        kcfg.max_iter = max_iter;
        kcfg.tol = 1e-4f;
        kcfg.n_init = n_init;
        /* subsample — fit the codebook on a random subset; final
         * assignment below runs on the FULL n_points via ib_kmeans_assign,
         * so quality is preserved. 50k is plenty for K=256 in 2D
         * (~195 points per cluster). Override via IB_PQV2_KMEANS_SUBSAMPLE. */
        kcfg.subsample = (n_points > fit_sample_cap) ? fit_sample_cap : 0;
        kcfg.seed = c->seed + (uint32_t)s * 1009u;
        kcfg.pool = NULL;  /* inner k-means stays single-threaded; the
                              outer slot loop is what's parallelized */
        if (ib_kmeans_fit(Xs, n_points, &kcfg, centers, labels, NULL) != 0) {
            c->errors_t[thread_id] = 1;
            return;
        }

        /* Quantize codewords to int8 + fp16 scale. */
        for (int k = 0; k < K; k++) {
            const float *cw = centers + (size_t)k * half;
            float ax = 0.0f;
            for (int h = 0; h < half; h++) {
                float v = cw[h]; v = v < 0 ? -v : v;
                if (v > ax) ax = v;
            }
            if (ax < 1e-12f) ax = 1e-12f;
            float scale = ax / 127.0f;
            float inv = 1.0f / scale;
            uint16_t sh = enc_f2h(scale);
            c->cb_scale_fp16_out[(size_t)s * K + k] = sh;
            int8_t *qrow = c->cb_int8_out + ((size_t)s * K + k) * half;
            for (int h = 0; h < half; h++) {
                float qf = roundf(cw[h] * inv);
                if (qf > 127.0f)  qf = 127.0f;
                if (qf < -128.0f) qf = -128.0f;
                qrow[h] = (int8_t)qf;
            }
        }

        /* Relabel against the int8+scale rounded codewords for encode ↔
         * decode consistency. */
        float *recon = centers;  /* reuse buffer */
        for (int k = 0; k < K; k++) {
            uint16_t sh = c->cb_scale_fp16_out[(size_t)s * K + k];
            float scl = enc_h2f(sh);
            const int8_t *q = c->cb_int8_out + ((size_t)s * K + k) * half;
            float *r = recon + (size_t)k * half;
            for (int h = 0; h < half; h++) r[h] = (float)q[h] * scl;
        }
        if (ib_kmeans_assign(Xs, n_points, half, recon, K, labels, NULL) != 0) {
            c->errors_t[thread_id] = 1;
            return;
        }

        /* Scatter labels into the caller's output buffers.
         *
         * Stage 5g.2: the u8 disk index scatter is gated on
         * idx_layout_rowmajor. The chunk-major branch is bit-identical
         * to legacy; the row-major branch swaps the linearization so
         * the GPU upload can zero-copy mmap the file region directly
         * into a MTLBuffer (the GPU SIMD kernel already reads in
         * row-major). int32 rowmajor output (used by the pyramid L1
         * recon pass) is independent of layout — always written in
         * row-major because the consumer is the per-row residual
         * inner loop. */
        const int rm = c->idx_layout_rowmajor;
        for (int m = 0; m < M; m++) {
            for (int ch = 0; ch < n_chunks; ch++) {
                int32_t lbl = labels[(size_t)m * n_chunks + ch];
                if (c->indices_chunkmajor_out) {
                    size_t off = rm
                        ? ((size_t)m * n_chunks + ch) * n_sub + s
                        : ((size_t)ch * n_sub + s) * M + m;
                    c->indices_chunkmajor_out[off] = (uint8_t)lbl;
                }
                if (c->indices_rowmajor_out) {
                    c->indices_rowmajor_out[((size_t)m * n_chunks + ch) * n_sub + s] =
                        lbl;
                }
            }
        }
    }
}

/* ── flat encode (l1) ─────────────────────────────────────────────── */
/*
 * For each sub-chunk slot s ∈ [0, n_subchunks): collect all
 * M × n_chunks length-`half` sub-vectors from W at that slot; fit a
 * K-codeword codebook via Lloyd's k-means (k-means++ init); quantize
 * each codeword to int8 + fp16 scale; assign each row's sub-vector to
 * the nearest codeword. Final indices laid out in chunk-major
 * [n_chunks][n_subchunks][M] to match the loader/Metal-upload contract.
 *
 * Row-scale (M fp16 values): the max absolute value across each row.
 * W is normalized by row_scale before encoding so the per-slot
 * codebook fits row *shape* rather than magnitude.
 *
 * indices_out_l1 may be NULL when the caller wants L1 labels written
 * back into the standard chunk-major layout (the pyramid path needs an
 * intermediate row-major copy; the flat path writes chunk-major
 * directly).
 *
 * If indices_rowmajor_out is non-NULL it receives the [M][n_chunks][n_subchunks]
 * row-major label array (caller-allocated). Used by the pyramid path to
 * compute the L1 reconstruction without re-running argmin.
 */
static int pqv2_encode_flat_impl(const float *W_in, int M, int N,
                                  int G, int K, int half,
                                  int apply_row_scale,
                                  int8_t   *cb_int8_out,
                                  uint16_t *cb_scale_fp16_out,
                                  uint16_t *row_scale_fp16_out,
                                  uint8_t  *indices_chunkmajor_out,
                                  int       idx_layout_rowmajor,
                                  int32_t  *indices_rowmajor_out,
                                  uint32_t  seed)
{
    if (!W_in || !cb_int8_out || !cb_scale_fp16_out ||
        M <= 0 || N <= 0 || G <= 0 || K <= 0 ||
        half <= 0 || (N % G) != 0 || (G % half) != 0) {
        ib_set_error("pqv2_encode_flat: invalid args (M=%d N=%d G=%d K=%d half=%d)",
                     M, N, G, K, half);
        return -1;
    }
    if (apply_row_scale && !row_scale_fp16_out) {
        ib_set_error("pqv2_encode_flat: apply_row_scale requires row_scale_fp16_out");
        return -1;
    }
    if (K > 256) {
        /* Loader-compatible flat layout only consumes uint8 L1 indices.
         * K>256 would require the wider-index path which the production
         * Metal kernels do not implement yet. */
        ib_set_error("pqv2_encode_flat: K=%d > 256 not supported by current loader/kernels", K);
        return -1;
    }
    int n_chunks = N / G;
    int n_sub = G / half;

    /* Per-row scale. The L1 path divides W by row_max so the codebook
     * fits row *shape* rather than magnitude; the decoder multiplies
     * acc_l1 by row_scale on the way out (see pqv2_kernel.c::matvec).
     *
     * The L2 path (pyramid residual) MUST NOT divide by a row scale,
     * because the decoder adds acc_l2 directly without any row factor.
     * Per-codeword fp16 scale handles dynamic range across rows.
     * Passing apply_row_scale=0 leaves W untouched and fills the (unused)
     * row_scale_fp16_out with ones — matches scripts/poc/pqv2_encode.py.
     */
    const float *W_to_fit = W_in;
    float *Wn = NULL;
    if (apply_row_scale) {
        float *row_scale_f32 = (float *)malloc((size_t)M * sizeof(float));
        if (!row_scale_f32) { ib_set_error("oom (row_scale_f32)"); return -1; }
        for (int m = 0; m < M; m++) {
            const float *row = W_in + (size_t)m * N;
            float ax = 0.0f;
            for (int n = 0; n < N; n++) {
                float v = row[n]; v = v < 0 ? -v : v;
                if (v > ax) ax = v;
            }
            if (ax < IB_PQV2_FP16_MIN_NORMAL) ax = IB_PQV2_FP16_MIN_NORMAL;
            uint16_t hbits = enc_f2h(ax);
            row_scale_fp16_out[m] = hbits;
            row_scale_f32[m] = enc_h2f(hbits);  /* round-trip through fp16 */
        }

        Wn = (float *)malloc((size_t)M * N * sizeof(float));
        if (!Wn) { free(row_scale_f32); ib_set_error("oom (Wn)"); return -1; }
        for (int m = 0; m < M; m++) {
            float rs = row_scale_f32[m];
            float inv = 1.0f / rs;
            const float *src = W_in + (size_t)m * N;
            float *dst = Wn + (size_t)m * N;
            for (int n = 0; n < N; n++) dst[n] = src[n] * inv;
        }
        free(row_scale_f32);
        W_to_fit = Wn;
    } else if (row_scale_fp16_out) {
        /* Caller passed a buffer but didn't ask for normalization.
         * Fill with 1.0 fp16 so any reader sees a no-op multiplier. */
        uint16_t one_h = enc_f2h(1.0f);
        for (int m = 0; m < M; m++) row_scale_fp16_out[m] = one_h;
    }

    /* Per-thread per-slot scratch — the slot loop below is now parallel
     * over s, so Xs/centers/labels are owned by each worker thread.
     * Determinism: each slot's k-means uses seed = base + s*1009, which
     * means the slot output is independent of which thread runs it. */
    int n_points = M * n_chunks;

    /* Encoding is offline + embarrassingly parallel across the n_sub
     * sub-chunk slots, with no inter-thread bandwidth contention. Use
     * ALL logical CPUs (P + E on Apple Silicon) — E-cores contribute
     * net speedup here because work-stealing (chunk_size=1 in
     * ib_pool_run) lets faster P-cores pull more slots. The slower
     * E-cores just chip in what they can. */
    int n_threads = ib_total_logical_cpus();
    {
        const char *env = getenv("IB_PQV2_ENCODE_THREADS");
        if (env && *env) {
            int v = atoi(env);
            if (v > 0) n_threads = v;
        }
    }
    if (n_threads < 1) n_threads = 1;
    if (n_threads > n_sub) n_threads = n_sub;  /* no point oversubscribing */

    float   **Xs_t      = (float **)calloc((size_t)n_threads, sizeof(float *));
    float   **centers_t = (float **)calloc((size_t)n_threads, sizeof(float *));
    int32_t **labels_t  = (int32_t **)calloc((size_t)n_threads, sizeof(int32_t *));
    int      *errors_t  = (int *)calloc((size_t)n_threads, sizeof(int));
    if (!Xs_t || !centers_t || !labels_t || !errors_t) {
        if (Wn) free(Wn);
        free(Xs_t); free(centers_t); free(labels_t); free(errors_t);
        ib_set_error("oom (per-thread scratch ptrs)");
        return -1;
    }
    for (int t = 0; t < n_threads; t++) {
        Xs_t[t]      = (float *)malloc((size_t)n_points * half * sizeof(float));
        centers_t[t] = (float *)malloc((size_t)K * half * sizeof(float));
        labels_t[t]  = (int32_t *)malloc((size_t)n_points * sizeof(int32_t));
        if (!Xs_t[t] || !centers_t[t] || !labels_t[t]) {
            for (int u = 0; u <= t; u++) {
                free(Xs_t[u]); free(centers_t[u]); free(labels_t[u]);
            }
            if (Wn) free(Wn);
            free(Xs_t); free(centers_t); free(labels_t); free(errors_t);
            ib_set_error("oom (per-thread slot buffers, n_threads=%d)", n_threads);
            return -1;
        }
    }

    /* Fitting subsample cap. K=256 clusters in 2D space; 50k random
     * points = ~195 per cluster, well above the rule-of-thumb 10×K floor.
     * The slot worker still does FINAL label assignment on the full
     * n_points via ib_kmeans_assign (so quality is preserved at the
     * cost of one extra distance scan), but FITTING runs on the
     * subsample for speed. Override via IB_PQV2_KMEANS_SUBSAMPLE. */
    int fit_sample_cap = 50000;
    {
        const char *env = getenv("IB_PQV2_KMEANS_SUBSAMPLE");
        if (env && *env) {
            int v = atoi(env);
            if (v > 0) fit_sample_cap = v;
        }
    }

    /* Closure for the slot worker. */
    pqv2_slot_ctx ctx;
    ctx.M = M; ctx.N = N; ctx.G = G; ctx.K = K; ctx.half = half;
    ctx.n_chunks = n_chunks; ctx.n_sub = n_sub; ctx.n_points = n_points;
    ctx.fit_sample_cap = fit_sample_cap;
    ctx.seed = seed;
    ctx.Wn = W_to_fit;
    ctx.cb_int8_out = cb_int8_out;
    ctx.cb_scale_fp16_out = cb_scale_fp16_out;
    ctx.indices_chunkmajor_out = indices_chunkmajor_out;
    ctx.idx_layout_rowmajor    = idx_layout_rowmajor;
    ctx.indices_rowmajor_out   = indices_rowmajor_out;
    ctx.Xs_t = Xs_t;
    ctx.centers_t = centers_t;
    ctx.labels_t  = labels_t;
    ctx.errors_t  = errors_t;
    /* The pyramid L2 residual pass is the only caller that disables the
     * per-row scale (the decoder adds acc_l2 with no row factor). Use that
     * as the L2 signal so we can apply the heavier k-means schedule to the
     * 16-entry residual codebook only — L1 (K=256) stays fast/unchanged. */
    ctx.is_l2 = (apply_row_scale == 0) ? 1 : 0;

    /* Run the slot loop in parallel. chunk_size = 1 → round-robin so the
     * larger slots get spread across workers rather than piling up on
     * one. Each call into the worker may handle [start, end) contiguous
     * slots; per-slot work is fully independent (deterministic seed). */
    ib_thread_pool *pool = (n_threads > 1) ? ib_pool_create(n_threads) : NULL;
    ib_pool_run(pool, pqv2_encode_slot_worker, &ctx, n_sub, 1);
    if (pool) ib_pool_destroy(pool);

    /* Reduce per-thread error flags. */
    int slot_err = 0;
    for (int t = 0; t < n_threads; t++) {
        if (errors_t[t]) { slot_err = 1; break; }
    }

    /* Free per-thread scratch. */
    for (int t = 0; t < n_threads; t++) {
        free(Xs_t[t]); free(centers_t[t]); free(labels_t[t]);
    }
    free(Xs_t); free(centers_t); free(labels_t); free(errors_t);

    if (slot_err) {
        if (Wn) free(Wn);
        ib_set_error("ib_kmeans_fit/assign failed in slot worker");
        return -1;
    }

    if (Wn) free(Wn);
    return 0;
}

int pqv2_encode_flat(const float *W, int M, int N,
                     int G, int K, int half,
                     int8_t   *cb_int8_out,
                     uint16_t *cb_scale_fp16_out,
                     uint16_t *row_scale_fp16_out,
                     uint8_t  *indices_out,
                     uint32_t  seed)
{
    /* Public legacy wrapper: chunk-major layout (idx_layout_rowmajor=0).
     * The row-major opt-in flows through the internal push_pqv2_tensor
     * path which calls pqv2_encode_flat_impl directly. */
    return pqv2_encode_flat_impl(W, M, N, G, K, half, /*apply_row_scale=*/1,
                                  cb_int8_out, cb_scale_fp16_out,
                                  row_scale_fp16_out, indices_out,
                                  /*idx_layout_rowmajor=*/0,
                                  NULL, seed);
}

/* ── pyramid encode (l1 + additive l2) ────────────────────────────── */

int pqv2_encode_pyramid(const float *W, int M, int N,
                        int G, int K_outer, int K_inner, int half,
                        int8_t   *cb_int8_l1_out,
                        uint16_t *cb_scale_fp16_l1_out,
                        int8_t   *cb_int8_l2_out,
                        uint16_t *cb_scale_fp16_l2_out,
                        uint16_t *row_scale_fp16_out,
                        uint8_t  *indices_l1_out,
                        uint8_t  *indices_l2_out,
                        uint32_t  seed)
{
    (void)K_outer;  /* reserved for future coarse-pre-clustering sweep */

    /* Pyramid L1 always at the production codebook size (K=256). The
     * caller's K_inner controls only the L2 codebook width. We hard-code
     * L1 to 256 because the loader/Metal/CPU K=256 kernels are the only
     * production matvec path; smaller L1 K would force a fallback to
     * pqv2_matvec_lut which isn't in the bench's hot path. */
    const int K_L1 = 256;
    if (!W || K_inner <= 0 || K_inner > 64 || (N % G) != 0 || (G % half) != 0) {
        ib_set_error("pqv2_encode_pyramid: bad args (K_inner must be in [1, 64]; "
                     "the production K=256 NEON kernel uses vqtbl4q_s8 (4×16=64) "
                     "for L2 — see pqv2_kernel.c:1010 and forward.c:399)");
        return -1;
    }

    int n_chunks = N / G;
    int n_sub = G / half;
    int K_L2 = K_inner;

    /* L1 pass: emit normal L1 indices (chunk-major) AND keep an
     * extra row-major copy so we can rebuild the recon without
     * re-running argmin per sub-vector. */
    int32_t *idx_rm = (int32_t *)malloc((size_t)M * n_chunks * n_sub * sizeof(int32_t));
    if (!idx_rm) { ib_set_error("oom (idx_rm)"); return -1; }

    int rc = pqv2_encode_flat_impl(W, M, N, G, K_L1, half, /*apply_row_scale=*/1,
                                    cb_int8_l1_out, cb_scale_fp16_l1_out,
                                    row_scale_fp16_out, indices_l1_out,
                                    /*idx_layout_rowmajor=*/0,
                                    idx_rm, seed);
    if (rc != 0) { free(idx_rm); return rc; }

    /* Reconstruct W_l1 (un-row-scaled, then row-scaled back) and
     * compute residual R = W - W_l1. */
    float *R = (float *)malloc((size_t)M * N * sizeof(float));
    if (!R) { free(idx_rm); ib_set_error("oom (R)"); return -1; }

    /* Pre-decode fp32 codebooks once. */
    float *cb_fp32 = (float *)malloc((size_t)n_sub * K_L1 * half * sizeof(float));
    if (!cb_fp32) { free(R); free(idx_rm); ib_set_error("oom (cb_fp32)"); return -1; }
    for (int s = 0; s < n_sub; s++) {
        for (int k = 0; k < K_L1; k++) {
            float scl = enc_h2f(cb_scale_fp16_l1_out[(size_t)s * K_L1 + k]);
            const int8_t *q = cb_int8_l1_out + ((size_t)s * K_L1 + k) * half;
            float *o = cb_fp32 + ((size_t)s * K_L1 + k) * half;
            for (int h = 0; h < half; h++) o[h] = (float)q[h] * scl;
        }
    }

    for (int m = 0; m < M; m++) {
        float rs = enc_h2f(row_scale_fp16_out[m]);
        const float *src = W + (size_t)m * N;
        float *dst = R + (size_t)m * N;
        for (int c = 0; c < n_chunks; c++) {
            for (int s = 0; s < n_sub; s++) {
                int32_t lbl = idx_rm[((size_t)m * n_chunks + c) * n_sub + s];
                const float *cw = cb_fp32 + ((size_t)s * K_L1 + (size_t)lbl) * half;
                float *out_slice = dst + (size_t)c * G + s * half;
                const float *in_slice = src + (size_t)c * G + s * half;
                for (int h = 0; h < half; h++) {
                    /* W_l1 = (codeword * row_scale). Residual = W - W_l1. */
                    out_slice[h] = in_slice[h] - cw[h] * rs;
                }
            }
        }
    }
    free(cb_fp32);
    free(idx_rm);

    /* L2 pass: fit k-means directly on the residual R (NO per-row
     * normalization). The decoder accumulates L2 contributions WITHOUT
     * any row-scale multiplier (see pqv2_kernel.c::matvec:
     *   y[m] = acc_l1 * row_scale[m] + acc_l2
     * — note acc_l2 has no row_scale factor). So the stored L2
     * codewords must equal the residual in the same magnitude basis as
     * the original W. Per-codeword fp16 scale absorbs the dynamic
     * range across all rows; row_scale_fp16 is unused at L2 (filled
     * with 1.0 fp16 by pqv2_encode_flat_impl when apply_row_scale=0).
     *
     * Earlier this path called pqv2_encode_flat_impl with
     * apply_row_scale=1 and then threw away the resulting row_scale_R;
     * that was a bug — it normalized R by row_max(R) before fitting,
     * so the stored codewords represented R/row_max(R) and the decoder
     * undershot the residual by a per-row factor of row_max(R). */
    uint16_t *row_scale_l2_throwaway =
        (uint16_t *)malloc((size_t)M * sizeof(uint16_t));
    if (!row_scale_l2_throwaway) { free(R); ib_set_error("oom (rs_l2)"); return -1; }

    rc = pqv2_encode_flat_impl(R, M, N, G, K_L2, half, /*apply_row_scale=*/0,
                                cb_int8_l2_out, cb_scale_fp16_l2_out,
                                row_scale_l2_throwaway, indices_l2_out,
                                /*idx_layout_rowmajor=*/0,
                                NULL, seed ^ 0xa5a5a5a5u);
    free(row_scale_l2_throwaway);
    free(R);
    return rc;
}

/* ─────────────────────────────────────────────────────────────────── */
/*  IBF v6 PQv2 writer                                                  */
/* ─────────────────────────────────────────────────────────────────── */

/* Replicates scripts/convert/pqv2_ibf_writer.py + reuses the convert.c
 * shared scaffolding (ib_tensor_source, ib_parse_config_json,
 * ib_quantize_int8, ib_copy_norm_fp16) for non-PQ tensors. */

#define IBF6_MAGIC "IBFV6PQ2"
#define IBF6_VERSION 1u
#define IBF6_ALIGN 64

static size_t align_up_z(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

/* ── manifest entry ── */

typedef struct {
    char *name;        /* owned */
    int kind;          /* IB_PQV2_KIND_* */
    int ndim;
    int32_t shape[4];
    /* Blob: built into a heap buffer, written at finalization. */
    void *blob;
    size_t blob_size;
    /* Allocated offset (filled in pass 2). */
    uint64_t blob_offset;
} ib6_entry;

typedef struct {
    ib6_entry *entries;
    int n;
    int cap;
} ib6_manifest;

static int ib6_push(ib6_manifest *mf, const char *name, int kind, int ndim,
                     const int32_t shape[4], void *blob, size_t blob_size)
{
    if (mf->n == mf->cap) {
        int nc = mf->cap ? mf->cap * 2 : 64;
        ib6_entry *ne = (ib6_entry *)realloc(mf->entries,
                                              (size_t)nc * sizeof(ib6_entry));
        if (!ne) return -1;
        mf->entries = ne; mf->cap = nc;
    }
    ib6_entry *e = &mf->entries[mf->n++];
    /* Portable strdup: MSVC names it _strdup and warns on the POSIX form. */
    {
        const char *src = name ? name : "";
        size_t nl = strlen(src);
        e->name = (char *)malloc(nl + 1);
        if (!e->name) { mf->n--; return -1; }
        memcpy(e->name, src, nl + 1);
    }
    e->kind = kind;
    e->ndim = ndim;
    for (int i = 0; i < 4; i++) e->shape[i] = shape[i];
    e->blob = blob;
    e->blob_size = blob_size;
    e->blob_offset = 0;
    return 0;
}

static void ib6_free(ib6_manifest *mf) {
    if (!mf) return;
    for (int i = 0; i < mf->n; i++) {
        free(mf->entries[i].name);
        free(mf->entries[i].blob);
    }
    free(mf->entries);
    memset(mf, 0, sizeof(*mf));
}

/* ── build the single-tensor "PQV2" blob layout consumed by
 *    pqv2_format.c::parse_pqv2_blob. ── */

/* Pack a row of M uint8 L2 indices (each in [0..63]) into ceil(M/4)*3
 * bytes using the LSB-first 6-bit layout from Stage 5h.1:
 *   byte0 = (i0 & 0x3F) | ((i1 & 0x03) << 6)
 *   byte1 = ((i1 >> 2) & 0x0F) | ((i2 & 0x0F) << 4)
 *   byte2 = ((i2 >> 4) & 0x03) | ((i3 & 0x3F) << 2)
 *
 * The 6-bit layout matches the random-access + inline-cost rules from
 * docs/v2/00_CORRECTION.md Stage 5h "file-size rule": the kernel reads
 * 3 bytes and unpacks 4 indices via two shifts each. Conceptually mirrors
 * the LSB-first bitstream packing in pq_decode.c:1078 but specialised
 * to a fixed 6-bit width along the M axis for SIMD-friendly access. */
static void pqv2_pack_l2_row_6bit(const uint8_t *src_m, uint32_t M, uint8_t *dst) {
    uint32_t m = 0;
    while (m + 4 <= M) {
        uint8_t i0 = src_m[m + 0] & 0x3F;
        uint8_t i1 = src_m[m + 1] & 0x3F;
        uint8_t i2 = src_m[m + 2] & 0x3F;
        uint8_t i3 = src_m[m + 3] & 0x3F;
        dst[0] = (uint8_t)(i0 | (i1 << 6));
        dst[1] = (uint8_t)((i1 >> 2) | (i2 << 4));
        dst[2] = (uint8_t)((i2 >> 4) | (i3 << 2));
        dst += 3;
        m += 4;
    }
    /* Tail: pad with zeros for slots m >= M. Decoded zeros are a valid
     * codeword index but those rows don't exist, so they're never read. */
    if (m < M) {
        uint8_t i0 = src_m[m + 0] & 0x3F;
        uint8_t i1 = (m + 1 < M) ? (src_m[m + 1] & 0x3F) : 0;
        uint8_t i2 = (m + 2 < M) ? (src_m[m + 2] & 0x3F) : 0;
        uint8_t i3 = 0;
        dst[0] = (uint8_t)(i0 | (i1 << 6));
        dst[1] = (uint8_t)((i1 >> 2) | (i2 << 4));
        dst[2] = (uint8_t)((i2 >> 4) | (i3 << 2));
    }
}

/* Goal N36 — pack a row of M uint8 L2 indices (each in [0..15]) into
 * ceil(M/2) bytes using 4-bit nibble packing:
 *   byte0 = (i0 & 0x0F) | ((i1 & 0x0F) << 4)
 * Mirror of the 6-bit packer above; two indices per byte, low nibble
 * first. Used when l2_K <= 16 (the kernel's single-bank vqtbl1q_s8
 * gather supports K up to 16). */
static void pqv2_pack_l2_row_4bit(const uint8_t *src_m, uint32_t M, uint8_t *dst) {
    uint32_t m = 0;
    while (m + 2 <= M) {
        uint8_t i0 = src_m[m + 0] & 0x0F;
        uint8_t i1 = src_m[m + 1] & 0x0F;
        dst[0] = (uint8_t)(i0 | (i1 << 4));
        dst += 1;
        m += 2;
    }
    if (m < M) {
        uint8_t i0 = src_m[m + 0] & 0x0F;
        dst[0] = (uint8_t)i0;  /* high nibble pads with zero */
    }
}

/* Goal N36 — L2 codebook size override. Default 64 preserves v0.4.x
 * behavior; set IB_L2_K=16 (or any 1..64 value, clamped) to shrink the
 * L2 codebook (and thus the per-row packed-index byte count). The
 * encoder uses 4-bit packing when the resolved value is ≤ 16, else
 * falls back to 6-bit (≤64). */
static int pqv2_resolve_l2_k(int pyramid) {
    if (!pyramid) return 0;
    const char *e = getenv("IB_L2_K");
    if (e && e[0]) {
        long v = strtol(e, NULL, 10);
        if (v >= 1 && v <= 64) return (int)v;
    }
    return 64;
}

static void *build_pqv2_blob(int M, int N, int G, int K, int n_sub, int half,
                              int l2_kind, int l2_K,
                              const uint16_t *row_scale_fp16,
                              const int8_t   *cb_q,
                              const uint16_t *cb_scale_fp16,
                              const uint8_t  *indices_chunkmajor,
                              const int8_t   *l2_cb_q,        /* nullable */
                              const uint16_t *l2_cb_scale_fp16,/* nullable */
                              const uint8_t  *l2_indices_chunkmajor, /* nullable */
                              int residency_hint,             /* Stage 5c */
                              int scale_precision,            /* Stage 5k */
                              int codebook_dedup,             /* Stage 5j */
                              int l1_idx_layout,              /* Stage 5g.2 */
                              size_t *out_size)
{
    size_t n_chunks = (size_t)N / G;
    /* Stage 5k scale layout (H2 sp2 redesign + I2 cb_scale rollback).
     *   sp = 0 : row_scale fp16[M]      (legacy, 2*M bytes)
     *            cb_scale  fp16[ns*K]   (legacy, 2*ns*K bytes)
     *   sp = 2 : row_scale fp8[M]       (E4M3, M bytes — saves M bytes
     *                                    per tensor; ~10-decade dynamic
     *                                    range, 2-6% per-row rel error.
     *                                    Probed safe on Llama-3 / TL
     *                                    row_scale distributions.)
     *            cb_scale  fp16[ns*K]   (legacy — Goal I2 fix). Previous
     *                                    sp=2 packed cb_scale as fp8 E4M3
     *                                    too, but cb_scale's distribution
     *                                    clusters around 1e-3..5e-3 with
     *                                    a 2-3 decade range; ~30% of
     *                                    codewords fell into E4M3's
     *                                    subnormal band where 5.9e-4
     *                                    flushes to 0. Probe RMSE_rel
     *                                    on TL gate_proj cb_scale was
     *                                    38.8% (with zero-flushes) vs
     *                                    2.6% for row_scale — confirming
     *                                    cb_scale, not row_scale, drove
     *                                    the sp2 PPL regression (PPL
     *                                    50.4 → 48.9 after row-scale-only
     *                                    fix; cb_scale fp16 expected to
     *                                    restore PPL to ~6.3 flat
     *                                    baseline). */
    int sp_fp8_rs   = (scale_precision >= 2);
    int sp_fp8_cbs  = 0;  /* Goal I2: cb_scale always fp16 — see note above. */
    size_t row_bytes;
    if (sp_fp8_rs) {
        row_bytes = (size_t)M;        /* fp8 E4M3[M] */
    } else {
        row_bytes = (size_t)M * 2;     /* legacy fp16[M] */
    }
    /* Stage 5j codebook pool. v1 scaffolding: pool_size = n_subchunks,
     * identity pool_id mapping. Same codebook bytes as legacy + an
     * n_subchunks-byte pool_id array. */
    int pool_on = (codebook_dedup != 0);
    uint32_t cb_pool_size_w  = pool_on ? (uint32_t)n_sub : 0u;
    uint32_t l2_cb_pool_size_w = (pool_on && l2_kind == 2) ? (uint32_t)n_sub : 0u;
    size_t cb_pool_rows = pool_on ? (size_t)n_sub : (size_t)n_sub;  /* same in v1 */
    size_t cb_q_bytes   = cb_pool_rows * (size_t)K * (size_t)half;
    size_t cb_s_bytes_fp16 = cb_pool_rows * (size_t)K * 2u;
    size_t cb_s_bytes   = sp_fp8_cbs ? cb_pool_rows * (size_t)K
                                       : cb_s_bytes_fp16;
    size_t cb_pool_id_bytes = pool_on ? (size_t)n_sub : 0u;
    size_t idx_bytes    = (size_t)M * n_chunks * n_sub;
    /* Decide whether to bit-pack the L2 index stream. The kernel hot path
     * is gated on l2_K ≤ 64 elsewhere; 6-bit-packing requires the same.
     * Goal N36: when l2_K ≤ 16, switch to 4-bit nibble packing (2 indices
     * per byte) for a further 33% L2-index savings. */
    int pack_l2_4bit = (l2_kind == 2 && l2_K > 0 && l2_K <= 16);
    int pack_l2_6bit = (l2_kind == 2 && l2_K > 0 && l2_K <= 64 && !pack_l2_4bit);
    int pack_l2 = pack_l2_4bit || pack_l2_6bit;
    uint32_t l2_idx_bits = pack_l2_4bit ? 4u : (pack_l2_6bit ? 6u : 8u);
    size_t l2_packed_bytes_per_row = pack_l2_4bit
        ? (((size_t)M + 1u) / 2u)
        : (((size_t)M + 3u) / 4u * 3u);
    size_t l2_idx_disk = pack_l2
        ? n_chunks * (size_t)n_sub * l2_packed_bytes_per_row
        : idx_bytes;
    /* Header layout (append-only growth):
     *   hdr[0..7]  M, N, G, K, n_sub, half, l2_kind, l2_K (legacy 8 u32)
     *   hdr[8]     l2_idx_bits          (Stage 5h.1)
     *   hdr[9]     residency_hint       (Stage 5c)
     *   hdr[10]    scale_precision      (Stage 5k)
     *   hdr[11]    cb_pool_size         (Stage 5j; 0 = no pool)
     *   hdr[12]    l2_cb_pool_size      (Stage 5j; 0 = no pool)
     *   hdr[13]    l1_idx_layout        (Stage 5g.2; 0 = chunk-major,
     *                                                 1 = row-major
     *                                                     for Metal
     *                                                     zero-copy)
     *
     * Picked-shortest-header policy: emit the SMALLEST header that
     * losslessly carries the active fields. Default config (sp == 0,
     * cd == 0, layout == 0) emits the 10-u32 header so old loaders
     * (Stage 5c readers) still consume the file. Files only grow when
     * the user opts into 5k / 5j / 5g.2 knobs. */
    int need_pool   = (cb_pool_size_w != 0u) || (l2_cb_pool_size_w != 0u);
    int need_sp     = (sp_fp8_rs || sp_fp8_cbs);
    int need_layout = (l1_idx_layout != 0);
    int header_u32s;
    size_t header_bytes;
    if (need_layout) {
        header_u32s = 14;
        header_bytes = 4 + 56;
    } else if (need_pool) {
        header_u32s = 13;
        header_bytes = 4 + 52;
    } else if (need_sp) {
        header_u32s = 11;
        header_bytes = 4 + 44;
    } else {
        header_u32s = 10;
        header_bytes = 4 + 40;
    }
    size_t total = header_bytes + row_bytes + cb_q_bytes + cb_s_bytes
                    + cb_pool_id_bytes + idx_bytes;
    size_t l2q_bytes = 0, l2s_bytes = 0, l2s_bytes_disk = 0;
    size_t l2_cb_pool_id_bytes = 0;
    if (l2_kind == 2) {
        size_t l2_rows = pool_on ? (size_t)n_sub : (size_t)n_sub;  /* same in v1 */
        l2q_bytes = l2_rows * (size_t)l2_K * (size_t)half;
        l2s_bytes_disk = sp_fp8_cbs ? (l2_rows * (size_t)l2_K)
                                       : (l2_rows * (size_t)l2_K * 2u);
        l2s_bytes = l2s_bytes_disk;
        l2_cb_pool_id_bytes = pool_on ? (size_t)n_sub : 0u;
        total += l2q_bytes + l2s_bytes + l2_cb_pool_id_bytes + l2_idx_disk;
    }
    uint8_t *buf = (uint8_t *)calloc(1, total);
    if (!buf) return NULL;
    size_t cur = 0;
    memcpy(buf + cur, "PQV2", 4); cur += 4;
    uint32_t rhint = (uint32_t)residency_hint;
    if (rhint > 2) rhint = 0;  /* clamp unknown values to AUTO */
    uint32_t sp = (uint32_t)scale_precision;
    /* Only modes 0 and 2 are implemented in v1; clamp anything else
     * down to 0 so a future writer that sets mode 1/3 won't trip the
     * loader. The loader also rejects unknown values. */
    if (sp != 0u && sp != 2u) sp = 0u;
    uint32_t layout_w = (l1_idx_layout == 1) ? 1u : 0u;
    uint32_t hdr[14] = {
        (uint32_t)M, (uint32_t)N, (uint32_t)G, (uint32_t)K,
        (uint32_t)n_sub, (uint32_t)half,
        (uint32_t)l2_kind, (uint32_t)l2_K,
        l2_idx_bits,
        rhint,
        sp,
        cb_pool_size_w,
        l2_cb_pool_size_w,
        layout_w,
    };
    /* Write only the active prefix of the header. Older loaders that
     * don't know the Stage 5k / 5j / 5g.2 fields still consume default-
     * off files (sp == 0 && cb_pool_size == 0 && layout == 0) because
     * we drop those slots. */
    memcpy(buf + cur, hdr, (size_t)header_u32s * 4u); cur += (size_t)header_u32s * 4u;

    /* row_scale: legacy fp16[M] OR fp8 E4M3[M] (Stage 5k, H2 sp2). */
    if (sp_fp8_rs) {
        enc_pack_row_scale_e4m3(row_scale_fp16, M, buf + cur);
        cur += (size_t)M;
    } else {
        memcpy(buf + cur, row_scale_fp16, row_bytes); cur += row_bytes;
    }
    /* Codebook: in v1 the pool layout writes the SAME bytes as legacy
     * (pool_size == n_subchunks; identity pool_id mapping appended
     * after cb_scale). */
    memcpy(buf + cur, cb_q, cb_q_bytes); cur += cb_q_bytes;
    if (sp_fp8_cbs) {
        size_t cb_n = cb_pool_rows * (size_t)K;
        enc_pack_cb_scale_e4m3(cb_scale_fp16, cb_n, buf + cur);
        cur += cb_n;
    } else {
        memcpy(buf + cur, cb_scale_fp16, cb_s_bytes); cur += cb_s_bytes;
    }
    /* Stage 5j pool_id[n_sub] — only present when cb_pool_size > 0.
     * v1 identity mapping: pool_id[s] = s, so cb[s] = pool[s] = cb[s]
     * (round-trips to legacy semantics). */
    if (cb_pool_id_bytes) {
        for (size_t s = 0; s < (size_t)n_sub; s++) {
            buf[cur + s] = (uint8_t)s;
        }
        cur += cb_pool_id_bytes;
    }
    memcpy(buf + cur, indices_chunkmajor, idx_bytes); cur += idx_bytes;
    if (l2_kind == 2) {
        memcpy(buf + cur, l2_cb_q, l2q_bytes); cur += l2q_bytes;
        if (sp_fp8_cbs) {
            size_t l2_cb_n = (size_t)n_sub * (size_t)l2_K;  /* pool_size == n_sub in v1 */
            enc_pack_cb_scale_e4m3(l2_cb_scale_fp16, l2_cb_n, buf + cur);
            cur += l2_cb_n;
        } else {
            memcpy(buf + cur, l2_cb_scale_fp16, l2s_bytes); cur += l2s_bytes;
        }
        if (l2_cb_pool_id_bytes) {
            for (size_t s = 0; s < (size_t)n_sub; s++) {
                buf[cur + s] = (uint8_t)s;
            }
            cur += l2_cb_pool_id_bytes;
        }
        if (pack_l2) {
            /* Walk chunks × subchunks and pack the M-axis row in place. */
            for (size_t c = 0; c < n_chunks; c++) {
                for (size_t s = 0; s < (size_t)n_sub; s++) {
                    const uint8_t *src_row =
                        l2_indices_chunkmajor
                        + (c * (size_t)n_sub + s) * (size_t)M;
                    uint8_t *dst_row = buf + cur
                        + (c * (size_t)n_sub + s) * l2_packed_bytes_per_row;
                    if (pack_l2_4bit)
                        pqv2_pack_l2_row_4bit(src_row, (uint32_t)M, dst_row);
                    else
                        pqv2_pack_l2_row_6bit(src_row, (uint32_t)M, dst_row);
                }
            }
            cur += l2_idx_disk;
        } else {
            memcpy(buf + cur, l2_indices_chunkmajor, idx_bytes);
            cur += idx_bytes;
        }
    }
    *out_size = total;
    return buf;
}

/* ── per-tensor convenience wrappers ── */

/* Encode an fp32 [M][N] matrix as PQv2, push a manifest entry.
 *
 * `residency_hint`: Stage 5c — written into the on-disk per-tensor blob
 * header. 0 = AUTO (loader decides), 1 = RAM, 2 = DRIVE. The encoder
 * heuristic + per-class overrides resolve this in pqv2_convert before
 * each call.
 *
 * `idx_layout_rowmajor`: Stage 5g.2 — when 1, write L1 indices in
 *   [M][n_chunks][n_subchunks] on disk and stamp l1_idx_layout=1 in the
 *   blob header so the Metal upload skips its [nc,ns,M]→[M,total]
 *   transpose and zero-copies the index region into a MTLBuffer. Default
 *   0 stays bit-identical to v0.4.x — chunk-major on disk, transposed at
 *   upload time. Caller resolves this via IB_PQV2_L1_ROWMAJOR=1 once
 *   per pqv2_convert and passes it down to every push_pqv2_tensor. */
static int push_pqv2_tensor(ib6_manifest *mf, const char *name,
                             const float *W, int M, int N,
                             int G, int K, int half,
                             int pyramid, int residency_hint,
                             int scale_precision, int codebook_dedup,
                             int idx_layout_rowmajor,
                             uint32_t seed)
{
    int n_chunks = N / G;
    int n_sub = G / half;
    int l2_kind = pyramid ? 2 : 0;
    /* L2 codebook size: the production K=256 NEON inner kernel uses
     * vqtbl4q_s8 for the L2 lookup table (4×16 = 64 lanes), so the L2
     * path is hard-coded for l2_K ≤ 64. Larger l2_K would silently get
     * skipped by pqv2_threaded_matvec_k256 (forward.c) and the
     * single-thread variant (pqv2_kernel.c::pqv2_matvec_tbl_int8_k256
     * line ~1010) — both guard on `l2_K <= 64`. Default 64; IB_L2_K=16
     * (Goal N36) switches encoder + kernel to 4-bit packing. */
    int l2_K = pqv2_resolve_l2_k(pyramid);

    size_t row_count    = (size_t)M;
    size_t cb_q_count   = (size_t)n_sub * K * half;
    size_t cb_s_count   = (size_t)n_sub * K;
    size_t idx_count    = (size_t)M * n_chunks * n_sub;

    size_t cb_q_l2_count = (size_t)n_sub * l2_K * half;
    size_t cb_s_l2_count = (size_t)n_sub * l2_K;

    int8_t   *cb_q     = (int8_t   *)calloc(cb_q_count, 1);
    uint16_t *cb_s     = (uint16_t *)calloc(cb_s_count, 2);
    uint16_t *row_s    = (uint16_t *)calloc(row_count, 2);
    uint8_t  *idx_l1   = (uint8_t  *)calloc(idx_count, 1);
    int8_t   *cb_q_l2  = NULL;
    uint16_t *cb_s_l2  = NULL;
    uint8_t  *idx_l2   = NULL;
    if (pyramid) {
        cb_q_l2 = (int8_t   *)calloc(cb_q_l2_count, 1);
        cb_s_l2 = (uint16_t *)calloc(cb_s_l2_count, 2);
        idx_l2  = (uint8_t  *)calloc(idx_count, 1);
    }
    if (!cb_q || !cb_s || !row_s || !idx_l1 ||
        (pyramid && (!cb_q_l2 || !cb_s_l2 || !idx_l2))) {
        free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        ib_set_error("push_pqv2_tensor: oom (%s)", name);
        return -1;
    }

    int rc;
    if (pyramid) {
        /* Pyramid path: L1 disk layout flows through the public
         * pyramid encoder (which writes chunk-major) when the legacy
         * default is in effect. Row-major requires bypassing the
         * public wrapper and calling pqv2_encode_flat_impl with the
         * layout flag (the pyramid L2 pass is always chunk-major on
         * disk — its kernel reads use slot-major packed layout that's
         * already zero-copy via the 5h.1 packed format). */
        if (idx_layout_rowmajor) {
            int32_t *idx_rm = (int32_t *)malloc((size_t)M * n_chunks * n_sub * sizeof(int32_t));
            if (!idx_rm) {
                free(cb_q); free(cb_s); free(row_s); free(idx_l1);
                free(cb_q_l2); free(cb_s_l2); free(idx_l2);
                ib_set_error("push_pqv2_tensor: oom (idx_rm pyramid)");
                return -1;
            }
            /* L1: row-major u8 on disk + int32 row-major for L2 recon. */
            rc = pqv2_encode_flat_impl(W, M, N, G, K, half, /*apply_row_scale=*/1,
                                        cb_q, cb_s, row_s, idx_l1,
                                        /*idx_layout_rowmajor=*/1,
                                        idx_rm, seed);
            if (rc == 0) {
                /* Reconstruct residual and L2-encode just like
                 * pqv2_encode_pyramid does, but using idx_rm for the
                 * L1 labels (already populated above). */
                float *R = (float *)malloc((size_t)M * N * sizeof(float));
                float *cb_fp32 = R ? (float *)malloc((size_t)n_sub * K * half * sizeof(float)) : NULL;
                if (!R || !cb_fp32) {
                    free(R); free(cb_fp32); free(idx_rm);
                    free(cb_q); free(cb_s); free(row_s); free(idx_l1);
                    free(cb_q_l2); free(cb_s_l2); free(idx_l2);
                    ib_set_error("push_pqv2_tensor: oom (pyramid R/cb_fp32)");
                    return -1;
                }
                for (int s = 0; s < n_sub; s++) {
                    for (int k = 0; k < K; k++) {
                        float scl = enc_h2f(cb_s[(size_t)s * K + k]);
                        const int8_t *q = cb_q + ((size_t)s * K + k) * half;
                        float *o = cb_fp32 + ((size_t)s * K + k) * half;
                        for (int h = 0; h < half; h++) o[h] = (float)q[h] * scl;
                    }
                }
                for (int m = 0; m < M; m++) {
                    float rs = enc_h2f(row_s[m]);
                    const float *src = W + (size_t)m * N;
                    float *dst = R + (size_t)m * N;
                    for (int c = 0; c < n_chunks; c++) {
                        for (int s = 0; s < n_sub; s++) {
                            int32_t lbl = idx_rm[((size_t)m * n_chunks + c) * n_sub + s];
                            const float *cw = cb_fp32 + ((size_t)s * K + (size_t)lbl) * half;
                            float *out_slice = dst + (size_t)c * G + s * half;
                            const float *in_slice = src + (size_t)c * G + s * half;
                            for (int h = 0; h < half; h++) {
                                out_slice[h] = in_slice[h] - cw[h] * rs;
                            }
                        }
                    }
                }
                free(cb_fp32);
                free(idx_rm);
                /* L2: residual encode, chunk-major on disk (the packed
                 * 6-bit L2 reader keeps slot-major, doesn't care about
                 * L1 layout). */
                uint16_t *rs_l2_throw = (uint16_t *)malloc((size_t)M * sizeof(uint16_t));
                if (!rs_l2_throw) {
                    free(R);
                    free(cb_q); free(cb_s); free(row_s); free(idx_l1);
                    free(cb_q_l2); free(cb_s_l2); free(idx_l2);
                    ib_set_error("push_pqv2_tensor: oom (rs_l2)");
                    return -1;
                }
                rc = pqv2_encode_flat_impl(R, M, N, G, l2_K, half, /*apply_row_scale=*/0,
                                            cb_q_l2, cb_s_l2, rs_l2_throw, idx_l2,
                                            /*idx_layout_rowmajor=*/0,
                                            NULL, seed ^ 0xa5a5a5a5u);
                free(rs_l2_throw);
                free(R);
            } else {
                free(idx_rm);
            }
        } else {
            rc = pqv2_encode_pyramid(W, M, N, G, 0, l2_K, half,
                                      cb_q, cb_s, cb_q_l2, cb_s_l2,
                                      row_s, idx_l1, idx_l2, seed);
        }
    } else {
        rc = pqv2_encode_flat_impl(W, M, N, G, K, half, /*apply_row_scale=*/1,
                                    cb_q, cb_s, row_s, idx_l1,
                                    idx_layout_rowmajor,
                                    NULL, seed);
    }
    if (rc != 0) {
        free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        return rc;
    }

    size_t blob_size = 0;
    void *blob = build_pqv2_blob(M, N, G, K, n_sub, half,
                                  l2_kind, l2_K,
                                  row_s, cb_q, cb_s, idx_l1,
                                  cb_q_l2, cb_s_l2, idx_l2,
                                  residency_hint,
                                  scale_precision, codebook_dedup,
                                  idx_layout_rowmajor,
                                  &blob_size);
    free(cb_q); free(cb_s); free(row_s); free(idx_l1);
    free(cb_q_l2); free(cb_s_l2); free(idx_l2);
    if (!blob) { ib_set_error("build_pqv2_blob oom (%s)", name); return -1; }

    int32_t shape[4] = { M, N, 1, 1 };
    if (ib6_push(mf, name, IB_PQV2_KIND_PQV2, 2, shape, blob, blob_size) != 0) {
        free(blob); ib_set_error("manifest oom (%s)", name); return -1;
    }
    return 0;
}

/* ── MoME (Stage 3a, docs/v2/00_CORRECTION.md) ────────────────────
 *
 * Split an FFN weight matrix into K equal sub-matrices along the
 * rows-axis (the "trivial row-split" v1 calibration baseline), encode
 * each as its own PQv2 tensor, and emit a zero-init router weight.
 *
 * Two split orientations:
 *
 *   row-split for gate/up: source W is [M_total = inter, N = hidden].
 *                          Each expert e gets contiguous row range
 *                          [e * M_per, (e+1) * M_per). Sub-tensor
 *                          shape [M_per, N]; no re-permutation needed
 *                          — rows are already contiguous in memory.
 *
 *   col-split for down:    source W is [M = hidden, N_total = inter].
 *                          Each expert e gets contiguous col range
 *                          [e * N_per, (e+1) * N_per). Sub-tensor
 *                          shape [M, N_per]; rows must be re-packed
 *                          because the source stride is N_total cols
 *                          per row, not N_per.
 *
 * The "row-split" name in the design doc applies to the
 * intermediate-size axis (M_total for gate/up, N_total for down) —
 * see docs/v2/00_CORRECTION.md §3a. */

/* Shared-codebook MoME encoder (Stage 3a.shared, supersedes the
 * round 1-5 per-expert independent codebook approach).
 *
 * The earlier path called pqv2_encode_flat / pqv2_encode_pyramid
 * separately for each expert's slice. Each k-means converged to its
 * own local minimum → quantization errors for different experts were
 * statistically INDEPENDENT. Sum-of-experts had variance K·Var(err),
 * with per-row RMSE of ~17% (E1 round-5 probe). Compounded over 22
 * layers, that's α^22 ≈ 800× PPL degradation (observed 5074 vs flat
 * 6.26).
 *
 * The fix: encode the FULL un-split W ONCE to get one shared
 * (cb_q, cb_scale, row_scale, indices) tuple. Then per expert we
 * slice the appropriate (row_scale, L1 idx, L2 idx) and emit a blob
 * that REUSES the same (cb_q, cb_scale, l2_cb_q, l2_cb_scale) bytes.
 *
 * Why this works:
 *   - PQv2 codebooks are indexed by SUBCHUNK SLOT (n_sub codebooks
 *     total), NOT by chunk or row. Every chunk and every row of the
 *     full tensor decodes against the same per-subchunk codebooks.
 *   - For ROW-SPLIT (gate/up, [inter, hidden]): each expert covers
 *     rows [e*M_per, (e+1)*M_per). Indices are chunk-major
 *     [n_chunks][n_sub][M]; we copy the M-axis slice [e*M_per..]. The
 *     row_scale for this slice is the FULL-tensor row_scale[e*M_per..]
 *     (same as flat) — preserves bit-exact decoding of every row.
 *   - For COL-SPLIT (down, [hidden, inter]): each expert covers a
 *     contiguous CHUNK range. Indices [n_chunks][n_sub][M] slice as a
 *     contiguous chunk range. row_scale stays the FULL hidden-length
 *     buffer (every expert sees every row of W_down; the codebook +
 *     row_scale combo from the full encode is already correct).
 *
 * Result: sum-of-experts == flat reconstruction (modulo float-add
 * order). Errors are CORRELATED (shared codebook → shared rounding
 * grid) so they cancel, not compound. Expected MoME PPL ≈ flat PPL.
 *
 * File-size cost: K-1 extra cb_q + cb_scale per FFN tensor (a few
 * KB per layer at K=2). Index byte count is identical to a single
 * flat encode (slicing only redistributes; total bytes = M·n_c·n_s
 * either way). row_scale is duplicated for col-split (K * hidden
 * fp16) but that's <5 KB per layer.
 */

/* Build a per-expert PQv2 blob from sliced row_scale + sliced L1/L2
 * indices, reusing shared (cb_q, cb_scale, [l2_cb_q, l2_cb_scale])
 * bytes from the full-tensor encode. Caller owns all input pointers.
 * Returns 0 on success, -1 on error. */
static int mome_emit_expert_blob(ib6_manifest *mf,
                                  const char *base_name, int expert_idx,
                                  int M_exp, int N_exp,
                                  int G, int K_cb, int n_sub, int half,
                                  int pyramid, int l2_K,
                                  const uint16_t *row_scale_slice,
                                  const int8_t   *cb_q_shared,
                                  const uint16_t *cb_s_shared,
                                  const uint8_t  *idx_l1_slice,
                                  const int8_t   *l2_cb_q_shared,
                                  const uint16_t *l2_cb_s_shared,
                                  const uint8_t  *idx_l2_slice,
                                  int residency_hint,
                                  int scale_precision,
                                  int codebook_dedup)
{
    size_t blob_size = 0;
    void *blob = build_pqv2_blob(M_exp, N_exp, G, K_cb, n_sub, half,
                                  pyramid ? 2 : 0, l2_K,
                                  row_scale_slice, cb_q_shared, cb_s_shared,
                                  idx_l1_slice,
                                  l2_cb_q_shared, l2_cb_s_shared, idx_l2_slice,
                                  residency_hint, scale_precision, codebook_dedup,
                                  /*l1_idx_layout=*/0,
                                  &blob_size);
    if (!blob) { ib_set_error("mome_emit_expert_blob: oom (build_pqv2_blob)"); return -1; }
    char nm[128];
    snprintf(nm, sizeof(nm), "%s.expert%d", base_name, expert_idx);
    int32_t shape[4] = { M_exp, N_exp, 1, 1 };
    if (ib6_push(mf, nm, IB_PQV2_KIND_PQV2, 2, shape, blob, blob_size) != 0) {
        free(blob); ib_set_error("mome_emit_expert_blob: manifest push (%s)", nm); return -1;
    }
    return 0;
}

/* Push a [K, hidden] raw fp16 router weight.
 *
 * Orientation: rows = K (experts, = output dim), cols = hidden (= input
 * dim). This matches the standard tensor_matmul contract
 * `out[i] = sum_j w[i*N + j] * input[j]` so the runtime can call
 * ib_tensor_matmul_cpu(model, &router, logits, x, K, hidden, …) with no
 * transpose.
 *
 * Heuristic (I4, no-calibration): if a gate_proj source is supplied,
 * we fill router[e, i] with the column-norm of expert e's contiguous
 * row-slice of W_gate, i.e. ||W_gate[e*M_per .. (e+1)*M_per, i]||_2 .
 * This makes `router_logits[e] = sum_i ||W_gate_e[:,i]|| * |x[i]|`
 * (modulo signs) — experts whose row-slice has high column-norms
 * aligned with the current activation's high-magnitude dims score
 * higher under the softmax+top-N gating in forward_single_ex.
 *
 * If gate_src is NULL we fall back to the original zero-init router
 * (runtime treats it as "no calibrated router" and uses uniform all-
 * experts). Real calibrated weights still require an offline pass;
 * see mome.c TODO #2. */
static int push_zero_router(ib6_manifest *mf, const char *base_name,
                             int hidden, int K,
                             const void *gate_src, const char *gate_dtype,
                             int gate_rows)
{
    size_t count = (size_t)K * (size_t)hidden;
    uint16_t *router = (uint16_t *)calloc(count, sizeof(uint16_t));
    if (!router) { ib_set_error("oom: router"); return -1; }

    int used_heuristic = 0;
    /* I4 heuristic: column-norm of each expert's gate_proj row-slice.
     * gate_proj is laid out [intermediate, hidden] row-major; expert e
     * owns rows [e*M_per .. (e+1)*M_per) for all `hidden` columns.    */
    if (gate_src && gate_dtype && gate_rows > 0 && K > 0 &&
        (gate_rows % K) == 0) {
        int M_per = gate_rows / K;
        size_t total = (size_t)gate_rows * (size_t)hidden;
        float *W = (float *)malloc(total * sizeof(float));
        if (W && pqv2_read_matrix_fp32(W, gate_src, gate_dtype,
                                        gate_rows, hidden) == 0) {
            /* sumsq_e[i] = sum_{r in expert e rows} W[r, hidden + i]^2  */
            double *sumsq = (double *)calloc((size_t)K * (size_t)hidden,
                                              sizeof(double));
            if (sumsq) {
                for (int e = 0; e < K; e++) {
                    int r0 = e * M_per;
                    int r1 = r0 + M_per;
                    for (int r = r0; r < r1; r++) {
                        const float *row = W + (size_t)r * (size_t)hidden;
                        double *acc = sumsq + (size_t)e * (size_t)hidden;
                        for (int i = 0; i < hidden; i++) {
                            float v = row[i];
                            acc[i] += (double)v * (double)v;
                        }
                    }
                }
                /* Take sqrt and write as fp16 in [K, hidden] layout.   */
                for (int e = 0; e < K; e++) {
                    const double *acc = sumsq + (size_t)e * (size_t)hidden;
                    uint16_t *dst = router + (size_t)e * (size_t)hidden;
                    for (int i = 0; i < hidden; i++) {
                        float n = (float)sqrt(acc[i]);
                        dst[i] = enc_f2h(n);
                    }
                }
                free(sumsq);
                used_heuristic = 1;
            }
        }
        free(W);
    }
    if (!used_heuristic) {
        memset(router, 0, count * sizeof(uint16_t));
    }

    char nm[128];
    snprintf(nm, sizeof(nm), "%s.router", base_name);
    int32_t shape[4] = { K, hidden, 1, 1 };
    if (ib6_push(mf, nm, IB_PQV2_KIND_RAW_FP16, 2, shape,
                  router, count * sizeof(uint16_t)) != 0) {
        free(router); ib_set_error("manifest oom (%s)", nm); return -1;
    }
    return 0;
}

/* INT8 → raw fp16 quantize-and-push (PQV2 path uses INT8 for q/k/v +
 * embedding + lm_head per the bit-allocation philosophy carried over
 * from convert.c). For loader simplicity in IBFv6 we store these as
 * raw fp16 — same data, no scale-array required. The INT8 path can be
 * added later if file size is a concern; INT8 vs fp16 for q/k/v on a
 * 7B model is ~120 MB of attention weight delta, far smaller than the
 * FFN PQ savings. Keeping it fp16 here avoids inventing a new IBFv6
 * tensor kind. */
static int push_raw_fp16_from_source(ib6_manifest *mf, const char *name,
                                       const void *src, const char *dtype,
                                       int rows, int cols)
{
    size_t total = (size_t)rows * (cols > 1 ? cols : 1);
    uint16_t *fp16 = (uint16_t *)malloc(total * sizeof(uint16_t));
    if (!fp16) { ib_set_error("push_raw_fp16: oom (%s)", name); return -1; }
    ib_copy_norm_fp16(fp16, src, dtype, (int)total);
    int ndim = (cols > 1) ? 2 : 1;
    int32_t shape[4] = { rows, cols > 1 ? cols : 1, 1, 1 };
    if (ib6_push(mf, name, IB_PQV2_KIND_RAW_FP16, ndim, shape,
                  fp16, total * sizeof(uint16_t)) != 0) {
        free(fp16); ib_set_error("manifest oom (%s)", name); return -1;
    }
    return 0;
}

/* Permute Q/K rows for libinferbit's interleaved RoPE layout. Mirrors
 * convert.c::permute_qk_rows_alloc but operates on already-fp32 data
 * (we read the source as fp32 first via pqv2_read_matrix_fp32). */
static void permute_qk_rows_fp32_inplace(float *W, int rows, int cols,
                                          int head_dim, int n_heads)
{
    if (n_heads <= 0 || head_dim <= 0 || rows != n_heads * head_dim) return;
    int half = head_dim / 2;
    size_t row_bytes = (size_t)cols * sizeof(float);
    float *tmp = (float *)malloc((size_t)rows * row_bytes);
    if (!tmp) return;
    memcpy(tmp, W, (size_t)rows * row_bytes);
    for (int h = 0; h < n_heads; h++) {
        for (int i = 0; i < half; i++) {
            memcpy(W + (size_t)(h * head_dim + 2 * i)     * cols,
                   tmp + (size_t)(h * head_dim + i)         * cols, row_bytes);
            memcpy(W + (size_t)(h * head_dim + 2 * i + 1) * cols,
                   tmp + (size_t)(h * head_dim + half + i)  * cols, row_bytes);
        }
    }
    free(tmp);
}

/* ── MoME cosine row-clustering (QUALITY PROBE, env-gated) ────────────
 *
 * Hypothesis (training-free "real MoME"): grouping FFN neurons whose
 * gate_proj row vectors point in similar input directions (high cosine
 * similarity) into the same expert makes each expert respond to a
 * coherent slice of input space. A runtime gate-energy router can then
 * skip low-energy experts (top_n < K) with far less quality loss than
 * the contiguous row-split, where each "expert" is just an arbitrary
 * slice of one big sum and carries no standalone meaning.
 *
 * This module computes a row permutation from the gate matrix:
 *   1. L2-normalise each of `inter` gate rows ([inter, hidden]).
 *   2. Cosine k-means (= Euclidean k-means on the normalised rows; the
 *      two are monotonically equivalent for unit vectors) with K_experts
 *      clusters.
 *   3. Build a permutation that groups rows by cluster, balanced to
 *      exactly M_per = inter / K_experts rows per expert (k-means gives
 *      uneven clusters; we assign the largest clusters first and overflow
 *      the remainder into the next expert slot so every expert is full).
 *
 * The SAME permutation is applied to gate (rows), up (rows) and down
 * (cols, since down's columns correspond to gate's rows). Because all
 * three are permuted consistently, the full-FFN sum is invariant — no
 * runtime change is needed for correctness; only WHICH rows land in each
 * expert changes. Default ON; opt out with IB_MOME_COSINE_CLUSTER=0.
 *
 * The permutation is computed once when the GATE projection is encoded
 * and stashed in a process-static keyed by row count, then reused by the
 * UP and DOWN encoders for the same layer (they run immediately after
 * gate in convert order). */
static int mome_cosine_cluster_enabled(void) {
    /* Default ON: cosine row-clustering is a free quality win (TinyLlama
     * K=2 PPL 6.65 vs contiguous 6.90, identical size, +0.6% convert
     * time). Opt out with IB_MOME_COSINE_CLUSTER=0 for the legacy
     * contiguous row-split. */
    const char *e = getenv("IB_MOME_COSINE_CLUSTER");
    if (e && e[0]) return (e[0] != '0') ? 1 : 0;
    return 1;
}

/* Process-static permutation handoff: gate writes, up/down read. Sized
 * for the largest FFN seen; reallocated on growth. Single-threaded
 * conversion, so no locking needed. */
static int  *g_mome_perm      = NULL;   /* length g_mome_perm_n */
static int   g_mome_perm_n    = 0;
static int   g_mome_perm_K    = 0;

/* Compute a balanced cosine-cluster row permutation for W_gate
 * [inter, hidden]. perm_out[new_row] = old_row (i.e. apply by gathering
 * src row perm_out[i] into dst row i). Returns 0 on success, -1 on
 * allocation failure (caller falls back to identity / contiguous). */
static int mome_compute_cosine_perm(const float *W_gate, int inter,
                                     int hidden, int K_experts, uint32_t seed,
                                     int *perm_out)
{
    if (inter <= 0 || hidden <= 0 || K_experts <= 1) return -1;
    if ((inter % K_experts) != 0) return -1;
    const int M_per = inter / K_experts;

    /* L2-normalise rows into a scratch copy (cosine k-means = Euclidean
     * k-means on unit vectors). */
    float *Wn = (float *)malloc((size_t)inter * hidden * sizeof(float));
    if (!Wn) return -1;
    for (int r = 0; r < inter; r++) {
        const float *src = W_gate + (size_t)r * hidden;
        float *dst = Wn + (size_t)r * hidden;
        double ss = 0.0;
        for (int i = 0; i < hidden; i++) ss += (double)src[i] * (double)src[i];
        float inv = (ss > 1e-20) ? (float)(1.0 / sqrt(ss)) : 0.0f;
        for (int i = 0; i < hidden; i++) dst[i] = src[i] * inv;
    }

    float   *centers = (float *)malloc((size_t)K_experts * hidden * sizeof(float));
    int32_t *labels  = (int32_t *)malloc((size_t)inter * sizeof(int32_t));
    if (!centers || !labels) { free(Wn); free(centers); free(labels); return -1; }

    /* Self-contained high-D cosine (spherical) k-means. The shared
     * ib_kmeans_fit is hardcoded for tiny D (PQ uses D=2; it bails when
     * D>64), so it cannot cluster D=hidden gate rows — we run a minimal
     * Lloyd's loop here. Cosine similarity = dot product on the already
     * L2-normalised rows, so "nearest center" = "largest dot", and the
     * centroid update is mean-then-renormalise (spherical k-means). */
    {
        /* Init: pick K well-separated seed rows (farthest-point / k-means++
         * style on cosine distance). Seed 0 = deterministic first row. */
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
            mindist[r] = 1.0f - dot;     /* cosine distance */
        }
        for (int kk = 1; kk < K_experts; kk++) {
            /* Farthest-point: pick the row with the largest min-cos-dist. */
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

        /* Lloyd iterations. */
        double *csum = (double *)malloc((size_t)K_experts * hidden * sizeof(double));
        if (!csum) { free(Wn); free(centers); free(labels); return -1; }
        for (int it = 0; it < 25; it++) {
            int changed = 0;
            for (int r = 0; r < inter; r++) {
                const float *x = Wn + (size_t)r * hidden;
                int bestk = 0; float bestdot = -2.0f;
                for (int k = 0; k < K_experts; k++) {
                    const float *c = centers + (size_t)k * hidden;
                    float dot = 0.0f;
                    for (int i = 0; i < hidden; i++) dot += x[i] * c[i];
                    if (dot > bestdot) { bestdot = dot; bestk = k; }
                }
                if (labels[r] != bestk) { changed = 1; }
                labels[r] = bestk;
            }
            /* Recompute centers = renormalised mean of assigned rows. */
            memset(csum, 0, (size_t)K_experts * hidden * sizeof(double));
            int *cnt = (int *)calloc((size_t)K_experts, sizeof(int));
            if (!cnt) { free(csum); free(Wn); free(centers); free(labels); return -1; }
            for (int r = 0; r < inter; r++) {
                int k = labels[r];
                const float *x = Wn + (size_t)r * hidden;
                double *acc = csum + (size_t)k * hidden;
                for (int i = 0; i < hidden; i++) acc[i] += x[i];
                cnt[k]++;
            }
            for (int k = 0; k < K_experts; k++) {
                float *c = centers + (size_t)k * hidden;
                if (cnt[k] == 0) continue;   /* keep prior center if empty */
                const double *acc = csum + (size_t)k * hidden;
                double ss = 0.0;
                for (int i = 0; i < hidden; i++) ss += acc[i] * acc[i];
                float inv = (ss > 1e-20) ? (float)(1.0 / sqrt(ss)) : 0.0f;
                for (int i = 0; i < hidden; i++) c[i] = (float)(acc[i] * inv);
            }
            free(cnt);
            if (!changed && it > 0) break;
        }
        free(csum);
    }
    free(Wn);
    free(centers);

    /* Count cluster sizes. */
    int *csize = (int *)calloc((size_t)K_experts, sizeof(int));
    if (!csize) { free(labels); return -1; }
    for (int r = 0; r < inter; r++) {
        int c = labels[r];
        if (c < 0 || c >= K_experts) c = 0;
        csize[c]++;
    }
    if (getenv("IB_MOME_CLUSTER_DEBUG")) {
        fprintf(stderr, "[mome_cosine] cluster sizes:");
        for (int c = 0; c < K_experts; c++) fprintf(stderr, " %d", csize[c]);
        fprintf(stderr, "\n");
    }

    /* Order clusters largest-first so the biggest coherent groups claim
     * whole expert slots before smaller clusters fill the remainder. */
    int *corder = (int *)malloc((size_t)K_experts * sizeof(int));
    if (!corder) { free(labels); free(csize); return -1; }
    for (int c = 0; c < K_experts; c++) corder[c] = c;
    for (int a = 0; a < K_experts; a++) {
        int best = a;
        for (int b = a + 1; b < K_experts; b++)
            if (csize[corder[b]] > csize[corder[best]]) best = b;
        int t = corder[a]; corder[a] = corder[best]; corder[best] = t;
    }

    /* Emit rows cluster-by-cluster (largest first) into a flat order, then
     * chop into K equal M_per blocks. Because k-means clusters are uneven,
     * a block boundary may split a cluster — that's the intended overflow:
     * boundary rows of an over-full cluster spill into the next expert. */
    int pos = 0;
    for (int oc = 0; oc < K_experts; oc++) {
        int c = corder[oc];
        for (int r = 0; r < inter && pos < inter; r++) {
            if (labels[r] == c) perm_out[pos++] = r;
        }
    }
    /* Safety: append any rows with out-of-range labels (shouldn't happen). */
    if (pos < inter) {
        char *seen = (char *)calloc((size_t)inter, 1);
        if (seen) {
            for (int i = 0; i < pos; i++) seen[perm_out[i]] = 1;
            for (int r = 0; r < inter && pos < inter; r++)
                if (!seen[r]) perm_out[pos++] = r;
            free(seen);
        }
    }

    (void)M_per;
    free(labels); free(csize); free(corder);
    return (pos == inter) ? 0 : -1;
}

/* Permute rows of W [rows, cols] in place using perm[new]=old. */
static int mome_apply_row_perm(float *W, int rows, int cols, const int *perm)
{
    float *tmp = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!tmp) return -1;
    for (int i = 0; i < rows; i++)
        memcpy(tmp + (size_t)i * cols, W + (size_t)perm[i] * cols,
               (size_t)cols * sizeof(float));
    memcpy(W, tmp, (size_t)rows * cols * sizeof(float));
    free(tmp);
    return 0;
}

/* Permute cols of W [rows, cols] in place using perm[new]=old. */
static int mome_apply_col_perm(float *W, int rows, int cols, const int *perm)
{
    float *tmp = (float *)malloc((size_t)cols * sizeof(float));
    if (!tmp) return -1;
    for (int r = 0; r < rows; r++) {
        float *row = W + (size_t)r * cols;
        for (int j = 0; j < cols; j++) tmp[j] = row[perm[j]];
        memcpy(row, tmp, (size_t)cols * sizeof(float));
    }
    free(tmp);
    return 0;
}

/* Classify a MoME projection from its tensor name. Returns 0=gate,
 * 1=up, 2=down, -1=unknown. */
static int mome_proj_kind(const char *name) {
    if (!name) return -1;
    if (strstr(name, "gate_proj")) return 0;
    if (strstr(name, "up_proj"))   return 1;
    if (strstr(name, "down_proj")) return 2;
    return -1;
}

/* MoME row-split FFN pusher (gate_proj / up_proj).
 *
 * Reads `name_in_source` (FFN gate/up matrix [intermediate, hidden]) as
 * fp32 once, slices it into K equal contiguous row-blocks, encodes each
 * as its own PQv2 tensor named `<base_name>.expert{e}`. Caller chooses
 * pyramid/flat exactly as for the non-MoME path.
 *
 * When IB_MOME_COSINE_CLUSTER=1 the rows are first reordered by a
 * cosine-cluster permutation (computed from the gate matrix, reused for
 * up/down) so each expert groups input-similar neurons.
 *
 * Returns 0 on success, -1 on error (ib_set_error filled), +1 if the
 * tensor cannot be MoME-split (e.g. M not divisible by K) — caller
 * should fall back to the non-MoME encoder on +1. */
static int read_and_push_pqv2_mome_rows(ib6_manifest *mf,
                                          const char *base_name,
                                          const ib_tensor_source *ts,
                                          int shard, int t,
                                          int G, int K_cb, int half,
                                          int pyramid, int residency_hint,
                                          int scale_precision, int codebook_dedup,
                                          int K_experts,
                                          uint32_t seed)
{
    const void *raw = ib_ts_tensor_data(ts, shard, t);
    const char *dtype = ib_ts_tensor_dtype(ts, shard, t);
    int rows = ib_ts_tensor_shape(ts, shard, t, 0);
    int cols = ib_ts_tensor_shape(ts, shard, t, 1);
    if (cols == 0) cols = 1;
    if (K_experts <= 1) return +1;
    if (rows <= 0 || (rows % K_experts) != 0) return +1;
    if ((cols % G) != 0) {
        ib_set_error("%s: cols=%d not divisible by G=%d", base_name, cols, G);
        return -1;
    }
    int M_full = rows;
    int N      = cols;
    int M_per  = rows / K_experts;
    int n_chunks = N / G;
    int n_sub    = G / half;
    int l2_K     = pqv2_resolve_l2_k(pyramid);

    float *W = (float *)malloc((size_t)M_full * N * sizeof(float));
    if (!W) { ib_set_error("oom reading %s", base_name); return -1; }
    if (pqv2_read_matrix_fp32(W, raw, dtype, M_full, N) != 0) {
        free(W);
        ib_set_error("%s: unsupported dtype %s", base_name, dtype);
        return -1;
    }

    /* ── Cosine row-clustering (QUALITY PROBE, IB_MOME_COSINE_CLUSTER) ──
     * gate: compute the permutation from W (the gate matrix) and stash it.
     * up:   reuse the stashed permutation (same row count as gate).
     * Both apply it by gathering rows so the per-expert blocks group
     * input-similar neurons instead of arbitrary contiguous slices. */
    if (mome_cosine_cluster_enabled()) {
        int kind = mome_proj_kind(base_name);
        if (getenv("IB_MOME_CLUSTER_DEBUG"))
            fprintf(stderr, "[mome_cosine] %s kind=%d M=%d N=%d K=%d\n",
                    base_name, kind, M_full, N, K_experts);
        if (kind == 0) {
            /* gate → (re)compute and stash. */
            if (g_mome_perm_n < M_full || !g_mome_perm) {
                free(g_mome_perm);
                g_mome_perm = (int *)malloc((size_t)M_full * sizeof(int));
                g_mome_perm_n = g_mome_perm ? M_full : 0;
            }
            if (g_mome_perm &&
                mome_compute_cosine_perm(W, M_full, N, K_experts,
                                          seed, g_mome_perm) == 0) {
                g_mome_perm_K = K_experts;
                if (getenv("IB_MOME_CLUSTER_DEBUG")) {
                    int moved = 0;
                    for (int i = 0; i < M_full; i++)
                        if (g_mome_perm[i] != i) moved++;
                    fprintf(stderr, "[mome_cosine] gate perm built: %d/%d rows moved\n",
                            moved, M_full);
                }
                if (mome_apply_row_perm(W, M_full, N, g_mome_perm) != 0) {
                    free(W); ib_set_error("mome cosine: row perm oom");
                    return -1;
                }
            } else {
                /* clustering failed → leave W contiguous, invalidate stash
                 * so up/down also stay contiguous (consistent fallback). */
                g_mome_perm_K = 0;
            }
        } else if (kind == 1) {
            /* up → reuse gate's permutation when valid for this row count. */
            if (g_mome_perm && g_mome_perm_n >= M_full &&
                g_mome_perm_K == K_experts) {
                if (mome_apply_row_perm(W, M_full, N, g_mome_perm) != 0) {
                    free(W); ib_set_error("mome cosine: up perm oom");
                    return -1;
                }
            }
        }
    }

    /* One full-tensor encode → shared codebook + full row_scale + full
     * indices. The per-expert blobs below slice the (row_scale,
     * L1 idx, L2 idx) and reuse the same (cb_q, cb_scale, l2_cb_*).
     */
    size_t cb_q_n   = (size_t)n_sub * (size_t)K_cb * (size_t)half;
    size_t cb_s_n   = (size_t)n_sub * (size_t)K_cb;
    size_t idx_n    = (size_t)M_full * (size_t)n_chunks * (size_t)n_sub;
    size_t cb_q_l2n = pyramid ? (size_t)n_sub * (size_t)l2_K * (size_t)half : 0;
    size_t cb_s_l2n = pyramid ? (size_t)n_sub * (size_t)l2_K : 0;

    int8_t   *cb_q     = (int8_t   *)calloc(cb_q_n ? cb_q_n : 1, 1);
    uint16_t *cb_s     = (uint16_t *)calloc(cb_s_n ? cb_s_n : 1, 2);
    uint16_t *row_s    = (uint16_t *)calloc((size_t)M_full, 2);
    uint8_t  *idx_l1   = (uint8_t  *)calloc(idx_n ? idx_n : 1, 1);
    int8_t   *cb_q_l2  = pyramid ? (int8_t   *)calloc(cb_q_l2n, 1) : NULL;
    uint16_t *cb_s_l2  = pyramid ? (uint16_t *)calloc(cb_s_l2n, 2) : NULL;
    uint8_t  *idx_l2   = pyramid ? (uint8_t  *)calloc(idx_n, 1)    : NULL;
    if (!cb_q || !cb_s || !row_s || !idx_l1 ||
        (pyramid && (!cb_q_l2 || !cb_s_l2 || !idx_l2))) {
        free(W); free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        ib_set_error("mome rows: oom (%s)", base_name); return -1;
    }
    int rc;
    if (pyramid) {
        rc = pqv2_encode_pyramid(W, M_full, N, G, 0, l2_K, half,
                                  cb_q, cb_s, cb_q_l2, cb_s_l2,
                                  row_s, idx_l1, idx_l2, seed);
    } else {
        rc = pqv2_encode_flat_impl(W, M_full, N, G, K_cb, half,
                                    /*apply_row_scale=*/1,
                                    cb_q, cb_s, row_s, idx_l1,
                                    /*idx_layout_rowmajor=*/0,
                                    NULL, seed);
    }
    free(W);
    if (rc != 0) {
        free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        return rc;
    }

    /* Per-expert scratch: sliced row_scale [M_per] + sliced L1 idx
     * [n_chunks][n_sub][M_per] (+ same shape for L2). Reused across
     * experts; sized for one expert. */
    size_t slice_idx_n = (size_t)M_per * (size_t)n_chunks * (size_t)n_sub;
    uint16_t *rs_e   = (uint16_t *)malloc((size_t)M_per * 2);
    uint8_t  *idx_e  = (uint8_t  *)malloc(slice_idx_n);
    uint8_t  *idxL2_e = pyramid ? (uint8_t *)malloc(slice_idx_n) : NULL;
    if (!rs_e || !idx_e || (pyramid && !idxL2_e)) {
        free(rs_e); free(idx_e); free(idxL2_e);
        free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        ib_set_error("mome rows: oom (slice)"); return -1;
    }

    int rc_final = 0;
    for (int e = 0; e < K_experts; e++) {
        size_t row_off = (size_t)e * (size_t)M_per;
        memcpy(rs_e, row_s + row_off, (size_t)M_per * 2);
        /* Slice L1 indices: chunk-major [n_chunks][n_sub][M_full] →
         * [n_chunks][n_sub][M_per] with offset row_off along the M axis. */
        for (int c = 0; c < n_chunks; c++) {
            for (int s = 0; s < n_sub; s++) {
                memcpy(idx_e + ((size_t)c * n_sub + s) * (size_t)M_per,
                       idx_l1 + ((size_t)c * n_sub + s) * (size_t)M_full + row_off,
                       (size_t)M_per);
            }
        }
        if (pyramid) {
            for (int c = 0; c < n_chunks; c++) {
                for (int s = 0; s < n_sub; s++) {
                    memcpy(idxL2_e + ((size_t)c * n_sub + s) * (size_t)M_per,
                           idx_l2 + ((size_t)c * n_sub + s) * (size_t)M_full + row_off,
                           (size_t)M_per);
                }
            }
        }
        if (mome_emit_expert_blob(mf, base_name, e,
                                    M_per, N, G, K_cb, n_sub, half,
                                    pyramid, l2_K,
                                    rs_e, cb_q, cb_s, idx_e,
                                    cb_q_l2, cb_s_l2, idxL2_e,
                                    residency_hint, scale_precision,
                                    codebook_dedup) != 0) {
            rc_final = -1; break;
        }
    }

    free(rs_e); free(idx_e); free(idxL2_e);
    free(cb_q); free(cb_s); free(row_s); free(idx_l1);
    free(cb_q_l2); free(cb_s_l2); free(idx_l2);
    return rc_final;
}

/* MoME col-split FFN pusher (down_proj).
 *
 * Reads `name_in_source` (down_proj [hidden, intermediate]) as fp32 once,
 * tiles it into K equal contiguous column-blocks ([hidden, inter/K]),
 * encodes each as its own PQv2 tensor named `<base_name>.expert{e}`.
 * Returns the same {-1, 0, +1} convention as the row-split helper. */
static int read_and_push_pqv2_mome_cols(ib6_manifest *mf,
                                          const char *base_name,
                                          const ib_tensor_source *ts,
                                          int shard, int t,
                                          int G, int K_cb, int half,
                                          int pyramid, int residency_hint,
                                          int scale_precision, int codebook_dedup,
                                          int K_experts,
                                          uint32_t seed)
{
    const void *raw = ib_ts_tensor_data(ts, shard, t);
    const char *dtype = ib_ts_tensor_dtype(ts, shard, t);
    int rows = ib_ts_tensor_shape(ts, shard, t, 0);
    int cols = ib_ts_tensor_shape(ts, shard, t, 1);
    if (cols == 0) cols = 1;
    if (K_experts <= 1) return +1;
    if (cols <= 0 || (cols % K_experts) != 0) return +1;
    int N_per = cols / K_experts;
    if ((N_per % G) != 0) return +1;
    int M_full   = rows;
    int N_full   = cols;
    int n_chunks = N_full / G;
    int n_sub    = G / half;
    int l2_K     = pqv2_resolve_l2_k(pyramid);
    if ((n_chunks % K_experts) != 0) {
        /* Shared-codebook col-split requires the chunk count to divide
         * cleanly so each expert owns a contiguous CHUNK range (PQv2
         * codebooks are per subchunk-slot, identical across chunks, so
         * chunk-slicing the indices preserves the shared codebook
         * contract). Fall back to non-MoME on this tensor. */
        return +1;
    }
    int n_chunks_per_e = n_chunks / K_experts;

    float *W = (float *)malloc((size_t)M_full * N_full * sizeof(float));
    if (!W) { ib_set_error("oom reading %s", base_name); return -1; }
    if (pqv2_read_matrix_fp32(W, raw, dtype, M_full, N_full) != 0) {
        free(W);
        ib_set_error("%s: unsupported dtype %s", base_name, dtype);
        return -1;
    }

    /* ── Cosine clustering (down): permute COLUMNS by gate's row perm.
     * down_proj is [hidden, inter]; its columns index the same FFN
     * neurons that gate/up rows do, so the identical permutation keeps
     * the FFN math consistent (gate row i ↔ up row i ↔ down col i). */
    if (mome_cosine_cluster_enabled() && g_mome_perm &&
        g_mome_perm_n >= N_full && g_mome_perm_K == K_experts) {
        if (mome_apply_col_perm(W, M_full, N_full, g_mome_perm) != 0) {
            free(W); ib_set_error("mome cosine: down col perm oom");
            return -1;
        }
    }

    /* Encode full down_proj [hidden, inter] once. */
    size_t cb_q_n   = (size_t)n_sub * (size_t)K_cb * (size_t)half;
    size_t cb_s_n   = (size_t)n_sub * (size_t)K_cb;
    size_t idx_n    = (size_t)M_full * (size_t)n_chunks * (size_t)n_sub;
    size_t cb_q_l2n = pyramid ? (size_t)n_sub * (size_t)l2_K * (size_t)half : 0;
    size_t cb_s_l2n = pyramid ? (size_t)n_sub * (size_t)l2_K : 0;

    int8_t   *cb_q    = (int8_t   *)calloc(cb_q_n ? cb_q_n : 1, 1);
    uint16_t *cb_s    = (uint16_t *)calloc(cb_s_n ? cb_s_n : 1, 2);
    uint16_t *row_s   = (uint16_t *)calloc((size_t)M_full, 2);
    uint8_t  *idx_l1  = (uint8_t  *)calloc(idx_n ? idx_n : 1, 1);
    int8_t   *cb_q_l2 = pyramid ? (int8_t   *)calloc(cb_q_l2n, 1) : NULL;
    uint16_t *cb_s_l2 = pyramid ? (uint16_t *)calloc(cb_s_l2n, 2) : NULL;
    uint8_t  *idx_l2  = pyramid ? (uint8_t  *)calloc(idx_n, 1)    : NULL;
    if (!cb_q || !cb_s || !row_s || !idx_l1 ||
        (pyramid && (!cb_q_l2 || !cb_s_l2 || !idx_l2))) {
        free(W); free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        ib_set_error("mome cols: oom (%s)", base_name); return -1;
    }
    int rc;
    if (pyramid) {
        rc = pqv2_encode_pyramid(W, M_full, N_full, G, 0, l2_K, half,
                                  cb_q, cb_s, cb_q_l2, cb_s_l2,
                                  row_s, idx_l1, idx_l2, seed);
    } else {
        rc = pqv2_encode_flat_impl(W, M_full, N_full, G, K_cb, half,
                                    /*apply_row_scale=*/1,
                                    cb_q, cb_s, row_s, idx_l1,
                                    /*idx_layout_rowmajor=*/0,
                                    NULL, seed);
    }
    free(W);
    if (rc != 0) {
        free(cb_q); free(cb_s); free(row_s); free(idx_l1);
        free(cb_q_l2); free(cb_s_l2); free(idx_l2);
        return rc;
    }

    /* Per-expert: slice a contiguous CHUNK range out of the full
     * chunk-major index buffer. row_scale is shared (every expert
     * sees every row of W_down; the full-tensor row_scale already
     * captures max|row| over all inter cols, which is the correct
     * scale because the codebook bytes were normalised against it). */
    size_t exp_chunk_bytes = (size_t)n_chunks_per_e * (size_t)n_sub * (size_t)M_full;
    int N_exp = n_chunks_per_e * G;

    int rc_final = 0;
    for (int e = 0; e < K_experts; e++) {
        const uint8_t *idx_l1_src = idx_l1 + (size_t)e * (size_t)n_chunks_per_e
                                                  * (size_t)n_sub * (size_t)M_full;
        const uint8_t *idx_l2_src = pyramid
            ? idx_l2 + (size_t)e * (size_t)n_chunks_per_e
                                  * (size_t)n_sub * (size_t)M_full
            : NULL;
        if (mome_emit_expert_blob(mf, base_name, e,
                                    M_full, N_exp, G, K_cb, n_sub, half,
                                    pyramid, l2_K,
                                    row_s, cb_q, cb_s, idx_l1_src,
                                    cb_q_l2, cb_s_l2, idx_l2_src,
                                    residency_hint, scale_precision,
                                    codebook_dedup) != 0) {
            rc_final = -1; break;
        }
        (void)exp_chunk_bytes;
    }

    free(cb_q); free(cb_s); free(row_s); free(idx_l1);
    free(cb_q_l2); free(cb_s_l2); free(idx_l2);
    return rc_final;
}

/* Read a (rows × cols) tensor from the source as fp32 with optional QK
 * permutation, then PQv2-encode and push it. */
static int read_and_push_pqv2(ib6_manifest *mf, const char *name,
                               const ib_tensor_source *ts, int shard, int t,
                               int G, int K, int half, int pyramid,
                               int residency_hint,
                               int scale_precision, int codebook_dedup,
                               int idx_layout_rowmajor,
                               int qk_n_heads, int head_dim, uint32_t seed)
{
    const void *raw = ib_ts_tensor_data(ts, shard, t);
    const char *dtype = ib_ts_tensor_dtype(ts, shard, t);
    int rows = ib_ts_tensor_shape(ts, shard, t, 0);
    int cols = ib_ts_tensor_shape(ts, shard, t, 1);
    if (cols == 0) cols = 1;
    if ((cols % G) != 0) {
        ib_set_error("%s: cols=%d not divisible by G=%d", name, cols, G);
        return -1;
    }
    float *W = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!W) { ib_set_error("oom reading %s", name); return -1; }
    if (pqv2_read_matrix_fp32(W, raw, dtype, rows, cols) != 0) {
        free(W);
        ib_set_error("%s: unsupported dtype %s", name, dtype);
        return -1;
    }
    if (qk_n_heads > 0 && head_dim > 0) {
        const char *e = getenv("IB_DISABLE_QK_PERM");
        int disabled = (e && e[0] && e[0] != '0');
        if (!disabled) permute_qk_rows_fp32_inplace(W, rows, cols, head_dim, qk_n_heads);
    }
    int rc = push_pqv2_tensor(mf, name, W, rows, cols, G, K, half,
                               pyramid, residency_hint,
                               scale_precision, codebook_dedup,
                               idx_layout_rowmajor, seed);
    free(W);
    return rc;
}

/* Read a tensor and push as raw fp16. Applies optional QK permutation. */
static int read_and_push_fp16(ib6_manifest *mf, const char *name,
                               const ib_tensor_source *ts, int shard, int t,
                               int qk_n_heads, int head_dim)
{
    const void *raw = ib_ts_tensor_data(ts, shard, t);
    const char *dtype = ib_ts_tensor_dtype(ts, shard, t);
    int rows = ib_ts_tensor_shape(ts, shard, t, 0);
    int cols = ib_ts_tensor_shape(ts, shard, t, 1);
    if (cols == 0) cols = 1;
    if (qk_n_heads > 0 && head_dim > 0) {
        /* QK permute requires fp32 buffer; convert → permute → push as fp16. */
        const char *e = getenv("IB_DISABLE_QK_PERM");
        int disabled = (e && e[0] && e[0] != '0');
        if (!disabled) {
            float *W = (float *)malloc((size_t)rows * cols * sizeof(float));
            if (!W) { ib_set_error("oom reading %s", name); return -1; }
            if (pqv2_read_matrix_fp32(W, raw, dtype, rows, cols) != 0) {
                free(W); ib_set_error("%s: unsupported dtype %s", name, dtype); return -1;
            }
            permute_qk_rows_fp32_inplace(W, rows, cols, head_dim, qk_n_heads);
            size_t total = (size_t)rows * cols;
            uint16_t *fp16 = (uint16_t *)malloc(total * 2);
            if (!fp16) { free(W); ib_set_error("oom %s", name); return -1; }
            for (size_t i = 0; i < total; i++) fp16[i] = enc_f2h(W[i]);
            free(W);
            int32_t shape[4] = { rows, cols > 1 ? cols : 1, 1, 1 };
            int ndim = (cols > 1) ? 2 : 1;
            if (ib6_push(mf, name, IB_PQV2_KIND_RAW_FP16, ndim, shape,
                          fp16, total * 2) != 0) {
                free(fp16); ib_set_error("manifest oom (%s)", name); return -1;
            }
            return 0;
        }
    }
    return push_raw_fp16_from_source(mf, name, raw, dtype, rows, cols);
}

/* Read 1-D norm tensor as raw fp16 (no permute, no PQ). */
static int read_and_push_norm(ib6_manifest *mf, const char *name,
                               const ib_tensor_source *ts, int shard, int t)
{
    const void *raw = ib_ts_tensor_data(ts, shard, t);
    const char *dtype = ib_ts_tensor_dtype(ts, shard, t);
    int rows = ib_ts_tensor_shape(ts, shard, t, 0);
    int cols = ib_ts_tensor_shape(ts, shard, t, 1);
    int n = rows * (cols > 0 ? cols : 1);
    uint16_t *fp16 = (uint16_t *)malloc((size_t)n * 2);
    if (!fp16) { ib_set_error("oom (%s)", name); return -1; }
    ib_copy_norm_fp16(fp16, raw, dtype, n);
    int32_t shape[4] = { rows, cols > 0 ? cols : 1, 1, 1 };
    int ndim = (cols > 0 && cols != 1) ? 2 : 1;
    if (ib6_push(mf, name, IB_PQV2_KIND_RAW_FP16, ndim, shape,
                  fp16, (size_t)n * 2) != 0) {
        free(fp16); ib_set_error("manifest oom (%s)", name); return -1;
    }
    return 0;
}

/* ── final write pass ── */

static int write_ibf6_file(const char *path, ib6_manifest *mf,
                            uint32_t pqv2_writer_flags)
{
    /* Pass 1: manifest size. */
    size_t manifest_size = 0;
    for (int i = 0; i < mf->n; i++) {
        const ib6_entry *e = &mf->entries[i];
        size_t nl = strlen(e->name);
        manifest_size += 2 + nl + 1 + 1 + 2 + 16 + 8 + 8;
    }
    size_t header_total = 24 + manifest_size;
    size_t first_blob_off = align_up_z(header_total, IBF6_ALIGN);

    /* Pass 2: blob offsets. */
    size_t cur = first_blob_off;
    for (int i = 0; i < mf->n; i++) {
        mf->entries[i].blob_offset = (uint64_t)cur;
        cur += mf->entries[i].blob_size;
        cur = align_up_z(cur, IBF6_ALIGN);
    }

    /* Pass 3: write. */
    FILE *f = fopen(path, "wb");
    if (!f) { ib_set_error("cannot open %s: %s", path, strerror(errno)); return -1; }
    if (fwrite(IBF6_MAGIC, 1, 8, f) != 8) goto wfail;
    uint32_t v = IBF6_VERSION;
    if (fwrite(&v, 4, 1, f) != 1) goto wfail;
    uint32_t n_tensors = (uint32_t)mf->n;
    if (fwrite(&n_tensors, 4, 1, f) != 1) goto wfail;
    uint32_t msz = (uint32_t)manifest_size;
    if (fwrite(&msz, 4, 1, f) != 1) goto wfail;
    /* "reserved (0)" slot in the file header (pqv2_format.h documents
     * the 4 bytes at [20..23] as reserved; the loader never reads
     * them). We overload it as a writer-flags field carrying the
     * format_str distinction: bit 0 = pyramid (any tensor uses
     * l2_kind=2). Older loaders ignore this — safe. */
    if (fwrite(&pqv2_writer_flags, 4, 1, f) != 1) goto wfail;

    for (int i = 0; i < mf->n; i++) {
        const ib6_entry *e = &mf->entries[i];
        uint16_t nl = (uint16_t)strlen(e->name);
        if (fwrite(&nl, 2, 1, f) != 1) goto wfail;
        if (fwrite(e->name, 1, nl, f) != nl) goto wfail;
        uint8_t kk = (uint8_t)e->kind;
        uint8_t dd = (uint8_t)e->ndim;
        if (fwrite(&kk, 1, 1, f) != 1) goto wfail;
        if (fwrite(&dd, 1, 1, f) != 1) goto wfail;
        uint8_t resv2[2] = {0, 0};
        if (fwrite(resv2, 1, 2, f) != 2) goto wfail;
        if (fwrite(e->shape, 4, 4, f) != 4) goto wfail;
        if (fwrite(&e->blob_offset, 8, 1, f) != 1) goto wfail;
        uint64_t bsz = (uint64_t)e->blob_size;
        if (fwrite(&bsz, 8, 1, f) != 1) goto wfail;
    }
    /* pad to first blob */
    {
        long pos = ftell(f);
        if (pos < 0) goto wfail;
        size_t pad = first_blob_off - (size_t)pos;
        static const uint8_t zeros[IBF6_ALIGN] = {0};
        while (pad > 0) {
            size_t chunk = pad > sizeof(zeros) ? sizeof(zeros) : pad;
            if (fwrite(zeros, 1, chunk, f) != chunk) goto wfail;
            pad -= chunk;
        }
    }
    for (int i = 0; i < mf->n; i++) {
        const ib6_entry *e = &mf->entries[i];
        long pos = ftell(f);
        if (pos < 0) goto wfail;
        if ((uint64_t)pos < e->blob_offset) {
            size_t pad = (size_t)e->blob_offset - (size_t)pos;
            static const uint8_t zeros[IBF6_ALIGN] = {0};
            while (pad > 0) {
                size_t chunk = pad > sizeof(zeros) ? sizeof(zeros) : pad;
                if (fwrite(zeros, 1, chunk, f) != chunk) goto wfail;
                pad -= chunk;
            }
        }
        if (fwrite(e->blob, 1, e->blob_size, f) != e->blob_size) goto wfail;
        /* trailing pad up to alignment for the NEXT blob is handled by
         * the leading-pad logic above on the next iteration. */
    }
    fclose(f);
    return 0;
wfail:
    ib_set_error("IBF v6 write failed: %s", strerror(errno));
    if (f) fclose(f);
    return -1;
}

/* ── architecture detection (shared with convert.c) ── */

typedef struct {
    char prefix[64];
    char layer_fmt[64];
    char q_proj[64];
    char k_proj[64];
    char v_proj[64];
    char o_proj[64];
    char gate_proj[64];
    char up_proj[64];
    char down_proj[64];
    char input_norm[64];
    char post_norm[64];
    char embed[128];
    char final_norm[128];
    char lm_head[128];
} pq6_names;

static int pq6_detect_naming(const ib_tensor_source *ts, pq6_names *n) {
    int s, t;
    if (ib_ts_find_suffix(ts, "model.layers.0.self_attn.q_proj.weight", &s, &t) == 0) {
        strcpy(n->prefix, "model.");
        strcpy(n->layer_fmt, "layers.%d.");
        strcpy(n->q_proj, "self_attn.q_proj.weight");
        strcpy(n->k_proj, "self_attn.k_proj.weight");
        strcpy(n->v_proj, "self_attn.v_proj.weight");
        strcpy(n->o_proj, "self_attn.o_proj.weight");
        strcpy(n->gate_proj, "mlp.gate_proj.weight");
        strcpy(n->up_proj, "mlp.up_proj.weight");
        strcpy(n->down_proj, "mlp.down_proj.weight");
        strcpy(n->input_norm, "input_layernorm.weight");
        strcpy(n->post_norm, "post_attention_layernorm.weight");
        strcpy(n->embed, "model.embed_tokens.weight");
        strcpy(n->final_norm, "model.norm.weight");
        strcpy(n->lm_head, "lm_head.weight");
        return 0;
    }
    if (ib_ts_find_suffix(ts, "layers.0.self_attn.q_proj.weight", &s, &t) == 0) {
        strcpy(n->prefix, "");
        strcpy(n->layer_fmt, "layers.%d.");
        strcpy(n->q_proj, "self_attn.q_proj.weight");
        strcpy(n->k_proj, "self_attn.k_proj.weight");
        strcpy(n->v_proj, "self_attn.v_proj.weight");
        strcpy(n->o_proj, "self_attn.o_proj.weight");
        strcpy(n->gate_proj, "mlp.gate_proj.weight");
        strcpy(n->up_proj, "mlp.up_proj.weight");
        strcpy(n->down_proj, "mlp.down_proj.weight");
        strcpy(n->input_norm, "input_layernorm.weight");
        strcpy(n->post_norm, "post_attention_layernorm.weight");
        strcpy(n->embed, "embed_tokens.weight");
        strcpy(n->final_norm, "norm.weight");
        strcpy(n->lm_head, "lm_head.weight");
        return 0;
    }
    ib_set_error("unrecognized tensor naming convention");
    return -1;
}

static int pq6_find_layer(const ib_tensor_source *ts, const pq6_names *n,
                            int li, const char *suffix, int *shard, int *t)
{
    char full[512]; char layer_part[64];
    snprintf(layer_part, sizeof(layer_part), n->layer_fmt, li);
    snprintf(full, sizeof(full), "%s%s%s", n->prefix, layer_part, suffix);
    return ib_ts_find(ts, full, shard, t);
}

/* ── Stage 5b/5c — per-tensor policy resolution ───────────────────
 *
 * resolve_format: returns the on-disk format for a tensor of `cls`.
 *   - If cfg->per_class_format[cls] is non-zero (= explicit override),
 *     use that value.
 *   - Else fall back to cfg->format (the global flag set by --format).
 * The encoder's pyramid/flat decision is then `(result == PQV2_PYRAMID)`
 * which is what `push_pqv2_tensor`'s `pyramid` boolean expects.
 *
 * Stage 5b.fix (regression filed 2026-05-17): an earlier draft of this
 * helper returned `INFERBIT_CONVERT_INT4` (= 0) on the fallback branch
 * instead of `cfg->format`, which made `--format pyramid` silently
 * produce flat output (symptom: pyramid and pqv2 .ibf files were the
 * same byte size and PPL). The fallback below MUST return cfg->format
 * to honor the global --format selector when no per-class override is
 * set. The semantic equivalence "per == 0 means INT4 means 'use global'"
 * is intentional — the per_class_format[] array's zero-init value
 * doubles as the "no override" sentinel.
 *
 * resolve_residency: returns the residency hint for a tensor of `cls`
 * in layer `layer_idx` (use -1 for non-layered tensors like embed/lm_head).
 *   - If cfg->per_class_residency[cls] is non-zero (= explicit override),
 *     use that value.
 *   - Else apply the encoder heuristic:
 *       * EMBED / LM_HEAD          → RAM (always hot)
 *       * 0 ≤ layer_idx < ram_layers (default 2; IB_RESIDENCY_RAM_LAYERS
 *         override) → RAM (hot path during early-stack streaming)
 *       * Otherwise                → AUTO (loader decides). */
static inline inferbit_convert_format
resolve_format(const inferbit_convert_config *cfg, inferbit_tensor_class cls)
{
    /* Defensive: out-of-range class index falls back to the global
     * cfg->format selector. Never silently downgrade to INT4. */
    if ((int)cls < 0 || (int)cls >= INFERBIT_TENSOR_CLASS_COUNT) {
        return cfg->format;
    }
    inferbit_convert_format per = cfg->per_class_format[cls];
    /* Any non-zero per-class value is an explicit caller-set override
     * that wins over `cfg->format` (and the class-default policy
     * below). Honor it verbatim — including the case where the user
     * explicitly opts attention/embed/lm_head into pyramid. */
    if (per != INFERBIT_CONVERT_INT4) return per;

    /* Class-default policy (Stage 5b.fix2, 2026-05-18): pyramid L2 is
     * a costly residual stream (~50–66% size bloat) that only buys
     * measurable quality on the large dense matmuls — i.e. the FFN
     * projections, which dominate both parameter count and the PPL
     * sensitivity to quantization error. Attention Q/K/V/O are
     * smaller and head-split (per-head error averages out in the
     * softmax); embedding and lm_head are row-lookup / final-projection
     * tensors where the loader already encodes them flat regardless.
     *
     * So when the user picks `--format pyramid` globally and does NOT
     * override per class, we apply pyramid ONLY to the FFN classes
     * and quietly downgrade everything else to PQv2 flat. This drops
     * the file size by roughly a third on a 1.1B model while keeping
     * the PPL essentially unchanged (FFN-dominated). The user can
     * still force pyramid on attention via the per-class override. */
    if (cfg->format == INFERBIT_CONVERT_PQV2_PYRAMID) {
        switch (cls) {
            case INFERBIT_TENSOR_CLASS_FFN_GATE:
            case INFERBIT_TENSOR_CLASS_FFN_UP:
            case INFERBIT_TENSOR_CLASS_FFN_DOWN:
                return INFERBIT_CONVERT_PQV2_PYRAMID;
            case INFERBIT_TENSOR_CLASS_ATTN_Q:
            case INFERBIT_TENSOR_CLASS_ATTN_K:
            case INFERBIT_TENSOR_CLASS_ATTN_V:
            case INFERBIT_TENSOR_CLASS_ATTN_O:
            case INFERBIT_TENSOR_CLASS_EMBED:
            case INFERBIT_TENSOR_CLASS_LM_HEAD:
            default:
                return INFERBIT_CONVERT_PQV2_FLAT;
        }
    }
    /* Non-pyramid global selectors (INT4 / PQV2_FLAT) propagate as-is. */
    return cfg->format;
}

static inline int
resolve_residency(const inferbit_convert_config *cfg,
                   inferbit_tensor_class cls,
                   int layer_idx, int ram_layers)
{
    if (cls >= 0 && cls < INFERBIT_TENSOR_CLASS_COUNT) {
        inferbit_residency per = cfg->per_class_residency[cls];
        if (per != INFERBIT_RESIDENCY_AUTO) return (int)per;
    }
    /* Heuristic defaults. */
    if (cls == INFERBIT_TENSOR_CLASS_EMBED ||
        cls == INFERBIT_TENSOR_CLASS_LM_HEAD) {
        return (int)INFERBIT_RESIDENCY_RAM;
    }
    if (layer_idx >= 0 && layer_idx < ram_layers) {
        return (int)INFERBIT_RESIDENCY_RAM;
    }
    return (int)INFERBIT_RESIDENCY_AUTO;
}

/* ── pqv2_convert: top-level orchestrator ── */

int pqv2_convert(const char *input_path,
                  const char *output_path,
                  const inferbit_convert_config *cfg)
{
    if (!input_path || !output_path || !cfg) {
        ib_set_error("pqv2_convert: NULL argument");
        return INFERBIT_ERROR_PARAM;
    }
    /* Global default pyramid flag (used at end of pass to OR-reduce
     * the writer_flags bit). Per-tensor resolution happens via
     * resolve_format() below. */
    int pyramid_default = (cfg->format == INFERBIT_CONVERT_PQV2_PYRAMID);
    const uint32_t seed = 42u;
    const int G = 64;
    const int K = 256;
    const int half = 2;

    /* Stage 5c — RAM-layers heuristic knob. Default 2 ("first two layers
     * hot"). Override via IB_RESIDENCY_RAM_LAYERS=N at encode time. The
     * loader honors the same env var to promote additional layers at
     * load — encoder-side default keeps the hot set small in the file
     * so any loader that reads the hint sees a sensible baseline. */
    int ram_layers = 2;
    {
        const char *e = getenv("IB_RESIDENCY_RAM_LAYERS");
        if (e && e[0]) {
            int v = atoi(e);
            if (v >= 0 && v < 1000) ram_layers = v;
        }
    }

    /* Format consolidation (2026-05-20): the supported on-disk formats are
     * exactly `flat` and `pyramid`, with `--mome K` as an orthogonal option.
     * The scale_precision=2 (sp2) and L1-row-major (rowmajor) variants are
     * REMOVED — both were Pareto-dominated (sp2: same size, worse PPL;
     * rowmajor: same size+quality as chunk-major, its only purpose was a
     * Metal zero-copy that proved a wash and was repeatedly buggy). We force
     * them off here so no file can ever be created with them; the now-dead
     * decode branches are scheduled for physical deletion. cfg->scale_precision
     * is ignored. IB_L2_K (4-bit L2) is intentionally retained — it's an
     * active line of work, not a dead variant. */
    int sp = 0;
    int cd = cfg->codebook_dedup ? 1 : 0;
    int idx_layout_rowmajor = 0;

    void (*progress)(float, const char *, void *) = cfg->progress;
    void *prog_ctx = cfg->progress_ctx;
    if (progress) progress(0.0f, "opening", prog_ctx);

    ib_tensor_source *ts = ib_ts_open(input_path);
    if (!ts) {
        ib_set_error("cannot open tensor source: %s", input_path);
        return INFERBIT_ERROR_LOAD;
    }

    /* Architecture: use config.json when available (authoritative);
     * otherwise infer from tensor shapes. Mirrors convert.c. */
    ib_model_config mc; memset(&mc, 0, sizeof(mc));
    int has_config = 0;
    {
        ib_struct_stat st;
        char cfg_path[1024];
        if (ib_stat(input_path, &st) == 0 && S_ISDIR(st.st_mode)) {
            snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", input_path);
        } else {
            const char *slash = strrchr(input_path, '/');
            if (slash) {
                size_t dlen = (size_t)(slash - input_path);
                snprintf(cfg_path, sizeof(cfg_path), "%.*s/config.json",
                         (int)dlen, input_path);
            } else {
                snprintf(cfg_path, sizeof(cfg_path), "config.json");
            }
        }
        has_config = (ib_parse_config_json(cfg_path, &mc) == 0);
    }

    pq6_names names;
    if (pq6_detect_naming(ts, &names) != 0) {
        ib_ts_close(ts);
        return INFERBIT_ERROR_FORMAT;
    }

    int num_layers, hidden, num_heads, num_kv_heads, head_dim;
    int vocab_size, intermediate;
    if (has_config) {
        num_layers   = mc.num_layers;
        hidden       = mc.hidden_size;
        num_heads    = mc.num_heads;
        num_kv_heads = mc.num_kv_heads;
        head_dim     = mc.head_dim;
        intermediate = mc.intermediate_size;
        vocab_size   = mc.vocab_size;
    } else {
        int s, t;
        if (ib_ts_find(ts, names.embed, &s, &t) != 0) {
            ib_set_error("cannot find embedding tensor");
            ib_ts_close(ts);
            return INFERBIT_ERROR_FORMAT;
        }
        vocab_size = ib_ts_tensor_shape(ts, s, t, 0);
        hidden     = ib_ts_tensor_shape(ts, s, t, 1);
        if (ib_ts_find_suffix(ts, names.q_proj, &s, &t) != 0) {
            ib_set_error("cannot find q_proj");
            ib_ts_close(ts);
            return INFERBIT_ERROR_FORMAT;
        }
        int q_out = ib_ts_tensor_shape(ts, s, t, 0);
        head_dim = (hidden > 2048) ? 128 : 64;
        num_heads = q_out / head_dim;
        num_kv_heads = num_heads;
        if (ib_ts_find_suffix(ts, names.k_proj, &s, &t) == 0) {
            int k_out = ib_ts_tensor_shape(ts, s, t, 0);
            num_kv_heads = k_out / head_dim;
        }
        intermediate = hidden * 4;
        if (ib_ts_find_suffix(ts, names.gate_proj, &s, &t) == 0) {
            intermediate = ib_ts_tensor_shape(ts, s, t, 0);
        }
        num_layers = 0;
        for (int i = 0; i < 1000; i++) {
            if (pq6_find_layer(ts, &names, i, names.q_proj, &s, &t) != 0) break;
            num_layers++;
        }
    }
    (void)intermediate;

    ib6_manifest mf; memset(&mf, 0, sizeof(mf));

    /* Embedding — PQv2 flat (Stage 5f). Encoded as flat (pyramid=0) even
     * when the caller requested pyramid: forward.c::cpu_embed_lookup only
     * consumes the L1 codebook for the embedding tensor, so an L2 stream
     * would be silently dropped at lookup time. No QK permutation
     * (qk_n_heads=0, head_dim=0). Loader auto-detects vocab_size from
     * this tensor's row count; pqv2_model.c::detect_arch_from_tensors.
     *
     * Safety guard: read_and_push_pqv2 requires cols (=hidden) divisible
     * by G. For Llama-3 (hidden=4096) and TinyLlama (hidden=2048) with
     * default G=64 this always holds; the guard exists so hypothetical
     * future architectures with awkward hidden sizes still convert
     * (falling back to raw fp16 for the embedding only). */
    if (progress) progress(0.02f, "embedding", prog_ctx);
    {
        int s, t;
        if (ib_ts_find(ts, names.embed, &s, &t) == 0) {
            int rc;
            /* Embedding is forced flat (pyramid=0) because
             * cpu_embed_lookup only consumes the L1 codebook; an L2
             * stream would be silently dropped. Per-class format override
             * for EMBED is therefore intentionally not honored — the
             * format choice is a no-op here. Residency hint IS honored;
             * EMBED defaults to RAM. */
            int rh = resolve_residency(cfg, INFERBIT_TENSOR_CLASS_EMBED,
                                        -1, ram_layers);
            if (G > 0 && (hidden % G) == 0) {
                rc = read_and_push_pqv2(&mf, "token_embedding", ts, s, t,
                                         G, K, half, /*pyramid=*/0, rh,
                                         sp, cd,
                                         idx_layout_rowmajor,
                                         /*qk_n_heads=*/0, /*head_dim=*/0,
                                         seed);
            } else {
                rc = read_and_push_fp16(&mf, "token_embedding", ts, s, t, 0, 0);
            }
            if (rc != 0) {
                ib6_free(&mf); ib_ts_close(ts); return INFERBIT_ERROR_INTERNAL;
            }
        }
    }

    /* Per-layer tensors. */
    const int verbose = (getenv("IB_PQV2_QUIET") == NULL);
    struct timespec ts_layer_start;
    for (int l = 0; l < num_layers; l++) {
        if (progress) {
            float pct = 0.05f + 0.85f * (float)l / (float)num_layers;
            progress(pct, "layers", prog_ctx);
        }
        if (verbose) {
            ib_clock_gettime(CLOCK_MONOTONIC, &ts_layer_start);
            fprintf(stderr,
                    "[pqv2] layer %d/%d encoding (FFN gate/up/down + o_proj, K=%d half=%d G=%d)\n",
                    l + 1, num_layers, K, half, G);
            fflush(stderr);
        }
        int s, t;
        char nm[64];

        /* Q/K/V/O all go through the PQv2 encoder now (was raw fp16 in the
         * agent's initial implementation — the CPU has no NEON fast path
         * for fp16-raw matmul, and Metal upload rejects raw-fp16 layer
         * matmuls outright). With PQv2 the same K=256/half=2 NEON kernel
         * handles attention as well, and file size drops ~5-8× vs raw fp16
         * for these tensors. Q and K still get the interleaved-RoPE row
         * permutation in fp32 *before* PQv2 encoding (read_and_push_pqv2
         * already applies it when qk_n_heads/head_dim are non-zero). */
        /* Stage 5b/5c — resolve per-class format + residency for this layer. */
        #define PY(cls)  (resolve_format(cfg, (cls)) == INFERBIT_CONVERT_PQV2_PYRAMID)
        #define RH(cls)  resolve_residency(cfg, (cls), l, ram_layers)

        if (pq6_find_layer(ts, &names, l, names.q_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.self_attn.q_proj", l);
            if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                    PY(INFERBIT_TENSOR_CLASS_ATTN_Q),
                                    RH(INFERBIT_TENSOR_CLASS_ATTN_Q),
                                    sp, cd,
                                    idx_layout_rowmajor,
                                    num_heads, head_dim, seed) != 0)
                goto fail;
        }
        if (pq6_find_layer(ts, &names, l, names.k_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.self_attn.k_proj", l);
            if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                    PY(INFERBIT_TENSOR_CLASS_ATTN_K),
                                    RH(INFERBIT_TENSOR_CLASS_ATTN_K),
                                    sp, cd,
                                    idx_layout_rowmajor,
                                    num_kv_heads, head_dim, seed) != 0)
                goto fail;
        }
        if (pq6_find_layer(ts, &names, l, names.v_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.self_attn.v_proj", l);
            if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                    PY(INFERBIT_TENSOR_CLASS_ATTN_V),
                                    RH(INFERBIT_TENSOR_CLASS_ATTN_V),
                                    sp, cd,
                                    idx_layout_rowmajor,
                                    0, 0, seed) != 0) goto fail;
        }
        if (pq6_find_layer(ts, &names, l, names.o_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.self_attn.o_proj", l);
            if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                    PY(INFERBIT_TENSOR_CLASS_ATTN_O),
                                    RH(INFERBIT_TENSOR_CLASS_ATTN_O),
                                    sp, cd,
                                    idx_layout_rowmajor,
                                    0, 0, seed) != 0) goto fail;
        }
        /* FFN tensors. MoME scaffolding (Stage 3a): when cfg->mome_experts
         * > 1, encode each of gate/up/down as K row/col-split sub-tensors
         * named `Lk.mlp.<proj>.expert{e}` instead of a single flat tensor.
         * The helpers return +1 when the split cannot be performed (M
         * not divisible by K, etc.); on +1 we fall back to the non-MoME
         * single-tensor emit for that layer's projection. The fallback
         * keeps the layer functional but downgrades it to mome_experts=1
         * at load time — fine in v1, which is correctness-first. */
        const int K_experts = (cfg->mome_experts > 1) ? cfg->mome_experts : 1;
        /* Capture gate_proj source for the I4 heuristic router below.   */
        const void *gate_router_src = NULL;
        const char *gate_router_dtype = NULL;
        int         gate_router_rows  = 0;
        if (pq6_find_layer(ts, &names, l, names.gate_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.mlp.gate_proj", l);
            int py_gate = PY(INFERBIT_TENSOR_CLASS_FFN_GATE);
            int rh_gate = RH(INFERBIT_TENSOR_CLASS_FFN_GATE);
            int rc_m = +1;   /* +1 = not MoME → fall through to flat */
            if (K_experts > 1) {
                /* Stash before encode (mome_rows doesn't mutate src).   */
                gate_router_src   = ib_ts_tensor_data(ts, s, t);
                gate_router_dtype = ib_ts_tensor_dtype(ts, s, t);
                gate_router_rows  = ib_ts_tensor_shape(ts, s, t, 0);
                rc_m = read_and_push_pqv2_mome_rows(&mf, nm, ts, s, t,
                                                     G, K, half, py_gate,
                                                     rh_gate, sp, cd,
                                                     K_experts, seed);
                if (rc_m == -1) goto fail;
                if (rc_m == +1) {
                    /* split rejected → don't pretend we have a router. */
                    gate_router_src = NULL;
                }
            }
            if (rc_m == +1) {
                if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                        py_gate, rh_gate, sp, cd,
                                        idx_layout_rowmajor,
                                        0, 0, seed) != 0) goto fail;
            }
        }
        if (pq6_find_layer(ts, &names, l, names.up_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.mlp.up_proj", l);
            int py_up = PY(INFERBIT_TENSOR_CLASS_FFN_UP);
            int rh_up = RH(INFERBIT_TENSOR_CLASS_FFN_UP);
            int rc_m = +1;
            if (K_experts > 1) {
                rc_m = read_and_push_pqv2_mome_rows(&mf, nm, ts, s, t,
                                                     G, K, half, py_up,
                                                     rh_up, sp, cd,
                                                     K_experts, seed);
                if (rc_m == -1) goto fail;
            }
            if (rc_m == +1) {
                if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                        py_up, rh_up, sp, cd,
                                        idx_layout_rowmajor,
                                        0, 0, seed) != 0) goto fail;
            }
        }
        if (pq6_find_layer(ts, &names, l, names.down_proj, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.mlp.down_proj", l);
            int py_dn = PY(INFERBIT_TENSOR_CLASS_FFN_DOWN);
            int rh_dn = RH(INFERBIT_TENSOR_CLASS_FFN_DOWN);
            int rc_m = +1;
            if (K_experts > 1) {
                rc_m = read_and_push_pqv2_mome_cols(&mf, nm, ts, s, t,
                                                     G, K, half, py_dn,
                                                     rh_dn, sp, cd,
                                                     K_experts, seed);
                if (rc_m == -1) goto fail;
            }
            if (rc_m == +1) {
                if (read_and_push_pqv2(&mf, nm, ts, s, t, G, K, half,
                                        py_dn, rh_dn, sp, cd,
                                        idx_layout_rowmajor,
                                        0, 0, seed) != 0) goto fail;
            }
        }

        #undef PY
        #undef RH
        /* MoME router: one [hidden, K] zero-init raw fp16 tensor per
         * layer when MoME is enabled AND at least one of gate/up/down
         * was actually split. The simplest correct gate is "K_experts
         * > 1 and hidden > 0 and K_experts <= IB_MOME_MAX_EXPERTS".
         * The runtime guards on `mome_router_is_nonzero` — a zero
         * router triggers the all-experts fallback regardless of
         * whether each FFN slot was actually split, so this is safe
         * even when one of the per-projection splits fell back to
         * flat. */
        if (K_experts > 1 && K_experts <= 32 && hidden > 0) {
            char base[64];
            snprintf(base, sizeof(base), "L%d.mlp", l);
            if (push_zero_router(&mf, base, hidden, K_experts,
                                  gate_router_src, gate_router_dtype,
                                  gate_router_rows) != 0) goto fail;
        }
        if (pq6_find_layer(ts, &names, l, names.input_norm, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.input_layernorm", l);
            if (read_and_push_norm(&mf, nm, ts, s, t) != 0) goto fail;
        }
        if (pq6_find_layer(ts, &names, l, names.post_norm, &s, &t) == 0) {
            snprintf(nm, sizeof(nm), "L%d.post_attention_layernorm", l);
            if (read_and_push_norm(&mf, nm, ts, s, t) != 0) goto fail;
        }

        if (verbose) {
            struct timespec ts_layer_end;
            ib_clock_gettime(CLOCK_MONOTONIC, &ts_layer_end);
            double dt = (ts_layer_end.tv_sec - ts_layer_start.tv_sec) +
                        (ts_layer_end.tv_nsec - ts_layer_start.tv_nsec) * 1e-9;
            int remaining = num_layers - (l + 1);
            double eta = dt * remaining;
            fprintf(stderr,
                    "[pqv2] layer %d/%d done in %.1fs (ETA ~%.0fs for remaining %d layers)\n",
                    l + 1, num_layers, dt, eta, remaining);
            fflush(stderr);
        }
    }

    /* Final norm + LM head. */
    if (progress) progress(0.92f, "output", prog_ctx);
    {
        int s, t;
        if (ib_ts_find(ts, names.final_norm, &s, &t) == 0) {
            if (read_and_push_norm(&mf, "output_norm", ts, s, t) != 0) goto fail;
        }
        int head_shard = -1, head_t = -1;
        if (ib_ts_find(ts, names.lm_head, &s, &t) == 0) {
            head_shard = s; head_t = t;
        }
        if (head_shard >= 0) {
            /* lm_head — PQv2 flat (Stage 5f). Encoded as flat (pyramid=0)
             * for parity with the embedding side: the loader's F1 fallback
             * (tied embeddings) aliases output_head to token_embedding,
             * and we want both code paths to behave identically. No QK
             * permutation. Same divisibility guard as the embedding —
             * shape is [vocab, hidden] and Llama-3 / TinyLlama satisfy
             * hidden % G == 0 for G=64. Residency hint IS honored via
             * resolve_residency — defaults to RAM (lm_head is always
             * hot on the last token). */
            int rh_head = resolve_residency(cfg,
                                              INFERBIT_TENSOR_CLASS_LM_HEAD,
                                              -1, ram_layers);
            int rc;
            if (G > 0 && (hidden % G) == 0) {
                rc = read_and_push_pqv2(&mf, "lm_head", ts, head_shard, head_t,
                                         G, K, half, /*pyramid=*/0, rh_head,
                                         sp, cd,
                                         idx_layout_rowmajor,
                                         /*qk_n_heads=*/0, /*head_dim=*/0,
                                         seed);
            } else {
                rc = read_and_push_fp16(&mf, "lm_head", ts, head_shard, head_t, 0, 0);
            }
            if (rc != 0) goto fail;
        }
        /* If tied embeddings (no lm_head tensor): the loader treats the
         * token_embedding as the head when output_head.pq is unset; this
         * matches the legacy IBF v5 behaviour. */
    }

    if (progress) progress(0.97f, "writing", prog_ctx);

    /* writer_flags bit 0 = "any tensor in this file uses pyramid L2".
     * Stage 5b makes the format per-tensor, so compute the bit by OR-
     * reducing across the resolved per-class formats (and the global
     * fallback). Older inspect tools that read this flag for
     * compatibility still see a correct value: bit 0 set iff at least
     * one tensor was emitted pyramid. */
    int any_pyramid = pyramid_default;
    for (int i = 0; i < INFERBIT_TENSOR_CLASS_COUNT && !any_pyramid; i++) {
        if (cfg->per_class_format[i] == INFERBIT_CONVERT_PQV2_PYRAMID)
            any_pyramid = 1;
    }
    uint32_t writer_flags = any_pyramid ? 0x1u : 0x0u;
    if (write_ibf6_file(output_path, &mf, writer_flags) != 0) {
        ib6_free(&mf);
        ib_ts_close(ts);
        return INFERBIT_ERROR_INTERNAL;
    }
    ib6_free(&mf);
    ib_ts_close(ts);
    if (progress) progress(1.0f, "done", prog_ctx);
    (void)num_kv_heads;  /* unused if k_proj missing — silence -Wunused */
    return INFERBIT_OK;

fail:
    ib6_free(&mf);
    ib_ts_close(ts);
    return INFERBIT_ERROR_INTERNAL;
}
