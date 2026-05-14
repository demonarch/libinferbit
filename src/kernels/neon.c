/*
 * neon.c — ARM NEON optimized kernels
 *
 * For Apple Silicon and other ARM64 platforms.
 */

#if defined(__aarch64__) || defined(_M_ARM64)

#include "../inferbit_internal.h"
#include <arm_neon.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* MSVC ARM64 doesn't grok GCC __attribute__ syntax. And the
 * function-level `target` attribute spells the dotprod feature
 * differently between compilers: clang takes `dotprod`, GCC requires
 * the `+dotprod` arch-extension form. */
#if defined(_MSC_VER)
#define IB_NOINLINE __declspec(noinline)
#define IB_DOTPROD_NOINLINE __declspec(noinline)
#define IB_UNUSED
#elif defined(__clang__)
#define IB_NOINLINE __attribute__((noinline))
#define IB_DOTPROD_NOINLINE __attribute__((target("dotprod"),noinline))
#define IB_UNUSED __attribute__((unused))
#else  /* GCC */
#define IB_NOINLINE __attribute__((noinline))
#define IB_DOTPROD_NOINLINE __attribute__((target("+dotprod"),noinline))
#define IB_UNUSED __attribute__((unused))
#endif

/* ── INT8 matmul ────────────────────────────────────────────── */

static void neon_matmul_int8(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const int8_t* w = (const int8_t*)weights;

#if defined(__ARM_FEATURE_DOTPROD)
    /* vdotq_s32 fast path: quantize fp32 input to INT8 with per-group
     * scales (group=IB_W4A8_GROUP=128, mirrors the w4a8 path), then run
     * 16-INT8-MAC-per-cycle dot product. ~3× faster than the older
     * widen-to-fp32 + vfmaq_f32 path. Quality cost from input quant is
     * the same as w4a8's, which is well-validated. */
    int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
    int8_t  stack_q[4096];
    float   stack_s[4096 / IB_W4A8_GROUP + 1];
    int8_t *x_q;
    float  *x_sc;
    if (N <= (int)(sizeof stack_q / sizeof *stack_q)) {
        x_q = stack_q; x_sc = stack_s;
    } else {
        x_q  = (int8_t*)malloc((size_t)N);
        x_sc = (float*)malloc((size_t)n_groups * sizeof(float));
    }
    ib_quantize_input_int8_g128(input, x_q, x_sc, N);

    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        float row_acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            int start = g * IB_W4A8_GROUP;
            int end   = (start + IB_W4A8_GROUP > N) ? N : start + IB_W4A8_GROUP;
            int32x4_t acc = vdupq_n_s32(0);
            int j = start;
            for (; j + 15 < end; j += 16) {
                int8x16_t w16 = vld1q_s8(row + j);
                int8x16_t x16 = vld1q_s8(x_q + j);
                acc = vdotq_s32(acc, w16, x16);
            }
            int32_t group_int = vaddvq_s32(acc);
            for (; j < end; j++) group_int += (int32_t)row[j] * (int32_t)x_q[j];
            row_acc += (float)group_int * x_sc[g];
        }
        out[i] = row_acc * scales[i];
    }

    if (x_q != stack_q) free(x_q);
    if (x_sc != stack_s) free(x_sc);
#else
    /* Fallback: original widen-to-fp32 + vfmaq_f32 path for older NEON
     * without the dotprod extension. */
    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 7 < N; j += 8) {
            int8x8_t w8 = vld1_s8(row + j);
            int16x8_t w16 = vmovl_s8(w8);
            int32x4_t w32_lo = vmovl_s16(vget_low_s16(w16));
            int32x4_t w32_hi = vmovl_s16(vget_high_s16(w16));
            float32x4_t wf_lo = vcvtq_f32_s32(w32_lo);
            float32x4_t wf_hi = vcvtq_f32_s32(w32_hi);
            float32x4_t in_lo = vld1q_f32(input + j);
            float32x4_t in_hi = vld1q_f32(input + j + 4);
            acc0 = vfmaq_f32(acc0, wf_lo, in_lo);
            acc1 = vfmaq_f32(acc1, wf_hi, in_hi);
        }

        float32x4_t sum = vaddq_f32(acc0, acc1);
        float result = vaddvq_f32(sum);

        for (; j < N; j++) {
            result += (float)row[j] * input[j];
        }

        out[i] = result * scales[i];
    }
#endif
}

/* ── INT4 matmul ────────────────────────────────────────────── */

static void neon_matmul_int4(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const uint8x8_t mask_lo = vdup_n_u8(0x0F);
    const int8x8_t bias = vdup_n_s8(8);

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        int j = 0;
        /* Process 16 values (8 packed bytes) per iteration */
        for (; j + 15 < N; j += 16) {
            /* Load 8 bytes = 16 nibbles */
            uint8x8_t packed = vld1_u8(row + j / 2);

            /* Extract low and high nibbles */
            uint8x8_t lo_u8 = vand_u8(packed, mask_lo);
            uint8x8_t hi_u8 = vshr_n_u8(packed, 4);

            /* Convert to signed and subtract bias: [0,15] → [-8,7] */
            int8x8_t lo_s8 = vsub_s8(vreinterpret_s8_u8(lo_u8), bias);
            int8x8_t hi_s8 = vsub_s8(vreinterpret_s8_u8(hi_u8), bias);

            /* Interleave: lo[0],hi[0],lo[1],hi[1]... to get original order */
            int8x8x2_t zipped = vzip_s8(lo_s8, hi_s8);

            /* Widen to int16 then int32 then float */
            int16x8_t wide0 = vmovl_s8(zipped.val[0]);
            int16x8_t wide1 = vmovl_s8(zipped.val[1]);

            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide0)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide0)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide1)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide1)));

            /* Load 16 input floats */
            float32x4_t i0 = vld1q_f32(input + j);
            float32x4_t i1 = vld1q_f32(input + j + 4);
            float32x4_t i2 = vld1q_f32(input + j + 8);
            float32x4_t i3 = vld1q_f32(input + j + 12);

            acc0 = vfmaq_f32(acc0, f0, i0);
            acc1 = vfmaq_f32(acc1, f1, i1);
            acc2 = vfmaq_f32(acc2, f2, i2);
            acc3 = vfmaq_f32(acc3, f3, i3);
        }

        float result = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        /* Scalar tail */
        for (; j < N; j += 2) {
            uint8_t byte = row[j / 2];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            result += (float)v0 * input[j];
            if (j + 1 < N) result += (float)v1 * input[j + 1];
        }

        out[i] = result * scales[i];
    }
}

/* ── RMSNorm ────────────────────────────────────────────────── */

static void neon_rmsnorm(
    float* out, const float* input, const float* weight,
    float eps, int N
) {
    float32x4_t ss_acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        ss_acc = vfmaq_f32(ss_acc, v, v);
    }
    float ss = vaddvq_f32(ss_acc);
    for (; i < N; i++) ss += input[i] * input[i];

    float scale = 1.0f / sqrtf(ss / (float)N + eps);
    float32x4_t vscale = vdupq_n_f32(scale);

    i = 0;
    for (; i + 3 < N; i += 4) {
        float32x4_t v = vld1q_f32(input + i);
        float32x4_t w = vld1q_f32(weight + i);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v, vscale), w));
    }
    for (; i < N; i++) {
        out[i] = input[i] * scale * weight[i];
    }
}

/* ── Softmax ────────────────────────────────────────────────── */

static void neon_softmax(float* data, int N) {
    /* Find max */
    float32x4_t vmax = vdupq_n_f32(-INFINITY);
    int i = 0;
    for (; i + 3 < N; i += 4) {
        vmax = vmaxq_f32(vmax, vld1q_f32(data + i));
    }
    float max_val = vmaxvq_f32(vmax);
    for (; i < N; i++) {
        if (data[i] > max_val) max_val = data[i];
    }

    /* exp and sum (scalar — expf hard to vectorize) */
    float sum = 0.0f;
    for (i = 0; i < N; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }

    /* Normalize */
    float32x4_t vinv = vdupq_n_f32(1.0f / sum);
    i = 0;
    for (; i + 3 < N; i += 4) {
        vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), vinv));
    }
    float inv = 1.0f / sum;
    for (; i < N; i++) data[i] *= inv;
}

/* ── SiLU multiply ──────────────────────────────────────────── */

static void neon_silu_mul(float* out, const float* gate, const float* up, int N) {
    int i = 0;
    for (; i + 3 < N; i += 4) {
        float silu_vals[4];
        for (int k = 0; k < 4; k++) {
            float x = gate[i + k];
            silu_vals[k] = x / (1.0f + expf(-x));
        }
        float32x4_t vs = vld1q_f32(silu_vals);
        float32x4_t vu = vld1q_f32(up + i);
        vst1q_f32(out + i, vmulq_f32(vs, vu));
    }
    for (; i < N; i++) {
        float x = gate[i];
        out[i] = (x / (1.0f + expf(-x))) * up[i];
    }
}

/* ── RoPE (scalar — sin/cos not vectorizable) ───────────────── */

static void neon_rope(
    float* q, float* k, int head_dim, int pos, float theta
) {
    for (int i = 0; i < head_dim; i += 2) {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);

        float q0 = q[i], q1 = q[i + 1];
        q[i]     = q0 * cos_a - q1 * sin_a;
        q[i + 1] = q0 * sin_a + q1 * cos_a;

        float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * cos_a - k1 * sin_a;
        k[i + 1] = k0 * sin_a + k1 * cos_a;
    }
}

/* ── W4A8 matmul (INT4 weight × INT8 activation) ─────────────
 *
 * Uses ARMv8.2-A dotprod (`sdot`) — 16 INT8 MACs per instruction.
 * Present on all Apple Silicon and all Neoverse. For older ARMv8.0 we fall
 * back to a pairwise widen+MLA (no sdot), still 2× vs scalar.
 */

#if defined(__ARM_FEATURE_DOTPROD)
#define IB_HAS_DOTPROD 1
#else
#define IB_HAS_DOTPROD 0
#endif

#if IB_HAS_DOTPROD
IB_DOTPROD_NOINLINE
#endif
static void neon_matmul_w4a8(
    float* out, const void* weights, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    const int G = IB_W4A8_GROUP;      /* 128 */
    const int groups = N / G;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float row_acc = 0.0f;

        for (int g = 0; g < groups; g++) {
            const int j0 = g * G;
            int32x4_t acc = vdupq_n_s32(0);

            /* Four iterations × 32 weights = 128 weights per group. */
            for (int k = 0; k < G; k += 32) {
                int j = j0 + k;
                uint8x16_t packed = vld1q_u8(row + j / 2);

                uint8x16_t lo_u8 = vandq_u8(packed, mask_lo);
                uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);
                int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias);
                int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias);
                int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8);

                int8x16_t a0 = vld1q_s8(input + j);
                int8x16_t a1 = vld1q_s8(input + j + 16);

#if IB_HAS_DOTPROD
                acc = vdotq_s32(acc, zipped.val[0], a0);
                acc = vdotq_s32(acc, zipped.val[1], a1);
#else
                int16x8_t p0 = vmull_s8(vget_low_s8(zipped.val[0]), vget_low_s8(a0));
                p0 = vmlal_s8(p0, vget_high_s8(zipped.val[0]), vget_high_s8(a0));
                int16x8_t p1 = vmull_s8(vget_low_s8(zipped.val[1]), vget_low_s8(a1));
                p1 = vmlal_s8(p1, vget_high_s8(zipped.val[1]), vget_high_s8(a1));
                acc = vpadalq_s16(acc, p0);
                acc = vpadalq_s16(acc, p1);
#endif
            }

            int32_t sum = vaddvq_s32(acc);
            row_acc += (float)sum * scales_a[g];
        }

        /* Tail: handle any leftover N % G columns in scalar. Uses the
         * last group's scale (caller guarantees that slot is populated). */
        int tail_start = groups * G;
        if (tail_start < N) {
            int32_t sum = 0;
            for (int j = tail_start; j < N; j += 2) {
                uint8_t byte = row[j / 2];
                int8_t v0 = (int8_t)(byte & 0x0F) - 8;
                int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
                sum += (int32_t)v0 * (int32_t)input[j];
                if (j + 1 < N) sum += (int32_t)v1 * (int32_t)input[j + 1];
            }
            row_acc += (float)sum * scales_a[groups];
        }

        out[i] = row_acc * scales_w[i];
    }
}

/* W4A8 with per-32-element block weight scales — unrolled by 4 blocks.
 *
 * scales_w has length M*(N/32) (one fp32 per 32 weight elements per row).
 * Activation grouping (128) is unchanged.
 *
 * The 4 weight blocks within each 128-element activation group are
 * processed in parallel: 4 separate int32x4_t accumulators, all loads
 * + vdotq issued before any horizontal reduction. The compiler/CPU
 * can then schedule the 4 vdotq pairs (8 instructions) and 4 vaddvq
 * reductions in parallel through Apple Silicon's wide SIMD pipelines,
 * hiding the ~6-cycle reduction latency that previously stalled the
 * loop after each block.
 *
 * Empirically this lifts the per-block-32 throughput close to per-row
 * NEON, eliminating most of the cost of finer-granularity scaling.
 */
#if IB_HAS_DOTPROD
IB_DOTPROD_NOINLINE
#endif
/* Single-row helper: used as the odd-row tail of the 2-row interleave path
 * and for when M < 2. Same semantics as the previous neon_matmul_w4a8_blk32. */
static inline void neon_matmul_w4a8_blk32_one_row(
    float* out_row,
    const uint8_t* row, const float* row_scales,
    const int8_t* input, const float* scales_a,
    int N, const uint8x16_t mask_lo, const int8x16_t bias
) {
    const int G_a = IB_W4A8_GROUP;
    const int groups = N / G_a;
    float row_acc = 0.0f;

    for (int g = 0; g < groups; g++) {
        const int j0 = g * G_a;
        const int wb_base = j0 / 32;
        const float a_scale = scales_a[g];

        int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
        int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

        #define BLOCK1(B, ACC) do { \
            int j = j0 + (B) * 32; \
            uint8x16_t packed = vld1q_u8(row + j / 2); \
            uint8x16_t lo_u8 = vandq_u8(packed, mask_lo); \
            uint8x16_t hi_u8 = vshrq_n_u8(packed, 4); \
            int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias); \
            int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias); \
            int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8); \
            int8x16_t a0 = vld1q_s8(input + j); \
            int8x16_t a1 = vld1q_s8(input + j + 16); \
            ACC = vdotq_s32(ACC, zipped.val[0], a0); \
            ACC = vdotq_s32(ACC, zipped.val[1], a1); \
        } while (0)

        #define BLOCK1_FALLBACK(B, ACC) do { \
            int j = j0 + (B) * 32; \
            uint8x16_t packed = vld1q_u8(row + j / 2); \
            uint8x16_t lo_u8 = vandq_u8(packed, mask_lo); \
            uint8x16_t hi_u8 = vshrq_n_u8(packed, 4); \
            int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias); \
            int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias); \
            int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8); \
            int8x16_t a0 = vld1q_s8(input + j); \
            int8x16_t a1 = vld1q_s8(input + j + 16); \
            int16x8_t p0 = vmull_s8(vget_low_s8(zipped.val[0]), vget_low_s8(a0)); \
            p0 = vmlal_s8(p0, vget_high_s8(zipped.val[0]), vget_high_s8(a0)); \
            int16x8_t p1 = vmull_s8(vget_low_s8(zipped.val[1]), vget_low_s8(a1)); \
            p1 = vmlal_s8(p1, vget_high_s8(zipped.val[1]), vget_high_s8(a1)); \
            ACC = vpadalq_s16(ACC, p0); \
            ACC = vpadalq_s16(ACC, p1); \
        } while (0)

#if IB_HAS_DOTPROD
        BLOCK1(0, acc0); BLOCK1(1, acc1); BLOCK1(2, acc2); BLOCK1(3, acc3);
#else
        BLOCK1_FALLBACK(0, acc0); BLOCK1_FALLBACK(1, acc1);
        BLOCK1_FALLBACK(2, acc2); BLOCK1_FALLBACK(3, acc3);
#endif
        #undef BLOCK1
        #undef BLOCK1_FALLBACK

        int32x4_t sums01 = vpaddq_s32(acc0, acc1);
        int32x4_t sums23 = vpaddq_s32(acc2, acc3);
        int32x4_t all_sums = vpaddq_s32(sums01, sums23);
        float32x4_t fp_sums = vcvtq_f32_s32(all_sums);
        float32x4_t w_scales_v = vld1q_f32(row_scales + wb_base);
        row_acc += vaddvq_f32(vmulq_f32(fp_sums, w_scales_v)) * a_scale;
    }
    *out_row = row_acc;
}

/* 2-row-interleave blk32 W4A8 matmul.
 *
 * Why: the scalar/per-row blk32 kernel is bandwidth-bound on weight + activation
 * loads. By processing two output rows at once we share the activation loads
 * (a0, a1) across both rows for every block, halving activation traffic and
 * letting the dotprod pipeline stay full while weight loads stream from L1.
 *
 * Register budget on M-class (32 NEON regs):
 *   - 4 int32x4_t accumulators × 2 rows = 8 regs (kept live across all blocks
 *     within a group to avoid serial dep on the reduction tree)
 *   - mask_lo + bias = 2 const regs
 *   - per-block transients (packed, lo_u8/hi_u8, lo_s8/hi_s8, zipped, a0/a1)
 *     ≈ 6-8 transient regs
 *   Comfortably fits, no spill.
 *
 * Caller guarantees N % 128 == 0 (and therefore N % 32 == 0). M can be odd —
 * the last row falls through to the single-row helper.
 */
static void neon_matmul_w4a8_blk32(
    float* out, const void* weights, const float* scales_w_blk32,
    const int8_t* input, const float* scales_a, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    const int G_a = IB_W4A8_GROUP;       /* 128 */
    const int n_w_blocks = N / 32;
    const int groups = N / G_a;

    int i = 0;
    for (; i + 1 < M; i += 2) {
        const uint8_t* row0 = w + (size_t)i       * (N / 2);
        const uint8_t* row1 = w + (size_t)(i + 1) * (N / 2);
        const float* scales0 = scales_w_blk32 + (size_t)i       * n_w_blocks;
        const float* scales1 = scales_w_blk32 + (size_t)(i + 1) * n_w_blocks;
        float row0_acc = 0.0f, row1_acc = 0.0f;

        for (int g = 0; g < groups; g++) {
            const int j0 = g * G_a;
            const int wb_base = j0 / 32;
            const float a_scale = scales_a[g];

            int32x4_t r0_b0 = vdupq_n_s32(0), r0_b1 = vdupq_n_s32(0);
            int32x4_t r0_b2 = vdupq_n_s32(0), r0_b3 = vdupq_n_s32(0);
            int32x4_t r1_b0 = vdupq_n_s32(0), r1_b1 = vdupq_n_s32(0);
            int32x4_t r1_b2 = vdupq_n_s32(0), r1_b3 = vdupq_n_s32(0);

            #define BLOCK2(B, R0_ACC, R1_ACC) do { \
                int j = j0 + (B) * 32; \
                int8x16_t a0 = vld1q_s8(input + j); \
                int8x16_t a1 = vld1q_s8(input + j + 16); \
                /* row 0 */ \
                uint8x16_t p0 = vld1q_u8(row0 + j / 2); \
                uint8x16_t lo0 = vandq_u8(p0, mask_lo); \
                uint8x16_t hi0 = vshrq_n_u8(p0, 4); \
                int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(lo0), bias); \
                int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(hi0), bias); \
                int8x16x2_t z0 = vzipq_s8(l0, h0); \
                R0_ACC = vdotq_s32(R0_ACC, z0.val[0], a0); \
                R0_ACC = vdotq_s32(R0_ACC, z0.val[1], a1); \
                /* row 1 — same a0, a1 */ \
                uint8x16_t p1 = vld1q_u8(row1 + j / 2); \
                uint8x16_t lo1 = vandq_u8(p1, mask_lo); \
                uint8x16_t hi1 = vshrq_n_u8(p1, 4); \
                int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(lo1), bias); \
                int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(hi1), bias); \
                int8x16x2_t z1 = vzipq_s8(l1, h1); \
                R1_ACC = vdotq_s32(R1_ACC, z1.val[0], a0); \
                R1_ACC = vdotq_s32(R1_ACC, z1.val[1], a1); \
            } while (0)

            #define BLOCK2_FALLBACK(B, R0_ACC, R1_ACC) do { \
                int j = j0 + (B) * 32; \
                int8x16_t a0 = vld1q_s8(input + j); \
                int8x16_t a1 = vld1q_s8(input + j + 16); \
                uint8x16_t p0 = vld1q_u8(row0 + j / 2); \
                uint8x16_t lo0 = vandq_u8(p0, mask_lo); \
                uint8x16_t hi0 = vshrq_n_u8(p0, 4); \
                int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(lo0), bias); \
                int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(hi0), bias); \
                int8x16x2_t z0 = vzipq_s8(l0, h0); \
                int16x8_t q00 = vmull_s8(vget_low_s8(z0.val[0]), vget_low_s8(a0)); \
                q00 = vmlal_s8(q00, vget_high_s8(z0.val[0]), vget_high_s8(a0)); \
                int16x8_t q01 = vmull_s8(vget_low_s8(z0.val[1]), vget_low_s8(a1)); \
                q01 = vmlal_s8(q01, vget_high_s8(z0.val[1]), vget_high_s8(a1)); \
                R0_ACC = vpadalq_s16(R0_ACC, q00); \
                R0_ACC = vpadalq_s16(R0_ACC, q01); \
                uint8x16_t p1 = vld1q_u8(row1 + j / 2); \
                uint8x16_t lo1 = vandq_u8(p1, mask_lo); \
                uint8x16_t hi1 = vshrq_n_u8(p1, 4); \
                int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(lo1), bias); \
                int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(hi1), bias); \
                int8x16x2_t z1 = vzipq_s8(l1, h1); \
                int16x8_t q10 = vmull_s8(vget_low_s8(z1.val[0]), vget_low_s8(a0)); \
                q10 = vmlal_s8(q10, vget_high_s8(z1.val[0]), vget_high_s8(a0)); \
                int16x8_t q11 = vmull_s8(vget_low_s8(z1.val[1]), vget_low_s8(a1)); \
                q11 = vmlal_s8(q11, vget_high_s8(z1.val[1]), vget_high_s8(a1)); \
                R1_ACC = vpadalq_s16(R1_ACC, q10); \
                R1_ACC = vpadalq_s16(R1_ACC, q11); \
            } while (0)

#if IB_HAS_DOTPROD
            BLOCK2(0, r0_b0, r1_b0);
            BLOCK2(1, r0_b1, r1_b1);
            BLOCK2(2, r0_b2, r1_b2);
            BLOCK2(3, r0_b3, r1_b3);
#else
            BLOCK2_FALLBACK(0, r0_b0, r1_b0);
            BLOCK2_FALLBACK(1, r0_b1, r1_b1);
            BLOCK2_FALLBACK(2, r0_b2, r1_b2);
            BLOCK2_FALLBACK(3, r0_b3, r1_b3);
#endif
            #undef BLOCK2
            #undef BLOCK2_FALLBACK

            /* Reduce row 0 */
            int32x4_t s01_0 = vpaddq_s32(r0_b0, r0_b1);
            int32x4_t s23_0 = vpaddq_s32(r0_b2, r0_b3);
            int32x4_t all_0 = vpaddq_s32(s01_0, s23_0);
            float32x4_t fps_0 = vcvtq_f32_s32(all_0);
            float32x4_t ws_0 = vld1q_f32(scales0 + wb_base);
            row0_acc += vaddvq_f32(vmulq_f32(fps_0, ws_0)) * a_scale;

            /* Reduce row 1 */
            int32x4_t s01_1 = vpaddq_s32(r1_b0, r1_b1);
            int32x4_t s23_1 = vpaddq_s32(r1_b2, r1_b3);
            int32x4_t all_1 = vpaddq_s32(s01_1, s23_1);
            float32x4_t fps_1 = vcvtq_f32_s32(all_1);
            float32x4_t ws_1 = vld1q_f32(scales1 + wb_base);
            row1_acc += vaddvq_f32(vmulq_f32(fps_1, ws_1)) * a_scale;
        }

        out[i]     = row0_acc;
        out[i + 1] = row1_acc;
    }

    /* Odd-M tail: last row uses the single-row helper. */
    if (i < M) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        const float* row_scales = scales_w_blk32 + (size_t)i * n_w_blocks;
        neon_matmul_w4a8_blk32_one_row(&out[i], row, row_scales, input,
                                       scales_a, N, mask_lo, bias);
    }
}

/* Old single-row inline kernel — kept as `_legacy` for differential debugging
 * if a numerical regression is suspected. Currently unused. */
IB_UNUSED
static void neon_matmul_w4a8_blk32_legacy(
    float* out, const void* weights, const float* scales_w_blk32,
    const int8_t* input, const float* scales_a, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    const int G_a = IB_W4A8_GROUP;       /* 128 */
    const int n_w_blocks = N / 32;
    const int groups = N / G_a;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        const float* row_scales = scales_w_blk32 + (size_t)i * n_w_blocks;
        float row_acc = 0.0f;

        for (int g = 0; g < groups; g++) {
            const int j0 = g * G_a;
            const int wb_base = j0 / 32;   /* 4 blocks per group */
            const float a_scale = scales_a[g];

            /* Load + unpack + dot for ALL 4 blocks before any reduction.
             * 4 independent int32 accumulators — register pressure stays
             * within the 32 NEON regs. */
            int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
            int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

            #define BLOCK(B, ACC) do { \
                int j = j0 + (B) * 32; \
                uint8x16_t packed = vld1q_u8(row + j / 2); \
                uint8x16_t lo_u8 = vandq_u8(packed, mask_lo); \
                uint8x16_t hi_u8 = vshrq_n_u8(packed, 4); \
                int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias); \
                int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias); \
                int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8); \
                int8x16_t a0 = vld1q_s8(input + j); \
                int8x16_t a1 = vld1q_s8(input + j + 16); \
                ACC = vdotq_s32(ACC, zipped.val[0], a0); \
                ACC = vdotq_s32(ACC, zipped.val[1], a1); \
            } while (0)

            #define BLOCK_FALLBACK(B, ACC) do { \
                int j = j0 + (B) * 32; \
                uint8x16_t packed = vld1q_u8(row + j / 2); \
                uint8x16_t lo_u8 = vandq_u8(packed, mask_lo); \
                uint8x16_t hi_u8 = vshrq_n_u8(packed, 4); \
                int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias); \
                int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias); \
                int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8); \
                int8x16_t a0 = vld1q_s8(input + j); \
                int8x16_t a1 = vld1q_s8(input + j + 16); \
                int16x8_t p0 = vmull_s8(vget_low_s8(zipped.val[0]), vget_low_s8(a0)); \
                p0 = vmlal_s8(p0, vget_high_s8(zipped.val[0]), vget_high_s8(a0)); \
                int16x8_t p1 = vmull_s8(vget_low_s8(zipped.val[1]), vget_low_s8(a1)); \
                p1 = vmlal_s8(p1, vget_high_s8(zipped.val[1]), vget_high_s8(a1)); \
                ACC = vpadalq_s16(ACC, p0); \
                ACC = vpadalq_s16(ACC, p1); \
            } while (0)

#if IB_HAS_DOTPROD
            BLOCK(0, acc0);
            BLOCK(1, acc1);
            BLOCK(2, acc2);
            BLOCK(3, acc3);
#else
            BLOCK_FALLBACK(0, acc0);
            BLOCK_FALLBACK(1, acc1);
            BLOCK_FALLBACK(2, acc2);
            BLOCK_FALLBACK(3, acc3);
#endif
            #undef BLOCK
            #undef BLOCK_FALLBACK

            /* Vectorized cross-lane reduction: acc{0,1,2,3} each have 4
             * partial sums in 4 lanes. Two vpaddq_s32 trees collapse
             * them into a single int32x4_t [s0, s1, s2, s3]. Then
             * convert + vector-multiply by the 4 weight scales (loaded
             * with one vld1q_f32) + final vaddvq_f32. Replaces 4×
             * vaddvq_s32 + 4 fmuls + 3 fadds with 3 vpaddq + 1 vcvt +
             * 1 vmul + 1 vaddvq — cheaper and pipelines better. */
            int32x4_t sums01 = vpaddq_s32(acc0, acc1);
            int32x4_t sums23 = vpaddq_s32(acc2, acc3);
            int32x4_t all_sums = vpaddq_s32(sums01, sums23);
            float32x4_t fp_sums = vcvtq_f32_s32(all_sums);
            float32x4_t w_scales_v = vld1q_f32(row_scales + wb_base);
            float32x4_t scaled = vmulq_f32(fp_sums, w_scales_v);
            float group_partial = vaddvq_f32(scaled);
            row_acc += group_partial * a_scale;
        }

        /* Caller guarantees N % IB_W4A8_GROUP == 0 (and N % 32 == 0). */
        out[i] = row_acc;
    }
}

/* ── W4A8 batched matmul (shared weights across B activation vectors) ─
 *
 * Weights are loaded once per row and applied against B independent
 * activation vectors. The win comes from amortizing the INT4-unpack cost
 * and, in the memory-bandwidth-bound regime, the weight load itself.
 *
 * Implementation note: accumulators MUST stay in NEON registers. If they
 * spill to stack the batching gain vanishes, because per-sdot memory
 * traffic dominates. We specialize common B (2, 4, 8) with named locals;
 * other B use a generic path that is slower but correct.
 */

/* Inner column-chunk: unpack 32 INT4 weights. Results left in `zL` and
 * `zH` which together hold 32 signed INT8 weights in the natural order. */
#define W4A8_UNPACK32(packed_u8x16, zL, zH) \
    do { \
        uint8x16_t _p = (packed_u8x16); \
        uint8x16_t _lo = vandq_u8(_p, mask_lo); \
        uint8x16_t _hi = vshrq_n_u8(_p, 4); \
        int8x16_t _lo_s = vsubq_s8(vreinterpretq_s8_u8(_lo), bias); \
        int8x16_t _hi_s = vsubq_s8(vreinterpretq_s8_u8(_hi), bias); \
        int8x16x2_t _z = vzipq_s8(_lo_s, _hi_s); \
        (zL) = _z.val[0]; \
        (zH) = _z.val[1]; \
    } while (0)

#if IB_HAS_DOTPROD
#define W4A8_DOT(acc, wL, wH, a0, a1) \
    do { (acc) = vdotq_s32(vdotq_s32((acc), (wL), (a0)), (wH), (a1)); } while (0)
#else
#define W4A8_DOT(acc, wL, wH, a0, a1) \
    do { \
        int16x8_t _p0 = vmull_s8(vget_low_s8(wL),  vget_low_s8(a0));  \
        _p0 = vmlal_s8(_p0, vget_high_s8(wL), vget_high_s8(a0));      \
        int16x8_t _p1 = vmull_s8(vget_low_s8(wH),  vget_low_s8(a1));  \
        _p1 = vmlal_s8(_p1, vget_high_s8(wH), vget_high_s8(a1));      \
        (acc) = vpadalq_s16((acc), _p0);                              \
        (acc) = vpadalq_s16((acc), _p1);                              \
    } while (0)
#endif

#if IB_HAS_DOTPROD
IB_DOTPROD_NOINLINE
#endif
static void neon_matmul_w4a8_batch_b4(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N, int M_stride
) {
    const int G = IB_W4A8_GROUP;
    const int groups = N / G;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);

    const int8_t* in0 = input + 0 * N;
    const int8_t* in1 = input + 1 * N;
    const int8_t* in2 = input + 2 * N;
    const int8_t* in3 = input + 3 * N;
    const float* sa0 = scales_a + 0 * groups;
    const float* sa1 = scales_a + 1 * groups;
    const float* sa2 = scales_a + 2 * groups;
    const float* sa3 = scales_a + 3 * groups;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float r0 = 0, r1 = 0, r2 = 0, r3 = 0;

        for (int g = 0; g < groups; g++) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            int j0 = g * G;

            for (int k = 0; k < G; k += 32) {
                int j = j0 + k;
                uint8x16_t packed = vld1q_u8(row + j / 2);
                int8x16_t zL, zH;
                W4A8_UNPACK32(packed, zL, zH);
                W4A8_DOT(a0, zL, zH, vld1q_s8(in0 + j), vld1q_s8(in0 + j + 16));
                W4A8_DOT(a1, zL, zH, vld1q_s8(in1 + j), vld1q_s8(in1 + j + 16));
                W4A8_DOT(a2, zL, zH, vld1q_s8(in2 + j), vld1q_s8(in2 + j + 16));
                W4A8_DOT(a3, zL, zH, vld1q_s8(in3 + j), vld1q_s8(in3 + j + 16));
            }
            r0 += (float)vaddvq_s32(a0) * sa0[g];
            r1 += (float)vaddvq_s32(a1) * sa1[g];
            r2 += (float)vaddvq_s32(a2) * sa2[g];
            r3 += (float)vaddvq_s32(a3) * sa3[g];
        }

        float sw = scales_w[i];
        out[0 * (size_t)M_stride + i] = r0 * sw;
        out[1 * (size_t)M_stride + i] = r1 * sw;
        out[2 * (size_t)M_stride + i] = r2 * sw;
        out[3 * (size_t)M_stride + i] = r3 * sw;
    }
}

#if IB_HAS_DOTPROD
IB_DOTPROD_NOINLINE
#endif
static void neon_matmul_w4a8_batch_b2(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N, int M_stride
) {
    const int G = IB_W4A8_GROUP;
    const int groups = N / G;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);

    const int8_t* in0 = input + 0 * N;
    const int8_t* in1 = input + 1 * N;
    const float* sa0 = scales_a + 0 * groups;
    const float* sa1 = scales_a + 1 * groups;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float r0 = 0, r1 = 0;

        for (int g = 0; g < groups; g++) {
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int j0 = g * G;
            for (int k = 0; k < G; k += 32) {
                int j = j0 + k;
                uint8x16_t packed = vld1q_u8(row + j / 2);
                int8x16_t zL, zH;
                W4A8_UNPACK32(packed, zL, zH);
                W4A8_DOT(a0, zL, zH, vld1q_s8(in0 + j), vld1q_s8(in0 + j + 16));
                W4A8_DOT(a1, zL, zH, vld1q_s8(in1 + j), vld1q_s8(in1 + j + 16));
            }
            r0 += (float)vaddvq_s32(a0) * sa0[g];
            r1 += (float)vaddvq_s32(a1) * sa1[g];
        }

        float sw = scales_w[i];
        out[0 * (size_t)M_stride + i] = r0 * sw;
        out[1 * (size_t)M_stride + i] = r1 * sw;
    }
}

/* Generic fallback — correct but slower than specialized variants because
 * the compiler spills per-batch accumulators to stack. */
#if IB_HAS_DOTPROD
IB_DOTPROD_NOINLINE
#endif
static void neon_matmul_w4a8_batch_generic(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N, int B, int M_stride
) {
    const int G = IB_W4A8_GROUP;
    const int groups = N / G;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    if (B > 32) B = 32;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float row_acc[32] = {0};
        for (int g = 0; g < groups; g++) {
            int32x4_t acc[32];
            for (int b = 0; b < B; b++) acc[b] = vdupq_n_s32(0);
            int j0 = g * G;
            for (int k = 0; k < G; k += 32) {
                int j = j0 + k;
                uint8x16_t packed = vld1q_u8(row + j / 2);
                int8x16_t zL, zH;
                W4A8_UNPACK32(packed, zL, zH);
                for (int b = 0; b < B; b++) {
                    const int8_t* arow = input + (size_t)b * N;
                    int8x16_t a0 = vld1q_s8(arow + j);
                    int8x16_t a1 = vld1q_s8(arow + j + 16);
                    W4A8_DOT(acc[b], zL, zH, a0, a1);
                }
            }
            for (int b = 0; b < B; b++) {
                int32_t sum = vaddvq_s32(acc[b]);
                row_acc[b] += (float)sum * scales_a[(size_t)b * groups + g];
            }
        }
        float sw = scales_w[i];
        for (int b = 0; b < B; b++) out[(size_t)b * M_stride + i] = row_acc[b] * sw;
    }
}

static void neon_matmul_w4a8_batch(
    float* out, const void* weights, const float* scales_w,
    const int8_t* input, const float* scales_a,
    int M, int N, int B, int M_stride
) {
    const uint8_t* w = (const uint8_t*)weights;
    const int groups = N / IB_W4A8_GROUP;

    if (B == 1) {
        neon_matmul_w4a8(out, weights, scales_w, input, scales_a, M, N);
        return;
    }
    if (B == 2) {
        neon_matmul_w4a8_batch_b2(out, w, scales_w, input, scales_a, M, N, M_stride);
        return;
    }
    if (B == 4) {
        neon_matmul_w4a8_batch_b4(out, w, scales_w, input, scales_a, M, N, M_stride);
        return;
    }
    /* For 3 ≤ B ≤ 8 and larger, decompose into chunks of 4, 2, 1 using
     * specialized helpers. Weights stay hot in L2 across the two passes,
     * so a B=8 case effectively runs as two B=4 calls with near-full
     * batching benefit. */
    int done = 0;
    while (done < B) {
        int rem = B - done;
        const int8_t* in_chunk  = input    + (size_t)done * N;
        const float*  sa_chunk  = scales_a + (size_t)done * groups;
        float*        out_chunk = out      + (size_t)done * M_stride;
        if (rem >= 4) {
            neon_matmul_w4a8_batch_b4(out_chunk, w, scales_w, in_chunk, sa_chunk, M, N, M_stride);
            done += 4;
        } else if (rem == 2 || rem == 3) {
            neon_matmul_w4a8_batch_b2(out_chunk, w, scales_w, in_chunk, sa_chunk, M, N, M_stride);
            done += 2;
        } else { /* rem == 1 */
            neon_matmul_w4a8(out_chunk, weights, scales_w, in_chunk, sa_chunk, M, N);
            done += 1;
        }
    }
}

/* ── Registration ───────────────────────────────────────────── */

/* ── INT8 × FP32 batched matmul ─────────────────────────────
 *
 * Same amortization idea as matmul_w4a8_batch: weights loaded once, applied
 * against B activation vectors. Used by the LM head during spec-decoding
 * verify, where we want B positions' logits from a single pass over the
 * vocab×hidden weight matrix. */
IB_NOINLINE
static void neon_matmul_int8_batch_b4(
    float* out, const int8_t* w, const float* scales_w,
    const float* input, int M, int N, int M_stride
) {
    const float* in0 = input + 0 * N;
    const float* in1 = input + 1 * N;
    const float* in2 = input + 2 * N;
    const float* in3 = input + 3 * N;

    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);

        int j = 0;
        for (; j + 7 < N; j += 8) {
            int8x8_t w8 = vld1_s8(row + j);
            int16x8_t w16 = vmovl_s8(w8);
            float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
            float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));

            a0 = vfmaq_f32(a0, wf_lo, vld1q_f32(in0 + j));
            a0 = vfmaq_f32(a0, wf_hi, vld1q_f32(in0 + j + 4));
            a1 = vfmaq_f32(a1, wf_lo, vld1q_f32(in1 + j));
            a1 = vfmaq_f32(a1, wf_hi, vld1q_f32(in1 + j + 4));
            a2 = vfmaq_f32(a2, wf_lo, vld1q_f32(in2 + j));
            a2 = vfmaq_f32(a2, wf_hi, vld1q_f32(in2 + j + 4));
            a3 = vfmaq_f32(a3, wf_lo, vld1q_f32(in3 + j));
            a3 = vfmaq_f32(a3, wf_hi, vld1q_f32(in3 + j + 4));
        }
        float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
        float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
        for (; j < N; j++) {
            float wv = (float)row[j];
            s0 += wv * in0[j]; s1 += wv * in1[j];
            s2 += wv * in2[j]; s3 += wv * in3[j];
        }
        float sw = scales_w[i];
        out[0 * (size_t)M_stride + i] = s0 * sw;
        out[1 * (size_t)M_stride + i] = s1 * sw;
        out[2 * (size_t)M_stride + i] = s2 * sw;
        out[3 * (size_t)M_stride + i] = s3 * sw;
    }
}

IB_NOINLINE
static void neon_matmul_int8_batch_b2(
    float* out, const int8_t* w, const float* scales_w,
    const float* input, int M, int N, int M_stride
) {
    const float* in0 = input + 0 * N;
    const float* in1 = input + 1 * N;

    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        int j = 0;
        for (; j + 7 < N; j += 8) {
            int8x8_t w8 = vld1_s8(row + j);
            int16x8_t w16 = vmovl_s8(w8);
            float32x4_t wf_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
            float32x4_t wf_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
            a0 = vfmaq_f32(a0, wf_lo, vld1q_f32(in0 + j));
            a0 = vfmaq_f32(a0, wf_hi, vld1q_f32(in0 + j + 4));
            a1 = vfmaq_f32(a1, wf_lo, vld1q_f32(in1 + j));
            a1 = vfmaq_f32(a1, wf_hi, vld1q_f32(in1 + j + 4));
        }
        float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
        for (; j < N; j++) {
            float wv = (float)row[j];
            s0 += wv * in0[j]; s1 += wv * in1[j];
        }
        float sw = scales_w[i];
        out[0 * (size_t)M_stride + i] = s0 * sw;
        out[1 * (size_t)M_stride + i] = s1 * sw;
    }
}

static void neon_matmul_int8_batch(
    float* out, const void* weights, const float* scales_w,
    const float* input, int M, int N, int B, int M_stride
) {
    const int8_t* w = (const int8_t*)weights;
    int done = 0;
    while (done < B) {
        int rem = B - done;
        const float* in_chunk = input + (size_t)done * N;
        float*       out_chunk = out  + (size_t)done * M_stride;
        if (rem >= 4) {
            neon_matmul_int8_batch_b4(out_chunk, w, scales_w, in_chunk, M, N, M_stride);
            done += 4;
        } else if (rem >= 2) {
            neon_matmul_int8_batch_b2(out_chunk, w, scales_w, in_chunk, M, N, M_stride);
            done += 2;
        } else {
            neon_matmul_int8(out_chunk, w, scales_w, in_chunk, M, N);
            done += 1;
        }
    }
}

void ib_init_kernels_neon(ib_kernels* kern) {
    kern->matmul_int4 = neon_matmul_int4;
    kern->matmul_int8 = neon_matmul_int8;
    kern->matmul_w4a8 = neon_matmul_w4a8;
    kern->matmul_w4a8_batch = neon_matmul_w4a8_batch;
    kern->matmul_w4a8_blk32 = neon_matmul_w4a8_blk32;
    kern->matmul_int8_batch = neon_matmul_int8_batch;
    kern->rmsnorm     = neon_rmsnorm;
    kern->rope        = neon_rope;
    kern->softmax     = neon_softmax;
    kern->silu_mul    = neon_silu_mul;
}

#else
/* Not ARM64 — provide stub */
#include "../inferbit_internal.h"
void ib_init_kernels_neon(ib_kernels* kern) { (void)kern; }
#endif
