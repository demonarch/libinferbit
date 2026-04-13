/*
 * neon.c — ARM NEON optimized kernels
 *
 * For Apple Silicon and other ARM64 platforms.
 */

#if defined(__aarch64__) || defined(_M_ARM64)

#include "../inferbit_internal.h"
#include <arm_neon.h>
#include <math.h>
#include <string.h>

/* ── INT8 matmul ────────────────────────────────────────────── */

static void neon_matmul_int8(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const int8_t* w = (const int8_t*)weights;

    for (int i = 0; i < M; i++) {
        const int8_t* row = w + (size_t)i * N;
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 7 < N; j += 8) {
            /* Load 8 INT8 weights */
            int8x8_t w8 = vld1_s8(row + j);
            /* Widen to INT16 */
            int16x8_t w16 = vmovl_s8(w8);
            /* Split and widen to INT32 */
            int32x4_t w32_lo = vmovl_s16(vget_low_s16(w16));
            int32x4_t w32_hi = vmovl_s16(vget_high_s16(w16));
            /* Convert to float */
            float32x4_t wf_lo = vcvtq_f32_s32(w32_lo);
            float32x4_t wf_hi = vcvtq_f32_s32(w32_hi);

            /* Load 8 input floats */
            float32x4_t in_lo = vld1q_f32(input + j);
            float32x4_t in_hi = vld1q_f32(input + j + 4);

            /* FMA */
            acc0 = vfmaq_f32(acc0, wf_lo, in_lo);
            acc1 = vfmaq_f32(acc1, wf_hi, in_hi);
        }

        /* Horizontal sum */
        float32x4_t sum = vaddq_f32(acc0, acc1);
        float result = vaddvq_f32(sum);

        /* Scalar tail */
        for (; j < N; j++) {
            result += (float)row[j] * input[j];
        }

        out[i] = result * scales[i];
    }
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
__attribute__((target("dotprod"),noinline))
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
__attribute__((target("dotprod"),noinline))
#endif
static void neon_matmul_w4a8_batch_b4(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N
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
        out[0 * M + i] = r0 * sw;
        out[1 * M + i] = r1 * sw;
        out[2 * M + i] = r2 * sw;
        out[3 * M + i] = r3 * sw;
    }
}

#if IB_HAS_DOTPROD
__attribute__((target("dotprod"),noinline))
#endif
static void neon_matmul_w4a8_batch_b2(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N
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
        out[0 * M + i] = r0 * sw;
        out[1 * M + i] = r1 * sw;
    }
}

/* Generic fallback — correct but slower than specialized variants because
 * the compiler spills per-batch accumulators to stack. */
#if IB_HAS_DOTPROD
__attribute__((target("dotprod"),noinline))
#endif
static void neon_matmul_w4a8_batch_generic(
    float* out, const uint8_t* w, const float* scales_w,
    const int8_t* input, const float* scales_a, int M, int N, int B
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
        for (int b = 0; b < B; b++) out[(size_t)b * M + i] = row_acc[b] * sw;
    }
}

static void neon_matmul_w4a8_batch(
    float* out, const void* weights, const float* scales_w,
    const int8_t* input, const float* scales_a,
    int M, int N, int B
) {
    const uint8_t* w = (const uint8_t*)weights;
    const int groups = N / IB_W4A8_GROUP;

    if (B == 1) {
        neon_matmul_w4a8(out, weights, scales_w, input, scales_a, M, N);
        return;
    }
    if (B == 2) {
        neon_matmul_w4a8_batch_b2(out, w, scales_w, input, scales_a, M, N);
        return;
    }
    if (B == 4) {
        neon_matmul_w4a8_batch_b4(out, w, scales_w, input, scales_a, M, N);
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
        float*        out_chunk = out      + (size_t)done * M;
        if (rem >= 4) {
            neon_matmul_w4a8_batch_b4(out_chunk, w, scales_w, in_chunk, sa_chunk, M, N);
            done += 4;
        } else if (rem == 2 || rem == 3) {
            neon_matmul_w4a8_batch_b2(out_chunk, w, scales_w, in_chunk, sa_chunk, M, N);
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
__attribute__((noinline))
static void neon_matmul_int8_batch_b4(
    float* out, const int8_t* w, const float* scales_w,
    const float* input, int M, int N
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
        out[0 * M + i] = s0 * sw;
        out[1 * M + i] = s1 * sw;
        out[2 * M + i] = s2 * sw;
        out[3 * M + i] = s3 * sw;
    }
}

__attribute__((noinline))
static void neon_matmul_int8_batch_b2(
    float* out, const int8_t* w, const float* scales_w,
    const float* input, int M, int N
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
        out[0 * M + i] = s0 * sw;
        out[1 * M + i] = s1 * sw;
    }
}

static void neon_matmul_int8_batch(
    float* out, const void* weights, const float* scales_w,
    const float* input, int M, int N, int B
) {
    const int8_t* w = (const int8_t*)weights;
    int done = 0;
    while (done < B) {
        int rem = B - done;
        const float* in_chunk = input + (size_t)done * N;
        float*       out_chunk = out  + (size_t)done * M;
        if (rem >= 4) {
            neon_matmul_int8_batch_b4(out_chunk, w, scales_w, in_chunk, M, N);
            done += 4;
        } else if (rem >= 2) {
            neon_matmul_int8_batch_b2(out_chunk, w, scales_w, in_chunk, M, N);
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
