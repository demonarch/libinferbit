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
__attribute__((target("dotprod")))
#endif
static void neon_matmul_w4a8(
    float* out, const void* weights, const float* scales_w,
    const int8_t* input, float scale_a, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        int32x4_t acc = vdupq_n_s32(0);

        int j = 0;
        /* Process 32 weights (16 packed bytes) per iteration. */
        for (; j + 31 < N; j += 32) {
            /* Load 16 packed bytes = 32 nibbles. */
            uint8x16_t packed = vld1q_u8(row + j / 2);

            /* Split nibbles. */
            uint8x16_t lo_u8 = vandq_u8(packed, mask_lo);
            uint8x16_t hi_u8 = vshrq_n_u8(packed, 4);
            int8x16_t lo_s8 = vsubq_s8(vreinterpretq_s8_u8(lo_u8), bias);
            int8x16_t hi_s8 = vsubq_s8(vreinterpretq_s8_u8(hi_u8), bias);

            /* Interleave back into original column order:
             * nibble j   = lo_s8[k], j+1 = hi_s8[k] (for k = j/2 mod 16). */
            int8x16x2_t zipped = vzipq_s8(lo_s8, hi_s8);

            /* Load 32 INT8 activations. */
            int8x16_t a0 = vld1q_s8(input + j);
            int8x16_t a1 = vld1q_s8(input + j + 16);

#if IB_HAS_DOTPROD
            acc = vdotq_s32(acc, zipped.val[0], a0);
            acc = vdotq_s32(acc, zipped.val[1], a1);
#else
            /* Fallback: widen + multiply + accumulate. */
            int16x8_t p0 = vmull_s8(vget_low_s8(zipped.val[0]), vget_low_s8(a0));
            p0 = vmlal_s8(p0, vget_high_s8(zipped.val[0]), vget_high_s8(a0));
            int16x8_t p1 = vmull_s8(vget_low_s8(zipped.val[1]), vget_low_s8(a1));
            p1 = vmlal_s8(p1, vget_high_s8(zipped.val[1]), vget_high_s8(a1));
            acc = vpadalq_s16(acc, p0);
            acc = vpadalq_s16(acc, p1);
#endif
        }

        int32_t sum = vaddvq_s32(acc);

        /* Scalar tail. */
        for (; j < N; j += 2) {
            uint8_t byte = row[j / 2];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            sum += (int32_t)v0 * (int32_t)input[j];
            if (j + 1 < N) sum += (int32_t)v1 * (int32_t)input[j + 1];
        }

        out[i] = (float)sum * scales_w[i] * scale_a;
    }
}

/* ── Registration ───────────────────────────────────────────── */

void ib_init_kernels_neon(ib_kernels* kern) {
    kern->matmul_int4 = neon_matmul_int4;
    kern->matmul_int8 = neon_matmul_int8;
    kern->matmul_w4a8 = neon_matmul_w4a8;
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
