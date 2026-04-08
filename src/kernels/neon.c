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

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 2);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 7 < N; j += 8) {
            /* Load 4 bytes = 8 nibbles */
            /* Unpack to int32 */
            int32_t vals[8];
            for (int k = 0; k < 4; k++) {
                uint8_t byte = row[j / 2 + k];
                vals[k * 2]     = (int32_t)(byte & 0x0F) - 8;
                vals[k * 2 + 1] = (int32_t)((byte >> 4) & 0x0F) - 8;
            }

            float32x4_t wf_lo = vcvtq_f32_s32(vld1q_s32(vals));
            float32x4_t wf_hi = vcvtq_f32_s32(vld1q_s32(vals + 4));

            float32x4_t in_lo = vld1q_f32(input + j);
            float32x4_t in_hi = vld1q_f32(input + j + 4);

            acc0 = vfmaq_f32(acc0, wf_lo, in_lo);
            acc1 = vfmaq_f32(acc1, wf_hi, in_hi);
        }

        float result = vaddvq_f32(vaddq_f32(acc0, acc1));

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

/* ── Registration ───────────────────────────────────────────── */

void ib_init_kernels_neon(ib_kernels* kern) {
    kern->matmul_int4 = neon_matmul_int4;
    kern->matmul_int8 = neon_matmul_int8;
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
