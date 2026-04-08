#include "../inferbit_internal.h"
#include <math.h>
#include <string.h>

/* ── Scalar fallback kernels ────────────────────────────────── */
/* These are the reference implementations. Correct but slow.    */
/* SIMD-optimized versions will replace them via dispatch table. */

static void scalar_matmul_int4(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j += 2) {
            int idx = i * (N / 2) + (j / 2);
            int8_t v0 = (int8_t)(w[idx] & 0x0F) - 8;
            int8_t v1 = (int8_t)((w[idx] >> 4) & 0x0F) - 8;
            sum += (float)v0 * input[j];
            if (j + 1 < N) {
                sum += (float)v1 * input[j + 1];
            }
        }
        out[i] = sum * scales[i];
    }
}

static void scalar_matmul_int8(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const int8_t* w = (const int8_t*)weights;
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += (float)w[i * N + j] * input[j];
        }
        out[i] = sum * scales[i];
    }
}

static void scalar_rmsnorm(
    float* out, const float* input, const float* weight,
    float eps, int N
) {
    float ss = 0.0f;
    for (int i = 0; i < N; i++) {
        ss += input[i] * input[i];
    }
    ss = 1.0f / sqrtf(ss / (float)N + eps);
    for (int i = 0; i < N; i++) {
        out[i] = input[i] * ss * weight[i];
    }
}

static void scalar_rope(
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

static void scalar_softmax(float* data, int N) {
    float max_val = data[0];
    for (int i = 1; i < N; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < N; i++) {
        data[i] *= inv_sum;
    }
}

static void scalar_silu_mul(float* out, const float* gate, const float* up, int N) {
    for (int i = 0; i < N; i++) {
        float x = gate[i];
        float silu = x / (1.0f + expf(-x));
        out[i] = silu * up[i];
    }
}

/* ── Kernel dispatch table ──────────────────────────────────── */

ib_kernels ib_kern;

/* Defined in avx2.c and neon.c */
void ib_init_kernels_avx2(ib_kernels* kern);
void ib_init_kernels_neon(ib_kernels* kern);

void ib_init_kernels(ib_simd_level level) {
    /* Always start with scalar fallbacks */
    ib_kern.matmul_int4 = scalar_matmul_int4;
    ib_kern.matmul_int8 = scalar_matmul_int8;
    ib_kern.rmsnorm     = scalar_rmsnorm;
    ib_kern.rope        = scalar_rope;
    ib_kern.softmax     = scalar_softmax;
    ib_kern.silu_mul    = scalar_silu_mul;

    /* Override with SIMD kernels if available */
    switch (level) {
        case IB_SIMD_AVX512:
            /* TODO: AVX-512 specific kernels */
            /* Fall through to AVX2 for now */
        case IB_SIMD_AVX2:
            ib_init_kernels_avx2(&ib_kern);
            break;
        case IB_SIMD_NEON:
            ib_init_kernels_neon(&ib_kern);
            break;
        case IB_SIMD_NONE:
        default:
            break;
    }
}
