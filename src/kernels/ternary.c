/*
 * ternary.c — INT2 ternary matmul kernels
 *
 * Weights are {-1, 0, +1} packed 4 per byte.
 * Encoding: 0 = -1, 1 = 0, 2 = +1 (stored in 2 bits)
 *
 * Matmul becomes:
 *   for each weight: if +1, add input; if -1, subtract input; if 0, skip
 *
 * No FMA/multiply needed — only addition and subtraction.
 * This is where BitNet's performance advantage comes from.
 */

#include "../inferbit_internal.h"
#include <string.h>

/* ── Scalar ternary matmul ──────────────────────────────────── */

static void scalar_matmul_int2(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 4);
        float sum = 0.0f;

        for (int j = 0; j < N; j += 4) {
            uint8_t byte = row[j / 4];
            /* Extract 4 ternary values: each 2 bits, 0=-1, 1=0, 2=+1 */
            int v0 = (byte & 0x03) - 1;
            int v1 = ((byte >> 2) & 0x03) - 1;
            int v2 = ((byte >> 4) & 0x03) - 1;
            int v3 = ((byte >> 6) & 0x03) - 1;

            /* Add/subtract/skip — no multiply */
            if (v0 == 1)       sum += input[j];
            else if (v0 == -1) sum -= input[j];

            if (j + 1 < N) {
                if (v1 == 1)       sum += input[j + 1];
                else if (v1 == -1) sum -= input[j + 1];
            }
            if (j + 2 < N) {
                if (v2 == 1)       sum += input[j + 2];
                else if (v2 == -1) sum -= input[j + 2];
            }
            if (j + 3 < N) {
                if (v3 == 1)       sum += input[j + 3];
                else if (v3 == -1) sum -= input[j + 3];
            }
        }

        out[i] = sum * scales[i];
    }
}

/* ── NEON ternary matmul ────────────────────────────────────── */

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

static void neon_matmul_int2(
    float* out, const void* weights, const float* scales,
    const float* input, int M, int N
) {
    const uint8_t* w = (const uint8_t*)weights;

    for (int i = 0; i < M; i++) {
        const uint8_t* row = w + (size_t)i * (N / 4);
        float32x4_t acc0 = vdupq_n_f32(0.0f);
        float32x4_t acc1 = vdupq_n_f32(0.0f);

        int j = 0;
        /* Process 8 values (2 bytes) per iteration */
        for (; j + 7 < N; j += 8) {
            /* Load 2 bytes = 8 ternary values */
            uint8_t b0 = row[j / 4];
            uint8_t b1 = row[j / 4 + 1];

            /* Unpack to int32 and convert: 0→-1, 1→0, 2→+1 */
            int32_t vals[8];
            vals[0] = (int32_t)(b0 & 0x03) - 1;
            vals[1] = (int32_t)((b0 >> 2) & 0x03) - 1;
            vals[2] = (int32_t)((b0 >> 4) & 0x03) - 1;
            vals[3] = (int32_t)((b0 >> 6) & 0x03) - 1;
            vals[4] = (int32_t)(b1 & 0x03) - 1;
            vals[5] = (int32_t)((b1 >> 2) & 0x03) - 1;
            vals[6] = (int32_t)((b1 >> 4) & 0x03) - 1;
            vals[7] = (int32_t)((b1 >> 6) & 0x03) - 1;

            /* Convert to float: only {-1, 0, +1} so this is fast */
            float32x4_t w0 = vcvtq_f32_s32(vld1q_s32(vals));
            float32x4_t w1 = vcvtq_f32_s32(vld1q_s32(vals + 4));

            /* Load input */
            float32x4_t i0 = vld1q_f32(input + j);
            float32x4_t i1 = vld1q_f32(input + j + 4);

            /* Multiply-accumulate (multiply by {-1,0,+1} is just sign flip) */
            acc0 = vfmaq_f32(acc0, w0, i0);
            acc1 = vfmaq_f32(acc1, w1, i1);
        }

        float result = vaddvq_f32(vaddq_f32(acc0, acc1));

        /* Scalar tail */
        for (; j < N; j += 4) {
            uint8_t byte = row[j / 4];
            int v0 = (byte & 0x03) - 1;
            int v1 = ((byte >> 2) & 0x03) - 1;
            int v2 = ((byte >> 4) & 0x03) - 1;
            int v3 = ((byte >> 6) & 0x03) - 1;
            if (v0)            result += v0 * input[j];
            if (v1 && j+1 < N) result += v1 * input[j+1];
            if (v2 && j+2 < N) result += v2 * input[j+2];
            if (v3 && j+3 < N) result += v3 * input[j+3];
        }

        out[i] = result * scales[i];
    }
}

#endif

/* ── Registration ───────────────────────────────────────────── */

void ib_init_kernels_int2(ib_kernels* kern) {
#if defined(__aarch64__) || defined(_M_ARM64)
    kern->matmul_int2 = neon_matmul_int2;
#else
    kern->matmul_int2 = scalar_matmul_int2;
#endif
}
