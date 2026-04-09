/*
 * ternary.c — INT2 ternary matmul kernels
 *
 * Weights are {-1, 0, +1} packed 4 per byte.
 * Encoding: 0 = -1, 1 = 0, 2 = +1 (stored in 2 bits)
 *
 * Key optimization: no multiply needed. For each weight:
 *   +1 → add input to accumulator
 *   -1 → subtract input from accumulator
 *    0 → nothing
 *
 * NEON version uses mask-based selection for branchless execution.
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
            int v0 = (byte & 0x03) - 1;
            int v1 = ((byte >> 2) & 0x03) - 1;
            int v2 = ((byte >> 4) & 0x03) - 1;
            int v3 = ((byte >> 6) & 0x03) - 1;

            /* Branchless: v * input is just add/subtract/zero since v is {-1,0,1} */
            sum += (float)v0 * input[j];
            if (j + 1 < N) sum += (float)v1 * input[j + 1];
            if (j + 2 < N) sum += (float)v2 * input[j + 2];
            if (j + 3 < N) sum += (float)v3 * input[j + 3];
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
        float32x4_t acc2 = vdupq_n_f32(0.0f);
        float32x4_t acc3 = vdupq_n_f32(0.0f);

        int j = 0;
        /* Process 16 values (4 bytes) per iteration */
        for (; j + 15 < N; j += 16) {
            uint8_t b0 = row[j / 4];
            uint8_t b1 = row[j / 4 + 1];
            uint8_t b2 = row[j / 4 + 2];
            uint8_t b3 = row[j / 4 + 3];

            /*
             * Branchless ternary using masks:
             * For each 2-bit value v: encoding 0=-1, 1=0, 2=+1
             *
             * pos_mask = (v == 2)  → select +input
             * neg_mask = (v == 0)  → select -input
             * zero: neither mask set → contributes 0
             *
             * result = (pos_mask & input) + (neg_mask & -input)
             *        = input * (+1 or -1 or 0)
             *
             * We use vbslq_f32 (bitwise select): picks input or -input or zero.
             * But the simplest branchless approach: convert {0,1,2}→{-1,0,+1} as float,
             * then multiply. On NEON, FMA throughput is high enough that the multiply
             * by {-1,0,1} is essentially free. The real win is 4x less data to load.
             */

            /* Unpack 4 bytes → 16 ternary values → 16 floats */
            /* Unrolled for speed */
            float32x4_t w0, w1, w2, w3;

            {
                int32_t v[4] = {
                    (b0 & 0x03) - 1, ((b0 >> 2) & 0x03) - 1,
                    ((b0 >> 4) & 0x03) - 1, ((b0 >> 6) & 0x03) - 1
                };
                w0 = vcvtq_f32_s32(vld1q_s32(v));
            }
            {
                int32_t v[4] = {
                    (b1 & 0x03) - 1, ((b1 >> 2) & 0x03) - 1,
                    ((b1 >> 4) & 0x03) - 1, ((b1 >> 6) & 0x03) - 1
                };
                w1 = vcvtq_f32_s32(vld1q_s32(v));
            }
            {
                int32_t v[4] = {
                    (b2 & 0x03) - 1, ((b2 >> 2) & 0x03) - 1,
                    ((b2 >> 4) & 0x03) - 1, ((b2 >> 6) & 0x03) - 1
                };
                w2 = vcvtq_f32_s32(vld1q_s32(v));
            }
            {
                int32_t v[4] = {
                    (b3 & 0x03) - 1, ((b3 >> 2) & 0x03) - 1,
                    ((b3 >> 4) & 0x03) - 1, ((b3 >> 6) & 0x03) - 1
                };
                w3 = vcvtq_f32_s32(vld1q_s32(v));
            }

            /* Load 16 input floats */
            float32x4_t i0 = vld1q_f32(input + j);
            float32x4_t i1 = vld1q_f32(input + j + 4);
            float32x4_t i2 = vld1q_f32(input + j + 8);
            float32x4_t i3 = vld1q_f32(input + j + 12);

            /* FMA: weight * input. Since weight is {-1,0,+1}, this is
             * effectively add/subtract/zero per element */
            acc0 = vfmaq_f32(acc0, w0, i0);
            acc1 = vfmaq_f32(acc1, w1, i1);
            acc2 = vfmaq_f32(acc2, w2, i2);
            acc3 = vfmaq_f32(acc3, w3, i3);
        }

        float result = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));

        /* Scalar tail */
        for (; j < N; j += 4) {
            if (j / 4 >= (int)((size_t)i * (N / 4) + N / 4)) break;
            uint8_t byte = row[j / 4];
            int v0 = (byte & 0x03) - 1;
            int v1 = ((byte >> 2) & 0x03) - 1;
            int v2 = ((byte >> 4) & 0x03) - 1;
            int v3 = ((byte >> 6) & 0x03) - 1;
            result += (float)v0 * input[j];
            if (j + 1 < N) result += (float)v1 * input[j + 1];
            if (j + 2 < N) result += (float)v2 * input[j + 2];
            if (j + 3 < N) result += (float)v3 * input[j + 3];
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
