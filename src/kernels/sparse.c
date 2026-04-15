/*
 * sparse.c — Column-sparse matmul kernels for DejaVu-style activation sparsity.
 *
 * Given:
 *   weights_cm:   column-major INT4-packed [N cols, each M/2 bytes]
 *   scales_w:     per-row FP16-source scales (already FP32), length M
 *   active_idx:   indices into the N columns that are active (non-zero input)
 *   active_vals:  corresponding input values at those columns
 *   n_active:     number of active columns (≤ N)
 *
 * Compute: out[i] = scales_w[i] * sum over a in 0..n_active-1
 *                                   of W[i, active_idx[a]] * active_vals[a]
 *
 * Out buffer must be zeroed by caller. Typical n_active is 10-40% of N;
 * below that, the sparse kernel dominates over dense.
 */

#include "../inferbit_internal.h"
#include <math.h>
#include <string.h>

/* Scalar reference — correct, but slow. Used for:
 *  (1) correctness comparison when developing SIMD variants
 *  (2) fallback on platforms without SIMD
 *
 * Layout: column j's weights occupy bytes weights_cm[j * (M/2) .. (j+1) * (M/2) - 1].
 *         Within that column, byte b encodes weights for rows (2b, 2b+1) as
 *         low nibble and high nibble respectively, in signed-INT4 via subtract-8. */
void ib_matmul_int4_col_sparse_scalar(
    float*         out,
    const uint8_t* weights_cm,
    const float*   scales_w,
    const int*     active_idx,
    const float*   active_vals,
    int M, int N, int n_active
) {
    (void)N;  /* N is not needed directly; column stride = M/2 */
    const int col_stride = M / 2;

    /* Accumulate per-column contributions. Out starts zeroed. */
    for (int a = 0; a < n_active; a++) {
        int j = active_idx[a];
        float v = active_vals[a];
        const uint8_t* col = weights_cm + (size_t)j * col_stride;
        for (int i = 0; i < M; i += 2) {
            uint8_t byte = col[i / 2];
            int8_t  w0 = (int8_t)(byte & 0x0F) - 8;   /* row i     */
            int8_t  w1 = (int8_t)((byte >> 4) & 0x0F) - 8;  /* row i+1 */
            out[i]     += (float)w0 * v;
            out[i + 1] += (float)w1 * v;
        }
    }

    /* Apply per-row scales at the end (cheaper than during accumulation). */
    for (int i = 0; i < M; i++) out[i] *= scales_w[i];
}

/* NEON SIMD variant — processes 32 rows per inner iteration. */
#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

void ib_matmul_int4_col_sparse_neon(
    float*         out,
    const uint8_t* weights_cm,
    const float*   scales_w,
    const int*     active_idx,
    const float*   active_vals,
    int M, int N, int n_active
) {
    (void)N;
    const int col_stride = M / 2;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t  bias    = vdupq_n_s8(8);

    /* For each active column, stream its packed bytes and FMA into out. */
    for (int a = 0; a < n_active; a++) {
        int j = active_idx[a];
        float v_scalar = active_vals[a];
        float32x4_t vv = vdupq_n_f32(v_scalar);
        const uint8_t* col = weights_cm + (size_t)j * col_stride;

        int i = 0;
        /* Main body: 32 rows at a time (16 packed bytes). */
        for (; i + 31 < M; i += 32) {
            uint8x16_t packed = vld1q_u8(col + i / 2);
            uint8x16_t lo_u = vandq_u8(packed, mask_lo);
            uint8x16_t hi_u = vshrq_n_u8(packed, 4);
            int8x16_t lo_s = vsubq_s8(vreinterpretq_s8_u8(lo_u), bias);
            int8x16_t hi_s = vsubq_s8(vreinterpretq_s8_u8(hi_u), bias);
            int8x16x2_t zipped = vzipq_s8(lo_s, hi_s);

            /* Widen to int16 then int32, convert to float, FMA with v. */
            int16x8_t w0 = vmovl_s8(vget_low_s8(zipped.val[0]));
            int16x8_t w1 = vmovl_s8(vget_high_s8(zipped.val[0]));
            int16x8_t w2 = vmovl_s8(vget_low_s8(zipped.val[1]));
            int16x8_t w3 = vmovl_s8(vget_high_s8(zipped.val[1]));

            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1)));
            float32x4_t f4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w2)));
            float32x4_t f5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w2)));
            float32x4_t f6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w3)));
            float32x4_t f7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w3)));

            float32x4_t o0 = vld1q_f32(out + i);
            float32x4_t o1 = vld1q_f32(out + i + 4);
            float32x4_t o2 = vld1q_f32(out + i + 8);
            float32x4_t o3 = vld1q_f32(out + i + 12);
            float32x4_t o4 = vld1q_f32(out + i + 16);
            float32x4_t o5 = vld1q_f32(out + i + 20);
            float32x4_t o6 = vld1q_f32(out + i + 24);
            float32x4_t o7 = vld1q_f32(out + i + 28);

            vst1q_f32(out + i,      vfmaq_f32(o0, f0, vv));
            vst1q_f32(out + i + 4,  vfmaq_f32(o1, f1, vv));
            vst1q_f32(out + i + 8,  vfmaq_f32(o2, f2, vv));
            vst1q_f32(out + i + 12, vfmaq_f32(o3, f3, vv));
            vst1q_f32(out + i + 16, vfmaq_f32(o4, f4, vv));
            vst1q_f32(out + i + 20, vfmaq_f32(o5, f5, vv));
            vst1q_f32(out + i + 24, vfmaq_f32(o6, f6, vv));
            vst1q_f32(out + i + 28, vfmaq_f32(o7, f7, vv));
        }
        /* Scalar tail. */
        for (; i < M; i += 2) {
            uint8_t byte = col[i / 2];
            int8_t w0 = (int8_t)(byte & 0x0F) - 8;
            int8_t w1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            out[i]     += (float)w0 * v_scalar;
            if (i + 1 < M) out[i + 1] += (float)w1 * v_scalar;
        }
    }

    /* Final per-row scale. */
    int i = 0;
    for (; i + 3 < M; i += 4) {
        float32x4_t o = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(scales_w + i);
        vst1q_f32(out + i, vmulq_f32(o, s));
    }
    for (; i < M; i++) out[i] *= scales_w[i];
}

#endif /* __aarch64__ */
