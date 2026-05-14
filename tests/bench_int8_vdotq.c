/* Honest INT8 NEON ceiling: a proper vdotq_s32-based int8 matmul vs
 * libinferbit's existing scalar-widen-FMA int8 path. Tells us what an
 * INT8 dot-product reformulation of PQv2 could ACTUALLY achieve.
 *
 * Two variants tested:
 *   (a) libinferbit's matmul_int8 (current — fp32 FMA after widening)
 *   (b) proper int8 matmul w/ vdotq_s32 (Apple Silicon dotprod ext)
 *
 * Same input data, same M×N, same warmup. Reports cycles/element ratio.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* (a) Libinferbit-style: widen + fp32 FMA. Same algorithm as
 * src/kernels/neon.c::neon_matmul_int8. */
static void int8_matmul_libinferbit(
    float *out, const int8_t *w, const float *w_scale,
    const float *x, int M, int N)
{
    for (int i = 0; i < M; i++) {
        const int8_t *row = w + (size_t)i * N;
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
            float32x4_t in_lo = vld1q_f32(x + j);
            float32x4_t in_hi = vld1q_f32(x + j + 4);
            acc0 = vfmaq_f32(acc0, wf_lo, in_lo);
            acc1 = vfmaq_f32(acc1, wf_hi, in_hi);
        }
        float32x4_t sum = vaddq_f32(acc0, acc1);
        float result = vaddvq_f32(sum);
        out[i] = result * w_scale[i];
    }
}

/* (b) Honest path: vdotq_s32 over int8 w × int8 x_quant. Per-row weight
 * scale + per-column input scale. Mirror w4a8 pattern but at int8.
 *
 * x_q is the input quantized to int8 with one fp32 scale `x_scale`. */
#if defined(__ARM_FEATURE_DOTPROD)
static void int8_matmul_vdotq(
    float *out, const int8_t *w, const float *w_scale,
    const int8_t *x_q, float x_scale, int M, int N)
{
    for (int i = 0; i < M; i++) {
        const int8_t *row = w + (size_t)i * N;
        int32x4_t acc = vdupq_n_s32(0);
        int j = 0;
        for (; j + 15 < N; j += 16) {
            int8x16_t w16 = vld1q_s8(row + j);
            int8x16_t x16 = vld1q_s8(x_q + j);
            acc = vdotq_s32(acc, w16, x16);   /* 16 INT8 MACs in 1 cycle */
        }
        int32_t sum = vaddvq_s32(acc);
        for (; j < N; j++) sum += (int32_t)row[j] * (int32_t)x_q[j];
        out[i] = (float)sum * w_scale[i] * x_scale;
    }
}
#else
static void int8_matmul_vdotq(float *out, const int8_t *w, const float *w_scale,
                                const int8_t *x_q, float x_scale, int M, int N) {
    /* dotprod-less fallback */
    int8_matmul_libinferbit(out, w, w_scale, (const float*)x_q, M, N);
    (void)x_scale;
}
#endif

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int iters = (argc > 3) ? atoi(argv[3]) : 1000;

    int8_t *w = aligned_alloc(64, ((size_t)M * N + 63) & ~(size_t)63);
    float  *w_scale = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    float  *x_fp32 = aligned_alloc(64, ((size_t)N * sizeof(float) + 63) & ~(size_t)63);
    int8_t *x_q   = aligned_alloc(64, ((size_t)N + 63) & ~(size_t)63);
    float  *y_lib = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    float  *y_dot = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);

    /* Random data */
    unsigned s = 42;
    for (size_t i = 0; i < (size_t)M * N; i++) {
        s = s * 1103515245 + 12345;
        w[i] = (int8_t)(((s >> 16) & 0xff) - 128);
    }
    for (int i = 0; i < M; i++) {
        s = s * 1103515245 + 12345;
        w_scale[i] = (float)((s >> 16) & 0xfff) * 1e-6f;
    }
    for (int j = 0; j < N; j++) {
        s = s * 1103515245 + 12345;
        x_fp32[j] = (float)((int)((s >> 16) & 0xfff) - 2048) * 0.01f;
    }
    /* Quantize x to int8 with one scale */
    float xmax = 0.0f;
    for (int j = 0; j < N; j++) {
        float a = fabsf(x_fp32[j]);
        if (a > xmax) xmax = a;
    }
    float x_scale = xmax / 127.0f;
    for (int j = 0; j < N; j++) {
        int q = (int)roundf(x_fp32[j] / x_scale);
        if (q > 127) q = 127; if (q < -128) q = -128;
        x_q[j] = (int8_t)q;
    }

    /* Warmup */
    for (int it = 0; it < 5; it++) {
        int8_matmul_libinferbit(y_lib, w, w_scale, x_fp32, M, N);
        int8_matmul_vdotq(y_dot, w, w_scale, x_q, x_scale, M, N);
    }

    /* Time path (a): libinferbit-style */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++)
        int8_matmul_libinferbit(y_lib, w, w_scale, x_fp32, M, N);
    double t_lib = (now_sec() - t0) / iters * 1000.0;

    /* Time path (b): vdotq_s32 */
    t0 = now_sec();
    for (int it = 0; it < iters; it++)
        int8_matmul_vdotq(y_dot, w, w_scale, x_q, x_scale, M, N);
    double t_dot = (now_sec() - t0) / iters * 1000.0;

    /* Correctness — vdotq path uses quantized input so won't be bit-exact
     * but cosine should be very high. */
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < M; i++) {
        dot += (double)y_lib[i] * (double)y_dot[i];
        na  += (double)y_lib[i] * (double)y_lib[i];
        nb  += (double)y_dot[i] * (double)y_dot[i];
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);

    printf("=== INT8 NEON ceiling derisk (M=%d, N=%d, iters=%d) ===\n", M, N, iters);
    printf("  libinferbit's neon_matmul_int8 (widen+fp32 FMA): %.3f ms/call\n", t_lib);
    printf("  proper int8 with vdotq_s32 + int8 input:         %.3f ms/call\n", t_dot);
    printf("  speedup of vdotq vs libinferbit: %.2f×\n", t_lib / t_dot);
    printf("  cosine (vdotq vs libinferbit): %.6f  (input quantization expected)\n", cos);
#if !defined(__ARM_FEATURE_DOTPROD)
    printf("  WARNING: __ARM_FEATURE_DOTPROD not defined; vdotq path was a fallback.\n");
#endif
    free(w); free(w_scale); free(x_fp32); free(x_q); free(y_lib); free(y_dot);
    return 0;
}
