/* Phase 2 derisk: Metal w4a8 vs NEON w4a8 on TinyLlama-sized matmul.
 *
 * Same input data, same packing, same scale model. Validates:
 *   1. Numerical equivalence: cosine + max abs diff
 *   2. Speed: ms/call, GB/s effective bandwidth
 *
 * If Metal is meaningfully faster (≥1.5×) and bit-equivalent (cos > 0.99999),
 * Phase 2 succeeds and we proceed to integration.
 */
#include "../src/metal/metal_runtime.h"
#include "../src/inferbit_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Convert fp32 to fp16 (storage as uint16). Matches the rest of libinferbit. */
static unsigned short fp32_to_fp16_bits(float f) {
    unsigned int x;
    memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000;
    int exp = (int)((x >> 23) & 0xFF) - 127 + 15;
    unsigned int mant = x & 0x7FFFFF;
    if (exp <= 0) return (unsigned short)sign;
    if (exp >= 31) return (unsigned short)(sign | 0x7C00);
    return (unsigned short)(sign | ((unsigned int)exp << 10) | (mant >> 13));
}

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    int iters = (argc > 3) ? atoi(argv[3]) : 200;

    if (!ib_metal_available()) {
        printf("Metal not available\n");
        return 1;
    }
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) return 1;
    printf("Device: %s\n", ib_metal_device_name(ctx));
    printf("Shape:  M=%d  N=%d  iters=%d\n", M, N, iters);

    /* Generate random INT4 weights (packed) and per-row fp16 scales. */
    int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
    size_t w_bytes  = (size_t)M * (N / 2);
    size_t ws_bytes = (size_t)M * sizeof(unsigned short);
    size_t xq_bytes = (size_t)N;
    size_t xs_bytes = (size_t)n_groups * sizeof(float);
    size_t out_bytes = (size_t)M * sizeof(float);

    /* Host buffers for setup. */
    unsigned char *h_w  = malloc(w_bytes);
    unsigned short *h_ws = malloc(ws_bytes);
    signed char *h_xq = malloc(xq_bytes);
    float *h_xs = malloc(xs_bytes);
    float *cpu_out = malloc(out_bytes);

    unsigned int seed = 42;
    for (size_t i = 0; i < w_bytes; i++) {
        seed = seed * 1103515245 + 12345;
        h_w[i] = (unsigned char)((seed >> 16) & 0xFF);
    }
    for (int i = 0; i < M; i++) {
        seed = seed * 1103515245 + 12345;
        float scale = (float)((seed >> 16) & 0xFFF) * 1e-5f;
        if (scale < 1e-6f) scale = 1e-6f;
        h_ws[i] = fp32_to_fp16_bits(scale);
    }
    for (int i = 0; i < N; i++) {
        seed = seed * 1103515245 + 12345;
        h_xq[i] = (signed char)(((seed >> 16) & 0xFF) - 128);
    }
    for (int g = 0; g < n_groups; g++) {
        seed = seed * 1103515245 + 12345;
        h_xs[g] = (float)((seed >> 16) & 0xFFF) * 1e-4f;
    }

    /* Allocate GPU buffers. ib_metal_alloc copies from host data once. */
    void *gpu_w   = ib_metal_alloc(ctx, w_bytes,   h_w);
    void *gpu_ws  = ib_metal_alloc(ctx, ws_bytes,  h_ws);
    void *gpu_xq  = ib_metal_alloc(ctx, xq_bytes,  h_xq);
    void *gpu_xs  = ib_metal_alloc(ctx, xs_bytes,  h_xs);
    void *gpu_out = ib_metal_alloc(ctx, out_bytes, NULL);
    if (!gpu_w || !gpu_ws || !gpu_xq || !gpu_xs || !gpu_out) {
        printf("alloc fail\n"); return 1;
    }

    /* Initialize CPU kernels for the reference path. */
    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);

    /* For ib_kern.matmul_w4a8 we need scales as fp32 (it dereferences as
     * fp16 stored in float* via reinterpret? actually it uses fp16 stored
     * as uint16_t — let's check. The scales_w param is `const float*`.
     * Looking at ib_parallel_matmul_w4a8: scales are fp32. But the
     * model's stored fp16 scales are converted to fp32 before kernel call.
     * So we pass fp32 here too. */
    float *h_ws_fp32 = malloc(M * sizeof(float));
    for (int i = 0; i < M; i++) {
        unsigned int s = h_ws[i];
        unsigned int sign = (s >> 15) << 31;
        int exp = (int)((s >> 10) & 0x1F);
        unsigned int mant = s & 0x3FF;
        unsigned int f;
        if (exp == 0)        f = sign | 0;
        else if (exp == 31)  f = sign | 0x7F800000 | (mant << 13);
        else                 f = sign | ((unsigned int)(exp + 112) << 23) | (mant << 13);
        memcpy(&h_ws_fp32[i], &f, 4);
    }

    /* CPU reference: ib_kern.matmul_w4a8 */
    ib_kern.matmul_w4a8(cpu_out, h_w, h_ws_fp32, h_xq, h_xs, M, N);

    /* GPU run + correctness */
    int rc = ib_metal_matmul_w4a8(ctx, gpu_w, gpu_ws, gpu_xq, gpu_xs, gpu_out, M, N);
    if (rc != 0) { printf("Metal matmul failed\n"); return 1; }
    float *gpu_out_view = (float *)gpu_out;
    double dot = 0, na = 0, nb = 0, max_err = 0;
    for (int i = 0; i < M; i++) {
        dot += (double)cpu_out[i] * (double)gpu_out_view[i];
        na  += (double)cpu_out[i] * (double)cpu_out[i];
        nb  += (double)gpu_out_view[i] * (double)gpu_out_view[i];
        double e = fabs(cpu_out[i] - gpu_out_view[i]);
        if (e > max_err) max_err = e;
    }
    double cos = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("Correctness: cos = %.6f, max abs diff = %.4e\n", cos, max_err);

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        ib_kern.matmul_w4a8(cpu_out, h_w, h_ws_fp32, h_xq, h_xs, M, N);
        ib_metal_matmul_w4a8(ctx, gpu_w, gpu_ws, gpu_xq, gpu_xs, gpu_out, M, N);
    }

    /* Time CPU */
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        ib_kern.matmul_w4a8(cpu_out, h_w, h_ws_fp32, h_xq, h_xs, M, N);
    }
    double t_cpu = (now_sec() - t0) / iters * 1000.0;

    /* Time GPU */
    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        ib_metal_matmul_w4a8(ctx, gpu_w, gpu_ws, gpu_xq, gpu_xs, gpu_out, M, N);
    }
    double t_gpu = (now_sec() - t0) / iters * 1000.0;

    double bw_cpu = ((double)w_bytes / 1e9) / (t_cpu * 1e-3);
    double bw_gpu = ((double)w_bytes / 1e9) / (t_gpu * 1e-3);
    printf("\n=== matmul_w4a8 (M=%d N=%d, %d iters) ===\n", M, N, iters);
    printf("  CPU NEON (vdotq+w4a8):   %7.4f ms/call  (%.2f GB/s)\n", t_cpu, bw_cpu);
    printf("  Metal GPU (matmul_w4a8): %7.4f ms/call  (%.2f GB/s)\n", t_gpu, bw_gpu);
    printf("  Metal/CPU ratio:         %.3f%s\n", t_gpu / t_cpu,
            t_gpu < t_cpu ? "  (Metal faster)" : "  (CPU faster)");

    free(h_w); free(h_ws); free(h_xq); free(h_xs); free(h_ws_fp32); free(cpu_out);
    ib_metal_free(ctx, gpu_w);
    ib_metal_free(ctx, gpu_ws);
    ib_metal_free(ctx, gpu_xq);
    ib_metal_free(ctx, gpu_xs);
    ib_metal_free(ctx, gpu_out);
    ib_metal_destroy(ctx);
    return 0;
}
