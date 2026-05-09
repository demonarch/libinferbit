/* Validate ib_metal_rmsnorm_fp16 against the CPU rmsnorm. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "metal/metal_runtime.h"
#include "inferbit_internal.h"

extern float ib_fp16_to_fp32(uint16_t h);

static uint16_t f32_to_fp16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint16_t s = (uint16_t)((b >> 16) & 0x8000);
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (uint16_t)(e << 10) | (uint16_t)(m >> 13);
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int run(int N) {
    ib_init_kernels(ib_detect_simd());

    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   N=%d\n", ib_metal_device_name(ctx), N);

    /* Random fp32 input + fp16 weight, plus a fp32 weight for the CPU path. */
    srand(0xC0FFEE);
    float *h_x = malloc((size_t)N * sizeof(float));
    uint16_t *h_w16 = malloc((size_t)N * sizeof(uint16_t));
    float    *h_w32 = malloc((size_t)N * sizeof(float));
    float *cpu_out = malloc((size_t)N * sizeof(float));
    float *gpu_host = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_x[i] = ((rand() & 0xFFFF) / 32767.0f - 1.0f) * 2.0f;
        float w = 0.5f + ((rand() & 0xFF) / 255.0f);
        h_w16[i] = f32_to_fp16(w);
        h_w32[i] = ib_fp16_to_fp32(h_w16[i]);  /* exact match w/ GPU path */
    }
    float eps = 1e-5f;

    ib_kern.rmsnorm(cpu_out, h_x, h_w32, eps, N);

    void *gpu_x = ib_metal_alloc(ctx, (size_t)N * sizeof(float),  h_x);
    void *gpu_w = ib_metal_alloc(ctx, (size_t)N * sizeof(uint16_t), h_w16);
    void *gpu_o = ib_metal_alloc(ctx, (size_t)N * sizeof(float), NULL);
    if (ib_metal_rmsnorm_fp16(ctx, gpu_x, gpu_w, gpu_o, N, eps) != 0) {
        fprintf(stderr, "rmsnorm dispatch failed\n");
        return 2;
    }
    memcpy(gpu_host, gpu_o, (size_t)N * sizeof(float));

    /* Compare. */
    double dot = 0, na = 0, nb = 0, max_diff = 0;
    for (int i = 0; i < N; i++) {
        double a = cpu_out[i], b = gpu_host[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a - b);
        if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("Correctness: cos=%.6f  max|diff|=%.4e\n", cos, max_diff);
    if (cos < 0.9999 || max_diff > 1e-3) {
        fprintf(stderr, "FAIL: rmsnorm GPU result does not match CPU\n");
        return 3;
    }

    /* Bench */
    int iters = 1000;
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        ib_kern.rmsnorm(cpu_out, h_x, h_w32, eps, N);
    }
    double t_cpu = (now_sec() - t0) / iters * 1000.0;
    /* warmup */
    for (int i = 0; i < 10; i++)
        ib_metal_rmsnorm_fp16(ctx, gpu_x, gpu_w, gpu_o, N, eps);
    t0 = now_sec();
    for (int i = 0; i < iters; i++) {
        ib_metal_rmsnorm_fp16(ctx, gpu_x, gpu_w, gpu_o, N, eps);
    }
    double t_gpu = (now_sec() - t0) / iters * 1000.0;
    printf("CPU rmsnorm:    %7.4f ms/call\n", t_cpu);
    printf("Metal rmsnorm:  %7.4f ms/call  (ratio %.2f%s)\n",
           t_gpu, t_gpu / t_cpu, t_gpu < t_cpu ? "  GPU faster" : "  CPU faster");

    free(h_x); free(h_w16); free(h_w32); free(cpu_out); free(gpu_host);
    ib_metal_free(ctx, gpu_x);
    ib_metal_free(ctx, gpu_w);
    ib_metal_free(ctx, gpu_o);
    ib_metal_destroy(ctx);
    return 0;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 2048;  /* TinyLlama hidden */
    return run(N);
}
