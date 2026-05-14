/* Validate ib_metal_matmul_int8_fp32_in vs CPU ib_kern.matmul_int8. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

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

int main(int argc, char **argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 2048;
    int N = (argc > 2) ? atoi(argv[2]) : 2048;
    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   M=%d N=%d\n", ib_metal_device_name(ctx), M, N);

    srand(0x18ECAFE);
    int8_t   *h_w  = malloc((size_t)M * N);
    uint16_t *h_ws = malloc((size_t)M * sizeof(uint16_t));
    float    *h_x  = malloc((size_t)N * sizeof(float));
    float    *cpu_out = malloc((size_t)M * sizeof(float));
    float    *gpu_out = malloc((size_t)M * sizeof(float));
    float    *h_ws_fp32 = malloc((size_t)M * sizeof(float));

    for (size_t i = 0; i < (size_t)M * N; i++) h_w[i] = (int8_t)((rand() & 0xFF) - 128);
    for (int i = 0; i < M; i++) {
        float s = 0.005f + ((rand() & 0xFFFF) / 65535.0f) * 0.05f;
        h_ws[i] = f32_to_fp16(s);
        h_ws_fp32[i] = ib_fp16_to_fp32(h_ws[i]);
    }
    for (int i = 0; i < N; i++) h_x[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;

    /* CPU. */
    ib_kern.matmul_int8(cpu_out, h_w, h_ws_fp32, h_x, M, N);

    /* GPU. */
    void *g_w  = ib_metal_alloc(ctx, (size_t)M * N, h_w);
    void *g_ws = ib_metal_alloc(ctx, (size_t)M * sizeof(uint16_t), h_ws);
    void *g_x  = ib_metal_alloc(ctx, (size_t)N * sizeof(float), h_x);
    void *g_out= ib_metal_alloc(ctx, (size_t)M * sizeof(float), NULL);
    if (ib_metal_matmul_int8_fp32_in(ctx, g_x, g_w, g_ws, g_out, M, N) != 0) return 2;
    memcpy(gpu_out, g_out, (size_t)M * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < M; i++) {
        double a = cpu_out[i], b = gpu_out[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("Correctness: cos=%.6f  max|diff|=%.4e\n", cos, max_diff);

    free(h_w); free(h_ws); free(h_x); free(cpu_out); free(gpu_out); free(h_ws_fp32);
    ib_metal_free(ctx, g_w);
    ib_metal_free(ctx, g_ws);
    ib_metal_free(ctx, g_x);
    ib_metal_free(ctx, g_out);
    ib_metal_destroy(ctx);
    return (cos < 0.9999) ? 3 : 0;
}
