/* Validate ib_metal_silu_mul against the CPU silu_mul. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "metal/metal_runtime.h"
#include "inferbit_internal.h"

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 5632;  /* TinyLlama intermediate */
    ib_init_kernels(ib_detect_simd());

    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   N=%d\n", ib_metal_device_name(ctx), N);

    srand(0xBEEF);
    float *gate = malloc((size_t)N * sizeof(float));
    float *up   = malloc((size_t)N * sizeof(float));
    float *cpu  = malloc((size_t)N * sizeof(float));
    float *gpu_h= malloc((size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) {
        gate[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 6.0f;
        up[i]   = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 4.0f;
    }
    ib_kern.silu_mul(cpu, gate, up, N);

    void *g_gate = ib_metal_alloc(ctx, (size_t)N * sizeof(float), gate);
    void *g_up   = ib_metal_alloc(ctx, (size_t)N * sizeof(float), up);
    void *g_out  = ib_metal_alloc(ctx, (size_t)N * sizeof(float), NULL);
    if (ib_metal_silu_mul(ctx, g_gate, g_up, g_out, N) != 0) {
        fprintf(stderr, "silu_mul dispatch failed\n");
        return 2;
    }
    memcpy(gpu_h, g_out, (size_t)N * sizeof(float));

    double dot = 0, na = 0, nb = 0, max_diff = 0;
    for (int i = 0; i < N; i++) {
        double a = cpu[i], b = gpu_h[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a - b);
        if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("Correctness: cos=%.6f  max|diff|=%.4e\n", cos, max_diff);
    int rc = (cos < 0.9999 || max_diff > 1e-3) ? 3 : 0;

    free(gate); free(up); free(cpu); free(gpu_h);
    ib_metal_free(ctx, g_gate);
    ib_metal_free(ctx, g_up);
    ib_metal_free(ctx, g_out);
    ib_metal_destroy(ctx);
    return rc;
}
