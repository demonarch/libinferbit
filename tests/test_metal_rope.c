/* Validate ib_metal_rope_inplace against scalar_rope across all heads. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "metal/metal_runtime.h"
#include "inferbit_internal.h"

int main(int argc, char **argv) {
    int n_heads  = (argc > 1) ? atoi(argv[1]) : 32;   /* TinyLlama Q heads */
    int head_dim = (argc > 2) ? atoi(argv[2]) : 64;
    int pos      = (argc > 3) ? atoi(argv[3]) : 17;
    float theta  = 10000.0f;
    int N = n_heads * head_dim;

    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   n_heads=%d head_dim=%d pos=%d\n",
           ib_metal_device_name(ctx), n_heads, head_dim, pos);

    srand(0xDEAD);
    float *cpu_q = malloc((size_t)N * sizeof(float));
    float *gpu_q = malloc((size_t)N * sizeof(float));
    float  k_dummy[64];  /* CPU rope rotates Q and K together; we feed a per-head dummy K and discard. */
    for (int i = 0; i < N; i++) {
        cpu_q[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 4.0f;
        gpu_q[i] = cpu_q[i];
    }

    /* CPU reference: rotate each head's Q (use the same kernel; the
     * scalar_rope rotates Q AND K, but we just call it per head with a
     * fresh dummy K each time and ignore the K result). */
    for (int h = 0; h < n_heads; h++) {
        memset(k_dummy, 0, sizeof(k_dummy));  /* irrelevant — discarded */
        ib_kern.rope(cpu_q + h * head_dim, k_dummy, head_dim, pos, theta, NULL, NULL);
    }

    void *g_t = ib_metal_alloc(ctx, (size_t)N * sizeof(float), gpu_q);
    if (ib_metal_rope_inplace(ctx, g_t, n_heads, head_dim, pos, theta) != 0) {
        fprintf(stderr, "rope dispatch failed\n");
        return 2;
    }
    memcpy(gpu_q, g_t, (size_t)N * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < N; i++) {
        double a = cpu_q[i], b = gpu_q[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("Correctness: cos=%.6f  max|diff|=%.4e\n", cos, max_diff);
    int rc = (cos < 0.9999 || max_diff > 1e-3) ? 3 : 0;

    free(cpu_q); free(gpu_q);
    ib_metal_free(ctx, g_t);
    ib_metal_destroy(ctx);
    return rc;
}
