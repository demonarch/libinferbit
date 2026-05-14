/* Validate ib_metal_softmax_rows + ib_metal_embed_lookup_fp16. */
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

static int test_softmax(ib_metal_ctx *ctx) {
    int n_rows = 32, row_len = 1024;
    int N = n_rows * row_len;
    float *cpu = malloc((size_t)N * sizeof(float));
    float *gpu_in = malloc((size_t)N * sizeof(float));
    srand(0xCAFE);
    for (int i = 0; i < N; i++) {
        cpu[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 8.0f;
        gpu_in[i] = cpu[i];
    }
    /* CPU: per-row softmax. */
    for (int r = 0; r < n_rows; r++) {
        ib_kern.softmax(cpu + r * row_len, row_len);
    }
    void *g = ib_metal_alloc(ctx, (size_t)N * sizeof(float), gpu_in);
    if (ib_metal_softmax_rows(ctx, g, n_rows, row_len) != 0) return 2;
    memcpy(gpu_in, g, (size_t)N * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < N; i++) {
        double a = cpu[i], b = gpu_in[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("softmax: rows=%d row_len=%d  cos=%.6f  max|diff|=%.4e\n",
           n_rows, row_len, cos, max_diff);
    int rc = (cos < 0.9999 || max_diff > 1e-3) ? 3 : 0;

    free(cpu); free(gpu_in);
    ib_metal_free(ctx, g);
    return rc;
}

static int test_embed(ib_metal_ctx *ctx) {
    int vocab = 32000, hidden = 2048;
    int token = 12345;
    uint16_t *emb = malloc((size_t)vocab * hidden * sizeof(uint16_t));
    /* fill the row we'll look up with known values, others with zero. */
    memset(emb, 0, (size_t)vocab * hidden * sizeof(uint16_t));
    for (int i = 0; i < hidden; i++) {
        emb[(size_t)token * hidden + i] = f32_to_fp16(((i * 17) % 256) / 256.0f - 0.5f);
    }
    void *g_emb = ib_metal_alloc(ctx, (size_t)vocab * hidden * sizeof(uint16_t), emb);
    void *g_out = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    if (ib_metal_embed_lookup_fp16(ctx, g_emb, token, hidden, g_out) != 0) return 2;
    float *out_h = malloc((size_t)hidden * sizeof(float));
    memcpy(out_h, g_out, (size_t)hidden * sizeof(float));

    double max_diff = 0;
    for (int i = 0; i < hidden; i++) {
        float expect = ib_fp16_to_fp32(emb[(size_t)token * hidden + i]);
        double d = fabs((double)out_h[i] - (double)expect);
        if (d > max_diff) max_diff = d;
    }
    printf("embed: vocab=%d hidden=%d  max|diff|=%.4e\n", vocab, hidden, max_diff);
    int rc = (max_diff > 1e-6) ? 3 : 0;

    free(emb); free(out_h);
    ib_metal_free(ctx, g_emb);
    ib_metal_free(ctx, g_out);
    return rc;
}

int main(void) {
    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s\n", ib_metal_device_name(ctx));
    int rc = 0;
    rc |= test_softmax(ctx);
    rc |= test_embed(ctx);
    ib_metal_destroy(ctx);
    return rc;
}
