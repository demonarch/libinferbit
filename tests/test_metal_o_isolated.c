/* Run o_proj in isolation: capture g_attnout from the layer recorder,
 * then run BOTH CPU and GPU o_proj on that exact same input, compare. */
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

int main(void) {
    int hidden = 2048;
    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();

    /* Synthesize a small attn_out-like input + o_proj weights. */
    srand(0x42);
    float *attn_out = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++)
        attn_out[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;

    uint8_t  *o_w  = malloc((size_t)hidden * (hidden / 2));
    uint16_t *o_s  = malloc((size_t)hidden * sizeof(uint16_t));
    float    *o_sf = malloc((size_t)hidden * sizeof(float));
    for (size_t i = 0; i < (size_t)hidden * (hidden / 2); i++) o_w[i] = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < hidden; i++) {
        float s = 0.005f + ((rand() & 0xFFFF) / 65535.0f) * 0.05f;
        o_s[i]  = f32_to_fp16(s);
        o_sf[i] = ib_fp16_to_fp32(o_s[i]);
    }

    /* CPU. */
    float *cpu_out = malloc((size_t)hidden * sizeof(float));
    int8_t *xq = malloc(hidden);
    float  *xs = malloc(((hidden + 127) / 128) * sizeof(float));
    ib_quantize_input_int8_g128(attn_out, xq, xs, hidden);
    ib_kern.matmul_w4a8(cpu_out, o_w, o_sf, xq, xs, hidden, hidden);

    /* GPU standalone via fp32_in. */
    void *g_in  = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), attn_out);
    void *g_out = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_w   = ib_metal_alloc(ctx, (size_t)hidden * (hidden / 2), o_w);
    void *g_ws  = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), o_s);
    void *g_xq  = ib_metal_alloc(ctx, (size_t)hidden, NULL);
    void *g_xs  = ib_metal_alloc(ctx, (size_t)((hidden + 127) / 128) * sizeof(float), NULL);
    ib_metal_matmul_w4a8_fp32_in(ctx, g_in, g_w, g_ws, g_out, g_xq, g_xs, hidden, hidden);

    float *gpu_out = malloc((size_t)hidden * sizeof(float));
    memcpy(gpu_out, g_out, (size_t)hidden * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < hidden; i++) {
        double a = cpu_out[i], b = gpu_out[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    printf("standalone o_proj  cos=%.6f  max|diff|=%.4e\n", dot/(sqrt(na)*sqrt(nb)), max_diff);

    /* Now via recorder. */
    void *g_out2 = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_in, g_w, g_ws, g_out2, g_xq, g_xs, hidden, hidden);
    ib_metal_recorder_commit(r);

    float *gpu_out2 = malloc((size_t)hidden * sizeof(float));
    memcpy(gpu_out2, g_out2, (size_t)hidden * sizeof(float));
    dot=0; na=0; nb=0; max_diff=0;
    for (int i = 0; i < hidden; i++) {
        double a = cpu_out[i], b = gpu_out2[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    printf("recorder   o_proj  cos=%.6f  max|diff|=%.4e\n", dot/(sqrt(na)*sqrt(nb)), max_diff);

    ib_metal_destroy(ctx);
    return 0;
}
