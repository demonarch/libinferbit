/* Validate ib_metal_attention_block_fp16 against a CPU reference.
 *
 * Setup: TinyLlama-shaped attention (n_heads=32, n_kv_heads=4,
 * head_dim=64). Seed the KV cache with random fp16 data for positions
 * 0..pos-1, run one new step at `pos` on both backends, compare the
 * attn_out vector. Bit-exact match isn't expected (fp16 KV truncation
 * happens on the GPU side before the dot product), but cosine should
 * be > 0.999 and max diff bounded.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "metal/metal_runtime.h"

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

/* CPU reference for one attention step against a fp16 KV cache, mirroring
 * exactly what the GPU kernels do. */
static void cpu_attn_block_fp16(
    const float *q, const float *k, const float *v,
    float *k_cache, float *v_cache,   /* in/out, [seq_len, kv_dim] (fp32 to match libinferbit kv_bits=16) */
    float *scores_scratch,                   /* [n_heads, pos+1] */
    float *attn_out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos)
{
    int kv_dim = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    int p1 = pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* 1. Write K, V at row `pos` of the cache (libinferbit kv_bits=16
     * stores fp32 — see ibf_loader.c line 240-241 — so no truncation). */
    for (int i = 0; i < kv_dim; i++) {
        k_cache[(size_t)pos * kv_dim + i] = k[i];
        v_cache[(size_t)pos * kv_dim + i] = v[i];
    }

    /* 2. Scores: scores[h, t] = (Q[h] · K_cache[t, kv_h]) * scale. */
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_h = q + h * head_dim;
        for (int t = 0; t <= pos; t++) {
            const float *k_t = k_cache + (size_t)t * kv_dim + kv_h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                s += q_h[d] * k_t[d];
            }
            scores_scratch[h * p1 + t] = s * scale;
        }
    }

    /* 3. Softmax over each row of scores (numerically stable). */
    for (int h = 0; h < n_heads; h++) {
        float *row = scores_scratch + h * p1;
        float m = row[0];
        for (int t = 1; t < p1; t++) if (row[t] > m) m = row[t];
        float sum = 0.0f;
        for (int t = 0; t < p1; t++) { row[t] = expf(row[t] - m); sum += row[t]; }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < p1; t++) row[t] *= inv_sum;
    }

    /* 4. Weighted V. */
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *s_row = scores_scratch + h * p1;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int t = 0; t < p1; t++) {
                const float *v_t = v_cache + (size_t)t * kv_dim + kv_h * head_dim + d;
                acc += s_row[t] * (*v_t);
            }
            attn_out[h * head_dim + d] = acc;
        }
    }
}

int main(int argc, char **argv) {
    int n_heads     = 32;
    int n_kv_heads  = 4;
    int head_dim    = 64;
    int seq_len     = 256;
    int pos         = (argc > 1) ? atoi(argv[1]) : 100;
    if (pos < 0 || pos >= seq_len) { fprintf(stderr, "bad pos\n"); return 1; }

    int kv_dim   = n_kv_heads * head_dim;
    int p1       = pos + 1;
    int n_q_dim  = n_heads * head_dim;

    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   pos=%d  n_heads=%d  n_kv_heads=%d  head_dim=%d  seq_len=%d\n",
           ib_metal_device_name(ctx), pos, n_heads, n_kv_heads, head_dim, seq_len);

    /* Random Q, K, V for the new step + random fp16 KV cache for the prefix. */
    srand(0xA77E);
    float    *h_q = malloc((size_t)n_q_dim * sizeof(float));
    float    *h_k = malloc((size_t)kv_dim  * sizeof(float));
    float    *h_v = malloc((size_t)kv_dim  * sizeof(float));
    float *h_kc = malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *h_vc = malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *cpu_scores = malloc((size_t)n_heads * p1 * sizeof(float));
    float *cpu_out    = malloc((size_t)n_q_dim * sizeof(float));
    float *gpu_out    = malloc((size_t)n_q_dim * sizeof(float));

    for (int i = 0; i < n_q_dim; i++) h_q[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
    for (int i = 0; i < kv_dim;  i++) h_k[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
    for (int i = 0; i < kv_dim;  i++) h_v[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
    /* Seed KV cache (fp32 — matches kv_bits=16 storage in libinferbit). */
    for (size_t i = 0; i < (size_t)seq_len * kv_dim; i++) {
        h_kc[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
        h_vc[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
    }

    /* CPU reference: take a copy of the cache so the GPU run sees the
     * identical prefix (the CPU run will mutate row `pos`). */
    float *cpu_kc = malloc((size_t)seq_len * kv_dim * sizeof(float));
    float *cpu_vc = malloc((size_t)seq_len * kv_dim * sizeof(float));
    memcpy(cpu_kc, h_kc, (size_t)seq_len * kv_dim * sizeof(float));
    memcpy(cpu_vc, h_vc, (size_t)seq_len * kv_dim * sizeof(float));
    cpu_attn_block_fp16(h_q, h_k, h_v, cpu_kc, cpu_vc, cpu_scores, cpu_out,
                        n_heads, n_kv_heads, head_dim, seq_len, pos);

    /* GPU buffers. */
    void *g_q  = ib_metal_alloc(ctx, (size_t)n_q_dim * sizeof(float),  h_q);
    void *g_k  = ib_metal_alloc(ctx, (size_t)kv_dim  * sizeof(float),  h_k);
    void *g_v  = ib_metal_alloc(ctx, (size_t)kv_dim  * sizeof(float),  h_v);
    void *g_kc = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(float), h_kc);
    void *g_vc = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(float), h_vc);
    void *g_s  = ib_metal_alloc(ctx, (size_t)n_heads * seq_len * sizeof(float), NULL);
    void *g_o  = ib_metal_alloc(ctx, (size_t)n_q_dim * sizeof(float), NULL);

    if (ib_metal_attention_block_fp16(ctx, g_q, g_k, g_v, g_kc, g_vc, g_s, g_o,
                                       n_heads, n_kv_heads, head_dim, seq_len, pos) != 0) {
        fprintf(stderr, "attention dispatch failed\n");
        return 2;
    }
    memcpy(gpu_out, g_o, (size_t)n_q_dim * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < n_q_dim; i++) {
        double a = cpu_out[i], b = gpu_out[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("Correctness: cos=%.6f  max|diff|=%.4e\n", cos, max_diff);

    int rc = (cos < 0.999 || max_diff > 1e-2) ? 3 : 0;

    free(h_q); free(h_k); free(h_v);
    free(h_kc); free(h_vc); free(cpu_kc); free(cpu_vc);
    free(cpu_scores); free(cpu_out); free(gpu_out);
    ib_metal_free(ctx, g_q);
    ib_metal_free(ctx, g_k);
    ib_metal_free(ctx, g_v);
    ib_metal_free(ctx, g_kc);
    ib_metal_free(ctx, g_vc);
    ib_metal_free(ctx, g_s);
    ib_metal_free(ctx, g_o);
    ib_metal_destroy(ctx);
    return rc;
}
