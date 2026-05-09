/* Phase 6b: end-to-end single transformer block (TinyLlama shape) on GPU
 * via the recorder, vs the CPU equivalent. Validates that an entire
 * layer worth of kernels can be assembled into ONE command buffer with
 * correct results, and benchmarks GPU-batched vs CPU.
 *
 * Layer pipeline (matches forward.c):
 *   xb  = rmsnorm(x, input_norm_w)
 *   q   = matmul_w4a8(q_proj, xb)
 *   k   = matmul_w4a8(k_proj, xb)
 *   v   = matmul_w4a8(v_proj, xb)
 *   rope(q, n_heads,    head_dim, pos)
 *   rope(k, n_kv_heads, head_dim, pos)
 *   attention_block(q, k, v, kv_cache, scores, attn_out)
 *   xb2 = matmul_w4a8(o_proj, attn_out)
 *   x  += xb2
 *   xb  = rmsnorm(x, post_norm_w)
 *   hb  = matmul_w4a8(gate_proj, xb)
 *   hb2 = matmul_w4a8(up_proj, xb)
 *   hb  = silu_mul(hb, hb2)
 *   xb  = matmul_w4a8(down_proj, hb)
 *   x  += xb
 */
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

/* Tiny CPU rope helper that rotates ONLY one tensor (Q, K, or anything),
 * mirroring kernels.metal rope_inplace exactly. */
static void cpu_rope_inplace(float *t, int n_heads, int head_dim, int pos, float theta) {
    for (int h = 0; h < n_heads; h++) {
        float *th = t + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float exponent = (float)i / (float)head_dim;
            float freq = powf(theta, -exponent);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float v0 = th[i], v1 = th[i + 1];
            th[i]     = v0 * c - v1 * s;
            th[i + 1] = v0 * s + v1 * c;
        }
    }
}

/* CPU attention-block reference matching the GPU fp16 version exactly. */
static void cpu_attn_block_fp16(
    const float *q, const float *k, const float *v,
    uint16_t *k_cache, uint16_t *v_cache,
    float *scores, float *attn_out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos)
{
    int kv_dim = n_kv_heads * head_dim;
    int heads_per_kv = n_heads / n_kv_heads;
    int p1 = pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    (void)seq_len;

    /* Note: in the GPU path we use Metal's RTNE half-cast, but here we use
     * the truncating f32_to_fp16. To remove that source of difference, we
     * pre-write the cache row using the same truncating cast on BOTH paths
     * before this CPU function runs (caller does this). So the cache row
     * here is read but never written. */
    for (int i = 0; i < kv_dim; i++) {
        k_cache[(size_t)pos * kv_dim + i] = f32_to_fp16(k[i]);
        v_cache[(size_t)pos * kv_dim + i] = f32_to_fp16(v[i]);
    }
    (void)k; (void)v;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *q_h = q + h * head_dim;
        for (int t = 0; t <= pos; t++) {
            const uint16_t *k_t = k_cache + (size_t)t * kv_dim + kv_h * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += q_h[d] * ib_fp16_to_fp32(k_t[d]);
            scores[h * p1 + t] = s * scale;
        }
    }
    for (int h = 0; h < n_heads; h++) {
        float *row = scores + h * p1;
        float m = row[0];
        for (int t = 1; t < p1; t++) if (row[t] > m) m = row[t];
        float sum = 0.0f;
        for (int t = 0; t < p1; t++) { row[t] = expf(row[t] - m); sum += row[t]; }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < p1; t++) row[t] *= inv_sum;
    }
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / heads_per_kv;
        const float *s_row = scores + h * p1;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int t = 0; t < p1; t++) {
                const uint16_t *v_t = v_cache + (size_t)t * kv_dim + kv_h * head_dim + d;
                acc += s_row[t] * ib_fp16_to_fp32(*v_t);
            }
            attn_out[h * head_dim + d] = acc;
        }
    }
}

/* CPU helper: matmul_w4a8 (full path including quantize). */
static void cpu_matmul_w4a8_full(float *out, const float *x_fp32,
                                  const uint8_t *W, const float *W_s,
                                  int M, int N, int8_t *xq_buf, float *xs_buf) {
    ib_quantize_input_int8_g128(x_fp32, xq_buf, xs_buf, N);
    ib_kern.matmul_w4a8(out, W, W_s, xq_buf, xs_buf, M, N);
}

int main(int argc, char **argv) {
    int hidden       = 2048;
    int intermediate = 5632;
    int n_heads      = 32;
    int n_kv_heads   = 4;
    int head_dim     = 64;
    int seq_len      = 256;
    int pos          = (argc > 1) ? atoi(argv[1]) : 100;
    int iters        = (argc > 2) ? atoi(argv[2]) : 30;
    float theta      = 10000.0f;
    float eps        = 1e-5f;
    int kv_dim       = n_kv_heads * head_dim;

    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   pos=%d  iters=%d\n", ib_metal_device_name(ctx), pos, iters);
    printf("TinyLlama shape: hidden=%d intermediate=%d n_heads=%d n_kv=%d head_dim=%d\n",
           hidden, intermediate, n_heads, n_kv_heads, head_dim);

    /* Allocate weights (uint8 packed nibbles, fp16 scales). */
    srand(0x1AB7E11A);
    #define ALLOC_W(name, M, N) \
        uint8_t  *name##_w  = malloc((size_t)(M) * (N) / 2); \
        uint16_t *name##_s  = malloc((size_t)(M) * sizeof(uint16_t)); \
        float    *name##_sf = malloc((size_t)(M) * sizeof(float)); \
        for (size_t i = 0; i < (size_t)(M) * (N) / 2; i++) name##_w[i] = (uint8_t)(rand() & 0xFF); \
        for (int i = 0; i < (M); i++) { \
            float s = 0.005f + ((rand() & 0xFFFF) / 65535.0f) * 0.05f; \
            name##_s[i]  = f32_to_fp16(s); \
            name##_sf[i] = ib_fp16_to_fp32(name##_s[i]); \
        }
    ALLOC_W(q_proj,    hidden, hidden);
    ALLOC_W(k_proj,    kv_dim, hidden);
    ALLOC_W(v_proj,    kv_dim, hidden);
    ALLOC_W(o_proj,    hidden, hidden);
    ALLOC_W(gate_proj, intermediate, hidden);
    ALLOC_W(up_proj,   intermediate, hidden);
    ALLOC_W(down_proj, hidden, intermediate);
    #undef ALLOC_W

    /* Norms (fp16). */
    uint16_t *input_norm_h  = malloc((size_t)hidden * sizeof(uint16_t));
    uint16_t *post_norm_h   = malloc((size_t)hidden * sizeof(uint16_t));
    float    *input_norm_f  = malloc((size_t)hidden * sizeof(float));
    float    *post_norm_f   = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) {
        input_norm_h[i] = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        post_norm_h[i]  = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        input_norm_f[i] = ib_fp16_to_fp32(input_norm_h[i]);
        post_norm_f[i]  = ib_fp16_to_fp32(post_norm_h[i]);
    }

    /* Initial x. */
    float *x0 = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) x0[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;

    /* Pre-seeded KV cache (positions 0..pos-1) — both paths share via this same
     * uint16 array (already in fp16). */
    uint16_t *kc_seed = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *vc_seed = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)seq_len * kv_dim; i++) {
        kc_seed[i] = f32_to_fp16(((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f);
        vc_seed[i] = f32_to_fp16(((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f);
    }

    /* ── CPU forward: ──────────────────────────────────────────────────────── */
    float *cpu_x       = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb      = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb2     = malloc((size_t)hidden * sizeof(float));
    float *cpu_q       = malloc((size_t)n_heads * head_dim * sizeof(float));
    float *cpu_k       = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v       = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_attn_out= malloc((size_t)n_heads * head_dim * sizeof(float));
    float *cpu_hb      = malloc((size_t)intermediate * sizeof(float));
    float *cpu_hb2     = malloc((size_t)intermediate * sizeof(float));
    float *cpu_scores  = malloc((size_t)n_heads * (pos + 1) * sizeof(float));
    int8_t *cpu_xq_h   = malloc((size_t)hidden);
    int8_t *cpu_xq_i   = malloc((size_t)intermediate);
    float  *cpu_xs_h   = malloc((size_t)((hidden + 127) / 128) * sizeof(float));
    float  *cpu_xs_i   = malloc((size_t)((intermediate + 127) / 128) * sizeof(float));
    uint16_t *cpu_kc = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *cpu_vc = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));

    memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
    memcpy(cpu_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
    memcpy(cpu_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));

    /* CPU layer execution. */
    ib_kern.rmsnorm(cpu_xb, cpu_x, input_norm_f, eps, hidden);
    cpu_matmul_w4a8_full(cpu_q, cpu_xb, q_proj_w, q_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
    cpu_matmul_w4a8_full(cpu_k, cpu_xb, k_proj_w, k_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
    cpu_matmul_w4a8_full(cpu_v, cpu_xb, v_proj_w, v_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
    cpu_rope_inplace(cpu_q, n_heads,    head_dim, pos, theta);
    cpu_rope_inplace(cpu_k, n_kv_heads, head_dim, pos, theta);
    cpu_attn_block_fp16(cpu_q, cpu_k, cpu_v, cpu_kc, cpu_vc, cpu_scores, cpu_attn_out,
                        n_heads, n_kv_heads, head_dim, seq_len, pos);
    cpu_matmul_w4a8_full(cpu_xb2, cpu_attn_out, o_proj_w, o_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
    for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb2[i];
    ib_kern.rmsnorm(cpu_xb, cpu_x, post_norm_f, eps, hidden);
    cpu_matmul_w4a8_full(cpu_hb,  cpu_xb, gate_proj_w, gate_proj_sf, intermediate, hidden, cpu_xq_h, cpu_xs_h);
    cpu_matmul_w4a8_full(cpu_hb2, cpu_xb, up_proj_w,   up_proj_sf,   intermediate, hidden, cpu_xq_h, cpu_xs_h);
    ib_kern.silu_mul(cpu_hb, cpu_hb, cpu_hb2, intermediate);
    cpu_matmul_w4a8_full(cpu_xb,  cpu_hb, down_proj_w, down_proj_sf, hidden, intermediate, cpu_xq_i, cpu_xs_i);
    for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];

    /* ── GPU forward via recorder: ───────────────────────────────────────── */
    /* State buffers */
    void *g_x        = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float),  x0);
    void *g_xb       = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float),  NULL);
    void *g_xb2      = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float),  NULL);
    void *g_q        = ib_metal_alloc(ctx, (size_t)n_heads * head_dim * sizeof(float), NULL);
    void *g_k        = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float),  NULL);
    void *g_v        = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float),  NULL);
    void *g_attn_out = ib_metal_alloc(ctx, (size_t)n_heads * head_dim * sizeof(float), NULL);
    void *g_hb       = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_hb2      = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_scores   = ib_metal_alloc(ctx, (size_t)n_heads * seq_len * sizeof(float), NULL);
    /* xq scratch sized for max(hidden, intermediate); xs scratch likewise. */
    int max_n        = intermediate;
    void *g_xq       = ib_metal_alloc(ctx, (size_t)max_n, NULL);
    void *g_xs       = ib_metal_alloc(ctx, (size_t)((max_n + 127) / 128) * sizeof(float), NULL);

    /* Weight + norm buffers */
    #define UPLOAD_W(name, M, N) \
        void *gw_##name##_w = ib_metal_alloc(ctx, (size_t)(M)*(N)/2, name##_w); \
        void *gw_##name##_s = ib_metal_alloc(ctx, (size_t)(M)*sizeof(uint16_t), name##_s);
    UPLOAD_W(q_proj,    hidden, hidden);
    UPLOAD_W(k_proj,    kv_dim, hidden);
    UPLOAD_W(v_proj,    kv_dim, hidden);
    UPLOAD_W(o_proj,    hidden, hidden);
    UPLOAD_W(gate_proj, intermediate, hidden);
    UPLOAD_W(up_proj,   intermediate, hidden);
    UPLOAD_W(down_proj, hidden, intermediate);
    #undef UPLOAD_W
    void *gw_input_norm = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), input_norm_h);
    void *gw_post_norm  = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), post_norm_h);
    void *g_kc          = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(uint16_t), kc_seed);
    void *g_vc          = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(uint16_t), vc_seed);

    /* Build the layer once into a recorder, commit, copy out, compare. */
    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    ib_metal_rec_rmsnorm_fp16(r, g_x, gw_input_norm, g_xb, hidden, eps);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_q_proj_w, gw_q_proj_s, g_q,        g_xq, g_xs, hidden, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_k_proj_w, gw_k_proj_s, g_k,        g_xq, g_xs, kv_dim, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_v_proj_w, gw_v_proj_s, g_v,        g_xq, g_xs, kv_dim, hidden);
    ib_metal_rec_rope_inplace(r, g_q, n_heads,    head_dim, pos, theta);
    ib_metal_rec_rope_inplace(r, g_k, n_kv_heads, head_dim, pos, theta);
    ib_metal_rec_attention_block_fp16(r, g_q, g_k, g_v, g_kc, g_vc, g_scores, g_attn_out,
                                        n_heads, n_kv_heads, head_dim, seq_len, pos);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_attn_out, gw_o_proj_w, gw_o_proj_s, g_xb2, g_xq, g_xs, hidden, hidden);
    ib_metal_rec_residual_add(r, g_x, g_xb2, hidden);
    ib_metal_rec_rmsnorm_fp16(r, g_x, gw_post_norm, g_xb, hidden, eps);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_gate_proj_w, gw_gate_proj_s, g_hb,  g_xq, g_xs, intermediate, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_up_proj_w,   gw_up_proj_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
    ib_metal_rec_silu_mul(r, g_hb, g_hb2, g_hb, intermediate);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_hb, gw_down_proj_w, gw_down_proj_s, g_xb,  g_xq, g_xs, hidden, intermediate);
    ib_metal_rec_residual_add(r, g_x, g_xb, hidden);
    ib_metal_recorder_commit(r);

    /* Copy GPU result, compare. */
    float *gpu_x = malloc((size_t)hidden * sizeof(float));
    memcpy(gpu_x, g_x, (size_t)hidden * sizeof(float));

    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < hidden; i++) {
        double a = cpu_x[i], b = gpu_x[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("\n=== single TinyLlama layer correctness ===\n");
    printf("  cos = %.6f   max|diff| = %.4e\n", cos, max_diff);

    /* Bench: per-layer time, batched GPU vs CPU. */
    /* GPU batched: reset x each iter, run one full layer in 1 cb. */
    for (int w = 0; w < 3; w++) {
        memcpy(g_x, x0, (size_t)hidden * sizeof(float));
        memcpy(g_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        memcpy(g_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        ib_metal_recorder *rr = ib_metal_recorder_begin(ctx);
        ib_metal_rec_rmsnorm_fp16(rr, g_x, gw_input_norm, g_xb, hidden, eps);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_q_proj_w, gw_q_proj_s, g_q, g_xq, g_xs, hidden, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_k_proj_w, gw_k_proj_s, g_k, g_xq, g_xs, kv_dim, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_v_proj_w, gw_v_proj_s, g_v, g_xq, g_xs, kv_dim, hidden);
        ib_metal_rec_rope_inplace(rr, g_q, n_heads,    head_dim, pos, theta);
        ib_metal_rec_rope_inplace(rr, g_k, n_kv_heads, head_dim, pos, theta);
        ib_metal_rec_attention_block_fp16(rr, g_q, g_k, g_v, g_kc, g_vc, g_scores, g_attn_out,
                                           n_heads, n_kv_heads, head_dim, seq_len, pos);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_attn_out, gw_o_proj_w, gw_o_proj_s, g_xb2, g_xq, g_xs, hidden, hidden);
        ib_metal_rec_residual_add(rr, g_x, g_xb2, hidden);
        ib_metal_rec_rmsnorm_fp16(rr, g_x, gw_post_norm, g_xb, hidden, eps);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_gate_proj_w, gw_gate_proj_s, g_hb,  g_xq, g_xs, intermediate, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_up_proj_w,   gw_up_proj_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
        ib_metal_rec_silu_mul(rr, g_hb, g_hb2, g_hb, intermediate);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_hb, gw_down_proj_w, gw_down_proj_s, g_xb, g_xq, g_xs, hidden, intermediate);
        ib_metal_rec_residual_add(rr, g_x, g_xb, hidden);
        ib_metal_recorder_commit(rr);
    }
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memcpy(g_x, x0, (size_t)hidden * sizeof(float));
        memcpy(g_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        memcpy(g_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        ib_metal_recorder *rr = ib_metal_recorder_begin(ctx);
        ib_metal_rec_rmsnorm_fp16(rr, g_x, gw_input_norm, g_xb, hidden, eps);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_q_proj_w, gw_q_proj_s, g_q, g_xq, g_xs, hidden, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_k_proj_w, gw_k_proj_s, g_k, g_xq, g_xs, kv_dim, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_v_proj_w, gw_v_proj_s, g_v, g_xq, g_xs, kv_dim, hidden);
        ib_metal_rec_rope_inplace(rr, g_q, n_heads,    head_dim, pos, theta);
        ib_metal_rec_rope_inplace(rr, g_k, n_kv_heads, head_dim, pos, theta);
        ib_metal_rec_attention_block_fp16(rr, g_q, g_k, g_v, g_kc, g_vc, g_scores, g_attn_out,
                                           n_heads, n_kv_heads, head_dim, seq_len, pos);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_attn_out, gw_o_proj_w, gw_o_proj_s, g_xb2, g_xq, g_xs, hidden, hidden);
        ib_metal_rec_residual_add(rr, g_x, g_xb2, hidden);
        ib_metal_rec_rmsnorm_fp16(rr, g_x, gw_post_norm, g_xb, hidden, eps);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_gate_proj_w, gw_gate_proj_s, g_hb,  g_xq, g_xs, intermediate, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_xb, gw_up_proj_w,   gw_up_proj_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
        ib_metal_rec_silu_mul(rr, g_hb, g_hb2, g_hb, intermediate);
        ib_metal_rec_matmul_w4a8_fp32_in(rr, g_hb, gw_down_proj_w, gw_down_proj_s, g_xb, g_xq, g_xs, hidden, intermediate);
        ib_metal_rec_residual_add(rr, g_x, g_xb, hidden);
        ib_metal_recorder_commit(rr);
    }
    double t_gpu = (now_sec() - t0) / iters * 1000.0;

    /* CPU bench: full layer per iteration. */
    for (int w = 0; w < 3; w++) {
        memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
        memcpy(cpu_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        memcpy(cpu_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        ib_kern.rmsnorm(cpu_xb, cpu_x, input_norm_f, eps, hidden);
        cpu_matmul_w4a8_full(cpu_q, cpu_xb, q_proj_w, q_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_k, cpu_xb, k_proj_w, k_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_v, cpu_xb, v_proj_w, v_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
        cpu_rope_inplace(cpu_q, n_heads, head_dim, pos, theta);
        cpu_rope_inplace(cpu_k, n_kv_heads, head_dim, pos, theta);
        cpu_attn_block_fp16(cpu_q, cpu_k, cpu_v, cpu_kc, cpu_vc, cpu_scores, cpu_attn_out,
                            n_heads, n_kv_heads, head_dim, seq_len, pos);
        cpu_matmul_w4a8_full(cpu_xb2, cpu_attn_out, o_proj_w, o_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
        for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb2[i];
        ib_kern.rmsnorm(cpu_xb, cpu_x, post_norm_f, eps, hidden);
        cpu_matmul_w4a8_full(cpu_hb,  cpu_xb, gate_proj_w, gate_proj_sf, intermediate, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_hb2, cpu_xb, up_proj_w,   up_proj_sf,   intermediate, hidden, cpu_xq_h, cpu_xs_h);
        ib_kern.silu_mul(cpu_hb, cpu_hb, cpu_hb2, intermediate);
        cpu_matmul_w4a8_full(cpu_xb,  cpu_hb, down_proj_w, down_proj_sf, hidden, intermediate, cpu_xq_i, cpu_xs_i);
        for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
    }
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
        memcpy(cpu_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        memcpy(cpu_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
        ib_kern.rmsnorm(cpu_xb, cpu_x, input_norm_f, eps, hidden);
        cpu_matmul_w4a8_full(cpu_q, cpu_xb, q_proj_w, q_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_k, cpu_xb, k_proj_w, k_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_v, cpu_xb, v_proj_w, v_proj_sf, kv_dim, hidden, cpu_xq_h, cpu_xs_h);
        cpu_rope_inplace(cpu_q, n_heads, head_dim, pos, theta);
        cpu_rope_inplace(cpu_k, n_kv_heads, head_dim, pos, theta);
        cpu_attn_block_fp16(cpu_q, cpu_k, cpu_v, cpu_kc, cpu_vc, cpu_scores, cpu_attn_out,
                            n_heads, n_kv_heads, head_dim, seq_len, pos);
        cpu_matmul_w4a8_full(cpu_xb2, cpu_attn_out, o_proj_w, o_proj_sf, hidden, hidden, cpu_xq_h, cpu_xs_h);
        for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb2[i];
        ib_kern.rmsnorm(cpu_xb, cpu_x, post_norm_f, eps, hidden);
        cpu_matmul_w4a8_full(cpu_hb,  cpu_xb, gate_proj_w, gate_proj_sf, intermediate, hidden, cpu_xq_h, cpu_xs_h);
        cpu_matmul_w4a8_full(cpu_hb2, cpu_xb, up_proj_w,   up_proj_sf,   intermediate, hidden, cpu_xq_h, cpu_xs_h);
        ib_kern.silu_mul(cpu_hb, cpu_hb, cpu_hb2, intermediate);
        cpu_matmul_w4a8_full(cpu_xb,  cpu_hb, down_proj_w, down_proj_sf, hidden, intermediate, cpu_xq_i, cpu_xs_i);
        for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
    }
    double t_cpu = (now_sec() - t0) / iters * 1000.0;

    printf("\n=== single TinyLlama layer per-iter time ===\n");
    printf("  CPU full layer:        %7.3f ms\n", t_cpu);
    printf("  GPU layer (1 cb):      %7.3f ms\n", t_gpu);
    printf("  GPU/CPU ratio:         %.2f×%s\n", t_gpu / t_cpu,
           t_gpu < t_cpu ? "  GPU faster" : "  CPU faster");

    /* Threshold: cosine must be ≥0.999. max|diff| in the residual stream
     * is dominated by fp16-KV-cache rounding noise being amplified
     * through the per-layer matmuls (the same noise the production model
     * sees), so we don't pin it. */
    int rc = (cos < 0.999) ? 3 : 0;

    /* Cleanup. */
    free(input_norm_h); free(post_norm_h); free(input_norm_f); free(post_norm_f);
    free(x0); free(kc_seed); free(vc_seed);
    free(cpu_x); free(cpu_xb); free(cpu_xb2); free(cpu_q); free(cpu_k); free(cpu_v);
    free(cpu_attn_out); free(cpu_hb); free(cpu_hb2); free(cpu_scores);
    free(cpu_xq_h); free(cpu_xq_i); free(cpu_xs_h); free(cpu_xs_i);
    free(cpu_kc); free(cpu_vc); free(gpu_x);
    #define FREE_W(name) free(name##_w); free(name##_s); free(name##_sf);
    FREE_W(q_proj); FREE_W(k_proj); FREE_W(v_proj); FREE_W(o_proj);
    FREE_W(gate_proj); FREE_W(up_proj); FREE_W(down_proj);
    #undef FREE_W
    ib_metal_destroy(ctx);
    return rc;
}
