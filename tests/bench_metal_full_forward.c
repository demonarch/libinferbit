/* Phase 6c: full N-layer forward pass, ONE command buffer per token.
 *
 * To stay compact: reuse the SAME synthesized weight set across all
 * layers (the kernel cost is identical to a real model, only the
 * uniqueness of values differs — the bench measures dispatch + compute,
 * not memory pressure across 22 distinct weight blocks). KV caches are
 * unique per layer though, since they're separately positionally-indexed.
 *
 * Runs N_LAYERS (default 22 for TinyLlama) layers per "token", times
 * the entire forward, compares against CPU.
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

int main(int argc, char **argv) {
    int hidden       = 2048;
    int intermediate = 5632;
    int n_heads      = 32;
    int n_kv_heads   = 4;
    int head_dim     = 64;
    int n_layers     = (argc > 1) ? atoi(argv[1]) : 22;
    int pos          = (argc > 2) ? atoi(argv[2]) : 100;
    int seq_len      = 256;
    int iters        = (argc > 3) ? atoi(argv[3]) : 20;
    float theta      = 10000.0f, eps = 1e-5f;
    int kv_dim       = n_kv_heads * head_dim;

    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) return 1;
    printf("Device: %s   n_layers=%d  pos=%d  iters=%d\n",
           ib_metal_device_name(ctx), n_layers, pos, iters);

    /* ── ONE shared weight set for all layers. ───────────────────── */
    srand(0xCAFEDEAD);

    /* Helper: alloc + fill an fp16 norm vector. */
    uint16_t *input_norm_h = malloc((size_t)hidden * sizeof(uint16_t));
    uint16_t *post_norm_h  = malloc((size_t)hidden * sizeof(uint16_t));
    float    *input_norm_f = malloc((size_t)hidden * sizeof(float));
    float    *post_norm_f  = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) {
        input_norm_h[i] = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        post_norm_h[i]  = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        input_norm_f[i] = ib_fp16_to_fp32(input_norm_h[i]);
        post_norm_f[i]  = ib_fp16_to_fp32(post_norm_h[i]);
    }

    #define ALLOC_W(name, M, N) \
        uint8_t  *name##_w  = malloc((size_t)(M)*(N)/2); \
        uint16_t *name##_s  = malloc((size_t)(M)*sizeof(uint16_t)); \
        float    *name##_sf = malloc((size_t)(M)*sizeof(float)); \
        for (size_t i = 0; i < (size_t)(M)*(N)/2; i++) name##_w[i] = (uint8_t)(rand() & 0xFF); \
        for (int i = 0; i < (M); i++) { \
            float s = 0.005f + ((rand() & 0xFFFF) / 65535.0f) * 0.05f; \
            name##_s[i]  = f32_to_fp16(s); \
            name##_sf[i] = ib_fp16_to_fp32(name##_s[i]); \
        }
    ALLOC_W(q,    hidden, hidden);
    ALLOC_W(k,    kv_dim, hidden);
    ALLOC_W(v,    kv_dim, hidden);
    ALLOC_W(o,    hidden, hidden);
    ALLOC_W(gate, intermediate, hidden);
    ALLOC_W(up,   intermediate, hidden);
    ALLOC_W(down, hidden, intermediate);
    #undef ALLOC_W

    float *x0 = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) x0[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f);

    /* Pre-seed KV caches (per layer, independent rand). */
    /* libinferbit kv_bits=16 stores fp32 (see ibf_loader.c). */
    float **kc_seed = malloc((size_t)n_layers * sizeof(float*));
    float **vc_seed = malloc((size_t)n_layers * sizeof(float*));
    for (int L = 0; L < n_layers; L++) {
        kc_seed[L] = malloc((size_t)seq_len * kv_dim * sizeof(float));
        vc_seed[L] = malloc((size_t)seq_len * kv_dim * sizeof(float));
        for (size_t i = 0; i < (size_t)seq_len * kv_dim; i++) {
            kc_seed[L][i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f);
            vc_seed[L][i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f);
        }
    }

    /* ── GPU buffers ──────────────────────────────────────────────── */
    void *g_x   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), x0);
    void *g_xb  = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_xb2 = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_q   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_k   = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float), NULL);
    void *g_v   = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float), NULL);
    void *g_attnout = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_hb  = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_hb2 = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_scores = ib_metal_alloc(ctx, (size_t)n_heads * seq_len * sizeof(float), NULL);
    void *g_xq  = ib_metal_alloc(ctx, (size_t)intermediate, NULL);
    void *g_xs  = ib_metal_alloc(ctx, (size_t)((intermediate + 127) / 128) * sizeof(float), NULL);

    void *gw_in   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), input_norm_h);
    void *gw_post = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), post_norm_h);
    #define UPLOAD(name, M, N) \
        void *gw_##name##_w = ib_metal_alloc(ctx, (size_t)(M)*(N)/2, name##_w); \
        void *gw_##name##_s = ib_metal_alloc(ctx, (size_t)(M)*sizeof(uint16_t), name##_s);
    UPLOAD(q, hidden, hidden);
    UPLOAD(k, kv_dim, hidden);
    UPLOAD(v, kv_dim, hidden);
    UPLOAD(o, hidden, hidden);
    UPLOAD(gate, intermediate, hidden);
    UPLOAD(up,   intermediate, hidden);
    UPLOAD(down, hidden, intermediate);
    #undef UPLOAD

    /* Per-layer KV cache buffers. */
    void **g_kc = malloc((size_t)n_layers * sizeof(void*));
    void **g_vc = malloc((size_t)n_layers * sizeof(void*));
    for (int L = 0; L < n_layers; L++) {
        g_kc[L] = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(float), kc_seed[L]);
        g_vc[L] = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(float), vc_seed[L]);
    }

    /* ── Bench: GPU full forward (n_layers in 1 cb). ───────────────── */
    /* warmup */
    for (int w = 0; w < 3; w++) {
        memcpy(g_x, x0, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            memcpy(g_kc[L], kc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(g_vc[L], vc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
        }
        ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rec_rmsnorm_fp16(r, g_x, gw_in, g_xb, hidden, eps);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_q_w, gw_q_s, g_q, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_k_w, gw_k_s, g_k, g_xq, g_xs, kv_dim, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_v_w, gw_v_s, g_v, g_xq, g_xs, kv_dim, hidden);
            ib_metal_rec_rope_inplace(r, g_q, n_heads,    head_dim, pos, theta);
            ib_metal_rec_rope_inplace(r, g_k, n_kv_heads, head_dim, pos, theta);
            ib_metal_rec_attention_block_fp16(r, g_q, g_k, g_v, g_kc[L], g_vc[L], g_scores, g_attnout,
                                                n_heads, n_kv_heads, head_dim, seq_len, pos, 0);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_attnout, gw_o_w, gw_o_s, g_xb2, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_residual_add(r, g_x, g_xb2, hidden);
            ib_metal_rec_rmsnorm_fp16(r, g_x, gw_post, g_xb, hidden, eps);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_gate_w, gw_gate_s, g_hb,  g_xq, g_xs, intermediate, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_up_w,   gw_up_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
            ib_metal_rec_silu_mul(r, g_hb, g_hb2, g_hb, intermediate);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_hb, gw_down_w, gw_down_s, g_xb, g_xq, g_xs, hidden, intermediate);
            ib_metal_rec_residual_add(r, g_x, g_xb, hidden);
        }
        ib_metal_recorder_commit(r);
    }
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memcpy(g_x, x0, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            memcpy(g_kc[L], kc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(g_vc[L], vc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
        }
        ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rec_rmsnorm_fp16(r, g_x, gw_in, g_xb, hidden, eps);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_q_w, gw_q_s, g_q, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_k_w, gw_k_s, g_k, g_xq, g_xs, kv_dim, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_v_w, gw_v_s, g_v, g_xq, g_xs, kv_dim, hidden);
            ib_metal_rec_rope_inplace(r, g_q, n_heads,    head_dim, pos, theta);
            ib_metal_rec_rope_inplace(r, g_k, n_kv_heads, head_dim, pos, theta);
            ib_metal_rec_attention_block_fp16(r, g_q, g_k, g_v, g_kc[L], g_vc[L], g_scores, g_attnout,
                                                n_heads, n_kv_heads, head_dim, seq_len, pos, 0);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_attnout, gw_o_w, gw_o_s, g_xb2, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_residual_add(r, g_x, g_xb2, hidden);
            ib_metal_rec_rmsnorm_fp16(r, g_x, gw_post, g_xb, hidden, eps);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_gate_w, gw_gate_s, g_hb,  g_xq, g_xs, intermediate, hidden);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, gw_up_w,   gw_up_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
            ib_metal_rec_silu_mul(r, g_hb, g_hb2, g_hb, intermediate);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_hb, gw_down_w, gw_down_s, g_xb, g_xq, g_xs, hidden, intermediate);
            ib_metal_rec_residual_add(r, g_x, g_xb, hidden);
        }
        ib_metal_recorder_commit(r);
    }
    double t_gpu = (now_sec() - t0) / iters * 1000.0;

    /* ── CPU full forward ────────────────────────────────────────── */
    float *cpu_x       = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb      = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb2     = malloc((size_t)hidden * sizeof(float));
    float *cpu_q       = malloc((size_t)hidden * sizeof(float));
    float *cpu_k       = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v       = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_attnout = malloc((size_t)hidden * sizeof(float));
    float *cpu_hb      = malloc((size_t)intermediate * sizeof(float));
    float *cpu_hb2     = malloc((size_t)intermediate * sizeof(float));
    float *cpu_scores  = malloc((size_t)n_heads * (pos + 1) * sizeof(float));
    int8_t *xqh = malloc((size_t)hidden);
    int8_t *xqi = malloc((size_t)intermediate);
    float  *xsh = malloc((size_t)((hidden + 127) / 128) * sizeof(float));
    float  *xsi = malloc((size_t)((intermediate + 127) / 128) * sizeof(float));
    float **cpu_kc = malloc((size_t)n_layers * sizeof(float*));
    float **cpu_vc = malloc((size_t)n_layers * sizeof(float*));
    for (int L = 0; L < n_layers; L++) {
        cpu_kc[L] = malloc((size_t)seq_len * kv_dim * sizeof(float));
        cpu_vc[L] = malloc((size_t)seq_len * kv_dim * sizeof(float));
    }

    for (int w = 0; w < 3; w++) {
        memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            memcpy(cpu_kc[L], kc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(cpu_vc[L], vc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
        }
        for (int L = 0; L < n_layers; L++) {
            ib_kern.rmsnorm(cpu_xb, cpu_x, input_norm_f, eps, hidden);
            ib_quantize_input_int8_g128(cpu_xb, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_q, q_w, q_sf, xqh, xsh, hidden, hidden);
            ib_kern.matmul_w4a8(cpu_k, k_w, k_sf, xqh, xsh, kv_dim, hidden);
            ib_kern.matmul_w4a8(cpu_v, v_w, v_sf, xqh, xsh, kv_dim, hidden);
            for (int h = 0; h < n_heads; h++) {
                float *q_h = cpu_q + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(angle), s = sinf(angle);
                    float v0 = q_h[i], v1 = q_h[i+1];
                    q_h[i]   = v0*c - v1*s; q_h[i+1] = v0*s + v1*c;
                }
            }
            for (int h = 0; h < n_kv_heads; h++) {
                float *k_h = cpu_k + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(angle), s = sinf(angle);
                    float v0 = k_h[i], v1 = k_h[i+1];
                    k_h[i]   = v0*c - v1*s; k_h[i+1] = v0*s + v1*c;
                }
            }
            /* attention */
            int p1 = pos + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int i = 0; i < kv_dim; i++) {
                cpu_kc[L][(size_t)pos * kv_dim + i] = cpu_k[i];
                cpu_vc[L][(size_t)pos * kv_dim + i] = cpu_v[i];
            }
            for (int h = 0; h < n_heads; h++) {
                int kvh = h / (n_heads / n_kv_heads);
                float *q_h = cpu_q + h * head_dim;
                for (int t = 0; t <= pos; t++) {
                    float *k_t = cpu_kc[L] + (size_t)t * kv_dim + kvh * head_dim;
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
                    cpu_scores[h * p1 + t] = s * scale;
                }
            }
            for (int h = 0; h < n_heads; h++) {
                float *row = cpu_scores + h * p1;
                float m = row[0];
                for (int t = 1; t < p1; t++) if (row[t] > m) m = row[t];
                float sum = 0.0f;
                for (int t = 0; t < p1; t++) { row[t] = expf(row[t]-m); sum += row[t]; }
                float inv = 1.0f/sum;
                for (int t = 0; t < p1; t++) row[t] *= inv;
            }
            for (int h = 0; h < n_heads; h++) {
                int kvh = h / (n_heads / n_kv_heads);
                float *s = cpu_scores + h * p1;
                for (int d = 0; d < head_dim; d++) {
                    float acc = 0.0f;
                    for (int t = 0; t < p1; t++)
                        acc += s[t] * cpu_vc[L][(size_t)t * kv_dim + kvh * head_dim + d];
                    cpu_attnout[h * head_dim + d] = acc;
                }
            }
            ib_quantize_input_int8_g128(cpu_attnout, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_xb2, o_w, o_sf, xqh, xsh, hidden, hidden);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb2[i];
            ib_kern.rmsnorm(cpu_xb, cpu_x, post_norm_f, eps, hidden);
            ib_quantize_input_int8_g128(cpu_xb, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_hb,  gate_w, gate_sf, xqh, xsh, intermediate, hidden);
            ib_kern.matmul_w4a8(cpu_hb2, up_w,   up_sf,   xqh, xsh, intermediate, hidden);
            ib_kern.silu_mul(cpu_hb, cpu_hb, cpu_hb2, intermediate);
            ib_quantize_input_int8_g128(cpu_hb, xqi, xsi, intermediate);
            ib_kern.matmul_w4a8(cpu_xb, down_w, down_sf, xqi, xsi, hidden, intermediate);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
        }
    }
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            memcpy(cpu_kc[L], kc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(cpu_vc[L], vc_seed[L], (size_t)seq_len * kv_dim * sizeof(float));
        }
        for (int L = 0; L < n_layers; L++) {
            ib_kern.rmsnorm(cpu_xb, cpu_x, input_norm_f, eps, hidden);
            ib_quantize_input_int8_g128(cpu_xb, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_q, q_w, q_sf, xqh, xsh, hidden, hidden);
            ib_kern.matmul_w4a8(cpu_k, k_w, k_sf, xqh, xsh, kv_dim, hidden);
            ib_kern.matmul_w4a8(cpu_v, v_w, v_sf, xqh, xsh, kv_dim, hidden);
            for (int h = 0; h < n_heads; h++) {
                float *q_h = cpu_q + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(angle), s = sinf(angle);
                    float v0 = q_h[i], v1 = q_h[i+1];
                    q_h[i] = v0*c - v1*s; q_h[i+1] = v0*s + v1*c;
                }
            }
            for (int h = 0; h < n_kv_heads; h++) {
                float *k_h = cpu_k + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
                    float c = cosf(angle), s = sinf(angle);
                    float v0 = k_h[i], v1 = k_h[i+1];
                    k_h[i] = v0*c - v1*s; k_h[i+1] = v0*s + v1*c;
                }
            }
            int p1 = pos + 1;
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int i = 0; i < kv_dim; i++) {
                cpu_kc[L][(size_t)pos * kv_dim + i] = cpu_k[i];
                cpu_vc[L][(size_t)pos * kv_dim + i] = cpu_v[i];
            }
            for (int h = 0; h < n_heads; h++) {
                int kvh = h / (n_heads / n_kv_heads);
                float *q_h = cpu_q + h * head_dim;
                for (int t = 0; t <= pos; t++) {
                    float *k_t = cpu_kc[L] + (size_t)t * kv_dim + kvh * head_dim;
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; d++) s += q_h[d] * k_t[d];
                    cpu_scores[h * p1 + t] = s * scale;
                }
            }
            for (int h = 0; h < n_heads; h++) {
                float *row = cpu_scores + h * p1;
                float m = row[0];
                for (int t = 1; t < p1; t++) if (row[t] > m) m = row[t];
                float sum = 0.0f;
                for (int t = 0; t < p1; t++) { row[t] = expf(row[t]-m); sum += row[t]; }
                float inv = 1.0f/sum;
                for (int t = 0; t < p1; t++) row[t] *= inv;
            }
            for (int h = 0; h < n_heads; h++) {
                int kvh = h / (n_heads / n_kv_heads);
                float *s = cpu_scores + h * p1;
                for (int d = 0; d < head_dim; d++) {
                    float acc = 0.0f;
                    for (int t = 0; t < p1; t++)
                        acc += s[t] * cpu_vc[L][(size_t)t * kv_dim + kvh * head_dim + d];
                    cpu_attnout[h * head_dim + d] = acc;
                }
            }
            ib_quantize_input_int8_g128(cpu_attnout, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_xb2, o_w, o_sf, xqh, xsh, hidden, hidden);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb2[i];
            ib_kern.rmsnorm(cpu_xb, cpu_x, post_norm_f, eps, hidden);
            ib_quantize_input_int8_g128(cpu_xb, xqh, xsh, hidden);
            ib_kern.matmul_w4a8(cpu_hb,  gate_w, gate_sf, xqh, xsh, intermediate, hidden);
            ib_kern.matmul_w4a8(cpu_hb2, up_w,   up_sf,   xqh, xsh, intermediate, hidden);
            ib_kern.silu_mul(cpu_hb, cpu_hb, cpu_hb2, intermediate);
            ib_quantize_input_int8_g128(cpu_hb, xqi, xsi, intermediate);
            ib_kern.matmul_w4a8(cpu_xb, down_w, down_sf, xqi, xsi, hidden, intermediate);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
        }
    }
    double t_cpu = (now_sec() - t0) / iters * 1000.0;

    /* Compare last GPU result to last CPU result. */
    float *gpu_x = malloc((size_t)hidden * sizeof(float));
    memcpy(gpu_x, g_x, (size_t)hidden * sizeof(float));
    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < hidden; i++) {
        double a = cpu_x[i], b = gpu_x[i];
        dot += a*b; na += a*a; nb += b*b;
        double d = fabs(a-b); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));

    printf("\n=== full %d-layer forward (TinyLlama shape) ===\n", n_layers);
    printf("  cos(CPU, GPU)         = %.6f\n", cos);
    printf("  max|diff|             = %.4e\n", max_diff);
    printf("  CPU full forward:    %8.3f ms/token\n", t_cpu);
    printf("  GPU full forward:    %8.3f ms/token  (1 cb of %d kernels)\n",
           t_gpu, n_layers * 16);
    printf("  Speedup:               %.2f×%s\n", t_cpu / t_gpu,
           t_gpu < t_cpu ? "  GPU faster" : "  CPU faster");
    printf("  Tokens/sec  CPU:      %7.1f\n", 1000.0 / t_cpu);
    printf("  Tokens/sec  GPU:      %7.1f\n", 1000.0 / t_gpu);

    /* Cleanup. */
    free(input_norm_h); free(post_norm_h); free(input_norm_f); free(post_norm_f);
    free(x0);
    for (int L = 0; L < n_layers; L++) { free(kc_seed[L]); free(vc_seed[L]); free(cpu_kc[L]); free(cpu_vc[L]); }
    free(kc_seed); free(vc_seed); free(cpu_kc); free(cpu_vc);
    free(cpu_x); free(cpu_xb); free(cpu_xb2); free(cpu_q); free(cpu_k); free(cpu_v);
    free(cpu_attnout); free(cpu_hb); free(cpu_hb2); free(cpu_scores);
    free(xqh); free(xqi); free(xsh); free(xsi); free(gpu_x);
    #define FREE_W(name) free(name##_w); free(name##_s); free(name##_sf);
    FREE_W(q); FREE_W(k); FREE_W(v); FREE_W(o); FREE_W(gate); FREE_W(up); FREE_W(down);
    #undef FREE_W
    ib_metal_destroy(ctx);
    return (cos < 0.999) ? 3 : 0;
}
