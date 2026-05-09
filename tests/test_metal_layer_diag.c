/* Layer diagnostic: snapshot GPU buffers at every stage; compare each. */
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

static void compare(const char *tag, const float *a, const float *b, int N) {
    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < N; i++) {
        double da = a[i], db = b[i];
        dot += da*db; na += da*da; nb += db*db;
        double d = fabs(da-db); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));
    printf("  %-15s  N=%5d  cos=%.6f  max|diff|=%.4e\n", tag, N, cos, max_diff);
}

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

static void cpu_matmul_w4a8_full(float *out, const float *x_fp32,
                                  const uint8_t *W, const float *W_s,
                                  int M, int N, int8_t *xq_buf, float *xs_buf) {
    ib_quantize_input_int8_g128(x_fp32, xq_buf, xs_buf, N);
    ib_kern.matmul_w4a8(out, W, W_s, xq_buf, xs_buf, M, N);
}

static void cpu_attn_block_fp16(
    const float *q, const float *k, const float *v,
    uint16_t *kc, uint16_t *vc, float *scores, float *out,
    int n_heads, int n_kv_heads, int head_dim, int seq_len, int pos)
{
    int kv_dim = n_kv_heads * head_dim, hpk = n_heads / n_kv_heads, p1 = pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    (void)seq_len;
    for (int i = 0; i < kv_dim; i++) {
        kc[(size_t)pos * kv_dim + i] = f32_to_fp16(k[i]);
        vc[(size_t)pos * kv_dim + i] = f32_to_fp16(v[i]);
    }
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / hpk;
        const float *q_h = q + h * head_dim;
        for (int t = 0; t <= pos; t++) {
            const uint16_t *k_t = kc + (size_t)t * kv_dim + kv_h * head_dim;
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
        float inv = 1.0f / sum;
        for (int t = 0; t < p1; t++) row[t] *= inv;
    }
    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / hpk;
        const float *s = scores + h * p1;
        for (int d = 0; d < head_dim; d++) {
            float acc = 0.0f;
            for (int t = 0; t < p1; t++)
                acc += s[t] * ib_fp16_to_fp32(vc[(size_t)t * kv_dim + kv_h * head_dim + d]);
            out[h * head_dim + d] = acc;
        }
    }
}

int main(void) {
    int hidden = 2048, intermediate = 5632;
    int n_heads = 32, n_kv_heads = 4, head_dim = 64;
    int kv_dim = n_kv_heads * head_dim;
    int seq_len = 256, pos = 50;
    float theta = 10000.0f, eps = 1e-5f;

    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) return 1;

    /* Allocate weights. */
    srand(0x1AB7E11A);
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

    uint16_t *norm_h = malloc((size_t)hidden * sizeof(uint16_t));
    float    *norm_f = malloc((size_t)hidden * sizeof(float));
    uint16_t *post_h = malloc((size_t)hidden * sizeof(uint16_t));
    float    *post_f = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) {
        norm_h[i] = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        post_h[i] = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
        norm_f[i] = ib_fp16_to_fp32(norm_h[i]);
        post_f[i] = ib_fp16_to_fp32(post_h[i]);
    }
    float *x0 = malloc((size_t)hidden * sizeof(float));
    for (int i = 0; i < hidden; i++) x0[i] = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;
    uint16_t *kc_seed = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *vc_seed = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    for (size_t i = 0; i < (size_t)seq_len * kv_dim; i++) {
        kc_seed[i] = f32_to_fp16(((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f);
        vc_seed[i] = f32_to_fp16(((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f);
    }

    /* CPU pipeline (snapshot all intermediates). */
    float *cpu_x       = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb_a    = malloc((size_t)hidden * sizeof(float));
    float *cpu_q_buf   = malloc((size_t)hidden * sizeof(float));
    float *cpu_k_buf   = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_v_buf   = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_q_rope  = malloc((size_t)hidden * sizeof(float));
    float *cpu_k_rope  = malloc((size_t)kv_dim * sizeof(float));
    float *cpu_attnout = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb2     = malloc((size_t)hidden * sizeof(float));
    float *cpu_xafter1 = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb_p    = malloc((size_t)hidden * sizeof(float));
    float *cpu_hb_buf  = malloc((size_t)intermediate * sizeof(float));
    float *cpu_hb2_buf = malloc((size_t)intermediate * sizeof(float));
    float *cpu_hb_silu = malloc((size_t)intermediate * sizeof(float));
    float *cpu_xb_d    = malloc((size_t)hidden * sizeof(float));
    float *cpu_x_final = malloc((size_t)hidden * sizeof(float));
    float *cpu_scores  = malloc((size_t)n_heads * (pos + 1) * sizeof(float));
    int8_t *xqh = malloc((size_t)hidden);
    int8_t *xqi = malloc((size_t)intermediate);
    float  *xsh = malloc((size_t)((hidden + 127) / 128) * sizeof(float));
    float  *xsi = malloc((size_t)((intermediate + 127) / 128) * sizeof(float));
    uint16_t *cpu_kc = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));
    uint16_t *cpu_vc = malloc((size_t)seq_len * kv_dim * sizeof(uint16_t));

    memcpy(cpu_x, x0, (size_t)hidden * sizeof(float));
    memcpy(cpu_kc, kc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
    memcpy(cpu_vc, vc_seed, (size_t)seq_len * kv_dim * sizeof(uint16_t));
    ib_kern.rmsnorm(cpu_xb_a, cpu_x, norm_f, eps, hidden);
    cpu_matmul_w4a8_full(cpu_q_buf, cpu_xb_a, q_w, q_sf, hidden, hidden, xqh, xsh);
    cpu_matmul_w4a8_full(cpu_k_buf, cpu_xb_a, k_w, k_sf, kv_dim, hidden, xqh, xsh);
    cpu_matmul_w4a8_full(cpu_v_buf, cpu_xb_a, v_w, v_sf, kv_dim, hidden, xqh, xsh);
    memcpy(cpu_q_rope, cpu_q_buf, (size_t)hidden * sizeof(float));
    memcpy(cpu_k_rope, cpu_k_buf, (size_t)kv_dim * sizeof(float));
    cpu_rope_inplace(cpu_q_rope, n_heads,    head_dim, pos, theta);
    cpu_rope_inplace(cpu_k_rope, n_kv_heads, head_dim, pos, theta);
    cpu_attn_block_fp16(cpu_q_rope, cpu_k_rope, cpu_v_buf, cpu_kc, cpu_vc, cpu_scores, cpu_attnout,
                        n_heads, n_kv_heads, head_dim, seq_len, pos);
    cpu_matmul_w4a8_full(cpu_xb2, cpu_attnout, o_w, o_sf, hidden, hidden, xqh, xsh);
    for (int i = 0; i < hidden; i++) cpu_xafter1[i] = cpu_x[i] + cpu_xb2[i];
    ib_kern.rmsnorm(cpu_xb_p, cpu_xafter1, post_f, eps, hidden);
    cpu_matmul_w4a8_full(cpu_hb_buf,  cpu_xb_p, gate_w, gate_sf, intermediate, hidden, xqh, xsh);
    cpu_matmul_w4a8_full(cpu_hb2_buf, cpu_xb_p, up_w,   up_sf,   intermediate, hidden, xqh, xsh);
    memcpy(cpu_hb_silu, cpu_hb_buf, (size_t)intermediate * sizeof(float));
    ib_kern.silu_mul(cpu_hb_silu, cpu_hb_silu, cpu_hb2_buf, intermediate);
    cpu_matmul_w4a8_full(cpu_xb_d, cpu_hb_silu, down_w, down_sf, hidden, intermediate, xqi, xsi);
    for (int i = 0; i < hidden; i++) cpu_x_final[i] = cpu_xafter1[i] + cpu_xb_d[i];

    /* GPU: build the layer with separate output buffers per stage so we
     * can read each one back individually. */
    void *g_x   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), x0);
    void *g_xba = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_q   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_k   = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float), NULL);
    void *g_v   = ib_metal_alloc(ctx, (size_t)kv_dim * sizeof(float), NULL);
    void *g_attnout = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_xb2 = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_xbp = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_hb  = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_hb2 = ib_metal_alloc(ctx, (size_t)intermediate * sizeof(float), NULL);
    void *g_xbd = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_scores = ib_metal_alloc(ctx, (size_t)n_heads * seq_len * sizeof(float), NULL);
    void *g_xq  = ib_metal_alloc(ctx, (size_t)intermediate, NULL);
    void *g_xs  = ib_metal_alloc(ctx, (size_t)((intermediate + 127) / 128) * sizeof(float), NULL);

    void *gw_norm = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), norm_h);
    void *gw_post = ib_metal_alloc(ctx, (size_t)hidden * sizeof(uint16_t), post_h);
    #define UPLOAD_W(name, M, N) \
        void *gw_##name##_w = ib_metal_alloc(ctx, (size_t)(M)*(N)/2, name##_w); \
        void *gw_##name##_s = ib_metal_alloc(ctx, (size_t)(M)*sizeof(uint16_t), name##_s);
    UPLOAD_W(q, hidden, hidden);
    UPLOAD_W(k, kv_dim, hidden);
    UPLOAD_W(v, kv_dim, hidden);
    UPLOAD_W(o, hidden, hidden);
    UPLOAD_W(gate, intermediate, hidden);
    UPLOAD_W(up,   intermediate, hidden);
    UPLOAD_W(down, hidden, intermediate);
    #undef UPLOAD_W
    void *g_kc = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(uint16_t), kc_seed);
    void *g_vc = ib_metal_alloc(ctx, (size_t)seq_len * kv_dim * sizeof(uint16_t), vc_seed);

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    ib_metal_rec_rmsnorm_fp16(r, g_x, gw_norm, g_xba, hidden, eps);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xba, gw_q_w, gw_q_s, g_q, g_xq, g_xs, hidden, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xba, gw_k_w, gw_k_s, g_k, g_xq, g_xs, kv_dim, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xba, gw_v_w, gw_v_s, g_v, g_xq, g_xs, kv_dim, hidden);
    ib_metal_rec_rope_inplace(r, g_q, n_heads,    head_dim, pos, theta);
    ib_metal_rec_rope_inplace(r, g_k, n_kv_heads, head_dim, pos, theta);
    ib_metal_rec_attention_block_fp16(r, g_q, g_k, g_v, g_kc, g_vc, g_scores, g_attnout,
                                       n_heads, n_kv_heads, head_dim, seq_len, pos);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_attnout, gw_o_w, gw_o_s, g_xb2, g_xq, g_xs, hidden, hidden);
    ib_metal_rec_residual_add(r, g_x, g_xb2, hidden);
    ib_metal_rec_rmsnorm_fp16(r, g_x, gw_post, g_xbp, hidden, eps);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xbp, gw_gate_w, gw_gate_s, g_hb,  g_xq, g_xs, intermediate, hidden);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_xbp, gw_up_w,   gw_up_s,   g_hb2, g_xq, g_xs, intermediate, hidden);
    ib_metal_rec_silu_mul(r, g_hb, g_hb2, g_hb, intermediate);
    ib_metal_rec_matmul_w4a8_fp32_in(r, g_hb, gw_down_w, gw_down_s, g_xbd, g_xq, g_xs, hidden, intermediate);
    ib_metal_rec_residual_add(r, g_x, g_xbd, hidden);
    ib_metal_recorder_commit(r);

    /* Per-stage compare. */
    printf("Stage-by-stage comparison (pos=%d):\n", pos);
    compare("rmsnorm",   cpu_xb_a,   (float*)g_xba,    hidden);
    compare("q_proj",    cpu_q_buf,  (float*)g_q,      hidden);  /* before rope */
    /* Note: g_q got rotated in-place, so cpu_q_buf doesn't match anymore.
     * Compare cpu_q_rope against g_q now. */
    compare("q after rope", cpu_q_rope, (float*)g_q,   hidden);
    compare("k_proj",    cpu_k_buf,  (float*)g_k,      kv_dim);
    compare("k after rope", cpu_k_rope, (float*)g_k,   kv_dim);
    compare("v_proj",    cpu_v_buf,  (float*)g_v,      kv_dim);
    compare("attn_out",  cpu_attnout,(float*)g_attnout,hidden);
    compare("o_proj",    cpu_xb2,    (float*)g_xb2,    hidden);
    compare("post_norm", cpu_xb_p,   (float*)g_xbp,    hidden);
    compare("gate_proj", cpu_hb_buf, (float*)g_hb,     intermediate);  /* but g_hb got silu'd */
    /* g_hb is now silu(g_hb)*g_hb2, so compare cpu_hb_silu */
    compare("after silu",cpu_hb_silu,(float*)g_hb,     intermediate);
    compare("up_proj",   cpu_hb2_buf,(float*)g_hb2,    intermediate);
    compare("down_proj", cpu_xb_d,   (float*)g_xbd,    hidden);
    compare("final x",   cpu_x_final,(float*)g_x,      hidden);

    ib_metal_destroy(ctx);
    return 0;
}
