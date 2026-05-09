/* Phase 7: IBF-model → Metal upload + per-token forward dispatcher.
 *
 * Walks an inferbit_model loaded by the standard IBF loader, allocates
 * Metal buffers for every weight/scale tensor and per-layer KV cache,
 * then exposes ib_metal_forward_token() which records the full forward
 * into ONE command buffer per token.
 *
 * Supports mixed-precision IBFs:
 *   - matmul tensors: bits=4 (INT4 packed nibbles, w4a8) or bits=8 (INT8)
 *   - norm tensors: bits=16 (fp16)
 *   - output_head: bits=4 or bits=8
 *   - kv_bits: 16 (fp32 storage) or 8 (per-head INT8 with fp32 scales)
 *   - token_embedding: any bit width — decoded on CPU per token
 */
#include "metal_runtime.h"
#include "../inferbit_internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Per-layer GPU buffer set. Each weight tensor's bits is recorded so the
 * forward dispatcher can pick the right kernel per matmul. */
struct layer_bufs {
    void *q_w, *q_s;     int q_bits;
    void *k_w, *k_s;     int k_bits;
    void *v_w, *v_s;     int v_bits;
    void *o_w, *o_s;     int o_bits;
    void *gate_w, *gate_s;  int gate_bits;
    void *up_w,   *up_s;    int up_bits;
    void *down_w, *down_s;  int down_bits;
    void *input_norm;
    void *post_norm;
    /* KV cache: layout depends on kv_bits.
     *   kv_bits=16: k_cache/v_cache hold fp32 [seq_len, kv_dim]; scales NULL.
     *   kv_bits=8:  k_cache/v_cache hold int8 [seq_len, kv_dim]; scales fp32 [seq_len, n_kv_heads]. */
    void *k_cache;
    void *v_cache;
    void *k_scales;
    void *v_scales;
};

struct ib_metal_model_buffers {
    int num_layers;
    int hidden;
    int intermediate;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int kv_dim;
    int vocab;
    int seq_len;
    int kv_bits;       /* 16 (fp32 KV) or 8 (int8 KV) */
    float rope_theta;
    float eps;

    const inferbit_model *model;

    struct layer_bufs *layers;

    void *output_norm;
    void *output_head_w;
    void *output_head_s;
    int   output_head_bits;

    /* State buffers (reused across layers). */
    void *x;
    void *xb;
    void *xb2;
    void *q;
    void *k;
    void *v;
    void *attn_out;
    void *hb;
    void *hb2;
    void *scores;
    void *xq;
    void *xs;
    void *logits;
};

/* Verifies the IBF is in a layout the GPU dispatcher supports. */
static int model_is_supported(const inferbit_model *m, char *err, size_t err_sz) {
    #define CHECK(cond, msg) do { \
        if (!(cond)) { snprintf(err, err_sz, "%s", msg); return 0; } \
    } while (0)

    int kvb = m->header.kv_bits;
    CHECK(kvb == 16 || kvb == 8, "Metal forward requires kv_bits=16 or kv_bits=8");
    CHECK(m->output_norm.bits == 16, "output_norm must be fp16");
    CHECK((m->output_head.bits == 4 || m->output_head.bits == 8) && m->output_head.pq == NULL,
          "output_head must be INT4 or INT8 without PQ");
    for (int L = 0; L < m->header.num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        #define BIT4or8(name) \
            CHECK((lm->name.bits == 4 || lm->name.bits == 8) && lm->name.pq == NULL, \
                  "layer matmul tensor must be INT4 or INT8 (no PQ): " #name)
        BIT4or8(q_proj); BIT4or8(k_proj); BIT4or8(v_proj); BIT4or8(o_proj);
        BIT4or8(gate_proj); BIT4or8(up_proj); BIT4or8(down_proj);
        #undef BIT4or8
        CHECK(lm->input_norm.bits == 16,    "input_norm must be fp16");
        CHECK(lm->post_attn_norm.bits == 16, "post_attn_norm must be fp16");
        CHECK(lm->sparsity_mask_size == 0,
              "sparsity not supported on the Metal forward path");
    }
    #undef CHECK
    return 1;
}

static void upload_w_pair(ib_metal_ctx *ctx, const inferbit_model *m,
                           const ib_tensor_meta *t,
                           void **out_w, void **out_s)
{
    const uint8_t *base = (const uint8_t *)m->weight_data;
    *out_w = ib_metal_alloc(ctx, t->size, base + t->offset);
    if (t->scale_size > 0) {
        *out_s = ib_metal_alloc(ctx, t->scale_size, base + t->scale_offset);
    } else {
        *out_s = NULL;
    }
}

static void upload_norm(ib_metal_ctx *ctx, const inferbit_model *m,
                         const ib_tensor_meta *t, void **out)
{
    const uint8_t *base = (const uint8_t *)m->weight_data;
    *out = ib_metal_alloc(ctx, t->size, base + t->offset);
}

extern "C" ib_metal_model_buffers *
ib_metal_upload_model(ib_metal_ctx *ctx, const void *model_handle)
{
    if (!ctx || !model_handle) return nullptr;
    const inferbit_model *m = (const inferbit_model *)model_handle;

    char err[160];
    if (!model_is_supported(m, err, sizeof(err))) {
        fprintf(stderr, "ib_metal_upload_model: %s\n", err);
        return nullptr;
    }

    ib_metal_model_buffers *b = (ib_metal_model_buffers *)
        calloc(1, sizeof(*b));
    if (!b) return nullptr;

    b->num_layers   = m->header.num_layers;
    b->hidden       = m->header.hidden_size;
    b->intermediate = m->header.intermediate_size;
    b->n_heads      = m->header.num_heads;
    b->n_kv_heads   = m->header.num_kv_heads;
    b->head_dim     = m->header.head_dim;
    b->kv_dim       = b->n_kv_heads * b->head_dim;
    b->vocab        = m->header.vocab_size;
    /* Use the configured KV-cache capacity (set via
     * inferbit_config_set_context_length), not the model's max — Llama-3
     * IBFs can advertise max_context_length=131072 which would balloon
     * the GPU KV cache to gigabytes. The CPU loader sizes its KV
     * buffers identically. */
    b->seq_len      = m->kv_caches ? m->kv_caches[0].capacity : m->header.max_context_length;
    if (b->seq_len <= 0) b->seq_len = m->header.max_context_length;
    b->kv_bits      = m->header.kv_bits;
    b->rope_theta   = m->header.rope_theta;
    b->eps          = m->header.norm_epsilon;
    b->model        = m;

    /* Per-layer weights + KV caches. */
    b->layers = (struct layer_bufs *)
        calloc((size_t)b->num_layers, sizeof(struct layer_bufs));
    if (!b->layers) { free(b); return nullptr; }

    /* KV cache element size depends on kv_bits. fp32 (kvb=16) is what
     * libinferbit actually stores in CPU; int8 + per-head scale for kvb=8. */
    size_t kv_elem = (b->kv_bits == 16) ? sizeof(float) : 1;
    size_t kv_bytes_per_layer =
        (size_t)b->seq_len * (size_t)b->kv_dim * kv_elem;
    size_t kv_scales_bytes = (size_t)b->seq_len * (size_t)b->n_kv_heads * sizeof(float);

    for (int L = 0; L < b->num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        struct layer_bufs *lb = &b->layers[L];
        upload_w_pair(ctx, m, &lm->q_proj,    &lb->q_w,    &lb->q_s);    lb->q_bits    = lm->q_proj.bits;
        upload_w_pair(ctx, m, &lm->k_proj,    &lb->k_w,    &lb->k_s);    lb->k_bits    = lm->k_proj.bits;
        upload_w_pair(ctx, m, &lm->v_proj,    &lb->v_w,    &lb->v_s);    lb->v_bits    = lm->v_proj.bits;
        upload_w_pair(ctx, m, &lm->o_proj,    &lb->o_w,    &lb->o_s);    lb->o_bits    = lm->o_proj.bits;
        upload_w_pair(ctx, m, &lm->gate_proj, &lb->gate_w, &lb->gate_s); lb->gate_bits = lm->gate_proj.bits;
        upload_w_pair(ctx, m, &lm->up_proj,   &lb->up_w,   &lb->up_s);   lb->up_bits   = lm->up_proj.bits;
        upload_w_pair(ctx, m, &lm->down_proj, &lb->down_w, &lb->down_s); lb->down_bits = lm->down_proj.bits;
        upload_norm  (ctx, m, &lm->input_norm,     &lb->input_norm);
        upload_norm  (ctx, m, &lm->post_attn_norm, &lb->post_norm);
        lb->k_cache = ib_metal_alloc(ctx, kv_bytes_per_layer, NULL);
        lb->v_cache = ib_metal_alloc(ctx, kv_bytes_per_layer, NULL);
        if (lb->k_cache) memset(lb->k_cache, 0, kv_bytes_per_layer);
        if (lb->v_cache) memset(lb->v_cache, 0, kv_bytes_per_layer);
        if (b->kv_bits == 8) {
            lb->k_scales = ib_metal_alloc(ctx, kv_scales_bytes, NULL);
            lb->v_scales = ib_metal_alloc(ctx, kv_scales_bytes, NULL);
            if (lb->k_scales) memset(lb->k_scales, 0, kv_scales_bytes);
            if (lb->v_scales) memset(lb->v_scales, 0, kv_scales_bytes);
        }
    }

    upload_norm(ctx, m, &m->output_norm, &b->output_norm);
    upload_w_pair(ctx, m, &m->output_head, &b->output_head_w, &b->output_head_s);
    b->output_head_bits = m->output_head.bits;

    int max_n = b->intermediate > b->hidden ? b->intermediate : b->hidden;
    int xs_groups = (max_n + 127) / 128;
    b->x        = ib_metal_alloc(ctx, (size_t)b->hidden * sizeof(float), NULL);
    b->xb       = ib_metal_alloc(ctx, (size_t)b->hidden * sizeof(float), NULL);
    b->xb2      = ib_metal_alloc(ctx, (size_t)b->hidden * sizeof(float), NULL);
    b->q        = ib_metal_alloc(ctx, (size_t)b->n_heads * b->head_dim * sizeof(float), NULL);
    b->k        = ib_metal_alloc(ctx, (size_t)b->kv_dim * sizeof(float), NULL);
    b->v        = ib_metal_alloc(ctx, (size_t)b->kv_dim * sizeof(float), NULL);
    b->attn_out = ib_metal_alloc(ctx, (size_t)b->n_heads * b->head_dim * sizeof(float), NULL);
    b->hb       = ib_metal_alloc(ctx, (size_t)b->intermediate * sizeof(float), NULL);
    b->hb2      = ib_metal_alloc(ctx, (size_t)b->intermediate * sizeof(float), NULL);
    b->scores   = ib_metal_alloc(ctx, (size_t)b->n_heads * b->seq_len * sizeof(float), NULL);
    b->xq       = ib_metal_alloc(ctx, (size_t)max_n, NULL);
    b->xs       = ib_metal_alloc(ctx, (size_t)xs_groups * sizeof(float), NULL);
    b->logits   = ib_metal_alloc(ctx, (size_t)b->vocab * sizeof(float), NULL);

    return b;
}

extern "C" void
ib_metal_release_model(ib_metal_ctx *ctx, ib_metal_model_buffers *b)
{
    if (!ctx || !b) return;
    #define FR(p) do { if (p) ib_metal_free(ctx, p); } while (0)
    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];
        FR(lb->q_w); FR(lb->q_s);
        FR(lb->k_w); FR(lb->k_s);
        FR(lb->v_w); FR(lb->v_s);
        FR(lb->o_w); FR(lb->o_s);
        FR(lb->gate_w); FR(lb->gate_s);
        FR(lb->up_w);   FR(lb->up_s);
        FR(lb->down_w); FR(lb->down_s);
        FR(lb->input_norm);
        FR(lb->post_norm);
        FR(lb->k_cache); FR(lb->v_cache);
        FR(lb->k_scales); FR(lb->v_scales);
    }
    free(b->layers);
    FR(b->output_norm); FR(b->output_head_w); FR(b->output_head_s);
    FR(b->x); FR(b->xb); FR(b->xb2);
    FR(b->q); FR(b->k); FR(b->v); FR(b->attn_out);
    FR(b->hb); FR(b->hb2); FR(b->scores);
    FR(b->xq); FR(b->xs); FR(b->logits);
    free(b);
    #undef FR
}

extern "C" void
ib_metal_reset_kv(ib_metal_model_buffers *b)
{
    if (!b) return;
    size_t kv_elem = (b->kv_bits == 16) ? sizeof(float) : 1;
    size_t bytes = (size_t)b->seq_len * (size_t)b->kv_dim * kv_elem;
    size_t scale_bytes = (size_t)b->seq_len * (size_t)b->n_kv_heads * sizeof(float);
    for (int L = 0; L < b->num_layers; L++) {
        if (b->layers[L].k_cache) memset(b->layers[L].k_cache, 0, bytes);
        if (b->layers[L].v_cache) memset(b->layers[L].v_cache, 0, bytes);
        if (b->layers[L].k_scales) memset(b->layers[L].k_scales, 0, scale_bytes);
        if (b->layers[L].v_scales) memset(b->layers[L].v_scales, 0, scale_bytes);
    }
}

/* Records ONE matmul into the recorder, picking the kernel based on
 * the tensor's bit width. xq/xs scratch is only used for INT4 (w4a8). */
static int rec_matmul(ib_metal_recorder *r,
                       int bits,
                       const void *x_fp32,
                       const void *weights, const void *w_scales,
                       void *out, void *xq, void *xs,
                       int M, int N)
{
    if (bits == 4) {
        return ib_metal_rec_matmul_w4a8_fp32_in(r, x_fp32, weights, w_scales,
                                                  out, xq, xs, M, N);
    } else if (bits == 8) {
        return ib_metal_rec_matmul_int8_fp32_in(r, x_fp32, weights, w_scales,
                                                  out, M, N);
    }
    return -1;
}

extern "C" int
ib_metal_forward_token(ib_metal_ctx *ctx,
                        ib_metal_model_buffers *b,
                        const float *cpu_embed_in,
                        int pos,
                        float *logits_out)
{
    if (!ctx || !b || !cpu_embed_in || !logits_out) return -1;
    if (pos < 0 || pos >= b->seq_len) return -1;

    memcpy(b->x, cpu_embed_in, (size_t)b->hidden * sizeof(float));

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) return -1;

    int hidden = b->hidden;
    int inter  = b->intermediate;
    int kv_dim = b->kv_dim;
    int nh     = b->n_heads;
    int nkh    = b->n_kv_heads;
    int hd     = b->head_dim;
    float th   = b->rope_theta;
    float eps  = b->eps;
    int sl     = b->seq_len;

    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];
        ib_metal_rec_rmsnorm_fp16(r, b->x, lb->input_norm, b->xb, hidden, eps);
        rec_matmul(r, lb->q_bits, b->xb, lb->q_w, lb->q_s, b->q, b->xq, b->xs, hidden, hidden);
        rec_matmul(r, lb->k_bits, b->xb, lb->k_w, lb->k_s, b->k, b->xq, b->xs, kv_dim, hidden);
        rec_matmul(r, lb->v_bits, b->xb, lb->v_w, lb->v_s, b->v, b->xq, b->xs, kv_dim, hidden);
        ib_metal_rec_rope_inplace(r, b->q, nh,  hd, pos, th);
        ib_metal_rec_rope_inplace(r, b->k, nkh, hd, pos, th);
        if (b->kv_bits == 16) {
            ib_metal_rec_attention_block_fp16(r, b->q, b->k, b->v,
                                                lb->k_cache, lb->v_cache,
                                                b->scores, b->attn_out,
                                                nh, nkh, hd, sl, pos);
        } else {
            ib_metal_rec_attention_block_int8(r, b->q, b->k, b->v,
                                                lb->k_cache, lb->v_cache,
                                                lb->k_scales, lb->v_scales,
                                                b->scores, b->attn_out,
                                                nh, nkh, hd, sl, pos);
        }
        rec_matmul(r, lb->o_bits, b->attn_out, lb->o_w, lb->o_s, b->xb2, b->xq, b->xs, hidden, hidden);
        ib_metal_rec_residual_add(r, b->x, b->xb2, hidden);
        ib_metal_rec_rmsnorm_fp16(r, b->x, lb->post_norm, b->xb, hidden, eps);
        rec_matmul(r, lb->gate_bits, b->xb, lb->gate_w, lb->gate_s, b->hb,  b->xq, b->xs, inter, hidden);
        rec_matmul(r, lb->up_bits,   b->xb, lb->up_w,   lb->up_s,   b->hb2, b->xq, b->xs, inter, hidden);
        ib_metal_rec_silu_mul(r, b->hb, b->hb2, b->hb, inter);
        rec_matmul(r, lb->down_bits, b->hb, lb->down_w, lb->down_s, b->xb, b->xq, b->xs, hidden, inter);
        ib_metal_rec_residual_add(r, b->x, b->xb, hidden);
    }
    ib_metal_rec_rmsnorm_fp16(r, b->x, b->output_norm, b->xb, hidden, eps);
    rec_matmul(r, b->output_head_bits, b->xb,
                b->output_head_w, b->output_head_s,
                b->logits, b->xq, b->xs, b->vocab, hidden);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
}
