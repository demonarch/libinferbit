/* Phase 7: IBF-model → Metal upload + per-token forward dispatcher.
 *
 * Walks an inferbit_model loaded by the standard IBF loader, allocates
 * Metal buffers for every weight/scale tensor and per-layer KV cache,
 * then exposes ib_metal_forward_token() which records the full
 * 22-layer (or whatever depth) forward into ONE command buffer.
 *
 * Embedding stays on CPU because it varies in bit-width across IBFs
 * (int4/int8/fp16) and is just an 8 KB copy per token — not worth a
 * dedicated GPU kernel set. Everything else (RMSNorm, all matmuls,
 * RoPE, attention block, residuals, final norm, lm_head) runs on GPU.
 */
#include "metal_runtime.h"
#include "../inferbit_internal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Per-layer GPU buffer set. */
struct layer_bufs {
    void *q_w, *q_s;
    void *k_w, *k_s;
    void *v_w, *v_s;
    void *o_w, *o_s;
    void *gate_w, *gate_s;
    void *up_w,   *up_s;
    void *down_w, *down_s;
    void *input_norm;
    void *post_norm;
    void *k_cache;
    void *v_cache;
};

struct ib_metal_model_buffers {
    /* Topology mirrored from the model header. */
    int num_layers;
    int hidden;
    int intermediate;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int kv_dim;
    int vocab;
    int seq_len;
    float rope_theta;
    float eps;

    /* CPU-side handle (just to keep alive / for the embedding lookup). */
    const inferbit_model *model;

    /* Per-layer GPU buffers. */
    struct layer_bufs *layers;

    /* Model-level GPU buffers. */
    void *output_norm;
    void *output_head_w;
    void *output_head_s;

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

/* Gates an IBF on the layout the GPU dispatcher supports today. */
static int model_is_supported(const inferbit_model *m, char *err, size_t err_sz) {
    #define CHECK(cond, msg) do { \
        if (!(cond)) { snprintf(err, err_sz, "%s", msg); return 0; } \
    } while (0)

    CHECK(m->header.kv_bits == 16, "Metal forward requires kv_bits=16");
    CHECK(m->output_norm.bits == 16, "output_norm must be fp16");
    CHECK(m->output_head.bits == 4 && m->output_head.pq == NULL,
          "output_head must be INT4 (w4a8) without PQ");
    for (int L = 0; L < m->header.num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        #define BIT4(name) \
            CHECK(lm->name.bits == 4 && lm->name.pq == NULL, \
                  "layer matmul tensor must be INT4 (w4a8) without PQ: " #name)
        BIT4(q_proj); BIT4(k_proj); BIT4(v_proj); BIT4(o_proj);
        BIT4(gate_proj); BIT4(up_proj); BIT4(down_proj);
        #undef BIT4
        CHECK(lm->input_norm.bits == 16,    "input_norm must be fp16");
        CHECK(lm->post_attn_norm.bits == 16, "post_attn_norm must be fp16");
        CHECK(lm->sparsity_mask_size == 0,
              "sparsity not supported on the Metal forward path");
    }
    #undef CHECK
    return 1;
}

/* Upload one tensor's weight bytes + scale bytes (if any). */
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
    b->seq_len      = m->header.max_context_length;
    b->rope_theta   = m->header.rope_theta;
    b->eps          = m->header.norm_epsilon;
    b->model        = m;

    /* Per-layer weights + KV caches. */
    b->layers = (struct layer_bufs *)
        calloc((size_t)b->num_layers, sizeof(struct layer_bufs));
    if (!b->layers) { free(b); return nullptr; }

    /* CPU stores kv_bits=16 as fp32 (see ibf_loader.c line 240-241), so
     * the GPU must match — fp32 KV cache, not fp16. */
    size_t kv_bytes_per_layer =
        (size_t)b->seq_len * (size_t)b->kv_dim * sizeof(float);

    for (int L = 0; L < b->num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        struct layer_bufs *lb = &b->layers[L];
        upload_w_pair(ctx, m, &lm->q_proj,    &lb->q_w,    &lb->q_s);
        upload_w_pair(ctx, m, &lm->k_proj,    &lb->k_w,    &lb->k_s);
        upload_w_pair(ctx, m, &lm->v_proj,    &lb->v_w,    &lb->v_s);
        upload_w_pair(ctx, m, &lm->o_proj,    &lb->o_w,    &lb->o_s);
        upload_w_pair(ctx, m, &lm->gate_proj, &lb->gate_w, &lb->gate_s);
        upload_w_pair(ctx, m, &lm->up_proj,   &lb->up_w,   &lb->up_s);
        upload_w_pair(ctx, m, &lm->down_proj, &lb->down_w, &lb->down_s);
        upload_norm  (ctx, m, &lm->input_norm,     &lb->input_norm);
        upload_norm  (ctx, m, &lm->post_attn_norm, &lb->post_norm);
        /* KV caches: zero-initialized fp16 buffers. */
        lb->k_cache = ib_metal_alloc(ctx, kv_bytes_per_layer, NULL);
        lb->v_cache = ib_metal_alloc(ctx, kv_bytes_per_layer, NULL);
        if (lb->k_cache) memset(lb->k_cache, 0, kv_bytes_per_layer);
        if (lb->v_cache) memset(lb->v_cache, 0, kv_bytes_per_layer);
    }

    /* Model-level. */
    upload_norm(ctx, m, &m->output_norm, &b->output_norm);
    upload_w_pair(ctx, m, &m->output_head, &b->output_head_w, &b->output_head_s);

    /* State buffers. */
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
    size_t bytes = (size_t)b->seq_len * (size_t)b->kv_dim * sizeof(float);
    for (int L = 0; L < b->num_layers; L++) {
        if (b->layers[L].k_cache) memset(b->layers[L].k_cache, 0, bytes);
        if (b->layers[L].v_cache) memset(b->layers[L].v_cache, 0, bytes);
    }
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

    /* Copy CPU embedding into the GPU x buffer (unified memory: just
     * memcpy the bytes — no upload primitive needed). */
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
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb, lb->q_w, lb->q_s, b->q, b->xq, b->xs, hidden, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb, lb->k_w, lb->k_s, b->k, b->xq, b->xs, kv_dim, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb, lb->v_w, lb->v_s, b->v, b->xq, b->xs, kv_dim, hidden);
        ib_metal_rec_rope_inplace(r, b->q, nh,  hd, pos, th);
        ib_metal_rec_rope_inplace(r, b->k, nkh, hd, pos, th);
        ib_metal_rec_attention_block_fp16(r, b->q, b->k, b->v,
                                            lb->k_cache, lb->v_cache,
                                            b->scores, b->attn_out,
                                            nh, nkh, hd, sl, pos);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->attn_out, lb->o_w, lb->o_s, b->xb2, b->xq, b->xs, hidden, hidden);
        ib_metal_rec_residual_add(r, b->x, b->xb2, hidden);
        ib_metal_rec_rmsnorm_fp16(r, b->x, lb->post_norm, b->xb, hidden, eps);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb, lb->gate_w, lb->gate_s, b->hb,  b->xq, b->xs, inter, hidden);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb, lb->up_w,   lb->up_s,   b->hb2, b->xq, b->xs, inter, hidden);
        ib_metal_rec_silu_mul(r, b->hb, b->hb2, b->hb, inter);
        ib_metal_rec_matmul_w4a8_fp32_in(r, b->hb, lb->down_w, lb->down_s, b->xb, b->xq, b->xs, hidden, inter);
        ib_metal_rec_residual_add(r, b->x, b->xb, hidden);
    }
    /* Final norm + LM head. */
    ib_metal_rec_rmsnorm_fp16(r, b->x, b->output_norm, b->xb, hidden, eps);
    ib_metal_rec_matmul_w4a8_fp32_in(r, b->xb,
                                      b->output_head_w, b->output_head_s,
                                      b->logits, b->xq, b->xs,
                                      b->vocab, hidden);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
}
