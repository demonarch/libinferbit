/*
 * forward.c — Transformer forward pass
 *
 * Implements: embedding → [RMSNorm → Attention → Residual → RMSNorm → MLP → Residual] × N → RMSNorm → Output head
 */

#include "inferbit_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Weight data access helpers ─────────────────────────────── */

/* Get pointer to weight data for a tensor */
static inline const void* tensor_data(const inferbit_model* m, const ib_tensor_meta* t) {
    return (const uint8_t*)m->weight_data + t->offset;
}

/* Get pointer to scale factors for a tensor (FP16 stored, we read as half→float) */
static inline const void* tensor_scales_raw(const inferbit_model* m, const ib_tensor_meta* t) {
    if (t->scale_offset == 0 && t->scale_size == 0) return NULL;
    return (const uint8_t*)m->weight_data + t->scale_offset;
}

/* ── FP16 conversion ────────────────────────────────────────── */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Subnormal */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        /* Inf/NaN */
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Convert FP16 scale array to FP32 (temporary buffer) */
static void scales_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* Read FP16 weight as FP32 (for norms) */
static void fp16_weights_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* ── Embedding lookup ───────────────────────────────────────── */

static void embedding_lookup(const inferbit_model* m, int token_id, float* out) {
    int hidden = m->header.hidden_size;
    const ib_tensor_meta* emb = &m->token_embedding;

    if (emb->bits == 8) {
        /* INT8 embedding: dequantize row */
        const int8_t* data = (const int8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        const int8_t* row = data + (size_t)token_id * hidden;

        if (scales_raw) {
            /* Per-row scale factor */
            const uint16_t* scales_fp16 = (const uint16_t*)scales_raw;
            float scale = fp16_to_fp32(scales_fp16[token_id]);
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i] * scale;
            }
        } else {
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i];
            }
        }
    } else if (emb->bits == 16) {
        /* FP16 embedding */
        const uint16_t* data = (const uint16_t*)tensor_data(m, emb);
        const uint16_t* row = data + (size_t)token_id * hidden;
        for (int i = 0; i < hidden; i++) {
            out[i] = fp16_to_fp32(row[i]);
        }
    } else if (emb->bits == 4) {
        /* INT4 embedding */
        const uint8_t* data = (const uint8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        size_t row_bytes = (size_t)hidden / 2;
        const uint8_t* row = data + (size_t)token_id * row_bytes;
        float scale = 1.0f;
        if (scales_raw) {
            scale = fp16_to_fp32(((const uint16_t*)scales_raw)[token_id]);
        }
        for (int i = 0; i < hidden; i += 2) {
            uint8_t byte = row[i / 2];
            out[i]     = (float)((int8_t)(byte & 0x0F) - 8) * scale;
            out[i + 1] = (float)((int8_t)((byte >> 4) & 0x0F) - 8) * scale;
        }
    }
}

/* ── Matmul dispatch ────────────────────────────────────────── */

/*
 * Run quantized matmul for a tensor: out[M] = weights[M,N] @ input[N]
 * Handles bit-width dispatch and scale conversion.
 * `scale_buf` is a caller-provided temporary buffer of at least M floats.
 */
static void tensor_matmul(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf
) {
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
    }

    if (t->bits == 4 || t->bits == 8) {
        /* Use parallel matmul if thread pool available and matrix is large enough */
        ib_parallel_matmul(m->thread_pool, out, weights, scale_buf, input, M, N, t->bits);
    } else if (t->bits == 16) {
        /* FP16 matmul: dequantize and multiply (norms — too small to parallelize) */
        const uint16_t* w = (const uint16_t*)weights;
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += fp16_to_fp32(w[i * N + j]) * input[j];
            }
            out[i] = sum;
        }
    }
}

/* ── RMSNorm with FP16 weights ──────────────────────────────── */

static void rmsnorm_fp16(float* out, const float* input,
                         const void* weight_fp16, float eps, int N,
                         float* weight_buf) {
    fp16_weights_to_fp32(weight_buf, weight_fp16, N);
    ib_kern.rmsnorm(out, input, weight_buf, eps, N);
}

/* ── KV cache write ─────────────────────────────────────────── */

static void kv_cache_write_fp32(ib_kv_cache* kv, int pos,
                                 const float* key, const float* value,
                                 int kv_dim) {
    /* For now, store as FP32 regardless of kv_bits (simplifies milestone 3) */
    /* TODO: Milestone 4 — quantize to KV cache bit-width */
    float* k_store = (float*)kv->key_data;
    float* v_store = (float*)kv->value_data;
    memcpy(k_store + (size_t)pos * kv_dim, key, kv_dim * sizeof(float));
    memcpy(v_store + (size_t)pos * kv_dim, value, kv_dim * sizeof(float));
}

/* ── Parallel attention task ─────────────────────────────────── */

typedef struct {
    float* q;
    float* att;
    float* xb2;
    float* k_cache;
    float* v_cache;
    int head_dim;
    int kv_dim;
    int heads_per_kv;
    int pos;
    float scale;
} ib_attn_ctx;

static void ib_attn_head_task(void* arg, int tid, int start, int end) {
    (void)tid;
    ib_attn_ctx* c = (ib_attn_ctx*)arg;
    for (int h = start; h < end; h++) {
        float* q_h = c->q + h * c->head_dim;
        int kv_h = h / c->heads_per_kv;

        /* Q*K scores */
        for (int t = 0; t <= c->pos; t++) {
            float* k_t = c->k_cache + (size_t)t * c->kv_dim + kv_h * c->head_dim;
            float score = 0.0f;
            for (int d = 0; d < c->head_dim; d++) {
                score += q_h[d] * k_t[d];
            }
            c->att[h * (c->pos + 1) + t] = score * c->scale;
        }

        /* Softmax */
        ib_kern.softmax(c->att + h * (c->pos + 1), c->pos + 1);

        /* Weighted sum of values */
        float* out_h = c->xb2 + h * c->head_dim;
        memset(out_h, 0, c->head_dim * sizeof(float));
        for (int t = 0; t <= c->pos; t++) {
            float a = c->att[h * (c->pos + 1) + t];
            float* v_t = c->v_cache + (size_t)t * c->kv_dim + kv_h * c->head_dim;
            for (int d = 0; d < c->head_dim; d++) {
                out_h[d] += a * v_t[d];
            }
        }
    }
}

/* ── Single-token forward pass ──────────────────────────────── */

static int forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    int hidden   = m->header.hidden_size;
    int n_layers = m->header.num_layers;
    int n_heads  = m->header.num_heads;
    int n_kv     = m->header.num_kv_heads;
    int head_dim = m->header.head_dim;
    int inter    = m->header.intermediate_size;
    int vocab    = m->header.vocab_size;
    float eps    = m->header.norm_epsilon;
    float theta  = m->header.rope_theta;

    int kv_dim   = n_kv * head_dim;
    int heads_per_kv = n_heads / n_kv;  /* For GQA */

    /* Activation buffers */
    float* x       = m->buf_residual;   /* [hidden] — residual stream */
    float* xb      = m->buf_hidden;     /* [hidden] — after norm */
    float* xb2     = m->buf_attn;       /* [hidden] — scratch */
    float* hb      = m->buf_mlp;        /* [inter]  — MLP scratch */
    float* hb2     = m->buf_mlp2;       /* [inter]  — MLP scratch 2 */
    float* qkv_buf = m->buf_qkv;        /* Scratch for projections + attention scores */

    /* Partition qkv_buf:
     * q:     [hidden]
     * k:     [kv_dim]
     * v:     [kv_dim]
     * att:   [n_heads * (pos+1)] — attention scores
     * scale: [max(hidden, inter, vocab)] — scale factor temp buffer
     */
    float* q     = qkv_buf;
    float* k     = q + hidden;
    float* v     = k + kv_dim;
    float* att   = v + kv_dim;
    int scale_sz = hidden > inter ? hidden : inter;
    if (vocab > scale_sz) scale_sz = vocab;
    float* scale_buf = att + (size_t)n_heads * (pos + 1);

    /* Embedding lookup */
    embedding_lookup(m, token_id, x);

    /* Transformer layers */
    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm before attention */
        rmsnorm_fp16(xb, x, tensor_data(m, &layer->input_norm),
                     eps, hidden, scale_buf);

        /* Q/K/V projections */
        tensor_matmul(m, &layer->q_proj, q, xb, hidden, hidden, scale_buf);
        tensor_matmul(m, &layer->k_proj, k, xb, kv_dim, hidden, scale_buf);
        tensor_matmul(m, &layer->v_proj, v, xb, kv_dim, hidden, scale_buf);

        /* RoPE: apply to each Q head paired with its corresponding K head.
         * For GQA, multiple Q heads share one K head. Apply RoPE to each
         * K head only once (on the first Q head that maps to it). */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;
            int is_first = (h % heads_per_kv == 0);
            if (is_first) {
                ib_kern.rope(q + h * head_dim, k + kv_h * head_dim,
                             head_dim, pos, theta);
            } else {
                /* Apply RoPE to Q only — use a scratch buffer for K */
                float k_scratch[256];
                memcpy(k_scratch, k + kv_h * head_dim, head_dim * sizeof(float));
                ib_kern.rope(q + h * head_dim, k_scratch, head_dim, pos, theta);
                /* Discard k_scratch — K was already rotated */
            }
        }

        /* Write K, V to cache */
        kv_cache_write_fp32(kv, pos, k, v, kv_dim);
        kv->length = pos + 1;

        /* Multi-head attention (parallelized across heads) */
        float attn_scale = 1.0f / sqrtf((float)head_dim);

        ib_attn_ctx attn_ctx = {
            .q = q, .att = att, .xb2 = xb2,
            .k_cache = (float*)kv->key_data,
            .v_cache = (float*)kv->value_data,
            .head_dim = head_dim, .kv_dim = kv_dim,
            .heads_per_kv = heads_per_kv,
            .pos = pos, .scale = attn_scale,
        };

        if (m->thread_pool && n_heads >= 4) {
            ib_pool_run(m->thread_pool, ib_attn_head_task, &attn_ctx, n_heads, 0);
        } else {
            ib_attn_head_task(&attn_ctx, 0, 0, n_heads);
        }

        /* Output projection: xb = O_proj @ xb2 */
        tensor_matmul(m, &layer->o_proj, xb, xb2, hidden, hidden, scale_buf);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }

        /* RMSNorm before MLP */
        rmsnorm_fp16(xb, x, tensor_data(m, &layer->post_attn_norm),
                     eps, hidden, scale_buf);

        /* MLP: gate + up + silu_mul + down */
        tensor_matmul(m, &layer->gate_proj, hb, xb, inter, hidden, scale_buf);
        tensor_matmul(m, &layer->up_proj, hb2, xb, inter, hidden, scale_buf);
        ib_kern.silu_mul(hb, hb, hb2, inter);
        tensor_matmul(m, &layer->down_proj, xb, hb, hidden, inter, scale_buf);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }
    }

    /* Final RMSNorm */
    rmsnorm_fp16(x, x, tensor_data(m, &m->output_norm),
                 eps, hidden, scale_buf);

    /* Output head: logits = head @ x */
    tensor_matmul(m, &m->output_head, logits, x, vocab, hidden, scale_buf);

    return INFERBIT_OK;
}

/* ── Public: forward pass ───────────────────────────────────── */

int ib_forward(inferbit_model* model, const int32_t* tokens, int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward");
        return INFERBIT_ERROR_PARAM;
    }

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d", kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }

    /* Process each token sequentially (prefill + single-token decode) */
    for (int i = 0; i < num_tokens; i++) {
        int pos = kv_pos + i;
        int token = tokens[i];

        if (token < 0 || token >= model->header.vocab_size) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)", token, model->header.vocab_size);
            return INFERBIT_ERROR_PARAM;
        }

        int rc = forward_single(model, token, pos, out_logits);
        if (rc != INFERBIT_OK) return rc;
    }

    return INFERBIT_OK;
}
