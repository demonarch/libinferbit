/*
 * forward.c — Transformer forward pass
 *
 * Implements: embedding → [RMSNorm → Attention → Residual → RMSNorm → MLP → Residual] × N → RMSNorm → Output head
 */

#include "inferbit_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* W4A8 path is on by default. Set IB_W4A8=0 in env to force the FP32
 * activation fallback (used for A/B comparison and debugging). */
static int w4a8_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("IB_W4A8");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* MLP activation-sparsity threshold. Post-silu_mul, neurons whose
 * |activation| < threshold * max(|activation|) are zeroed, so their
 * contribution to the subsequent down_proj matmul is zero.
 *
 * On a bandwidth-bound CPU path with packed INT4 row-major weights this
 * does NOT save memory reads — the kernel still streams every row of
 * down_proj. It only saves arithmetic on the zeroed columns. The real
 * bandwidth win from activation sparsity requires a column-major
 * down_proj layout and a column-sparse matmul kernel, which is Phase 2.
 *
 * Phase 1 ships this as a runtime toggle so we can measure quality
 * impact, tune the threshold, and have the plumbing in place for when
 * Phase 2 rewires the kernel path.
 *
 * Reasonable values per Script 4 (TinyLlama, wikitext):
 *   0.01  — ~60%% neurons kept, <1%% PPL cost (conservative)
 *   0.05  — ~17%% neurons kept, ~1-3%% PPL cost (aggressive)
 *   0.10  — ~5.4%% neurons kept, higher PPL cost (extreme)
 *   0.0   — disabled (default) */
static float act_sparsity_threshold(void) {
    static float cached = -1.0f;
    if (cached < 0.0f) {
        const char* e = getenv("IB_ACT_SPARSITY");
        cached = (e && *e) ? (float)atof(e) : 0.0f;
        if (cached < 0.0f) cached = 0.0f;
        if (cached > 1.0f) cached = 1.0f;
    }
    return cached;
}

/* Apply threshold-based activation sparsity in place.
 *
 * Computes max |h[i]| then zeros entries whose magnitude is below
 * threshold * max. Returns number of neurons kept (for diagnostics). */
static int apply_act_sparsity(float* h, int n, float threshold) {
    if (threshold <= 0.0f || n <= 0) return n;
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = h[i] < 0 ? -h[i] : h[i];
        if (a > m) m = a;
    }
    float cutoff = m * threshold;
    int kept = 0;
    for (int i = 0; i < n; i++) {
        float a = h[i] < 0 ? -h[i] : h[i];
        if (a < cutoff) h[i] = 0.0f;
        else kept++;
    }
    return kept;
}

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

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8) {
        /* Quantize input to INT8 per-group (IB_W4A8_GROUP elements per
         * scale). Small N uses stack; larger N heap-allocates. */
        int8_t stack_q[4096];
        float  stack_s[4096 / IB_W4A8_GROUP + 1];
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        int8_t* q_buf = (N <= 4096) ? stack_q : (int8_t*)malloc((size_t)N);
        float*  s_buf = (n_groups <= (int)(sizeof stack_s / sizeof *stack_s))
                        ? stack_s
                        : (float*)malloc((size_t)n_groups * sizeof(float));
        ib_quantize_input_int8_g128(input, q_buf, s_buf, N);
        ib_parallel_matmul_w4a8(m->thread_pool, out, weights, scale_buf,
                                q_buf, s_buf, M, N);
        if (q_buf != stack_q) free(q_buf);
        if (s_buf != stack_s) free(s_buf);
    } else if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
        ib_parallel_matmul(m->thread_pool, out, weights, scale_buf, input, M, N, t->bits);
    } else if (t->bits == 16) {
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

/* Batched variant of tensor_matmul.
 *
 *   out    [B * M]  row-major, out[b*M + i]
 *   input  [B * N]  row-major, input[b*N + j]
 *   scale_buf: caller-provided, at least M floats (weight scales).
 *   q_scratch: only used for INT4+W4A8 path — B * N int8 bytes for quantized
 *              activations, plus B * ceil(N/IB_W4A8_GROUP) floats for scales.
 *              Caller supplies both to avoid malloc in the hot loop. Pass
 *              NULL for paths that don't need them (INT8, FP16).
 *
 * Same dispatch policy as tensor_matmul: INT4 routes through W4A8 batched
 * kernel when enabled; INT8 uses matmul_int8_batch; FP16 falls back to the
 * sequential FP16 path because we don't have a batched FP16 kernel. */
static void tensor_matmul_batch(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N, int B,
    float* scale_buf, int8_t* q_scratch, float* sa_scratch
) {
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
    }

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8_batch && q_scratch && sa_scratch) {
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        for (int b = 0; b < B; b++) {
            ib_quantize_input_int8_g128(input + (size_t)b * N,
                                        q_scratch + (size_t)b * N,
                                        sa_scratch + (size_t)b * n_groups, N);
        }
        ib_parallel_matmul_w4a8_batch(m->thread_pool, out, weights, scale_buf,
                                      q_scratch, sa_scratch, M, N, B);
    } else if (t->bits == 8 && ib_kern.matmul_int8_batch) {
        ib_parallel_matmul_int8_batch(m->thread_pool, out, weights, scale_buf,
                                      input, M, N, B);
    } else {
        /* Fallback: per-position sequential. */
        for (int b = 0; b < B; b++) {
            float* out_b = out + (size_t)b * M;
            const float* in_b = input + (size_t)b * N;
            if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
                ib_parallel_matmul(m->thread_pool, out_b, weights, scale_buf,
                                   in_b, M, N, t->bits);
            } else if (t->bits == 16) {
                const uint16_t* w = (const uint16_t*)weights;
                for (int i = 0; i < M; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        sum += fp16_to_fp32(w[(size_t)i * N + j]) * in_b[j];
                    }
                    out_b[i] = sum;
                }
            }
        }
    }
}

/*
 * Sparse matmul: same as tensor_matmul but skips rows where mask[row] == 0.
 * Outputs zero for skipped rows. mask is a byte array of length M.
 */
static void tensor_matmul_sparse(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf, const uint8_t* mask
) {
    if (!mask) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        return;
    }

    /* Count active rows and build index */
    int active = 0;
    for (int i = 0; i < M; i++) {
        if (mask[i]) active++;
    }

    /* If most rows are active (>80%), just run dense — overhead of sparse indexing isn't worth it */
    if (active > M * 4 / 5) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        /* Zero out masked rows */
        for (int i = 0; i < M; i++) {
            if (!mask[i]) out[i] = 0.0f;
        }
        return;
    }

    /* Sparse path: only compute active rows */
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
    }

    /* Zero entire output first */
    memset(out, 0, M * sizeof(float));

    /* Compute only active rows */
    for (int i = 0; i < M; i++) {
        if (!mask[i]) continue;

        float sum = 0.0f;
        if (t->bits == 8) {
            const int8_t* w = (const int8_t*)weights + (size_t)i * N;
            for (int j = 0; j < N; j++) sum += (float)w[j] * input[j];
        } else if (t->bits == 4) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 2);
            for (int j = 0; j < N; j += 2) {
                uint8_t byte = w[j / 2];
                sum += (float)((int8_t)(byte & 0x0F) - 8) * input[j];
                if (j + 1 < N) sum += (float)((int8_t)((byte >> 4) & 0x0F) - 8) * input[j + 1];
            }
        } else if (t->bits == 2) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 4);
            for (int j = 0; j < N; j += 4) {
                uint8_t byte = w[j / 4];
                sum += (float)((byte & 0x03) - 1) * input[j];
                if (j+1 < N) sum += (float)(((byte >> 2) & 0x03) - 1) * input[j+1];
                if (j+2 < N) sum += (float)(((byte >> 4) & 0x03) - 1) * input[j+2];
                if (j+3 < N) sum += (float)(((byte >> 6) & 0x03) - 1) * input[j+3];
            }
        }
        out[i] = sum * scale_buf[i];
    }
}

/* ── RMSNorm with FP16 weights ──────────────────────────────── */

static void rmsnorm_fp16(float* out, const float* input,
                         const void* weight_fp16, float eps, int N,
                         float* weight_buf) {
    fp16_weights_to_fp32(weight_buf, weight_fp16, N);
    ib_kern.rmsnorm(out, input, weight_buf, eps, N);
}

/* ── KV cache quantization helpers ──────────────────────────── */

static inline void kv_write_int4_row(uint8_t* dst, const float* src, float scale, int n) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i += 2) {
        int q0 = (int)roundf(src[i] * inv);
        int q1 = (i + 1 < n) ? (int)roundf(src[i + 1] * inv) : 0;
        if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
        if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;
        uint8_t lo = (uint8_t)(q0 + 8) & 0x0F;
        uint8_t hi = (uint8_t)(q1 + 8) & 0x0F;
        dst[i / 2] = lo | (hi << 4);
    }
}

static inline void kv_read_int4_row(float* out, const uint8_t* src, float scale, int n) {
    for (int i = 0; i < n; i += 2) {
        uint8_t b = src[i / 2];
        out[i] = ((float)((int)(b & 0x0F) - 8)) * scale;
        if (i + 1 < n) out[i + 1] = ((float)((int)((b >> 4) & 0x0F) - 8)) * scale;
    }
}

static void kv_cache_write(ib_kv_cache* kv, int pos,
                           const float* key, const float* value,
                           int kv_dim, int n_kv_heads, int head_dim, int kv_bits) {
    if (kv_bits >= 16) {
        float* k_store = (float*)kv->key_data;
        float* v_store = (float*)kv->value_data;
        memcpy(k_store + (size_t)pos * kv_dim, key, kv_dim * sizeof(float));
        memcpy(v_store + (size_t)pos * kv_dim, value, kv_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        int8_t* k_store = (int8_t*)kv->key_data + (size_t)pos * kv_dim;
        int8_t* v_store = (int8_t*)kv->value_data + (size_t)pos * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 127.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 127.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)pos * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)pos * n_kv_heads + h] = v_scale;
            float k_inv = 1.0f / k_scale;
            float v_inv = 1.0f / v_scale;
            for (int d = 0; d < head_dim; d++) {
                int kq = (int)roundf(k_h[d] * k_inv);
                int vq = (int)roundf(v_h[d] * v_inv);
                if (kq < -127) kq = -127; if (kq > 127) kq = 127;
                if (vq < -127) vq = -127; if (vq > 127) vq = 127;
                k_store[h * head_dim + d] = (int8_t)kq;
                v_store[h * head_dim + d] = (int8_t)vq;
            }
        }
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        uint8_t* k_store = (uint8_t*)kv->key_data + (size_t)pos * row_bytes;
        uint8_t* v_store = (uint8_t*)kv->value_data + (size_t)pos * row_bytes;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 7.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 7.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)pos * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)pos * n_kv_heads + h] = v_scale;
            kv_write_int4_row(k_store + (size_t)h * ((head_dim + 1) / 2), k_h, k_scale, head_dim);
            kv_write_int4_row(v_store + (size_t)h * ((head_dim + 1) / 2), v_h, v_scale, head_dim);
        }
        return;
    }
}

static void kv_cache_read_head(const ib_kv_cache* kv, int is_key, int pos, int kv_head,
                               int kv_dim, int n_kv_heads, int head_dim, int kv_bits,
                               float* out_head) {
    if (kv_bits >= 16) {
        const float* src = is_key ? (const float*)kv->key_data : (const float*)kv->value_data;
        const float* row = src + (size_t)pos * kv_dim + kv_head * head_dim;
        memcpy(out_head, row, head_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        const int8_t* src = is_key ? (const int8_t*)kv->key_data : (const int8_t*)kv->value_data;
        const int8_t* row = src + (size_t)pos * kv_dim + kv_head * head_dim;
        float scale = is_key
            ? kv->key_scales[(size_t)pos * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)pos * n_kv_heads + kv_head];
        for (int d = 0; d < head_dim; d++) out_head[d] = (float)row[d] * scale;
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        const uint8_t* src = is_key ? (const uint8_t*)kv->key_data : (const uint8_t*)kv->value_data;
        const uint8_t* row = src + (size_t)pos * row_bytes + (size_t)kv_head * ((head_dim + 1) / 2);
        float scale = is_key
            ? kv->key_scales[(size_t)pos * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)pos * n_kv_heads + kv_head];
        kv_read_int4_row(out_head, row, scale, head_dim);
        return;
    }
}

/* ── Parallel attention task ─────────────────────────────────── */

typedef struct {
    float* q;
    float* att;
    float* xb2;
    ib_kv_cache* kv;
    int head_dim;
    int kv_dim;
    int n_kv_heads;
    int heads_per_kv;
    int pos;
    int kv_bits;
    float scale;
} ib_attn_ctx;

static void ib_attn_head_task(void* arg, int tid, int start, int end) {
    (void)tid;
    ib_attn_ctx* c = (ib_attn_ctx*)arg;
    float k_tmp[256];
    float v_tmp[256];

    for (int h = start; h < end; h++) {
        float* q_h = c->q + h * c->head_dim;
        int kv_h = h / c->heads_per_kv;

        for (int t = 0; t <= c->pos; t++) {
            float score = 0.0f;
            if (c->kv_bits >= 16) {
                float* k_cache = (float*)c->kv->key_data;
                float* k_t = k_cache + (size_t)t * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_t[d];
            } else {
                kv_cache_read_head(c->kv, 1, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, k_tmp);
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_tmp[d];
            }
            c->att[h * (c->pos + 1) + t] = score * c->scale;
        }

        ib_kern.softmax(c->att + h * (c->pos + 1), c->pos + 1);

        float* out_h = c->xb2 + h * c->head_dim;
        memset(out_h, 0, c->head_dim * sizeof(float));
        for (int t = 0; t <= c->pos; t++) {
            float a = c->att[h * (c->pos + 1) + t];
            if (c->kv_bits >= 16) {
                float* v_cache = (float*)c->kv->value_data;
                float* v_t = v_cache + (size_t)t * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_t[d];
            } else {
                kv_cache_read_head(c->kv, 0, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, v_tmp);
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_tmp[d];
            }
        }
    }
}

/* ── Single-token forward pass ──────────────────────────────── */

/* Extended single-token forward.
 *
 * compute_logits:
 *   0 — skip the final RMSNorm and LM head (prefill path, advances KV only)
 *   1 — compute logits via the LM head (default decode path)
 *
 * hidden_out: if non-NULL, writes the post-final-RMSNorm hidden state into
 *   hidden_out[hidden_size]. Used by the batched verify path to stack B
 *   positions' hidden states before a single batched LM head matmul. When
 *   hidden_out is supplied the final RMSNorm runs regardless of compute_logits. */
static int forward_single_ex(inferbit_model* m, int token_id, int pos,
                             float* logits, int compute_logits,
                             float* hidden_out) {
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
        kv_cache_write(kv, pos, k, v, kv_dim, n_kv, head_dim, m->header.kv_bits);
        kv->length = pos + 1;

        /* Multi-head attention (parallelized across heads) */
        float attn_scale = 1.0f / sqrtf((float)head_dim);

        ib_attn_ctx attn_ctx = {
            .q = q, .att = att, .xb2 = xb2,
            .kv = kv,
            .head_dim = head_dim, .kv_dim = kv_dim,
            .n_kv_heads = n_kv,
            .heads_per_kv = heads_per_kv,
            .pos = pos,
            .kv_bits = m->header.kv_bits,
            .scale = attn_scale,
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

        /* MLP: gate + up + silu_mul + down
         * With sparsity: skip masked intermediate neurons entirely */
        {
            const uint8_t* sp_mask = NULL;
            if (layer->sparsity_mask_size > 0) {
                sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            }
            tensor_matmul_sparse(m, &layer->gate_proj, hb, xb, inter, hidden, scale_buf, sp_mask);
            tensor_matmul_sparse(m, &layer->up_proj, hb2, xb, inter, hidden, scale_buf, sp_mask);
            ib_kern.silu_mul(hb, hb, hb2, inter);
            /* DejaVu-style activation sparsity: zero small post-silu
             * activations so down_proj multiplies them by zero. Phase 1
             * is compute-only; bandwidth savings need column-major layout. */
            apply_act_sparsity(hb, inter, act_sparsity_threshold());
            /* down_proj reads from hb which already has zeros for masked rows —
             * the multiply by zero propagates naturally, no sparse path needed */
            tensor_matmul(m, &layer->down_proj, xb, hb, hidden, inter, scale_buf);
        }

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }
    }

    if (compute_logits || hidden_out) {
        /* Final RMSNorm */
        rmsnorm_fp16(x, x, tensor_data(m, &m->output_norm),
                     eps, hidden, scale_buf);

        if (hidden_out) {
            memcpy(hidden_out, x, (size_t)hidden * sizeof(float));
        }
        if (compute_logits) {
            /* Output head: logits = head @ x */
            tensor_matmul(m, &m->output_head, logits, x, vocab, hidden, scale_buf);
        }
    }

    return INFERBIT_OK;
}

static int forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    return forward_single_ex(m, token_id, pos, logits, 1, NULL);
}

/* ── Batched forward pass ───────────────────────────────────── */

/* Process B tokens (at contiguous positions positions[0..B-1]) through the
 * transformer, using batched matmul for projections, MLP, and LM head. Each
 * position's attention runs sequentially (each attends to its own prefix of
 * the KV cache, so there's no matmul-shape win from batching attention).
 *
 * If out_logits is non-NULL, writes per-position logits [B * vocab] row-major.
 *
 * Invariant: positions[b] = inferbit_kv_length(m) + b on entry (each position
 * gets appended to the KV cache as processed). Caller is responsible for
 * ensuring that's true. */
static int forward_batch(inferbit_model* m, const int32_t* tokens,
                         const int* positions, int B, float* out_logits) {
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
    int heads_per_kv = n_heads / n_kv;

    /* Per-batch activations. Layout: outer index = batch, inner = hidden/etc. */
    size_t x_sz   = (size_t)B * hidden;
    size_t kv_sz  = (size_t)B * kv_dim;
    size_t inter_sz = (size_t)B * inter;

    float* x   = (float*)malloc(x_sz   * sizeof(float));
    float* xb  = (float*)malloc(x_sz   * sizeof(float));
    float* xb2 = (float*)malloc(x_sz   * sizeof(float));
    float* q   = (float*)malloc(x_sz   * sizeof(float));
    float* k   = (float*)malloc(kv_sz  * sizeof(float));
    float* v   = (float*)malloc(kv_sz  * sizeof(float));
    float* hb  = (float*)malloc(inter_sz * sizeof(float));
    float* hb2 = (float*)malloc(inter_sz * sizeof(float));

    int scale_sz = hidden;
    if (inter > scale_sz) scale_sz = inter;
    if (vocab > scale_sz) scale_sz = vocab;
    float* scale_buf = (float*)malloc((size_t)scale_sz * sizeof(float));

    /* Attention scratch (per-position) */
    int max_pos = 0;
    for (int b = 0; b < B; b++) if (positions[b] > max_pos) max_pos = positions[b];
    size_t att_sz = (size_t)n_heads * (max_pos + 1);
    float* att = (float*)malloc(att_sz * sizeof(float));

    /* Batched-matmul scratch: for W4A8 path we quantize B activations of
     * length up to max(hidden, inter). Allocate max size. */
    int n_max = hidden > inter ? hidden : inter;
    int groups_max = (n_max + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
    int8_t* q_scratch = (int8_t*)malloc((size_t)B * n_max * sizeof(int8_t));
    float*  sa_scratch = (float*)malloc((size_t)B * groups_max * sizeof(float));

    if (!x || !xb || !xb2 || !q || !k || !v || !hb || !hb2 ||
        !scale_buf || !att || !q_scratch || !sa_scratch) {
        free(x); free(xb); free(xb2); free(q); free(k); free(v);
        free(hb); free(hb2); free(scale_buf); free(att);
        free(q_scratch); free(sa_scratch);
        ib_set_error("forward_batch oom");
        return INFERBIT_ERROR_PARAM;
    }

    /* Embed each token (cheap, per-position). */
    for (int b = 0; b < B; b++) {
        embedding_lookup(m, tokens[b], x + (size_t)b * hidden);
    }

    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm per position. Uses rmsnorm_fp16 to handle FP16 weights. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                         tensor_data(m, &layer->input_norm),
                         eps, hidden, scale_buf);
        }

        /* Q/K/V projections — batched. */
        tensor_matmul_batch(m, &layer->q_proj, q, xb, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->k_proj, k, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->v_proj, v, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Per-position: RoPE, KV-cache write, attention, O-proj-input
         * (accumulated per-position into xb2[b]). */
        for (int b = 0; b < B; b++) {
            int pos = positions[b];
            float* qb = q + (size_t)b * hidden;
            float* kb = k + (size_t)b * kv_dim;
            float* vb = v + (size_t)b * kv_dim;

            /* RoPE: same pattern as forward_single_ex. */
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / heads_per_kv;
                int is_first = (h % heads_per_kv == 0);
                if (is_first) {
                    ib_kern.rope(qb + h * head_dim, kb + kv_h * head_dim,
                                 head_dim, pos, theta);
                } else {
                    float k_scratch[256];
                    memcpy(k_scratch, kb + kv_h * head_dim, head_dim * sizeof(float));
                    ib_kern.rope(qb + h * head_dim, k_scratch, head_dim, pos, theta);
                }
            }

            kv_cache_write(kv, pos, kb, vb, kv_dim, n_kv, head_dim, m->header.kv_bits);
            kv->length = pos + 1;

            float attn_scale = 1.0f / sqrtf((float)head_dim);
            ib_attn_ctx ctx = {
                .q = qb, .att = att, .xb2 = xb2 + (size_t)b * hidden,
                .kv = kv,
                .head_dim = head_dim, .kv_dim = kv_dim,
                .n_kv_heads = n_kv,
                .heads_per_kv = heads_per_kv,
                .pos = pos,
                .kv_bits = m->header.kv_bits,
                .scale = attn_scale,
            };
            if (m->thread_pool && n_heads >= 4) {
                ib_pool_run(m->thread_pool, ib_attn_head_task, &ctx, n_heads, 0);
            } else {
                ib_attn_head_task(&ctx, 0, 0, n_heads);
            }
        }

        /* O projection — batched. */
        tensor_matmul_batch(m, &layer->o_proj, xb, xb2, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];

        /* RMSNorm before MLP, per position. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                         tensor_data(m, &layer->post_attn_norm),
                         eps, hidden, scale_buf);
        }

        /* MLP — batched gate/up/down. Sparsity is not applied here; if a
         * layer has sparsity we fall back to the single-position path per
         * batch via tensor_matmul_sparse (rare for Llama, so OK). */
        if (layer->sparsity_mask_size > 0) {
            const uint8_t* sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            for (int b = 0; b < B; b++) {
                float* xb_b = xb + (size_t)b * hidden;
                float* hb_b = hb + (size_t)b * inter;
                float* hb2_b = hb2 + (size_t)b * inter;
                float* xb_out = xb + (size_t)b * hidden;  /* overwrite xb[b] with down output */
                tensor_matmul_sparse(m, &layer->gate_proj, hb_b, xb_b, inter, hidden, scale_buf, sp_mask);
                tensor_matmul_sparse(m, &layer->up_proj, hb2_b, xb_b, inter, hidden, scale_buf, sp_mask);
                ib_kern.silu_mul(hb_b, hb_b, hb2_b, inter);
                apply_act_sparsity(hb_b, inter, act_sparsity_threshold());
                tensor_matmul(m, &layer->down_proj, xb_out, hb_b, hidden, inter, scale_buf);
            }
        } else {
            tensor_matmul_batch(m, &layer->gate_proj, hb, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            tensor_matmul_batch(m, &layer->up_proj, hb2, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            float thr = act_sparsity_threshold();
            for (int b = 0; b < B; b++) {
                ib_kern.silu_mul(hb + (size_t)b * inter,
                                 hb + (size_t)b * inter,
                                 hb2 + (size_t)b * inter, inter);
                apply_act_sparsity(hb + (size_t)b * inter, inter, thr);
            }
            tensor_matmul_batch(m, &layer->down_proj, xb, hb, hidden, inter, B,
                                scale_buf, q_scratch, sa_scratch);
        }

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];
    }

    /* Final RMSNorm + LM head. */
    if (out_logits) {
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16(x + (size_t)b * hidden, x + (size_t)b * hidden,
                         tensor_data(m, &m->output_norm),
                         eps, hidden, scale_buf);
        }
        tensor_matmul_batch(m, &m->output_head, out_logits, x, vocab, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
    }

    free(x); free(xb); free(xb2); free(q); free(k); free(v);
    free(hb); free(hb2); free(scale_buf); free(att);
    free(q_scratch); free(sa_scratch);
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

    /* Validate all tokens first */
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= model->header.vocab_size) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)", tokens[i], model->header.vocab_size);
            return INFERBIT_ERROR_PARAM;
        }
    }

    if (num_tokens == 1) {
        /* Single token — standard decode path */
        return forward_single(model, tokens[0], kv_pos, out_logits);
    }

    /*
     * Batch prefill optimization:
     * For multi-token input, we only need logits from the LAST token.
     * Process tokens 0..N-2 through embed + projections + KV cache write only
     * (skip attention output, MLP contributes to residual but we only need
     * the final token's state for generation).
     *
     * Simplified approach: process all tokens sequentially but skip the
     * output head matmul (the most expensive single op, vocab_size rows)
     * for all but the last token.
     */
    /*
     * Batch prefill: skip the output head matmul (vocab_size x hidden)
     * for all tokens except the last. The output head is typically the
     * single most expensive matmul (32K x 4K = 128M ops for Mistral-7B).
     * For a 100-token prompt, this saves 99 output head computations.
     */
    for (int i = 0; i < num_tokens - 1; i++) {
        int pos = kv_pos + i;
        int rc = forward_single_ex(model, tokens[i], pos, out_logits, 0, NULL);
        if (rc != INFERBIT_OK) return rc;
    }

    /* Last token — full forward with logits */
    return forward_single_ex(model, tokens[num_tokens - 1], kv_pos + num_tokens - 1, out_logits, 1, NULL);
}

int ib_forward_positions(inferbit_model* model, const int32_t* tokens,
                         int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward_positions");
        return INFERBIT_ERROR_PARAM;
    }

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;
    int vocab   = model->header.vocab_size;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d",
                     kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= vocab) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)",
                         tokens[i], vocab);
            return INFERBIT_ERROR_PARAM;
        }
    }

    /* Build absolute positions for forward_batch (each token is appended to
     * the KV cache in sequence starting at kv_pos). */
    int* positions = (int*)malloc((size_t)num_tokens * sizeof(int));
    if (!positions) { ib_set_error("oom"); return INFERBIT_ERROR_PARAM; }
    for (int i = 0; i < num_tokens; i++) positions[i] = kv_pos + i;

    int rc = forward_batch(model, tokens, positions, num_tokens, out_logits);
    free(positions);
    return rc;
}
