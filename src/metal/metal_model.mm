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
#include "../platform.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/mman.h>

/* Per-layer GPU buffer set. Each weight tensor's (bits, is_blk32) pair
 * is recorded so the forward dispatcher can pick the right kernel. */
struct layer_bufs {
    void *q_w, *q_s;     int q_bits;     int q_blk32;
    void *k_w, *k_s;     int k_bits;     int k_blk32;
    void *v_w, *v_s;     int v_bits;     int v_blk32;
    void *o_w, *o_s;     int o_bits;     int o_blk32;
    void *gate_w, *gate_s;  int gate_bits;  int gate_blk32;
    void *up_w,   *up_s;    int up_bits;    int up_blk32;
    void *down_w, *down_s;  int down_bits;  int down_blk32;
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
    int   output_head_blk32;

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

    /* Batched prefill scratch — sized for up to b_max tokens at once.
     * Layout for each is row-major [b][hidden_or_inter]: row b at byte
     * offset b * row_bytes. Allocated lazily on first ib_metal_forward_
     * prefill call (or at upload if IB_PREFILL_PREALLOC is set). */
    int   b_max;
    void *x_b;       /* [b_max][hidden]            fp32 */
    void *xb_b;      /* [b_max][hidden]            fp32 */
    void *xb2_b;     /* [b_max][hidden]            fp32 */
    void *q_b;       /* [b_max][n_heads*head_dim]  fp32 */
    void *k_b;       /* [b_max][kv_dim]            fp32 */
    void *v_b;       /* [b_max][kv_dim]            fp32 */
    void *attn_out_b;/* [b_max][n_heads*head_dim]  fp32 */
    void *hb_b;      /* [b_max][intermediate]      fp32 */
    void *hb2_b;     /* [b_max][intermediate]      fp32 */
    void *xq_b;      /* [b_max][max_in]            int8 */
    void *xs_b;      /* [b_max][max_in/128]        fp32 */
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

/* After a tensor's bytes have been memcpy'd into a Metal buffer, the source
 * mmap pages are no longer needed for GPU forward. madvise(MADV_DONTNEED) on
 * the page-aligned interior of the source range tells the kernel those pages
 * can be reclaimed — so the process never holds both the mmap pages AND the
 * Metal buffer for the same tensor at once, capping peak RSS during upload
 * at ~1× file size + 1 working tensor instead of ~2× file size.
 *
 * Safety: align inward (lo rounded up, hi rounded down) so we never touch
 * pages that may contain bytes belonging to neighbouring tensors. Skip
 * ranges smaller than one page after alignment.
 */
static void release_mmap_range(const void *src, size_t len) {
    if (!src || len == 0) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    uintptr_t lo = (uintptr_t)src;
    uintptr_t hi = lo + len;
    uintptr_t lo_aligned = (lo + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi_aligned = hi & ~((uintptr_t)pg - 1);
    if (hi_aligned <= lo_aligned) return;
    /* macOS notes: MADV_DONTNEED is mostly advisory and rarely reduces RSS.
     * MADV_FREE_REUSABLE is the Darwin pattern that actually returns pages
     * to the OS (used by malloc internals). For file-backed PROT_READ
     * mappings the kernel may map MADV_FREE → MADV_FREE_REUSABLE
     * internally, but call the more aggressive form explicitly. */
    size_t len_a = hi_aligned - lo_aligned;
    void *addr = (void*)lo_aligned;
    /* Note: on macOS, MADV_FREE on a PROT_READ file-backed mapping returns 0
     * but the kernel only reclaims pages under memory pressure — peak RSS
     * (high-water mark from /usr/bin/time -l) does not drop. On Linux,
     * MADV_DONTNEED reclaims immediately. Try Linux-style first, then fall
     * back to Darwin's softer hints. */
#if defined(MADV_DONTNEED)
    if (madvise(addr, len_a, MADV_DONTNEED) == 0) return;
#endif
#if defined(MADV_FREE)
    madvise(addr, len_a, MADV_FREE);
#endif
}

static void upload_w_pair(ib_metal_ctx *ctx, const inferbit_model *m,
                           const ib_tensor_meta *t,
                           void **out_w, void **out_s)
{
    const uint8_t *base = (const uint8_t *)m->weight_data;
    const uint8_t *w_src = base + t->offset;
    *out_w = ib_metal_alloc(ctx, t->size, w_src);
    release_mmap_range(w_src, t->size);
    if (t->scale_size > 0) {
        const uint8_t *s_src = base + t->scale_offset;
        *out_s = ib_metal_alloc(ctx, t->scale_size, s_src);
        release_mmap_range(s_src, t->scale_size);
    } else {
        *out_s = NULL;
    }
}

static void upload_norm(ib_metal_ctx *ctx, const inferbit_model *m,
                         const ib_tensor_meta *t, void **out)
{
    const uint8_t *base = (const uint8_t *)m->weight_data;
    const uint8_t *src = base + t->offset;
    *out = ib_metal_alloc(ctx, t->size, src);
    release_mmap_range(src, t->size);
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
        /* For each INT4 weight tensor, detect per-block-32 layout via
         * scale_size > rows * 2. shape[0] is the row count. */
        #define IS_BLK32(t) ((t).bits == 4 && (t).scale_size > (size_t)(t).shape[0] * 2)
        upload_w_pair(ctx, m, &lm->q_proj,    &lb->q_w,    &lb->q_s);    lb->q_bits    = lm->q_proj.bits;    lb->q_blk32    = IS_BLK32(lm->q_proj);
        upload_w_pair(ctx, m, &lm->k_proj,    &lb->k_w,    &lb->k_s);    lb->k_bits    = lm->k_proj.bits;    lb->k_blk32    = IS_BLK32(lm->k_proj);
        upload_w_pair(ctx, m, &lm->v_proj,    &lb->v_w,    &lb->v_s);    lb->v_bits    = lm->v_proj.bits;    lb->v_blk32    = IS_BLK32(lm->v_proj);
        upload_w_pair(ctx, m, &lm->o_proj,    &lb->o_w,    &lb->o_s);    lb->o_bits    = lm->o_proj.bits;    lb->o_blk32    = IS_BLK32(lm->o_proj);
        upload_w_pair(ctx, m, &lm->gate_proj, &lb->gate_w, &lb->gate_s); lb->gate_bits = lm->gate_proj.bits; lb->gate_blk32 = IS_BLK32(lm->gate_proj);
        upload_w_pair(ctx, m, &lm->up_proj,   &lb->up_w,   &lb->up_s);   lb->up_bits   = lm->up_proj.bits;   lb->up_blk32   = IS_BLK32(lm->up_proj);
        upload_w_pair(ctx, m, &lm->down_proj, &lb->down_w, &lb->down_s); lb->down_bits = lm->down_proj.bits; lb->down_blk32 = IS_BLK32(lm->down_proj);
        #undef IS_BLK32
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
    b->output_head_blk32 = (m->output_head.bits == 4 &&
                             m->output_head.scale_size > (size_t)m->output_head.shape[0] * 2);

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

    /* Prefill batched scratch. B_max chosen to comfortably cover prompts
     * up to a few hundred tokens on the M4 — total scratch is ~B_max ×
     * (5*hidden + 2*kv_dim + 2*intermediate) × 4 bytes; ~30 MB for the
     * 8B model at B_max=128, ~5 MB for TinyLlama. Tunable via
     * IB_PREFILL_BMAX env var. */
    {
        const char *bmax_env = getenv("IB_PREFILL_BMAX");
        b->b_max = bmax_env ? atoi(bmax_env) : 128;
        if (b->b_max < 1) b->b_max = 128;
        size_t bm = (size_t)b->b_max;
        size_t H  = (size_t)b->hidden, I = (size_t)b->intermediate;
        size_t QH = (size_t)b->n_heads * b->head_dim;
        size_t KV = (size_t)b->kv_dim;
        b->x_b        = ib_metal_alloc(ctx, bm * H  * sizeof(float), NULL);
        b->xb_b       = ib_metal_alloc(ctx, bm * H  * sizeof(float), NULL);
        b->xb2_b      = ib_metal_alloc(ctx, bm * H  * sizeof(float), NULL);
        b->q_b        = ib_metal_alloc(ctx, bm * QH * sizeof(float), NULL);
        b->k_b        = ib_metal_alloc(ctx, bm * KV * sizeof(float), NULL);
        b->v_b        = ib_metal_alloc(ctx, bm * KV * sizeof(float), NULL);
        b->attn_out_b = ib_metal_alloc(ctx, bm * QH * sizeof(float), NULL);
        b->hb_b       = ib_metal_alloc(ctx, bm * I  * sizeof(float), NULL);
        b->hb2_b      = ib_metal_alloc(ctx, bm * I  * sizeof(float), NULL);
        b->xq_b       = ib_metal_alloc(ctx, bm * (size_t)max_n, NULL);
        b->xs_b       = ib_metal_alloc(ctx, bm * (size_t)xs_groups * sizeof(float), NULL);
    }

    return b;
}

/* Strip the original mmap'd IBF down to just the token-embedding bytes.
 * Used after ib_metal_upload_model when running GPU-only — every other
 * weight tensor lives on the GPU side at this point, so the rest of the
 * mmap is dead weight (peak RSS of file_size + Metal_buffer_size).
 *
 * Approach (works on both Linux and macOS, unlike madvise):
 *   1. Compute the byte-extent the embedding occupies in the IBF
 *      (data range + scale range, taking the union).
 *   2. malloc a buffer of that extent and memcpy the bytes out.
 *   3. Rebind model->weight_data so the existing offset arithmetic
 *      (cpu_embed_lookup uses base + token_embedding.offset) still resolves
 *      to the right address inside the new buffer.
 *   4. munmap the original IBF and close the fd.
 *
 * Caller MUST be done reading any non-embedding tensor through model->
 * weight_data after this call. The GPU forward path is fine — it only
 * touches GPU buffers post-upload.
 */
extern "C" int
ib_metal_strip_cpu_mmap(void *model_handle)
{
    if (!model_handle) return -1;
    inferbit_model *m = (inferbit_model *)model_handle;
    if (!m->weight_data_mmap || !m->weight_data) return 0; /* nothing to do */

    const ib_tensor_meta *e = &m->token_embedding;
    size_t lo = e->offset;
    size_t hi = e->offset + e->size;
    if (e->scale_size > 0) {
        if (e->scale_offset < lo) lo = e->scale_offset;
        size_t s_hi = e->scale_offset + e->scale_size;
        if (s_hi > hi) hi = s_hi;
    }
    size_t extent = hi - lo;
    if (extent == 0) return -1;

    uint8_t *buf = (uint8_t *)malloc(extent);
    if (!buf) return -1;

    const uint8_t *src = (const uint8_t *)m->weight_data + lo;
    memcpy(buf, src, extent);

    /* Compute mmap base + size for the upcoming munmap, while the original
     * mapping is still live (we need m->weight_data and mmap_fd). */
    size_t weight_offset = m->header.weight_data_offset;
    void *base = (uint8_t *)m->weight_data - weight_offset;
    size_t mmap_size = 0;
    if (m->mmap_fd >= 0) {
        ib_struct_stat st;
        if (ib_fstat(m->mmap_fd, &st) == 0) mmap_size = (size_t)st.st_size;
    }

    /* Rebind: new_weight_data + e->offset must equal &buf[e->offset - lo].
     * So new_weight_data = buf - lo. */
    m->embed_strip_buffer = buf;
    m->weight_data = (uint8_t *)buf - lo;
    m->weight_data_mmap = false;

    if (mmap_size > 0) {
        ib_munmap(base, mmap_size);
    }
    if (m->mmap_fd >= 0) {
        ib_close(m->mmap_fd);
        m->mmap_fd = -1;
    }
    return (int)extent;
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
    FR(b->x_b); FR(b->xb_b); FR(b->xb2_b);
    FR(b->q_b); FR(b->k_b); FR(b->v_b); FR(b->attn_out_b);
    FR(b->hb_b); FR(b->hb2_b);
    FR(b->xq_b); FR(b->xs_b);
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
 * the tensor's bit width AND blk32-ness. xq/xs scratch is only used
 * for INT4 (w4a8) variants. */
static int rec_matmul(ib_metal_recorder *r,
                       int bits, int blk32,
                       const void *x_fp32,
                       const void *weights, const void *w_scales,
                       void *out, void *xq, void *xs,
                       int M, int N)
{
    if (bits == 4) {
        if (blk32) {
            return ib_metal_rec_matmul_w4a8_blk32_fp32_in(
                r, x_fp32, weights, w_scales, out, xq, xs, M, N);
        }
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
        rec_matmul(r, lb->q_bits, lb->q_blk32, b->xb, lb->q_w, lb->q_s, b->q, b->xq, b->xs, hidden, hidden);
        rec_matmul(r, lb->k_bits, lb->k_blk32, b->xb, lb->k_w, lb->k_s, b->k, b->xq, b->xs, kv_dim, hidden);
        rec_matmul(r, lb->v_bits, lb->v_blk32, b->xb, lb->v_w, lb->v_s, b->v, b->xq, b->xs, kv_dim, hidden);
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
        rec_matmul(r, lb->o_bits, lb->o_blk32, b->attn_out, lb->o_w, lb->o_s, b->xb2, b->xq, b->xs, hidden, hidden);
        ib_metal_rec_residual_add(r, b->x, b->xb2, hidden);
        ib_metal_rec_rmsnorm_fp16(r, b->x, lb->post_norm, b->xb, hidden, eps);
        rec_matmul(r, lb->gate_bits, lb->gate_blk32, b->xb, lb->gate_w, lb->gate_s, b->hb,  b->xq, b->xs, inter, hidden);
        rec_matmul(r, lb->up_bits,   lb->up_blk32,   b->xb, lb->up_w,   lb->up_s,   b->hb2, b->xq, b->xs, inter, hidden);
        ib_metal_rec_silu_mul(r, b->hb, b->hb2, b->hb, inter);
        rec_matmul(r, lb->down_bits, lb->down_blk32, b->hb, lb->down_w, lb->down_s, b->xb, b->xq, b->xs, hidden, inter);
        ib_metal_rec_residual_add(r, b->x, b->xb, hidden);
    }
    ib_metal_rec_rmsnorm_fp16(r, b->x, b->output_norm, b->xb, hidden, eps);
    rec_matmul(r, b->output_head_bits, b->output_head_blk32, b->xb,
                b->output_head_w, b->output_head_s,
                b->logits, b->xq, b->xs, b->vocab, hidden);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
}

/* Records ONE batched matmul into the recorder, picking the kernel
 * based on the tensor's bit width AND blk32-ness. xq/xs scratch is
 * only used for INT4 (w4a8) variants. Mirrors rec_matmul above.
 *
 * The blk32 path uses the non-tiled variant by default. The tiled
 * variant (weight row loaded to threadgroup memory once and shared
 * across TILE_B SIMD groups) is available via IB_PREFILL_TILED=1 but
 * empirically gives no speedup on Apple M4 — the L1 cache already
 * deduplicates same-row weight loads across SIMD groups in the same
 * threadgroup, so explicit shared memory just adds barrier overhead.
 * Kept around because it may help on other Apple GPUs / batch sizes. */
static int rec_matmul_batched(ib_metal_recorder *r,
                                int bits, int blk32,
                                const void *x_fp32,
                                const void *weights, const void *w_scales,
                                void *out, void *xq, void *xs,
                                int B, int M, int N)
{
    if (bits == 4) {
        if (blk32) {
            /* Variant selection (read once, cached):
             *   IB_PREFILL_SIMDMAT=1 → simdgroup_matrix kernel (Apple's
             *     8x8 fp16 matrix-multiply hardware intrinsic)
             *   IB_PREFILL_TILED=1   → threadgroup-memory tiled kernel
             *   default              → non-tiled batched kernel
             * Precedence: SIMDMAT > TILED > default. */
            static int variant = -2;
            if (variant == -2) {
                const char *simd_env  = getenv("IB_PREFILL_SIMDMAT");
                const char *tiled_env = getenv("IB_PREFILL_TILED");
                if (simd_env && simd_env[0] == '1') variant = 2;
                else if (tiled_env && tiled_env[0] == '1') variant = 1;
                else variant = 0;
            }
            if (variant == 2) {
                /* Try simdmat variants in descending tile size — bigger
                 * tile = more dequant amortization, but needs M & B
                 * divisible by tile dim. Falls through if shape doesn't
                 * fit any simdmat variant. */
                int rc;
                rc = ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg32_fp32_in(
                    r, x_fp32, weights, w_scales, out, xq, xs, B, M, N);
                if (rc == 0) return 0;
                rc = ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg_fp32_in(
                    r, x_fp32, weights, w_scales, out, xq, xs, B, M, N);
                if (rc == 0) return 0;
                rc = ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_fp32_in(
                    r, x_fp32, weights, w_scales, out, xq, xs, B, M, N);
                if (rc != -2) return rc;
                /* All simdmat variants refused — fall back to non-tiled. */
            }
            if (variant == 1) {
                return ib_metal_rec_matmul_w4a8_blk32_batched_tiled_fp32_in(
                    r, x_fp32, weights, w_scales, out, xq, xs, B, M, N);
            }
            return ib_metal_rec_matmul_w4a8_blk32_batched_fp32_in(
                r, x_fp32, weights, w_scales, out, xq, xs, B, M, N);
        }
        /* TODO: per-row INT4 batched matmul. Today fall through. */
        return -1;
    } else if (bits == 8) {
        return ib_metal_rec_matmul_int8_fp32_in_batched(
            r, x_fp32, weights, w_scales, out, B, M, N);
    }
    return -1;
}

/* Helper: prefill currently supports INT4-blk32 and INT8 per-layer
 * tensors (mixed OK). Output head is unrestricted (it runs only on the
 * last token, so batching isn't needed there). Returns 0 if any layer
 * tensor is per-row INT4 — that variant doesn't have a batched kernel
 * yet, caller should fall back to per-token. */
static int model_supports_batched_prefill(const ib_metal_model_buffers *b) {
    for (int L = 0; L < b->num_layers; L++) {
        const struct layer_bufs *lb = &b->layers[L];
        #define CHK(NAME) do { \
            int bits = lb->NAME##_bits; \
            int blk32 = lb->NAME##_blk32; \
            if (bits != 8 && !(bits == 4 && blk32)) return 0; \
        } while (0)
        CHK(q); CHK(k); CHK(v); CHK(o); CHK(gate); CHK(up); CHK(down);
        #undef CHK
    }
    return 1;
}

extern "C" int
ib_metal_forward_prefill(ib_metal_ctx *ctx,
                          ib_metal_model_buffers *b,
                          const float *cpu_embeds_in,
                          int n_tokens, int start_pos,
                          float *last_logits_out)
{
    if (!ctx || !b || !cpu_embeds_in || !last_logits_out) return -1;
    if (n_tokens < 1 || n_tokens > b->b_max) return -1;
    if (start_pos < 0 || start_pos + n_tokens > b->seq_len) return -1;
    if (!model_supports_batched_prefill(b)) return -2;

    int hidden = b->hidden;
    int inter  = b->intermediate;
    int kv_dim = b->kv_dim;
    int qh     = b->n_heads * b->head_dim;
    int nh     = b->n_heads;
    int nkh    = b->n_kv_heads;
    int hd     = b->head_dim;
    float th   = b->rope_theta;
    float eps  = b->eps;
    int sl     = b->seq_len;
    int B      = n_tokens;

    /* Copy embeddings into the batched x buffer (host-visible Shared mode). */
    memcpy(b->x_b, cpu_embeds_in, (size_t)B * hidden * sizeof(float));

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) return -1;

    #define ROW_F(buf, n, dim) ((float*)(buf) + (size_t)(n) * (dim))

    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];

        /* Pre-attention RMSNorm (batched): x_b -> xb_b */
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->input_norm, b->xb_b, B, hidden, eps);

        /* Batched Q/K/V matmul (INT4 blk32 or INT8) */
        rec_matmul_batched(r, lb->q_bits, lb->q_blk32, b->xb_b,
            lb->q_w, lb->q_s, b->q_b, b->xq_b, b->xs_b, B, qh, hidden);
        rec_matmul_batched(r, lb->k_bits, lb->k_blk32, b->xb_b,
            lb->k_w, lb->k_s, b->k_b, b->xq_b, b->xs_b, B, kv_dim, hidden);
        rec_matmul_batched(r, lb->v_bits, lb->v_blk32, b->xb_b,
            lb->v_w, lb->v_s, b->v_b, b->xq_b, b->xs_b, B, kv_dim, hidden);

        /* Batched RoPE: each row b at pos = start_pos + b */
        ib_metal_rec_rope_inplace_batched(r, b->q_b, B, nh,  hd, start_pos, th);
        ib_metal_rec_rope_inplace_batched(r, b->k_b, B, nkh, hd, start_pos, th);

        /* Per-token attention block (KV cache fill at distinct positions;
         * causal scoring against [0..pos]). Attention is hard to batch
         * cleanly because each position attends to a different prefix. */
        for (int bb = 0; bb < B; bb++) {
            int pos = start_pos + bb;
            float *q_row = ROW_F(b->q_b, bb, qh);
            float *k_row = ROW_F(b->k_b, bb, kv_dim);
            float *v_row = ROW_F(b->v_b, bb, kv_dim);
            float *attn_out_row = ROW_F(b->attn_out_b, bb, qh);

            if (b->kv_bits == 16) {
                ib_metal_rec_attention_block_fp16(r,
                    q_row, k_row, v_row,
                    lb->k_cache, lb->v_cache,
                    b->scores, attn_out_row,
                    nh, nkh, hd, sl, pos);
            } else {
                ib_metal_rec_attention_block_int8(r,
                    q_row, k_row, v_row,
                    lb->k_cache, lb->v_cache,
                    lb->k_scales, lb->v_scales,
                    b->scores, attn_out_row,
                    nh, nkh, hd, sl, pos);
            }
        }

        /* Batched O matmul: attn_out_b -> xb2_b */
        rec_matmul_batched(r, lb->o_bits, lb->o_blk32, b->attn_out_b,
            lb->o_w, lb->o_s, b->xb2_b, b->xq_b, b->xs_b, B, hidden, qh);

        /* Batched residual: x_b += xb2_b */
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb2_b, B, hidden);

        /* Batched post-attn RMSNorm: x_b -> xb_b */
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->post_norm, b->xb_b, B, hidden, eps);

        /* Batched gate / up matmul */
        rec_matmul_batched(r, lb->gate_bits, lb->gate_blk32, b->xb_b,
            lb->gate_w, lb->gate_s, b->hb_b,  b->xq_b, b->xs_b, B, inter, hidden);
        rec_matmul_batched(r, lb->up_bits, lb->up_blk32, b->xb_b,
            lb->up_w,   lb->up_s,   b->hb2_b, b->xq_b, b->xs_b, B, inter, hidden);

        /* Batched silu_mul: hb_b = silu(hb_b) * hb2_b */
        ib_metal_rec_silu_mul_batched(r, b->hb_b, b->hb2_b, b->hb_b, B, inter);

        /* Batched down matmul: hb_b -> xb_b */
        rec_matmul_batched(r, lb->down_bits, lb->down_blk32, b->hb_b,
            lb->down_w, lb->down_s, b->xb_b, b->xq_b, b->xs_b, B, hidden, inter);

        /* Batched residual: x_b += xb_b */
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb_b, B, hidden);
    }

    /* Final RMSNorm + output_head only on the last token (typical
     * prefill: caller wants logits for next-token prediction). Reuse
     * the per-token scratch (b->xb, b->logits) since we only need one. */
    ib_metal_rec_rmsnorm_fp16(r,
        ROW_F(b->x_b, B - 1, hidden), b->output_norm,
        b->xb, hidden, eps);
    rec_matmul(r, b->output_head_bits, b->output_head_blk32, b->xb,
                b->output_head_w, b->output_head_s,
                b->logits, b->xq, b->xs, b->vocab, hidden);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(last_logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
    #undef ROW_F
}
