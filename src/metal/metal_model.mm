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
#include <errno.h>
#include <sys/mman.h>

/* Per-tensor GPU buffer set. EITHER (w, s, bits, blk32) — for INT4/INT8 —
 * OR (pq_rs, pq_cb, pq_idx, pq_M, pq_N, pq_G, pq_ns) — for PQv2 — is
 * populated. `is_pq` selects which path the dispatcher uses. */
struct tensor_bufs {
    /* INT4/INT8 path */
    void *w;
    void *s;
    int   bits;
    int   blk32;
    /* PQv2 path (the stacked 2D codebook pyramid) */
    int   is_pq;
    void *pq_rs;        /* [M] fp16 row_scale */
    void *pq_cb;        /* [n_subchunks * K * half] fp16 pre-decoded codebooks */
    void *pq_idx;       /* [n_chunks * n_subchunks * M] u8 indices */
    int   pq_M;
    int   pq_N;
    int   pq_G;
    int   pq_ns;
    /* Path D GPU drive mode: when set, pq_idx points at the SHARED
     * gpu_drive_idx_scratch (not a per-tensor MTLBuffer). The forward
     * path preads from this file offset (within the IBF) into the
     * scratch right before this tensor's matmul dispatch. */
    size_t pq_drive_file_offset;   /* 0 = not streamed (RAM mode) */
};

/* Per-layer GPU buffer set. */
struct layer_bufs {
    struct tensor_bufs q;
    struct tensor_bufs k;
    struct tensor_bufs v;
    struct tensor_bufs o;
    struct tensor_bufs gate;
    struct tensor_bufs up;
    struct tensor_bufs down;
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
    /* Output head: tensor_bufs unifies INT4/INT8/FP16 and PQv2 paths. */
    struct tensor_bufs output_head;

    /* GPU-side token-embedding buffers, populated only if the IBF's
     * token_embedding is PQ-encoded. Used by ib_metal_forward_decode_n
     * for on-GPU embedding feedback in the autoregressive loop. */
    int   token_embedding_is_pq;
    void *token_embedding_pq_rs;
    void *token_embedding_pq_cb;
    void *token_embedding_pq_idx;
    int   token_embedding_pq_G;
    int   token_embedding_pq_ns;

    /* Path D GPU drive mode (doc 32 follow-up): when the model is in
     * residency_mode==drive AND uploaded to Metal, the PQ-indices for
     * every streamed PQv2 tensor share a 2-slot ring of MTLBuffer
     * scratches. Slot i is refilled (pread + transpose) just before
     * the matmul that reads it; with 2 slots, the GPU dispatch reading
     * slot 0 runs concurrently with the CPU refilling slot 1, so we
     * checkpoint only once per pair of streamed matmuls (halves the
     * GPU syncs vs the original single-scratch implementation).
     *
     * Bounded GPU RAM = max-matmul-indices × 2 — still independent of
     * model size, still tiny vs full per-tensor residency. */
    void   *gpu_drive_idx_scratch[2];    /* 2 shared MTLBuffer slots */
    size_t  gpu_drive_idx_scratch_size;  /* size of EACH slot */
    void   *gpu_drive_idx_staging[2];    /* 2 CPU pread → transpose buffers */
    int     gpu_drive_idx_n_slots;       /* 0 = drive mode off, else N (currently 2) */
    int     gpu_drive_idx_slot;          /* next slot to refill: 0 or 1 */
    int     gpu_drive_idx_in_flight;     /* dispatches in current CB reading drive slots */

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
    void *scores_b;  /* [b_max][n_heads][seq_len]  fp32 — batched-attn scratch */
};

/* Verifies the IBF is in a layout the GPU dispatcher supports. */
static int model_is_supported(const inferbit_model *m, char *err, size_t err_sz) {
    #define CHECK(cond, msg) do { \
        if (!(cond)) { snprintf(err, err_sz, "%s", msg); return 0; } \
    } while (0)

    int kvb = m->header.kv_bits;
    CHECK(kvb == 16 || kvb == 8, "Metal forward requires kv_bits=16 or kv_bits=8");
    CHECK(m->output_norm.bits == 16, "output_norm must be fp16");
    if (m->output_head.pq) {
        CHECK(m->output_head.pq->K == 256 && m->output_head.pq->half == 2,
              "PQv2 GPU path requires K=256 and half=2 on output_head");
        CHECK(m->output_head.pq->l2_kind == 0,
              "PQv2 L2 residual not yet supported on GPU for output_head");
    } else {
        CHECK(m->output_head.bits == 4 || m->output_head.bits == 8 || m->output_head.bits == 16,
              "output_head must be INT4/INT8/FP16 or PQv2");
    }
    for (int L = 0; L < m->header.num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        /* Accept INT4/INT8 OR PQv2 (K=256, half=2) for each matmul tensor. */
        #define TENSOR_OK(name) do { \
            const ib_tensor_meta *tt = &lm->name; \
            if (tt->pq) { \
                CHECK(tt->pq->K == 256 && tt->pq->half == 2, \
                      "PQv2 GPU path requires K=256 and half=2: " #name); \
                CHECK(tt->pq->l2_kind == 0, \
                      "PQv2 L2 residual not yet supported on GPU: " #name); \
            } else { \
                CHECK(tt->bits == 4 || tt->bits == 8, \
                      "layer matmul tensor must be INT4/INT8 or PQv2: " #name); \
            } \
        } while (0)
        TENSOR_OK(q_proj); TENSOR_OK(k_proj); TENSOR_OK(v_proj); TENSOR_OK(o_proj);
        TENSOR_OK(gate_proj); TENSOR_OK(up_proj); TENSOR_OK(down_proj);
        #undef TENSOR_OK
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

/* fp16 helper: scalar fp32 → fp16 IEEE 754 round-to-nearest. */
static inline uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int      exp  = (int)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* PQv2 tensor upload: produces 3 GPU buffers (row_scale, decoded cb,
 * indices) + metadata. cb is pre-decoded from int8+scale → fp16 once on
 * the CPU so the kernel skips the dequant. */
/* keep_indices_resident: when set in drive mode, override the streaming
 * behavior and upload the full indices to a per-tensor MTLBuffer. Used
 * by the token_embedding upload because cpu_embed_lookup needs
 * randomly indexed reads that don't fit the per-matmul-pread pattern. */
static void upload_pqv2_tensor_ex(ib_metal_ctx *ctx, const inferbit_model *m,
                                    const pqv2_t *pq, struct tensor_bufs *out,
                                    int keep_indices_resident);

static void upload_pqv2_tensor(ib_metal_ctx *ctx, const inferbit_model *m,
                                const pqv2_t *pq, struct tensor_bufs *out)
{
    upload_pqv2_tensor_ex(ctx, m, pq, out, /*keep_indices_resident=*/0);
}

static void upload_pqv2_tensor_ex(ib_metal_ctx *ctx, const inferbit_model *m,
                                    const pqv2_t *pq, struct tensor_bufs *out,
                                    int keep_indices_resident)
{
    memset(out, 0, sizeof(*out));
    out->is_pq = 1;
    out->pq_M  = (int)pq->M;
    out->pq_N  = (int)pq->N;
    out->pq_G  = (int)pq->G;
    out->pq_ns = (int)pq->n_subchunks;

    /* row_scale is already fp16 (stored as uint16_t). Upload as-is. */
    out->pq_rs = ib_metal_alloc(ctx, (size_t)pq->M * sizeof(uint16_t), pq->row_scale);

    /* Decode codebooks: cb_q (int8) * cb_scale (fp16, per-K) → fp16 cb[ns][K][half]. */
    size_t cb_elts  = (size_t)pq->n_subchunks * pq->K * pq->half;
    size_t cb_bytes = cb_elts * sizeof(uint16_t);
    uint16_t *cb_fp16 = (uint16_t *)malloc(cb_bytes);
    if (!cb_fp16) { fprintf(stderr, "upload_pqv2_tensor: oom\n"); return; }
    for (uint32_t s = 0; s < pq->n_subchunks; s++) {
        for (uint32_t k = 0; k < pq->K; k++) {
            /* cb_scale stored as raw uint16 fp16 — convert via union/memcpy. */
            uint16_t scl_bits = pq->cb_scale[s * pq->K + k];
            uint32_t bits32 =
                ((scl_bits & 0x8000u) << 16) |
                ((((uint32_t)(scl_bits & 0x7C00u) >> 10) + 0x70u) << 23) |
                ((uint32_t)(scl_bits & 0x03FFu) << 13);
            /* zero / denorm / inf handling */
            if ((scl_bits & 0x7FFFu) == 0) bits32 = (uint32_t)(scl_bits & 0x8000u) << 16;
            else if ((scl_bits & 0x7C00u) == 0x7C00u) bits32 = ((uint32_t)(scl_bits & 0x8000u) << 16) | 0x7F800000u;
            float scl;
            memcpy(&scl, &bits32, 4);
            for (uint32_t h = 0; h < pq->half; h++) {
                int8_t q = pq->cb_q[(s * pq->K + k) * pq->half + h];
                float v = (float)q * scl;
                cb_fp16[(s * pq->K + k) * pq->half + h] = fp32_to_fp16_bits(v);
            }
        }
    }
    out->pq_cb = ib_metal_alloc(ctx, cb_bytes, cb_fp16);
    free(cb_fp16);

    /* Indices on disk are [n_chunks][n_subchunks][M] u8 (laid out by the
     * Python writer). For the GPU SIMD kernel we transpose to [M][total]
     * where total = n_chunks * n_subchunks. This makes the inner-loop
     * `indices[m * total + i]` reads coalesced within a SIMDgroup (32
     * lanes, each consuming i = lane, lane+32, ...). */
    uint32_t total = (uint32_t)((pq->N / pq->G) * pq->n_subchunks);
    size_t idx_bytes = (size_t)pq->M * total;

    /* GPU drive mode: don't allocate a per-tensor MTLBuffer for the
     * indices. Record the file offset; the model finalizer will point
     * pq_idx at a shared scratch MTLBuffer that the forward path
     * refills per matmul. Keeps GPU weight RAM bounded regardless of
     * model size. token_embedding overrides via keep_indices_resident=1
     * because embed_lookup needs random-access reads. */
    if (m && m->residency_mode == 1 && pq->indices_file_offset != 0
        && !keep_indices_resident) {
        out->pq_idx = NULL;
        out->pq_drive_file_offset = pq->indices_file_offset;
        return;
    }
    out->pq_drive_file_offset = 0;

    uint8_t *idx_t = (uint8_t *)malloc(idx_bytes);
    if (!idx_t) { fprintf(stderr, "upload_pqv2_tensor: oom on indices\n"); return; }
    /* In CPU drive mode (doc 32), pq->indices has been redirected to a
     * shared scratch buffer that is empty at upload time. The original
     * file offset is stashed in pq->indices_file_offset. pread() the
     * real bytes from disk into a temp buffer for GPU upload — GPU
     * forward keeps a full-residency copy in MTLBuffer (drive mode is
     * a CPU-side capability only). */
    uint8_t *src_buf = NULL;
    const uint8_t *src;
    if (m && m->residency_mode == 1 && pq->indices_file_offset != 0
        && m->drive_fd >= 0) {
        src_buf = (uint8_t *)malloc(idx_bytes);
        if (!src_buf) { free(idx_t); fprintf(stderr, "upload_pqv2_tensor: oom on drive-pread\n"); return; }
        size_t done = 0;
        off_t off = (off_t)pq->indices_file_offset;
        while (done < idx_bytes) {
            ssize_t r = pread(m->drive_fd, src_buf + done, idx_bytes - done, off + (off_t)done);
            if (r <= 0) {
                if (r == -1 && errno == EINTR) continue;
                fprintf(stderr, "upload_pqv2_tensor: pread failed in drive mode\n");
                free(src_buf); free(idx_t); return;
            }
            done += (size_t)r;
        }
        src = src_buf;
    } else {
        src = (const uint8_t *)pq->indices; /* [nc][ns][M] */
    }
    {
        uint32_t nc = pq->N / pq->G;
        for (uint32_t m_ = 0; m_ < pq->M; m_++) {
            for (uint32_t c = 0; c < nc; c++) {
                for (uint32_t s = 0; s < pq->n_subchunks; s++) {
                    idx_t[(size_t)m_ * total + c * pq->n_subchunks + s] =
                        src[((size_t)c * pq->n_subchunks + s) * pq->M + m_];
                }
            }
        }
    }
    if (src_buf) free(src_buf);
    out->pq_idx = ib_metal_alloc(ctx, idx_bytes, idx_t);
    free(idx_t);
}

/* Unified tensor upload: PQv2 if t->pq is set, else INT4/INT8 (w + s).
 * Releases mmap pages for the source bytes after copy. */
static void upload_tensor(ib_metal_ctx *ctx, const inferbit_model *m,
                           const ib_tensor_meta *t, struct tensor_bufs *out)
{
    memset(out, 0, sizeof(*out));
    if (t->pq) {
        upload_pqv2_tensor(ctx, m, t->pq, out);
        return;
    }
    upload_w_pair(ctx, m, t, &out->w, &out->s);
    out->bits  = t->bits;
    out->blk32 = (t->bits == 4 && t->scale_size > (size_t)t->shape[0] * 2);
    out->is_pq = 0;
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
        upload_tensor(ctx, m, &lm->q_proj,    &lb->q);
        upload_tensor(ctx, m, &lm->k_proj,    &lb->k);
        upload_tensor(ctx, m, &lm->v_proj,    &lb->v);
        upload_tensor(ctx, m, &lm->o_proj,    &lb->o);
        upload_tensor(ctx, m, &lm->gate_proj, &lb->gate);
        upload_tensor(ctx, m, &lm->up_proj,   &lb->up);
        upload_tensor(ctx, m, &lm->down_proj, &lb->down);
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
    upload_tensor(ctx, m, &m->output_head, &b->output_head);

    /* If token_embedding is PQ-encoded, upload it as PQv2 buffers so the
     * paginated decode path can do GPU embedding lookups. CPU still uses
     * the inferbit_model's pq pointer for the per-token CPU path. */
    if (m->token_embedding.pq) {
        struct tensor_bufs emb_tb;
        /* token_embedding needs full-residency indices (embed_lookup
         * does random-access reads not amenable to per-matmul stream). */
        upload_pqv2_tensor_ex(ctx, m, m->token_embedding.pq, &emb_tb, 1);
        b->token_embedding_is_pq = 1;
        b->token_embedding_pq_rs  = emb_tb.pq_rs;
        b->token_embedding_pq_cb  = emb_tb.pq_cb;
        b->token_embedding_pq_idx = emb_tb.pq_idx;
        b->token_embedding_pq_G   = emb_tb.pq_G;
        b->token_embedding_pq_ns  = emb_tb.pq_ns;
    } else {
        b->token_embedding_is_pq = 0;
    }

    /* Path D GPU drive mode finalizer: allocate a 2-slot ring of
     * MTLBuffer scratches, each sized for the largest streamed PQv2
     * matmul. tb->pq_idx is set to slot 0 as a sentinel — the actual
     * dispatch picks the slot dynamically via drive_prepare_pq. */
    if (m->residency_mode == 1) {
        size_t max_idx = 0;
        #define IDX_BYTES(tb) ((size_t)(tb).pq_M * ((tb).pq_N / (tb).pq_G) * (tb).pq_ns)
        #define CONSIDER(tb)  do { \
            if ((tb).is_pq && (tb).pq_drive_file_offset != 0) { \
                size_t s = IDX_BYTES(tb); \
                if (s > max_idx) max_idx = s; \
            } \
        } while (0)
        for (int L = 0; L < b->num_layers; L++) {
            struct layer_bufs *lb = &b->layers[L];
            CONSIDER(lb->q); CONSIDER(lb->k); CONSIDER(lb->v); CONSIDER(lb->o);
            CONSIDER(lb->gate); CONSIDER(lb->up); CONSIDER(lb->down);
        }
        CONSIDER(b->output_head);
        if (max_idx > 0) {
            /* Page-align scratch size. */
            size_t scratch_sz = (max_idx + 16383u) & ~((size_t)16383u);
            const int n_slots = 2;
            int alloc_ok = 1;
            for (int i = 0; i < n_slots; i++) {
                b->gpu_drive_idx_scratch[i] = ib_metal_alloc(ctx, scratch_sz, NULL);
                b->gpu_drive_idx_staging[i] = malloc(scratch_sz);
                if (!b->gpu_drive_idx_scratch[i] || !b->gpu_drive_idx_staging[i]) {
                    alloc_ok = 0;
                }
            }
            b->gpu_drive_idx_scratch_size = scratch_sz;
            b->gpu_drive_idx_n_slots = alloc_ok ? n_slots : 0;
            b->gpu_drive_idx_slot = 0;
            b->gpu_drive_idx_in_flight = 0;
            if (!alloc_ok) {
                fprintf(stderr, "ib_metal: GPU drive scratch alloc failed (%zu B × %d)\n",
                        scratch_sz, n_slots);
            } else {
                int n_repointed = 0;
                #define REPOINT(tb) do { \
                    if ((tb).is_pq && (tb).pq_drive_file_offset != 0) { \
                        (tb).pq_idx = b->gpu_drive_idx_scratch[0]; \
                        n_repointed++; \
                    } \
                } while (0)
                for (int L = 0; L < b->num_layers; L++) {
                    struct layer_bufs *lb = &b->layers[L];
                    REPOINT(lb->q); REPOINT(lb->k); REPOINT(lb->v); REPOINT(lb->o);
                    REPOINT(lb->gate); REPOINT(lb->up); REPOINT(lb->down);
                }
                REPOINT(b->output_head);
                fprintf(stderr, "ib_metal: GPU drive mode ON. %d-slot ring, %zu B/slot, %d tensors streamed\n",
                        n_slots, scratch_sz, n_repointed);
                #undef REPOINT
            }
        }
        #undef CONSIDER
        #undef IDX_BYTES
    }

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
        b->scores_b   = ib_metal_alloc(ctx,
            bm * (size_t)b->n_heads * (size_t)b->seq_len * sizeof(float), NULL);
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
    /* In GPU drive mode, every streamed PQv2 tensor's pq_idx points at
     * the same shared scratch — free only RAM-mode per-tensor indices
     * here, the shared scratch is freed once below. */
    #define FR_IDX(tb) do { \
        if ((tb).is_pq && (tb).pq_drive_file_offset == 0) FR((tb).pq_idx); \
    } while (0)
    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];
        #define FREE_TB(tb) do { \
            FR((tb).w); FR((tb).s); \
            FR((tb).pq_rs); FR((tb).pq_cb); \
            FR_IDX(tb); \
        } while (0)
        FREE_TB(lb->q); FREE_TB(lb->k); FREE_TB(lb->v); FREE_TB(lb->o);
        FREE_TB(lb->gate); FREE_TB(lb->up); FREE_TB(lb->down);
        #undef FREE_TB
        FR(lb->input_norm);
        FR(lb->post_norm);
        FR(lb->k_cache); FR(lb->v_cache);
        FR(lb->k_scales); FR(lb->v_scales);
    }
    free(b->layers);
    FR(b->output_norm);
    FR(b->output_head.w); FR(b->output_head.s);
    FR(b->output_head.pq_rs); FR(b->output_head.pq_cb);
    FR_IDX(b->output_head);
    #undef FR_IDX
    FR(b->token_embedding_pq_rs);
    FR(b->token_embedding_pq_cb);
    FR(b->token_embedding_pq_idx);
    /* Free shared GPU drive mode 2-slot ring exactly once. */
    for (int i = 0; i < 2; i++) {
        FR(b->gpu_drive_idx_scratch[i]);
        if (b->gpu_drive_idx_staging[i]) free(b->gpu_drive_idx_staging[i]);
    }
    FR(b->x); FR(b->xb); FR(b->xb2);
    FR(b->q); FR(b->k); FR(b->v); FR(b->attn_out);
    FR(b->hb); FR(b->hb2); FR(b->scores);
    FR(b->xq); FR(b->xs); FR(b->logits);
    FR(b->x_b); FR(b->xb_b); FR(b->xb2_b);
    FR(b->q_b); FR(b->k_b); FR(b->v_b); FR(b->attn_out_b);
    FR(b->hb_b); FR(b->hb2_b);
    FR(b->xq_b); FR(b->xs_b);
    FR(b->scores_b);
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

/* Path D GPU drive mode helper: pread this tensor's PQ indices from
 * the IBF into the slot's staging buffer, then transpose into the
 * slot's MTLBuffer scratch ([nc][ns][M] → [M][total]). Caller must
 * have already ensured the slot is not being read by any in-flight
 * GPU dispatch. Returns 0 on success, -1 on error. */
static int drive_load_pq_idx_to_slot(ib_metal_model_buffers *b,
                                       const struct tensor_bufs *tb,
                                       int slot)
{
    if (!tb || !tb->is_pq || tb->pq_drive_file_offset == 0) return 0;
    if (!b || !b->model) return -1;
    if (slot < 0 || slot >= b->gpu_drive_idx_n_slots) return -1;
    if (!b->gpu_drive_idx_scratch[slot] || !b->gpu_drive_idx_staging[slot]) return -1;
    const inferbit_model *m = (const inferbit_model *)b->model;
    if (m->drive_fd < 0) return -1;

    uint32_t M     = (uint32_t)tb->pq_M;
    uint32_t nc    = (uint32_t)(tb->pq_N / tb->pq_G);
    uint32_t ns    = (uint32_t)tb->pq_ns;
    uint32_t total = nc * ns;
    size_t idx_bytes = (size_t)M * total;
    if (idx_bytes > b->gpu_drive_idx_scratch_size) {
        fprintf(stderr, "drive_load_pq_idx_to_slot: %zu B > slot %zu B\n",
                idx_bytes, b->gpu_drive_idx_scratch_size);
        return -1;
    }
    uint8_t *staging = (uint8_t *)b->gpu_drive_idx_staging[slot];
    size_t done = 0;
    off_t off = (off_t)tb->pq_drive_file_offset;
    while (done < idx_bytes) {
        ssize_t r = pread(m->drive_fd, staging + done, idx_bytes - done,
                          off + (off_t)done);
        if (r <= 0) {
            if (r == -1 && errno == EINTR) continue;
            fprintf(stderr, "drive_load_pq_idx_to_slot: pread failed (off=%lld, want=%zu)\n",
                    (long long)off, idx_bytes - done);
            return -1;
        }
        done += (size_t)r;
    }
    uint8_t *dst = (uint8_t *)b->gpu_drive_idx_scratch[slot];
    /* Transpose [nc][ns][M] → [M][total = nc*ns]. */
    for (uint32_t m_ = 0; m_ < M; m_++) {
        uint8_t *row = dst + (size_t)m_ * total;
        for (uint32_t c = 0; c < nc; c++) {
            for (uint32_t s = 0; s < ns; s++) {
                row[c * ns + s] =
                    staging[((size_t)c * ns + s) * M + m_];
            }
        }
    }
    return 0;
}

/* GPU drive mode 2-slot ring wrapper. Returns the MTLBuffer pointer
 * the matmul should read pq_idx from — either the ring slot we just
 * loaded into (drive-streamed) or tb->pq_idx unchanged (RAM-mode).
 * Returns NULL on hard error.
 *
 * Ring discipline:
 *   - Each PQ matmul consumes one slot.
 *   - With 2 slots, we can fill the next slot while the GPU is reading
 *     the prior one. The CB stays open across the pair.
 *   - When BOTH slots are in flight, we must checkpoint (commit + wait)
 *     before refilling slot 0 — that drains the CB and frees both
 *     slots. Sync cost = once per pair of streamed matmuls, instead of
 *     once per matmul. */
static void *drive_prepare_pq(ib_metal_recorder *r,
                                ib_metal_model_buffers *b,
                                const struct tensor_bufs *tb)
{
    if (!tb || !tb->is_pq || tb->pq_drive_file_offset == 0) {
        return tb ? tb->pq_idx : NULL;
    }
    if (!b || b->gpu_drive_idx_n_slots < 1) return NULL;
    int n = b->gpu_drive_idx_n_slots;
    if (b->gpu_drive_idx_in_flight >= n) {
        if (ib_metal_recorder_checkpoint(r) != 0) return NULL;
        b->gpu_drive_idx_in_flight = 0;
    }
    int slot = b->gpu_drive_idx_slot;
    if (drive_load_pq_idx_to_slot(b, tb, slot) != 0) return NULL;
    b->gpu_drive_idx_slot = (slot + 1) % n;
    b->gpu_drive_idx_in_flight++;
    return b->gpu_drive_idx_scratch[slot];
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
    } else if (bits == 16) {
        return ib_metal_rec_matmul_fp16w_fp32x(r, x_fp32, weights, out, M, N);
    }
    return -1;
}

/* PQv2-aware matmul dispatch — pick PQv2 kernel when tensor is PQ-encoded,
 * else fall through to the INT4/INT8 path. M and N are the matmul shape
 * (caller's responsibility). In GPU drive mode this checkpoints + reloads
 * the shared scratch with this tensor's indices before recording. */
static int rec_matmul_tb(ib_metal_recorder *r,
                          ib_metal_model_buffers *b,
                          const struct tensor_bufs *tb,
                          const void *x_fp32,
                          void *out, void *xq, void *xs,
                          int M, int N)
{
    if (tb->is_pq) {
        void *idx = drive_prepare_pq(r, b, tb);
        if (!idx) return -1;
        /* Optional simdmat decode (IB_PQV2_SIMDMAT_DECODE=1). Uses
         * Apple simdgroup_matrix; on small M (TinyLlama) the x-broadcast
         * 8× compute waste outweighs the matrix-HW win, so off by default.
         * May help on larger M (lm_head M=32000) — kept available. */
        static int simdmat_dec_setting = -1;
        if (simdmat_dec_setting < 0) {
            const char *env = getenv("IB_PQV2_SIMDMAT_DECODE");
            simdmat_dec_setting = (env && env[0] == '1') ? 1 : 0;
        }
        if (simdmat_dec_setting) {
            int rc = ib_metal_rec_matmul_pqv2_simdmat_decode(r,
                tb->pq_rs, tb->pq_cb, idx, x_fp32, out,
                tb->pq_M, tb->pq_N, tb->pq_G, tb->pq_ns);
            if (rc == 0) return 0;
        }
        return ib_metal_rec_matmul_pqv2_k256_half2(r,
            tb->pq_rs, tb->pq_cb, idx, x_fp32, out,
            tb->pq_M, tb->pq_N, tb->pq_G, tb->pq_ns);
    }
    return rec_matmul(r, tb->bits, tb->blk32, x_fp32, tb->w, tb->s,
                       out, xq, xs, M, N);
}

/* Record the per-step body of a single-token forward pass (rmsnorm,
 * QKV, RoPE, attention, residuals, FFN, final rmsnorm, lm_head) into
 * the recorder. Assumes b->x already holds the fp32 input embedding.
 * Writes the next-token logits into b->logits. */
static void record_single_forward_step(ib_metal_recorder *r,
                                        ib_metal_model_buffers *b,
                                        int pos)
{
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
        /* Try fused Q+K+V matmul (one dispatch instead of three).
         * Requires all three to be the same kernel family (all INT4
         * blk32 or all INT8) and same N. Falls back to 3 separate
         * matmuls otherwise. */
        int rc_qkv = -1;
        int qh = b->n_heads * b->head_dim;
        if (!lb->q.is_pq && !lb->k.is_pq && !lb->v.is_pq
            && lb->q.bits == 4 && lb->q.blk32
            && lb->k.bits == 4 && lb->k.blk32
            && lb->v.bits == 4 && lb->v.blk32) {
            rc_qkv = ib_metal_rec_matmul_w4a8_blk32_dr_a32_qkv_fp32_in(r,
                b->xb,
                lb->q.w, lb->q.s,
                lb->k.w, lb->k.s,
                lb->v.w, lb->v.s,
                b->q, b->k, b->v,
                qh, kv_dim, hidden);
        } else if (lb->q.is_pq && lb->k.is_pq && lb->v.is_pq
                   && lb->q.pq_drive_file_offset == 0
                   && lb->k.pq_drive_file_offset == 0
                   && lb->v.pq_drive_file_offset == 0
                   && lb->q.pq_M == qh && lb->k.pq_M == kv_dim && lb->v.pq_M == kv_dim
                   && lb->q.pq_N == hidden && lb->k.pq_N == hidden && lb->v.pq_N == hidden
                   && lb->q.pq_G == lb->k.pq_G && lb->q.pq_G == lb->v.pq_G
                   && lb->q.pq_ns == lb->k.pq_ns && lb->q.pq_ns == lb->v.pq_ns) {
            rc_qkv = ib_metal_rec_matmul_pqv2_qkv_k256_half2(r,
                b->xb,
                lb->q.pq_rs, lb->q.pq_cb, lb->q.pq_idx, b->q,
                lb->k.pq_rs, lb->k.pq_cb, lb->k.pq_idx, b->k,
                lb->v.pq_rs, lb->v.pq_cb, lb->v.pq_idx, b->v,
                qh, kv_dim, hidden, lb->q.pq_G, lb->q.pq_ns);
        } else if (!lb->q.is_pq && !lb->k.is_pq && !lb->v.is_pq
                   && lb->q.bits == 8 && lb->k.bits == 8 && lb->v.bits == 8) {
            rc_qkv = ib_metal_rec_matmul_int8_fp32_in_qkv(r,
                b->xb,
                lb->q.w, lb->q.s,
                lb->k.w, lb->k.s,
                lb->v.w, lb->v.s,
                b->q, b->k, b->v,
                qh, kv_dim, hidden);
        }
        if (rc_qkv != 0) {
            rec_matmul_tb(r, b, &lb->q, b->xb, b->q, b->xq, b->xs, hidden, hidden);
            rec_matmul_tb(r, b, &lb->k, b->xb, b->k, b->xq, b->xs, kv_dim, hidden);
            rec_matmul_tb(r, b, &lb->v, b->xb, b->v, b->xq, b->xs, kv_dim, hidden);
        }
        ib_metal_rec_rope_inplace_qk(r, b->q, b->k, nh, nkh, hd, pos, th);
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
        /* Try fused o_proj+residual: matmul writes x += attn_out·o_w.
         * Saves one residual_add dispatch per layer. Falls back to the
         * 2-step path for non-blk32 / PQv2 IBFs. */
        {
            int rc_o = -1;
            if (!lb->o.is_pq && lb->o.bits == 4 && lb->o.blk32) {
                rc_o = ib_metal_rec_matmul_w4a8_blk32_dr_a32_add_fp32_in(
                    r, b->attn_out, lb->o.w, lb->o.s, b->x, hidden, hidden);
            }
            if (rc_o != 0) {
                rec_matmul_tb(r, b, &lb->o, b->attn_out, b->xb2, b->xq, b->xs, hidden, hidden);
                ib_metal_rec_residual_add(r, b->x, b->xb2, hidden);
            }
        }
        /* Optional fused rmsnorm + gate+up via IB_DECODE_FUSE_RMS_GU=1.
         * Tried but slower (-12%) — the redundant RMSNorm pass across
         * ~2800 TGs (per-matmul TG count) costs more than the saved
         * RMSNorm dispatch. Kept available, off by default. */
        static int fuse_rms_gu_setting = -1;
        if (fuse_rms_gu_setting < 0) {
            const char *env = getenv("IB_DECODE_FUSE_RMS_GU");
            fuse_rms_gu_setting = (env && env[0] == '1') ? 1 : 0;
        }
        int rc_rgu = -1;
        if (fuse_rms_gu_setting
            && !lb->gate.is_pq && lb->gate.bits == 4 && lb->gate.blk32
            && !lb->up.is_pq && lb->up.bits == 4 && lb->up.blk32) {
            rc_rgu = ib_metal_rec_matmul_w4a8_blk32_dr_a32_rmsnorm_gateup_fp32_in(r,
                b->x,
                lb->post_norm,
                lb->gate.w, lb->gate.s,
                lb->up.w,   lb->up.s,
                b->hb, b->hb2,
                inter, hidden, eps);
        }
        if (rc_rgu != 0) {
            ib_metal_rec_rmsnorm_fp16(r, b->x, lb->post_norm, b->xb, hidden, eps);
            int rc_gu = -1;
            if (!lb->gate.is_pq && lb->gate.bits == 4 && lb->gate.blk32
                && !lb->up.is_pq && lb->up.bits == 4 && lb->up.blk32) {
                rc_gu = ib_metal_rec_matmul_w4a8_blk32_dr_a32_gateup_fp32_in(r,
                    b->xb,
                    lb->gate.w, lb->gate.s,
                    lb->up.w,   lb->up.s,
                    b->hb, b->hb2,
                    inter, hidden);
            } else if (lb->gate.is_pq && lb->up.is_pq
                       && lb->gate.pq_drive_file_offset == 0
                       && lb->up.pq_drive_file_offset == 0
                       && lb->gate.pq_M == inter && lb->up.pq_M == inter
                       && lb->gate.pq_N == hidden && lb->up.pq_N == hidden
                       && lb->gate.pq_G == lb->up.pq_G
                       && lb->gate.pq_ns == lb->up.pq_ns) {
                rc_gu = ib_metal_rec_matmul_pqv2_gateup_k256_half2(r,
                    b->xb,
                    lb->gate.pq_rs, lb->gate.pq_cb, lb->gate.pq_idx, b->hb,
                    lb->up.pq_rs,   lb->up.pq_cb,   lb->up.pq_idx,   b->hb2,
                    inter, hidden, lb->gate.pq_G, lb->gate.pq_ns);
            }
            if (rc_gu != 0) {
                rec_matmul_tb(r, b, &lb->gate, b->xb, b->hb,  b->xq, b->xs, inter, hidden);
                rec_matmul_tb(r, b, &lb->up,   b->xb, b->hb2, b->xq, b->xs, inter, hidden);
            }
        }
        /* Silu+down fusion (IB_DECODE_FUSE_SILU=1) tried but slower
         * (-17%) — redundant exp() across 512 TGs costs more than the
         * saved dispatch. Available for benchmarking. */
        static int fuse_silu_setting = -1;
        if (fuse_silu_setting < 0) {
            const char *env = getenv("IB_DECODE_FUSE_SILU");
            fuse_silu_setting = (env && env[0] == '1') ? 1 : 0;
        }
        int fused_rc = -1;
        if (fuse_silu_setting && !lb->down.is_pq && lb->down.bits == 4 && lb->down.blk32) {
            fused_rc = ib_metal_rec_matmul_w4a8_blk32_dr_a32_silu_fp32_in(
                r, b->hb, b->hb2, lb->down.w, lb->down.s, b->xb, hidden, inter);
        }
        if (fused_rc != 0) {
            ib_metal_rec_silu_mul(r, b->hb, b->hb2, b->hb, inter);
            /* Fused down_proj+residual: writes x += hb·down_w. */
            int rc_d = -1;
            if (!lb->down.is_pq && lb->down.bits == 4 && lb->down.blk32) {
                rc_d = ib_metal_rec_matmul_w4a8_blk32_dr_a32_add_fp32_in(
                    r, b->hb, lb->down.w, lb->down.s, b->x, hidden, inter);
            }
            if (rc_d != 0) {
                rec_matmul_tb(r, b, &lb->down, b->hb, b->xb, b->xq, b->xs, hidden, inter);
                ib_metal_rec_residual_add(r, b->x, b->xb, hidden);
            }
        }
    }
    ib_metal_rec_rmsnorm_fp16(r, b->x, b->output_norm, b->xb, hidden, eps);
    rec_matmul_tb(r, b, &b->output_head, b->xb, b->logits, b->xq, b->xs, b->vocab, hidden);
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
    /* Drive-mode ring counters start clean on each new recorder/CB. */
    b->gpu_drive_idx_in_flight = 0;
    b->gpu_drive_idx_slot = 0;

    record_single_forward_step(r, b, pos);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
}

extern "C" int
ib_metal_forward_decode_n(ib_metal_ctx *ctx,
                           ib_metal_model_buffers *b,
                           const float *init_input_embed_fp32,
                           int start_pos, int n_steps,
                           int *out_tokens)
{
    if (!ctx || !b || !init_input_embed_fp32 || !out_tokens || n_steps <= 0) return -1;
    if (start_pos < 0 || start_pos + n_steps > b->seq_len) return -1;
    /* GPU embed feedback requires PQ-encoded token embedding. */
    if (!b->token_embedding_is_pq) return -2;

    /* Load first step's input into b->x. Subsequent steps use the GPU
     * embed_lookup to fill b->x from out_tokens[step-1]. */
    memcpy(b->x, init_input_embed_fp32, (size_t)b->hidden * sizeof(float));

    /* Scratch for the int32 token IDs produced on the GPU. */
    void *gpu_tokens = ib_metal_alloc(ctx, (size_t)n_steps * sizeof(int), NULL);
    if (!gpu_tokens) return -1;

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) { ib_metal_free(ctx, gpu_tokens); return -1; }

    for (int step = 0; step < n_steps; step++) {
        int pos = start_pos + step;
        if (step > 0) {
            /* Decode token_id[step-1] → b->x (the next forward's input). */
            int *tok_slot = (int *)gpu_tokens + (step - 1);
            ib_metal_rec_embed_lookup_pqv2(r,
                tok_slot,
                b->token_embedding_pq_rs,
                b->token_embedding_pq_cb,
                b->token_embedding_pq_idx,
                b->x,
                b->vocab, b->hidden,
                b->token_embedding_pq_G,
                b->token_embedding_pq_ns);
        }
        record_single_forward_step(r, b, pos);
        /* argmax b->logits → gpu_tokens[step]. */
        int *out_slot = (int *)gpu_tokens + step;
        ib_metal_rec_argmax_logits(r, b->logits, out_slot, b->vocab);
    }

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) { ib_metal_free(ctx, gpu_tokens); return rc; }

    memcpy(out_tokens, gpu_tokens, (size_t)n_steps * sizeof(int));
    ib_metal_free(ctx, gpu_tokens);
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
/* Batched PQv2-or-INT matmul dispatch. PQv2 path dispatched when tb is
 * non-NULL and is_pq is set; otherwise the int-bits path runs. In GPU
 * drive mode the PQ indices are streamed into the shared scratch
 * before the dispatch is recorded. */
static int rec_matmul_batched_tb(ib_metal_recorder *r,
                                  ib_metal_model_buffers *b,
                                  const struct tensor_bufs *tb,
                                  const void *x_fp32,
                                  void *out, void *xq, void *xs,
                                  int B, int M, int N)
{
    if (tb && tb->is_pq) {
        void *idx = drive_prepare_pq(r, b, tb);
        if (!idx) return -1;
        /* Optional tile-geometry override for the throughput sweep.
         *   IB_PQV2_TILE=tg32  (32×32, default, M%32+B%32+N%64)
         *   IB_PQV2_TILE=tg16  (16×32, higher occupancy, M%16+B%32+N%64)
         *   IB_PQV2_TILE=tg64  (64×32, more reuse, M%64+B%32+N%64)
         * When set and applicable, tries the chosen variant first. */
        static int tile_pref = -2;  /* -2 = unread, -1 = no pref, 0=32, 1=16, 2=64 */
        if (tile_pref == -2) {
            const char *env = getenv("IB_PQV2_TILE");
            if (!env)                    tile_pref = -1;
            else if (!strcmp(env, "tg16")) tile_pref = 1;
            else if (!strcmp(env, "tg64")) tile_pref = 2;
            else                            tile_pref = 0;
        }
        int rc;
        if (tile_pref == 1) {
            rc = ib_metal_rec_matmul_pqv2_batched_simdmat_tg16(r,
                tb->pq_rs, tb->pq_cb, idx, x_fp32, out,
                B, M, N, tb->pq_G, tb->pq_ns);
            if (rc == 0) return 0;
        } else if (tile_pref == 2) {
            rc = ib_metal_rec_matmul_pqv2_batched_simdmat_tg64(r,
                tb->pq_rs, tb->pq_cb, idx, x_fp32, out,
                B, M, N, tb->pq_G, tb->pq_ns);
            if (rc == 0) return 0;
        }
        /* Auto-split: decompose B into (k×32) + (k×16) + (k×8) + r.
         * Each chunk records a separate dispatch with offset pointers
         * so any prompt size B≥8 hits the simdmat fast path for as
         * much of B as possible. Tiny remainder r (<8) uses SIMD-coop. */
        int b_off = 0, rem = B;
        int split_rc = 0;
        struct { int unit; int (*rec)(ib_metal_recorder*, const void*, const void*,
                                       const void*, const void*, void*,
                                       int, int, int, int, int); } steps[] = {
            { 32, ib_metal_rec_matmul_pqv2_batched_simdmat },
            { 16, ib_metal_rec_matmul_pqv2_batched_simdmat_b16 },
            {  8, ib_metal_rec_matmul_pqv2_batched_simdmat_b8 },
        };
        for (int si = 0; si < 3 && split_rc == 0; si++) {
            int unit = steps[si].unit;
            if (rem < unit) continue;     /* try smaller units */
            int chunk = (rem / unit) * unit;
            const void *xp = (const float *)x_fp32 + (size_t)b_off * N;
            void       *op = (float *)out          + (size_t)b_off * M;
            split_rc = steps[si].rec(r,
                tb->pq_rs, tb->pq_cb, idx, xp, op,
                chunk, M, N, tb->pq_G, tb->pq_ns);
            if (split_rc != 0) break;
            b_off += chunk; rem -= chunk;
        }
        if (split_rc == 0 && rem > 0) {
            const void *xp = (const float *)x_fp32 + (size_t)b_off * N;
            void       *op = (float *)out          + (size_t)b_off * M;
            split_rc = ib_metal_rec_matmul_pqv2_k256_half2_batched(r,
                tb->pq_rs, tb->pq_cb, idx, xp, op,
                rem, M, N, tb->pq_G, tb->pq_ns);
            if (split_rc == 0) { b_off += rem; rem = 0; }
        }
        if (split_rc == 0 && b_off == B) return 0;
        /* Auto-split refused (e.g. shape constraint failed mid-way).
         * Fall through to SIMD-coop full-B fallback below. */
        return ib_metal_rec_matmul_pqv2_k256_half2_batched(r,
            tb->pq_rs, tb->pq_cb, idx, x_fp32, out,
            B, M, N, tb->pq_G, tb->pq_ns);
    }
    return -1;  /* caller falls through to bits-based variant */
}

static int rec_matmul_batched(ib_metal_recorder *r,
                                int bits, int blk32,
                                const void *x_fp32,
                                const void *weights, const void *w_scales,
                                void *out, void *xq, void *xs,
                                int B, int M, int N)
{
    if (bits == 4) {
        if (blk32) {
            /* Variant selection (read once, cached). Default is simdmat
             * (Apple's 8x8 fp16 matrix-multiply HW intrinsic) — empirically
             * ~5× faster than the scalar-style batched kernel at typical
             * prefill shapes. Env overrides:
             *   IB_PREFILL_SIMDMAT=0 → force non-simdmat (legacy default)
             *   IB_PREFILL_TILED=1   → threadgroup-memory tiled kernel
             * Precedence: SIMDMAT > TILED > default. */
            static int variant = -2;
            if (variant == -2) {
                const char *simd_env  = getenv("IB_PREFILL_SIMDMAT");
                const char *tiled_env = getenv("IB_PREFILL_TILED");
                if (simd_env && simd_env[0] == '0') variant = 0;
                else if (simd_env && simd_env[0] == '1') variant = 2;
                else if (tiled_env && tiled_env[0] == '1') variant = 1;
                else variant = 2;  /* default ON */
            }
            if (variant == 2) {
                /* tg32 with INT8 activation quantize (16 SGs / 32×32
                 * tile) is the empirical sweet spot on M4. The a16
                 * variant (fused fp32 input, no INT8 quantize pass)
                 * was tried but is marginally slower — the 4× extra
                 * fp32 activation bandwidth costs more than the saved
                 * dispatch. Available as IB_PREFILL_A16=1 for
                 * benchmarking. */
                static int a16_setting = -1;
                if (a16_setting < 0) {
                    const char *env = getenv("IB_PREFILL_A16");
                    a16_setting = (env && env[0] == '1') ? 1 : 0;
                }
                int rc;
                if (a16_setting) {
                    rc = ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg32_a16_fp32_in(
                        r, x_fp32, weights, w_scales, out, B, M, N);
                    if (rc == 0) return 0;
                }
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
            const struct tensor_bufs *tb = &lb->NAME; \
            if (tb->is_pq) break; /* PQv2 batched kernel exists */ \
            int bits = tb->bits; \
            int blk32 = tb->blk32; \
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
    if (n_tokens < 1) return -1;
    if (start_pos < 0 || start_pos + n_tokens > b->seq_len) return -1;
    if (!model_supports_batched_prefill(b)) return -2;

    /* Auto-chunk when the prompt exceeds the preallocated batched
     * scratch (b_max). Each chunk advances start_pos by chunk_size;
     * only the LAST chunk's logits are kept (caller only consumes the
     * final logits anyway). This makes arbitrary prompt sizes work
     * without bloating the default scratch allocation. */
    if (n_tokens > b->b_max) {
        int chunk_size = b->b_max;
        int sent = 0;
        while (sent < n_tokens) {
            int this_chunk = n_tokens - sent;
            if (this_chunk > chunk_size) this_chunk = chunk_size;
            int rc = ib_metal_forward_prefill(ctx, b,
                cpu_embeds_in + (size_t)sent * b->hidden,
                this_chunk, start_pos + sent, last_logits_out);
            if (rc != 0) return rc;
            sent += this_chunk;
        }
        return 0;
    }

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
    /* Drive-mode ring counters start clean on each new recorder/CB. */
    b->gpu_drive_idx_in_flight = 0;
    b->gpu_drive_idx_slot = 0;

    #define ROW_F(buf, n, dim) ((float*)(buf) + (size_t)(n) * (dim))

    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];

        /* Pre-attention RMSNorm (batched): x_b -> xb_b */
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->input_norm, b->xb_b, B, hidden, eps);

        /* Batched Q/K/V matmul (PQv2 or INT4 blk32 / INT8) */
        if (rec_matmul_batched_tb(r, b, &lb->q, b->xb_b, b->q_b, b->xq_b, b->xs_b, B, qh, hidden) != 0)
            rec_matmul_batched(r, lb->q.bits, lb->q.blk32, b->xb_b,
                lb->q.w, lb->q.s, b->q_b, b->xq_b, b->xs_b, B, qh, hidden);
        if (rec_matmul_batched_tb(r, b, &lb->k, b->xb_b, b->k_b, b->xq_b, b->xs_b, B, kv_dim, hidden) != 0)
            rec_matmul_batched(r, lb->k.bits, lb->k.blk32, b->xb_b,
                lb->k.w, lb->k.s, b->k_b, b->xq_b, b->xs_b, B, kv_dim, hidden);
        if (rec_matmul_batched_tb(r, b, &lb->v, b->xb_b, b->v_b, b->xq_b, b->xs_b, B, kv_dim, hidden) != 0)
            rec_matmul_batched(r, lb->v.bits, lb->v.blk32, b->xb_b,
                lb->v.w, lb->v.s, b->v_b, b->xq_b, b->xs_b, B, kv_dim, hidden);

        /* Batched RoPE: each row b at pos = start_pos + b */
        ib_metal_rec_rope_inplace_batched(r, b->q_b, B, nh,  hd, start_pos, th);
        ib_metal_rec_rope_inplace_batched(r, b->k_b, B, nkh, hd, start_pos, th);

        if (b->kv_bits == 16) {
            /* Batched fp16-KV attention: 4 dispatches per layer total
             * (vs 4*B in the per-position fallback). Causal mask is
             * handled inside the scores kernel. */
            ib_metal_rec_attention_block_fp16_batched(r,
                b->q_b, b->k_b, b->v_b,
                lb->k_cache, lb->v_cache,
                b->scores_b, b->attn_out_b,
                B, nh, nkh, hd, sl, start_pos);
        } else {
            /* INT8 KV: batched path not implemented yet, fall back to
             * per-position loop. */
            for (int bb = 0; bb < B; bb++) {
                int pos = start_pos + bb;
                float *q_row = ROW_F(b->q_b, bb, qh);
                float *k_row = ROW_F(b->k_b, bb, kv_dim);
                float *v_row = ROW_F(b->v_b, bb, kv_dim);
                float *attn_out_row = ROW_F(b->attn_out_b, bb, qh);
                ib_metal_rec_attention_block_int8(r,
                    q_row, k_row, v_row,
                    lb->k_cache, lb->v_cache,
                    lb->k_scales, lb->v_scales,
                    b->scores, attn_out_row,
                    nh, nkh, hd, sl, pos);
            }
        }

        /* Batched O matmul: attn_out_b -> xb2_b */
        if (rec_matmul_batched_tb(r, b, &lb->o, b->attn_out_b, b->xb2_b, b->xq_b, b->xs_b, B, hidden, qh) != 0)
            rec_matmul_batched(r, lb->o.bits, lb->o.blk32, b->attn_out_b,
                lb->o.w, lb->o.s, b->xb2_b, b->xq_b, b->xs_b, B, hidden, qh);

        /* Batched residual: x_b += xb2_b */
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb2_b, B, hidden);

        /* Batched post-attn RMSNorm: x_b -> xb_b */
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->post_norm, b->xb_b, B, hidden, eps);

        /* Batched gate / up matmul */
        if (rec_matmul_batched_tb(r, b, &lb->gate, b->xb_b, b->hb_b, b->xq_b, b->xs_b, B, inter, hidden) != 0)
            rec_matmul_batched(r, lb->gate.bits, lb->gate.blk32, b->xb_b,
                lb->gate.w, lb->gate.s, b->hb_b,  b->xq_b, b->xs_b, B, inter, hidden);
        if (rec_matmul_batched_tb(r, b, &lb->up, b->xb_b, b->hb2_b, b->xq_b, b->xs_b, B, inter, hidden) != 0)
            rec_matmul_batched(r, lb->up.bits, lb->up.blk32, b->xb_b,
                lb->up.w,   lb->up.s,   b->hb2_b, b->xq_b, b->xs_b, B, inter, hidden);

        /* Batched silu_mul: hb_b = silu(hb_b) * hb2_b */
        ib_metal_rec_silu_mul_batched(r, b->hb_b, b->hb2_b, b->hb_b, B, inter);

        /* Batched down matmul: hb_b -> xb_b */
        if (rec_matmul_batched_tb(r, b, &lb->down, b->hb_b, b->xb_b, b->xq_b, b->xs_b, B, hidden, inter) != 0)
            rec_matmul_batched(r, lb->down.bits, lb->down.blk32, b->hb_b,
                lb->down.w, lb->down.s, b->xb_b, b->xq_b, b->xs_b, B, hidden, inter);

        /* Batched residual: x_b += xb_b */
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb_b, B, hidden);
    }

    /* Final RMSNorm + output_head only on the last token (typical
     * prefill: caller wants logits for next-token prediction). Reuse
     * the per-token scratch (b->xb, b->logits) since we only need one. */
    ib_metal_rec_rmsnorm_fp16(r,
        ROW_F(b->x_b, B - 1, hidden), b->output_norm,
        b->xb, hidden, eps);
    rec_matmul_tb(r, b, &b->output_head, b->xb, b->logits, b->xq, b->xs, b->vocab, hidden);

    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(last_logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
    #undef ROW_F
}
