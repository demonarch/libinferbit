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

/* pqv2_kernel.h is wrapped in extern "C" so pqv2_h2f / pqv2_f2h are
 * C-linkage when included from this .mm — no forward decls needed. */

/* Per-tensor GPU buffer set. EITHER (w, s, bits, blk32) — for INT4/INT8 —
 * OR (pq_rs, pq_cb, pq_idx, pq_M, pq_N, pq_G, pq_ns) — for PQv2 — is
 * populated. `is_pq` selects which path the dispatcher uses. */
struct tensor_bufs {
    /* INT4/INT8 path */
    void *w;
    void *s;
    int   bits;
    int   blk32;
    /* Optional fp16 dequant of INT4-blk32 weights for the MPS-hybrid
     * prefill kernel (IB_PREFILL_FP16_W=1). NULL when not pre-dequanted.
     * Doubles the per-tensor GPU memory (acceptable on small models). */
    void *w_fp16;
    /* PQv2 path (the stacked 2D codebook pyramid) */
    int   is_pq;
    void *pq_rs;        /* [M] fp16 row_scale */
    void *pq_cb;        /* [n_subchunks * K * half] fp16 pre-decoded codebooks */
    void *pq_idx;       /* [n_chunks * n_subchunks * M] u8 indices */
    int   pq_M;
    int   pq_N;
    int   pq_G;
    int   pq_ns;
    /* PQv2 pyramid L2-PQ residual stage (only populated when the
     * source tensor has l2_kind == 2; pq_K_l2 == 0 means flat PQv2).
     * The kernel dispatcher checks pq_K_l2 > 0 to route to the
     * l2residual decode kernel instead of the flat decoder.
     *
     * Layout matches the L1 buffers — codebook is pre-decoded fp16
     * with stride K_L2 entries per subchunk (vs 256 for L1), and the
     * indices are transposed to [M][total] just like L1. */
    void *pq_cb_l2;     /* [n_subchunks * K_L2 * half] fp16 pre-decoded */
    void *pq_idx_l2;    /* L2 indices: layout depends on pq_l2_idx_bits */
    int   pq_K_l2;      /* 0 = no L2 residual (flat); else K_L2 (≤ 64) */
    /* L2 index on-disk bit-width (Stage 5h.1). 8 = legacy uint8
     * [M][total] transposed copy. 6 = bit-packed 4-in-3-bytes along the
     * M axis, kept in the on-disk slot-major layout
     * [total][ceil(M/4)*3] so 4 rows share a 3-byte triple. */
    int   pq_l2_idx_bits;
    /* Stage 5g.2 — on-disk L1 index layout.
     *   0 = chunk-major [n_chunks][n_subchunks][M] (legacy).
     *   1 = row-major   [M][n_chunks][n_subchunks] (kernel-native; the
     *       drive-mode pread can land directly in the MTLBuffer scratch
     *       with no transpose, mirroring the doc-35 sidecar fast path). */
    int   pq_l1_idx_layout;
    /* Path D GPU drive mode: when set, pq_idx points at the SHARED
     * gpu_drive_idx_scratch (not a per-tensor MTLBuffer). The forward
     * path preads from this file offset (within the IBF) into the
     * scratch right before this tensor's matmul dispatch. */
    size_t pq_drive_file_offset;   /* 0 = not streamed (RAM mode) */
    /* PEAK-RAM fix: pyramid L2 indices streamed from disk in GPU drive
     * mode (mirrors pq_drive_file_offset for L1). When set, pq_idx_l2 is
     * NULL at upload time (no resident MTLBuffer) and the forward path
     * preads the L2 indices into the shared gpu_drive_l2_idx_scratch ring
     * right before each pyramid matmul. The L2 CODEBOOK (pq_cb_l2) stays
     * resident (small). 0 = L2 indices resident (RAM mode). */
    size_t pq_l2_drive_file_offset; /* 0 = L2 idx resident */
    size_t pq_l2_drive_disk_bytes;  /* on-disk L2 idx byte count to pread */
    size_t pq_l2_drive_scratch_bytes; /* scratch (kernel-native) byte count */
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
    int seq_len;       /* physical KV ring size (== kv_window when windowed) */
    int max_logical_pos; /* logical context bound for start_pos checks */
    int kv_window;     /* 0 = full causal; >0 = ring buffer size (== seq_len) */
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
    void   *gpu_drive_idx_scratch[4];    /* up to 4 shared MTLBuffer slots */
    size_t  gpu_drive_idx_scratch_size;  /* size of EACH slot */
    void   *gpu_drive_idx_staging[4];    /* up to 4 CPU pread → transpose buffers */
    int     gpu_drive_idx_n_slots;       /* 0 = drive mode off, else N (1..4) */
    int     gpu_drive_idx_slot;          /* legacy: kept-reset to 0, not read */
    int     gpu_drive_idx_in_flight;     /* dispatches in current sub-ring */
    /* Async commit (doc-35 feature 4) with 2-sub-ring layout: slots are
     * split into N sub-rings of sr_size each (e.g. N=4 → 2 sub-rings of 2).
     * Each sub-ring is consumed by ONE in-flight CB; when a sub-ring fills,
     * we commit_async its CB, swap to the other sub-ring, and wait on its
     * prior CB before reusing those physical slots. Keeps two CBs in flight
     * without racing slot reads against slot writes.
     *
     * Validated 2026-05-14 on llama-3.2-1B PQv2 drive mode:
     *   sync N=1 (no async):   PPL=12.641506, 6.64 tok/s
     *   async N=4 (2sr × 2):   PPL=12.641506, 9.79 tok/s  → +47.4%
     * PPL identical to 6 dec → zero correctness regression. */
    int     gpu_drive_n_subrings;        /* 1 (sync fallback) or 2 (async) */
    int     gpu_drive_sr_size;           /* slots per sub-ring (n_slots / n_subrings) */
    int     gpu_drive_cur_sr;            /* current sub-ring (0..n_subrings-1) */
    void   *gpu_drive_pending_cb;        /* CB of the OTHER sub-ring (in flight) */

    /* PEAK-RAM fix: parallel L2-index scratch ring for pyramid drive
     * mode. Same slot count / sub-ring discipline as the L1 ring (each
     * pyramid matmul consumes slot i of BOTH rings in lockstep), so no
     * extra synchronization is needed — drive_prepare_pq loads L1 then
     * L2 into the same slot index. Sized for the largest tensor's
     * kernel-native L2 indices. 0 slots = no pyramid tensors streamed. */
    void   *gpu_drive_l2_idx_scratch[4]; /* L2 idx MTLBuffer slots */
    void   *gpu_drive_l2_idx_staging[4]; /* CPU pread → transpose buffers */
    size_t  gpu_drive_l2_idx_scratch_size; /* size of EACH L2 slot */
    int     gpu_drive_l2_idx_n_slots;    /* 0 = no L2 streaming */

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
    /* MoME FFN is not implemented on the Metal forward kernel. The upload
     * loop intentionally leaves lb->gate/up/down zeroed for MoME layers
     * (the per-expert tensors live in lm->*_proj_experts), but
     * ib_metal_forward_token / ib_metal_forward_prefill unconditionally
     * dispatch matmuls against those slots — producing garbage logits
     * (PPL=1e5+) without crashing because the calloc-zero buffers are
     * valid Metal allocations. Refuse Metal up-front so the caller falls
     * back to CPU, where mome_dispatch_ffn handles MoME correctly. */
     for (int L = 0; L < m->header.num_layers; L++) {
         if (m->layers[L].mome_experts > 1) {
             snprintf(err, err_sz,
                 "Metal forward not supported for MoME models (layer %d has %d experts); "
                 "set IB_BACKEND=cpu or use a non-MoME IBF",
                 L, m->layers[L].mome_experts);
             return 0;
         }
     }
    if (m->output_head.pq) {
        CHECK(m->output_head.pq->K == 256 && m->output_head.pq->half == 2,
              "PQv2 GPU path requires K=256 and half=2 on output_head");
        /* l2_kind == 0 (flat) and l2_kind == 2 (pyramid, K_L2 ≤ 64) both
         * supported on GPU after Stage 5a (docs/v2/00_CORRECTION.md).
         *
         * Goal H3 (partial): pyramid + GPU drive (IB_RESIDENCY_MODE=drive)
         * is now permitted. L1 indices continue to stream through the
         * shared scratch ring as in flat drive mode; the L2 codebook and
         * L2 indices are uploaded fully resident at model upload time
         * (preading from drive_fd at l2_indices_file_offset when the
         * mmap-backed pq->l2_indices pointer is the empty drive scratch).
         * L2 streaming would need a second shared-scratch ring — kept as
         * a follow-up; for now we trade some Metal RAM (L2 indices are
         * ~1/8th to 1/4 of L1 idx bytes depending on l2_idx_bits) for the
         * much larger savings of L1 streaming. */
        if (m->output_head.pq->l2_kind != 0) {
            CHECK(m->output_head.pq->l2_kind == 2 && m->output_head.pq->l2_K > 0
                  && m->output_head.pq->l2_K <= 64,
                  "PQv2 L2 residual on output_head requires l2_kind=2 and 1 ≤ K_L2 ≤ 64");
        }
    } else {
        CHECK(m->output_head.bits == 4 || m->output_head.bits == 8 || m->output_head.bits == 16,
              "output_head must be INT4/INT8/FP16 or PQv2");
    }
    for (int L = 0; L < m->header.num_layers; L++) {
        const ib_layer_meta *lm = &m->layers[L];
        /* Accept INT4/INT8 OR PQv2 (K=256, half=2; l2_kind ∈ {0,2}). */
        #define TENSOR_OK(name) do { \
            const ib_tensor_meta *tt = &lm->name; \
            if (tt->pq) { \
                CHECK(tt->pq->K == 256 && tt->pq->half == 2, \
                      "PQv2 GPU path requires K=256 and half=2: " #name); \
                if (tt->pq->l2_kind != 0) { \
                    CHECK(tt->pq->l2_kind == 2 && tt->pq->l2_K > 0 \
                          && tt->pq->l2_K <= 64, \
                          "PQv2 L2 residual requires l2_kind=2 and 1 ≤ K_L2 ≤ 64: " #name); \
                    /* Goal H3: drive-mode pyramid is supported — see the \
                     * matching block above for output_head. L1 streams; \
                     * L2 codebook + L2 indices are uploaded resident in \
                     * upload_pqv2_tensor_ex (preading from drive_fd at \
                     * pq->l2_indices_file_offset when the mmap pointer is \
                     * the empty drive scratch). */ \
                } \
            } else { \
                CHECK(tt->bits == 4 || tt->bits == 8, \
                      "layer matmul tensor must be INT4/INT8 or PQv2: " #name); \
            } \
        } while (0)
        /* FFN-only variant: accepts an EMPTY slot when the layer carries
         * MoME experts (mome_experts > 1) — the legacy gate/up/down slots
         * are unused on MoME layers and the per-expert tensors live in
         * *_proj_experts arrays. forward.c routes MoME FFN through
         * mome_dispatch_ffn on CPU, so Metal upload simply skips these
         * slots; see also the upload loop further below. */
        #define TENSOR_OK_FFN(name) do { \
            const ib_tensor_meta *tt = &lm->name; \
            if (lm->mome_experts > 1 && !tt->pq && tt->bits == 0 \
                && tt->size == 0) { \
                /* MoME layer with empty legacy FFN slot — OK, skip. */ \
            } else if (tt->pq) { \
                CHECK(tt->pq->K == 256 && tt->pq->half == 2, \
                      "PQv2 GPU path requires K=256 and half=2: " #name); \
                if (tt->pq->l2_kind != 0) { \
                    CHECK(tt->pq->l2_kind == 2 && tt->pq->l2_K > 0 \
                          && tt->pq->l2_K <= 64, \
                          "PQv2 L2 residual requires l2_kind=2 and 1 ≤ K_L2 ≤ 64: " #name); \
                    /* Goal H3: drive-mode pyramid allowed — see TENSOR_OK. */ \
                } \
            } else { \
                CHECK(tt->bits == 4 || tt->bits == 8, \
                      "layer matmul tensor must be INT4/INT8 or PQv2: " #name); \
            } \
        } while (0)
        TENSOR_OK(q_proj); TENSOR_OK(k_proj); TENSOR_OK(v_proj); TENSOR_OK(o_proj);
        TENSOR_OK_FFN(gate_proj); TENSOR_OK_FFN(up_proj); TENSOR_OK_FFN(down_proj);
        #undef TENSOR_OK
        #undef TENSOR_OK_FFN
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

/* Forward decl — defined later in this file. */
static inline uint16_t fp32_to_fp16_bits(float f);

/* Dequantize INT4-blk32 weights to fp16 [M][N], for the MPS-hybrid
 * prefill kernel. Per-32-element block has a single fp16 scale; each
 * nibble decoded as ((byte >> 0/4) & 0x0F) - 8. Returns malloc'd buffer
 * the caller owns (uploaded to MTLBuffer, then freed). */
static uint16_t *dequant_int4_blk32_to_fp16(const uint8_t *w_src,
                                              const uint16_t *ws_fp16_src,
                                              int M, int N)
{
    if (M <= 0 || N <= 0 || (N % 32) != 0) return NULL;
    size_t total = (size_t)M * N;
    uint16_t *dst = (uint16_t *)malloc(total * sizeof(uint16_t));
    if (!dst) return NULL;
    int row_bytes  = N / 2;
    int scales_per_row = N / 32;
    for (int m = 0; m < M; m++) {
        const uint8_t  *row    = w_src + (size_t)m * row_bytes;
        const uint16_t *scales = ws_fp16_src + (size_t)m * scales_per_row;
        for (int n = 0; n < N; n++) {
            uint8_t byte = row[n / 2];
            int w_int = (n & 1) ? ((int)((byte >> 4) & 0x0F) - 8)
                                  : ((int)(byte & 0x0F) - 8);
            uint16_t s_bits = scales[n / 32];
            /* Decode fp16 scale → fp32, multiply, re-encode as fp16. */
            uint32_t sign = (s_bits & 0x8000) << 16;
            uint32_t expo = (s_bits >> 10) & 0x1F;
            uint32_t mant = s_bits & 0x3FF;
            float w_scale;
            if (expo == 0) {
                /* Subnormal: 2^-14 * mant/1024 */
                w_scale = (float)mant / 1024.0f / 16384.0f;
            } else if (expo == 31) {
                w_scale = 0.0f;  /* inf/NaN → treat as 0 */
            } else {
                uint32_t f32_bits = sign | ((expo - 15 + 127) << 23) | (mant << 13);
                memcpy(&w_scale, &f32_bits, 4);
            }
            float val = (float)w_int * w_scale;
            dst[(size_t)m * N + n] = fp32_to_fp16_bits(val);
        }
    }
    return dst;
}

/* Pre-shuffle INT4-blk32 weights from row-major [M][N/2] into tile-major
 * [(M/32)*(N/64)][32*32] bytes (32-row × 64-col tile flattened). Matches
 * the K_TILE=64 pipelined kernel's tile geometry so each TG reads its
 * 1024-byte tile as one contiguous chunk. Caller owns the returned buffer. */
static uint8_t *shuffle_int4_blk32_k64(const uint8_t *src, int M, int N) {
    if (M <= 0 || N <= 0 || (M % 32) || (N % 64)) return NULL;
    size_t total = (size_t)M * (size_t)(N / 2);
    uint8_t *dst = (uint8_t *)malloc(total);
    if (!dst) return NULL;
    int num_k_tiles = N / 64;
    int row_bytes = N / 2;       /* bytes per src row */
    for (int m = 0; m < M; m++) {
        int m_t = m / 32;
        int m_in_t = m % 32;
        const uint8_t *src_row = src + (size_t)m * row_bytes;
        for (int kb = 0; kb < row_bytes; kb++) {
            int k_in_bytes = kb * 2;          /* the col-index of the LOW nibble */
            int k_t = k_in_bytes / 64;
            int kb_in_t = (k_in_bytes % 64) / 2;  /* = kb % 32 */
            size_t tile_idx = (size_t)m_t * num_k_tiles + k_t;
            size_t dst_off = tile_idx * 1024 + (size_t)m_in_t * 32 + kb_in_t;
            dst[dst_off] = src_row[kb];
        }
    }
    return dst;
}

static void upload_w_pair(ib_metal_ctx *ctx, const inferbit_model *m,
                           const ib_tensor_meta *t,
                           void **out_w, void **out_s)
{
    const uint8_t *base = (const uint8_t *)m->weight_data;
    const uint8_t *w_src = base + t->offset;
    /* IB_PREFILL_SHUFFLED=1 — EXPERIMENTAL / KNOWN BROKEN FOR DECODE.
     *
     * The shuffled prefill kernel (variant=3) reads INT4 weights in
     * tile-major [(M/32)*(N/64)][32×32-bytes] layout instead of
     * row-major. Setting this env flag rewrites the upload-time
     * weights in place — but the decode kernels (dr_a32 family) still
     * expect row-major, so DECODE PRODUCES GARBAGE in this mode. The
     * empirical PP perf is also -13% vs the default K=64 pipelined
     * (negative result), so this flag should only be used for kernel
     * micro-benching, never in production. Dual-buffer (keep both
     * layouts) is the path forward if shuffled kernel is ever revised
     * to win. */
    int can_shuffle = (t->bits == 4 &&
                       t->scale_size > (size_t)t->shape[0] * 2 &&  /* blk32 */
                       t->shape[0] > 0 && t->shape[1] > 0 &&
                       (t->shape[0] % 32) == 0 && (t->shape[1] % 64) == 0);
    static int shuffle_setting = -1;
    if (shuffle_setting < 0) {
        const char *env = getenv("IB_PREFILL_SHUFFLED");
        shuffle_setting = (env && env[0] == '1') ? 1 : 0;
        if (shuffle_setting) {
            fprintf(stderr, "ib_metal: WARNING IB_PREFILL_SHUFFLED=1 — decode kernels will misread weights; use for prefill micro-bench only.\n");
        }
    }
    if (shuffle_setting && can_shuffle) {
        uint8_t *sh = shuffle_int4_blk32_k64(w_src, t->shape[0], t->shape[1]);
        if (sh) {
            /* sh is malloc'd + freed below — must use the copying path. */
            *out_w = ib_metal_alloc(ctx, t->size, sh);
            free(sh);
        } else {
            *out_w = ib_metal_alloc_mmap(ctx, t->size, w_src);
        }
    } else {
        *out_w = ib_metal_alloc_mmap(ctx, t->size, w_src);
    }
    if (!*out_w) {
        fprintf(stderr, "ib_metal: upload_w_pair: ib_metal_alloc_mmap returned NULL for weight bytes (size=%zu, bits=%d, shape=[%d,%d]) — Metal weight buffer allocation failed\n",
                t->size, t->bits, t->shape[0], t->shape[1]);
        return;
    }
    /* Note: when the zero-copy path engages, Metal references the mmap'd
     * pages directly. release_mmap_range() requests reclaim via madvise,
     * but the pages are still referenced by the MTLBuffer so the kernel
     * will keep them resident — madvise becomes a no-op rather than a
     * correctness issue. When the copy path engages, the pages are
     * unreferenced after memcpy and madvise behaves as before. */
    release_mmap_range(w_src, t->size);
    if (t->scale_size > 0) {
        const uint8_t *s_src = base + t->scale_offset;
        *out_s = ib_metal_alloc_mmap(ctx, t->scale_size, s_src);
        if (!*out_s) {
            fprintf(stderr, "ib_metal: upload_w_pair: ib_metal_alloc_mmap returned NULL for scale bytes (scale_size=%zu, bits=%d) — Metal scale buffer allocation failed\n",
                    t->scale_size, t->bits);
        }
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
    /* src is an mmap'd pointer (m->weight_data + offset). Zero-copy when
     * the offset happens to land on a page boundary; falls back to copy
     * otherwise. Norms are tiny (<= hidden_dim * 2 bytes), so they will
     * almost always be smaller than one page and take the copy path. */
    *out = ib_metal_alloc_mmap(ctx, t->size, src);
    if (!*out) {
        fprintf(stderr, "ib_metal: upload_norm: ib_metal_alloc_mmap returned NULL (size=%zu) — norm tensor buffer allocation failed\n",
                t->size);
    }
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
    out->pq_l1_idx_layout = (int)pq->l1_idx_layout;

    /* row_scale is already fp16 (stored as uint16_t). Source is the
     * mmap'd PQv2 blob (parsed in pqv2_format.c::parse_pqv2_blob). Try
     * zero-copy; falls back to copy when the in-blob offset (typically
     * blob_start + 36 bytes) is not page-aligned, which is the common
     * case — but the fallback is the same memcpy we did before, so no
     * regression. */
    out->pq_rs = ib_metal_alloc_mmap(ctx, (size_t)pq->M * sizeof(uint16_t), pq->row_scale);
    if (!out->pq_rs) {
        fprintf(stderr, "ib_metal: upload_pqv2_tensor_ex: row_scale alloc returned NULL (M=%u, %zu bytes) — PQv2 tensor unusable on GPU\n",
                (unsigned)pq->M, (size_t)pq->M * sizeof(uint16_t));
        return;
    }

    /* Decode codebooks: cb_q (int8) * cb_scale (fp16, per-K) → fp16 cb[ns][K][half]. */
    size_t cb_elts  = (size_t)pq->n_subchunks * pq->K * pq->half;
    size_t cb_bytes = cb_elts * sizeof(uint16_t);
    uint16_t *cb_fp16 = (uint16_t *)malloc(cb_bytes);
    if (!cb_fp16) { fprintf(stderr, "upload_pqv2_tensor: oom\n"); return; }
    for (uint32_t s = 0; s < pq->n_subchunks; s++) {
        for (uint32_t k = 0; k < pq->K; k++) {
            /* cb_scale stored as raw uint16 fp16. Previously decoded via
             * a (exp + 0x70)<<23 shift that only handled fp16 NORMALS
             * correctly — fp16 denormals (exp == 0, mant != 0) decoded
             * 7-12× too large. L1 cb_scales stay normal in current
             * encoders, so this didn't bite L1 in practice, but the same
             * decode is shared with the L2 path below where denormal
             * scales are routine. Use the canonical pqv2_h2f helper
             * (matches pqv2_kernel.c::pqv2_h2f) so both paths handle
             * denormals correctly. */
            uint16_t scl_bits = pq->cb_scale[s * pq->K + k];
            float scl = pqv2_h2f(scl_bits);
            for (uint32_t h = 0; h < pq->half; h++) {
                int8_t q = pq->cb_q[(s * pq->K + k) * pq->half + h];
                float v = (float)q * scl;
                cb_fp16[(s * pq->K + k) * pq->half + h] = fp32_to_fp16_bits(v);
            }
        }
    }
    out->pq_cb = ib_metal_alloc(ctx, cb_bytes, cb_fp16);
    free(cb_fp16);
    if (!out->pq_cb) {
        fprintf(stderr, "ib_metal: upload_pqv2_tensor_ex: codebook alloc returned NULL (ns=%u, K=%u, half=%u, %zu bytes) — PQv2 tensor unusable on GPU\n",
                (unsigned)pq->n_subchunks, (unsigned)pq->K, (unsigned)pq->half, cb_bytes);
        return;
    }

    /* Indices on disk are either [n_chunks][n_subchunks][M] u8 (legacy
     * chunk-major; `pq->l1_idx_layout == 0`) or [M][n_chunks][n_subchunks]
     * (row-major, opt-in `IB_PQV2_L1_ROWMAJOR=1` at encode time;
     * `pq->l1_idx_layout == 1`).
     *
     * The GPU SIMD kernel reads `indices[m * total + i]` (row-major)
     * where total = n_chunks * n_subchunks, so for `layout == 1` the
     * mmap'd file region already matches the kernel layout and we
     * zero-copy via newBufferWithBytesNoCopy. For `layout == 0` we
     * transpose at upload time into a malloc'd staging buffer and
     * Metal allocates a fresh MTLBuffer (legacy 2× file-size path). */
    uint32_t total = (uint32_t)((pq->N / pq->G) * pq->n_subchunks);
    size_t idx_bytes = (size_t)pq->M * total;

    /* GPU drive mode: don't allocate a per-tensor MTLBuffer for the
     * indices. Record the file offset; the model finalizer will point
     * pq_idx at a shared scratch MTLBuffer that the forward path
     * refills per matmul. Keeps GPU weight RAM bounded regardless of
     * model size. token_embedding overrides via keep_indices_resident=1
     * because embed_lookup needs random-access reads.
     *
     * Goal I1: when the tensor is a pyramid (l2_kind == 2), we must
     * NOT early-return here — the L2 codebook + L2 indices upload at
     * the bottom of this function is required for the dispatcher
     * (rec_matmul_tb) to route to the l2residual kernel. Without it
     * pq_K_l2 stays 0 and the flat decoder runs on pyramid data
     * (Round 8 H3 symptom: PPL 6.26 instead of 5.886). Set the L1
     * drive metadata, then fall through to the L2 upload block. */
    int skip_l1_upload = 0;
    if (m && m->residency_mode == 1 && pq->indices_file_offset != 0
        && !keep_indices_resident) {
        out->pq_idx = NULL;
        /* Prefer the pre-transposed sidecar offset if available (doc 35
         * feature 3 — skips per-matmul transpose). Falls back to the
         * legacy in-file chunk-major offset if sidecar build failed.
         * Selector is the SIDECAR FD presence (the first tensor's
         * sidecar offset is 0, so we can't use offset != 0 as the
         * selector). */
        out->pq_drive_file_offset = (m->drive_fd_pretransposed >= 0)
            ? pq->indices_pretransposed_offset
            : pq->indices_file_offset;
        skip_l1_upload = 1;
    } else {
        out->pq_drive_file_offset = 0;
    }

    /* Goal I1: when skip_l1_upload is set (GPU drive mode), the L1
     * MTLBuffer is shared scratch — bypass the per-tensor L1 upload
     * but still fall through to the L2 pyramid upload below. */
  if (!skip_l1_upload) {
    /* Stage 5g.2 — when the encoder wrote L1 indices in row-major
     * ([M][n_chunks][n_subchunks]) on disk, the layout already matches
     * exactly what the GPU SIMD kernel reads. Skip the transpose +
     * malloc + Metal copy and zero-copy the mmap'd region directly
     * into a MTLBuffer via newBufferWithBytesNoCopy. This is the whole
     * point of IB_PQV2_L1_ROWMAJOR=1 — drops peak Metal RAM from
     * ~2× file size to ~1× file size for PQv2-encoded weights.
     *
     * Only applies when we're not in GPU drive mode (handled above)
     * and not in CPU drive mode (the drive_fd path below). In CPU
     * drive mode pq->indices points at a shared scratch buffer, not
     * the mmap'd disk region, so we can't zero-copy from it. */
    int can_zero_copy_rm = (pq->l1_idx_layout == 1) && pq->indices
        && !(m && m->residency_mode == 1 && pq->indices_file_offset != 0
              && m->drive_fd >= 0);
    if (can_zero_copy_rm) {
        out->pq_idx = ib_metal_alloc_mmap(ctx, idx_bytes,
                                            (const void *)pq->indices);
        if (out->pq_idx) {
            /* Successful zero-copy or copy-fallback inside the helper.
             * Either way no transpose was needed. */
            /* Continue into the L2 path below. */
        } else {
            fprintf(stderr, "upload_pqv2_tensor: alloc_mmap returned NULL for L1 row-major indices\n");
            return;
        }
    } else {
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
            src = (const uint8_t *)pq->indices; /* [nc][ns][M] in legacy chunk-major */
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
  } /* end if (!skip_l1_upload) — Goal I1 */

    /* PQv2 pyramid (Stage 5a, docs/v2/00_CORRECTION.md): when the
     * source tensor carries a second-level codebook (l2_kind == 2,
     * K_L2 ≤ 64), upload it as a parallel set of GPU buffers so the
     * matmul_pqv2_k256_half2_l2residual kernel can pick them up.
     *
     * Layouts match the L1 path:
     *   - cb_l2: int8 + fp16 scales → pre-decoded fp16 [ns][K_L2][half]
     *   - idx_l2: on-disk [n_chunks][n_subchunks][M] u8, transposed
     *     to [M][n_chunks * n_subchunks] u8 for coalesced GPU reads.
     *     (l2_idx_bits == 6: the on-disk packed [total][ceil(M/4)*3]
     *     layout is uploaded as-is.)
     *
     * Goal H3 (partial drive support): pyramid + GPU drive is now
     * permitted. L1 indices continue to stream through the shared
     * scratch ring (handled above); the L2 codebook + L2 indices are
     * uploaded fully resident here. In drive mode the source buffer
     * pq->l2_indices is the (empty) drive scratch slot — we pread the
     * real bytes from the IBF at pq->l2_indices_file_offset into a
     * temporary host buffer first, then hand them to Metal. Full L2
     * streaming (a second shared-scratch ring) is a follow-up; for now
     * we keep L2 indices fully resident on Metal because they are
     * ~1/4 to 1× the L1 idx bytes (depends on l2_idx_bits) — small
     * compared to the L1 footprint we just kept off the GPU. */
    if (pq->l2_kind == 2 && pq->l2_K > 0 && pq->l2_K <= 64
        && pq->l2_cb_q && pq->l2_cb_scale && pq->l2_indices) {
        size_t cb2_elts  = (size_t)pq->n_subchunks * pq->l2_K * pq->half;
        size_t cb2_bytes = cb2_elts * sizeof(uint16_t);
        uint16_t *cb2_fp16 = (uint16_t *)malloc(cb2_bytes);
        if (!cb2_fp16) {
            fprintf(stderr, "ib_metal: upload_pqv2_tensor_ex: OOM on PQv2 pyramid L2 codebook staging (%zu bytes) — pyramid tensor will decode without residual (results will be wrong)\n",
                    cb2_bytes);
        }
        if (cb2_fp16) {
            for (uint32_t s = 0; s < pq->n_subchunks; s++) {
                for (uint32_t k = 0; k < pq->l2_K; k++) {
                    /* Decode the per-codeword fp16 scale to fp32. The
                     * Naive (exp+0x70)<<23 formula above is correct only
                     * for fp16 NORMAL values (exp ∈ [1, 30]). L2 residual
                     * scales are routinely DENORMAL (exp == 0, mant != 0,
                     * value = mant/1024 * 2^-14), and the naive formula
                     * decodes those as `(1 + mant/1024) * 2^-15`, which
                     * is ~7-12× too large. That inflated scale made every
                     * L2 codeword 7-12× larger than CPU, turning the L2
                     * "small residual" into a destructive contribution
                     * (matches pyramid Metal RAM PPL 9.47 vs CPU 5.91).
                     * Use the same explicit decode that
                     * pqv2_kernel.c::pqv2_h2f does. The L1 cb_scale path
                     * isn't hit by this bug in practice because L1 scales
                     * stay in the normal range, but feed it through the
                     * same helper so future small-scale L1 codebooks
                     * (e.g. Stage 5k INT8-row-scale variants) decode
                     * correctly too. */
                    uint16_t scl_bits = pq->l2_cb_scale[s * pq->l2_K + k];
                    float scl = pqv2_h2f(scl_bits);
                    for (uint32_t h = 0; h < pq->half; h++) {
                        int8_t q = pq->l2_cb_q[(s * pq->l2_K + k) * pq->half + h];
                        float v = (float)q * scl;
                        cb2_fp16[(s * pq->l2_K + k) * pq->half + h] =
                            fp32_to_fp16_bits(v);
                    }
                }
            }
            out->pq_cb_l2 = ib_metal_alloc(ctx, cb2_bytes, cb2_fp16);
            free(cb2_fp16);
        }
        /* L2 indices: layout depends on pq->l2_idx_bits.
         *
         *   l2_idx_bits == 8 (legacy): on-disk is [nc][ns][M] u8 ⇒
         *     transpose to [M][total] u8 so the GPU kernel reads M-major
         *     coalesced (one byte per lane).
         *   l2_idx_bits == 6 (Stage 5h.1 packed): on-disk is
         *     [nc][ns][ceil(M/4)*3] bytes — slot-major, M-axis packed.
         *     Keep this layout as-is on the GPU; the kernel reads 3 bytes
         *     for a (slot, m/4) group and extracts the m%4-th 6-bit lane.
         *     The 4-row sharing means consecutive rows in a warp hit the
         *     same 3-byte triple — coalesced enough for the residual
         *     pass, which is bandwidth-secondary to L1 anyway.
         *
         * PEAK-RAM fix: when the model is in GPU drive mode, do NOT make
         * the L2 indices resident. Like L1, they are the big resident
         * chunk; stream them per matmul through gpu_drive_l2_idx_scratch.
         * Here we just record the file offset + byte counts and leave
         * pq_idx_l2 NULL — the model finalizer allocates the shared L2
         * scratch ring and drive_prepare_pq preads into it. The L2
         * CODEBOOK above stays resident (few KB/tensor). */
        uint32_t l2_bits = (pq->l2_idx_bits == 6) ? 6u
                         : (pq->l2_idx_bits == 4) ? 4u : 8u;
        out->pq_l2_idx_bits = (int)l2_bits;
        uint8_t *idx_l2_t = NULL;
        size_t idx_l2_bytes;
        uint32_t nc_l2 = pq->N / pq->G;
        size_t l2_disk_bytes;
        if (l2_bits == 6) {
            size_t packed_row = ((size_t)pq->M + 3u) / 4u * 3u;
            idx_l2_bytes = (size_t)nc_l2 * pq->n_subchunks * packed_row;
            l2_disk_bytes = idx_l2_bytes;  /* same packed layout on disk */
        } else if (l2_bits == 4) {
            /* Goal N36 — 4-bit packed (2 indices per byte, low nibble
             * first; l2_K ≤ 16). Per (chunk,subchunk) slot row is
             * ceil(M/2) bytes. Mirror pqv2_l2_packed_row_bytes_4bit.
             * Upload the on-disk slot-major [nc][ns][ceil(M/2)] layout
             * as-is; the kernel unpacks the m%2-th nibble per lane. */
            size_t packed_row = ((size_t)pq->M + 1u) / 2u;
            idx_l2_bytes = (size_t)nc_l2 * pq->n_subchunks * packed_row;
            l2_disk_bytes = idx_l2_bytes;  /* same packed layout on disk */
        } else {
            idx_l2_bytes = idx_bytes;                                   /* [M][total] u8 */
            l2_disk_bytes = (size_t)pq->M * nc_l2 * pq->n_subchunks;     /* [nc][ns][M] u8 */
        }
        int l2_drive = (m && m->residency_mode == 1
                        && pq->l2_indices_file_offset != 0
                        && m->drive_fd >= 0);
        if (l2_drive) {
            /* Stream the L2 indices: record the redirect; pq_idx_l2 stays
             * NULL (filled per-matmul). Codebook already resident above. */
            out->pq_l2_drive_file_offset   = pq->l2_indices_file_offset;
            out->pq_l2_drive_disk_bytes     = l2_disk_bytes;
            out->pq_l2_drive_scratch_bytes  = idx_l2_bytes;
            out->pq_idx_l2 = NULL;
        } else {
        /* RAM mode: the mmap'd l2_indices pointer is live; upload it
         * resident (transposing for the 8-bit layout). */
        const uint8_t *l2_src_bytes = (const uint8_t *)pq->l2_indices;
        if (!l2_src_bytes) {
            /* No source available. Skip upload; pq_K_l2 stays 0 and the
             * warning below fires. */
        } else if (l2_bits == 6 || l2_bits == 4) {
            /* Upload directly from the mmap'd packed buffer — same layout
             * the GPU kernel expects (6-bit: 4-in-3 bytes; 4-bit: 2-in-1
             * byte). The kernel unpacks per lane. */
            out->pq_idx_l2 =
                ib_metal_alloc(ctx, idx_l2_bytes, l2_src_bytes);
        } else {
            idx_l2_t = (uint8_t *)malloc(idx_l2_bytes);
            if (idx_l2_t) {
                for (uint32_t m_ = 0; m_ < pq->M; m_++) {
                    for (uint32_t c = 0; c < nc_l2; c++) {
                        for (uint32_t s = 0; s < pq->n_subchunks; s++) {
                            idx_l2_t[(size_t)m_ * total
                                     + c * pq->n_subchunks + s] =
                                l2_src_bytes[((size_t)c * pq->n_subchunks + s)
                                              * pq->M + m_];
                        }
                    }
                }
                out->pq_idx_l2 = ib_metal_alloc(ctx, idx_l2_bytes, idx_l2_t);
                free(idx_l2_t);
            }
        }
        } /* end RAM-mode L2 upload */
        /* Only flip pq_K_l2 ON when BOTH uploads succeeded — the
         * dispatcher (rec_matmul_tb) checks pq_K_l2 > 0 to route to the
         * L2residual kernel. If either upload failed we leave pq_K_l2 = 0
         * so the flat decoder runs (which will be silently wrong on
         * pyramid data, but model_is_supported has already gated this
         * tensor through, so failing closed here would crash the model).
         * Emit a warning so the failure is at least visible. */
        /* In drive mode pq_idx_l2 is intentionally NULL (streamed); the
         * presence of pq_l2_drive_file_offset means the L2 ring will fill
         * it per-matmul. Treat that as success too. */
        int l2_ready = out->pq_idx_l2 || out->pq_l2_drive_file_offset != 0;
        if (out->pq_cb_l2 && l2_ready) {
            out->pq_K_l2 = (int)pq->l2_K;
        } else {
            fprintf(stderr,
                "ib_metal: upload_pqv2_tensor_ex: L2 buffer alloc failed; "
                "pyramid tensor will decode without residual (results will be wrong)\n");
        }
    }
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
    out->w_fp16 = NULL;
    /* MPS-hybrid prefill (IB_PREFILL_FP16_W=1): pre-dequantize INT4-blk32
     * weights to fp16 and upload as a parallel buffer. Doubles GPU RAM
     * for the weight tensor — only viable on small models. Decode still
     * uses the original INT4 buffer; only the fp16-weight prefill kernel
     * reads out->w_fp16. */
    static int fp16w_setting = -1;
    if (fp16w_setting < 0) {
        const char *env = getenv("IB_PREFILL_FP16_W");
        fp16w_setting = (env && env[0] == '1') ? 1 : 0;
    }
    if (fp16w_setting && out->bits == 4 && out->blk32
        && t->shape[0] > 0 && t->shape[1] > 0
        && (t->shape[1] % 32) == 0) {
        const uint8_t *base = (const uint8_t *)m->weight_data;
        const uint8_t *w_src = base + t->offset;
        const uint16_t *ws_src = (const uint16_t *)(base + t->scale_offset);
        uint16_t *fp16w = dequant_int4_blk32_to_fp16(w_src, ws_src,
                                                       t->shape[0], t->shape[1]);
        if (fp16w) {
            size_t bytes = (size_t)t->shape[0] * t->shape[1] * sizeof(uint16_t);
            out->w_fp16 = ib_metal_alloc(ctx, bytes, fp16w);
            free(fp16w);
        }
    }
}

extern "C" ib_metal_model_buffers *
ib_metal_upload_model(ib_metal_ctx *ctx, const void *model_handle)
{
    if (!ctx || !model_handle) {
        fprintf(stderr, "ib_metal: ib_metal_upload_model: NULL %s — cannot upload model to Metal\n",
                !ctx ? "ib_metal_ctx" : "model_handle");
        return nullptr;
    }
    const inferbit_model *m = (const inferbit_model *)model_handle;

    char err[160];
    if (!model_is_supported(m, err, sizeof(err))) {
        fprintf(stderr, "ib_metal_upload_model: %s\n", err);
        return nullptr;
    }

    ib_metal_model_buffers *b = (ib_metal_model_buffers *)
        calloc(1, sizeof(*b));
    if (!b) {
        fprintf(stderr, "ib_metal: ib_metal_upload_model: calloc(ib_metal_model_buffers) failed — out of host memory before any tensor uploaded\n");
        return nullptr;
    }

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
    /* Rotating KV window (doc 36 phase 2.2). m->kv_window is normalized
     * by ib_alloc_kv_caches: 0 = full causal, else == the physical ring
     * size (which is what kv_caches[0].capacity / b->seq_len already
     * reflect). When windowed, start_pos may exceed the physical ring,
     * so bounds checks use max_logical_pos (the model's true context),
     * while KV addressing uses pos % seq_len. */
    b->kv_window    = m->kv_window;
    b->max_logical_pos = (b->kv_window > 0)
        ? m->header.max_context_length
        : b->seq_len;
    b->kv_bits      = m->header.kv_bits;
    b->rope_theta   = m->header.rope_theta;
    b->eps          = m->header.norm_epsilon;
    b->model        = m;

    /* Per-layer weights + KV caches. */
    b->layers = (struct layer_bufs *)
        calloc((size_t)b->num_layers, sizeof(struct layer_bufs));
    if (!b->layers) {
        fprintf(stderr, "ib_metal: ib_metal_upload_model: calloc for %d layer_bufs failed — out of host memory before any layer uploaded\n",
                b->num_layers);
        free(b); return nullptr;
    }

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
        /* MoME layers (mome_experts > 1) keep their FFN trio empty on
         * Metal — the legacy gate/up/down slots aren't populated by the
         * loader, and the per-expert tensors live in lm->*_proj_experts.
         * forward.c routes MoME FFN through mome_dispatch_ffn (CPU); the
         * Metal recorder never reads lb->gate/up/down for such layers,
         * so leaving them zeroed (from the parent calloc) is safe. */
        if (lm->mome_experts > 1) {
            fprintf(stderr, "ib_metal: layer %d uses MoME (experts=%d); FFN routes to CPU\n",
                    L, lm->mome_experts);
        } else {
            upload_tensor(ctx, m, &lm->gate_proj, &lb->gate);
            upload_tensor(ctx, m, &lm->up_proj,   &lb->up);
            upload_tensor(ctx, m, &lm->down_proj, &lb->down);
        }
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
            /* Doc-35 feature 4: bump ring depth from 2 → 4 to keep more
             * matmuls in flight before each commit-wait. Each slot adds
             * scratch_sz GPU RAM; on 8B that's 263 MB × 4 = ~1 GB scratch
             * (vs ~526 MB at N=2). Trade RAM for fewer GPU syncs. Env
             * IB_DRIVE_RING_N can override (1..4). */
            int n_slots = 4;
            {
                const char *env = getenv("IB_DRIVE_RING_N");
                if (env) {
                    int v = atoi(env);
                    if (v >= 1 && v <= 4) n_slots = v;
                }
            }
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
            /* 2-sub-ring iff n_slots is even and >=2 (i.e. 2 or 4).
             * sr_size = n_slots/2. Otherwise (n_slots==1 or 3): single
             * sub-ring (sync checkpoint fallback). */
            if (alloc_ok && n_slots >= 2 && (n_slots % 2) == 0) {
                b->gpu_drive_n_subrings = 2;
                b->gpu_drive_sr_size = n_slots / 2;
            } else {
                b->gpu_drive_n_subrings = 1;
                b->gpu_drive_sr_size = n_slots;
            }
            b->gpu_drive_cur_sr = 0;
            b->gpu_drive_pending_cb = NULL;
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
                fprintf(stderr, "ib_metal: GPU drive mode ON. %d slots (%d sub-rings × %d), %zu B/slot, %d tensors streamed\n",
                        n_slots, b->gpu_drive_n_subrings, b->gpu_drive_sr_size, scratch_sz, n_repointed);
                #undef REPOINT

                /* PEAK-RAM fix: parallel L2-index scratch ring for pyramid
                 * tensors. Allocate ONLY if any tensor streams its L2
                 * indices (pq_l2_drive_file_offset != 0). Uses the SAME
                 * n_slots so the L2 ring advances in lockstep with the L1
                 * ring (drive_prepare_pq fills slot i of both). Sized to
                 * the largest tensor's kernel-native L2 idx bytes. */
                size_t max_l2 = 0;
                #define CONSIDER_L2(tb) do { \
                    if ((tb).is_pq && (tb).pq_l2_drive_file_offset != 0) { \
                        size_t s2 = (tb).pq_l2_drive_scratch_bytes; \
                        if (s2 > max_l2) max_l2 = s2; \
                    } \
                } while (0)
                for (int L = 0; L < b->num_layers; L++) {
                    struct layer_bufs *lb = &b->layers[L];
                    CONSIDER_L2(lb->q); CONSIDER_L2(lb->k); CONSIDER_L2(lb->v); CONSIDER_L2(lb->o);
                    CONSIDER_L2(lb->gate); CONSIDER_L2(lb->up); CONSIDER_L2(lb->down);
                }
                CONSIDER_L2(b->output_head);
                #undef CONSIDER_L2
                if (max_l2 > 0) {
                    size_t l2_scratch_sz = (max_l2 + 16383u) & ~((size_t)16383u);
                    int l2_ok = 1;
                    for (int i = 0; i < n_slots; i++) {
                        b->gpu_drive_l2_idx_scratch[i] = ib_metal_alloc(ctx, l2_scratch_sz, NULL);
                        b->gpu_drive_l2_idx_staging[i] = malloc(l2_scratch_sz);
                        if (!b->gpu_drive_l2_idx_scratch[i] || !b->gpu_drive_l2_idx_staging[i]) {
                            l2_ok = 0;
                        }
                    }
                    b->gpu_drive_l2_idx_scratch_size = l2_scratch_sz;
                    b->gpu_drive_l2_idx_n_slots = l2_ok ? n_slots : 0;
                    if (!l2_ok) {
                        fprintf(stderr, "ib_metal: GPU drive L2 scratch alloc failed (%zu B × %d)\n",
                                l2_scratch_sz, n_slots);
                    } else {
                        fprintf(stderr, "ib_metal: GPU drive L2-index streaming ON. %zu B/slot × %d slots\n",
                                l2_scratch_sz, n_slots);
                    }
                }
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
    /* Scores stay LOGICAL-indexed (position 0..max_logical_pos), even
     * with a rotating KV window — the kernels mask out-of-window
     * positions to -INF. Only the KV cache itself is bounded to the
     * physical ring (b->seq_len). The scores buffer is a single transient
     * scratch, so O(context) here is acceptable; O(context) per-layer KV
     * is what windowing eliminates. */
    b->scores   = ib_metal_alloc(ctx, (size_t)b->n_heads * b->max_logical_pos * sizeof(float), NULL);
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
            bm * (size_t)b->n_heads * (size_t)b->max_logical_pos * sizeof(float), NULL);
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
            FR((tb).w); FR((tb).s); FR((tb).w_fp16); \
            FR((tb).pq_rs); FR((tb).pq_cb); \
            FR((tb).pq_cb_l2); FR((tb).pq_idx_l2); \
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
    FR(b->output_head.w); FR(b->output_head.s); FR(b->output_head.w_fp16);
    FR(b->output_head.pq_rs); FR(b->output_head.pq_cb);
    FR(b->output_head.pq_cb_l2); FR(b->output_head.pq_idx_l2);
    FR_IDX(b->output_head);
    #undef FR_IDX
    FR(b->token_embedding_pq_rs);
    FR(b->token_embedding_pq_cb);
    FR(b->token_embedding_pq_idx);
    /* Free shared GPU drive mode 2-slot ring exactly once. */
    for (int i = 0; i < 4; i++) {
        FR(b->gpu_drive_idx_scratch[i]);
        if (b->gpu_drive_idx_staging[i]) free(b->gpu_drive_idx_staging[i]);
        /* PEAK-RAM fix: parallel L2-index ring (pyramid drive mode). */
        FR(b->gpu_drive_l2_idx_scratch[i]);
        if (b->gpu_drive_l2_idx_staging[i]) free(b->gpu_drive_l2_idx_staging[i]);
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
    /* Doc-35 feature 3: prefer the pre-transposed sidecar (fd in
     * m->drive_fd_pretransposed) — those bytes are already in
     * kernel-native [M][total] layout, so we pread directly to the
     * MTLBuffer scratch with zero transpose work on the critical path.
     * Lifts the drive-mode CPU floor from ~2.3 tok/s.
     *
     * Stage 5g.2: the in-file region itself is kernel-native when
     * `pq_l1_idx_layout == 1` (encoder opt-in IB_PQV2_L1_ROWMAJOR=1).
     * In that case we can skip both the sidecar and the legacy
     * transpose — pread directly from the main IBF into the MTLBuffer
     * scratch.
     *
     * Falls back to the legacy in-file [c][s][m] path (with per-matmul
     * transpose) only when neither the sidecar nor the on-disk
     * row-major layout is available. */
    int use_sidecar = (m->drive_fd_pretransposed >= 0);
    int in_file_rowmajor = (tb->pq_l1_idx_layout == 1);
    int direct_pread = use_sidecar || in_file_rowmajor;
    int fd = use_sidecar ? m->drive_fd_pretransposed : m->drive_fd;
    if (fd < 0) return -1;
    off_t off = (off_t)tb->pq_drive_file_offset;

    if (direct_pread) {
        /* Direct-to-scratch pread; no transpose. Source is either the
         * sidecar (pre-transposed copy) or the main IBF with on-disk
         * row-major layout (Stage 5g.2). */
        uint8_t *dst = (uint8_t *)b->gpu_drive_idx_scratch[slot];
        size_t done = 0;
        while (done < idx_bytes) {
            ssize_t r = pread(fd, dst + done, idx_bytes - done,
                              off + (off_t)done);
            if (r <= 0) {
                if (r == -1 && errno == EINTR) continue;
                fprintf(stderr, "drive_load_pq_idx_to_slot[%s]: pread failed (off=%lld, want=%zu)\n",
                        use_sidecar ? "sidecar" : "in-file-rowmajor",
                        (long long)off, idx_bytes - done);
                return -1;
            }
            done += (size_t)r;
        }
        return 0;
    }

    /* Legacy fallback: pread into staging, transpose to scratch. */
    uint8_t *staging = (uint8_t *)b->gpu_drive_idx_staging[slot];
    size_t done = 0;
    while (done < idx_bytes) {
        ssize_t r = pread(fd, staging + done, idx_bytes - done,
                          off + (off_t)done);
        if (r <= 0) {
            if (r == -1 && errno == EINTR) continue;
            fprintf(stderr, "drive_load_pq_idx_to_slot[legacy]: pread failed (off=%lld, want=%zu)\n",
                    (long long)off, idx_bytes - done);
            return -1;
        }
        done += (size_t)r;
    }
    /* Transpose [nc][ns][M] → [M][total = nc*ns]. */
    uint8_t *dst = (uint8_t *)b->gpu_drive_idx_scratch[slot];
    for (uint32_t m_ = 0; m_ < M; m_++) {
        uint8_t *row = dst + (size_t)m_ * total;
        for (uint32_t c = 0; c < nc; c++) {
            for (uint32_t s = 0; s < ns; s++) {
                row[c * ns + s] = staging[((size_t)c * ns + s) * M + m_];
            }
        }
    }
    return 0;
}

/* PEAK-RAM fix: pyramid L2-index analog of drive_load_pq_idx_to_slot.
 * preads this tensor's L2 indices from the IBF at
 * pq_l2_drive_file_offset into the L2 ring's slot, transposing
 * [nc][ns][M] → [M][total] for the 8-bit layout (matching the resident
 * upload), or preading the packed 6-bit layout directly. Returns 0 on
 * success, -1 on error. Slot index is the SAME as the L1 slot so the two
 * rings stay in lockstep. */
static int drive_load_pq_l2_idx_to_slot(ib_metal_model_buffers *b,
                                         const struct tensor_bufs *tb,
                                         int slot)
{
    if (!tb || !tb->is_pq || tb->pq_l2_drive_file_offset == 0) return 0;
    if (!b || !b->model) return -1;
    if (slot < 0 || slot >= b->gpu_drive_l2_idx_n_slots) return -1;
    if (!b->gpu_drive_l2_idx_scratch[slot] || !b->gpu_drive_l2_idx_staging[slot]) return -1;
    const inferbit_model *m = (const inferbit_model *)b->model;
    int fd = m->drive_fd;
    if (fd < 0) return -1;

    size_t disk_bytes    = tb->pq_l2_drive_disk_bytes;
    size_t scratch_bytes = tb->pq_l2_drive_scratch_bytes;
    if (scratch_bytes > b->gpu_drive_l2_idx_scratch_size) {
        fprintf(stderr, "drive_load_pq_l2_idx_to_slot: %zu B > slot %zu B\n",
                scratch_bytes, b->gpu_drive_l2_idx_scratch_size);
        return -1;
    }
    off_t off = (off_t)tb->pq_l2_drive_file_offset;

    if (tb->pq_l2_idx_bits == 6 || tb->pq_l2_idx_bits == 4) {
        /* Packed layout (6-bit 4-in-3, or 4-bit 2-in-1): on-disk ==
         * kernel-native; pread to scratch unchanged. */
        uint8_t *dst = (uint8_t *)b->gpu_drive_l2_idx_scratch[slot];
        size_t done = 0;
        while (done < disk_bytes) {
            ssize_t r = pread(fd, dst + done, disk_bytes - done, off + (off_t)done);
            if (r <= 0) {
                if (r == -1 && errno == EINTR) continue;
                fprintf(stderr, "drive_load_pq_l2_idx_to_slot[6bit]: pread failed (off=%lld, want=%zu)\n",
                        (long long)off, disk_bytes - done);
                return -1;
            }
            done += (size_t)r;
        }
        return 0;
    }

    /* 8-bit layout: pread [nc][ns][M] into staging, transpose to scratch
     * [M][total]. total = nc*ns. */
    uint32_t M     = (uint32_t)tb->pq_M;
    uint32_t nc    = (uint32_t)(tb->pq_N / tb->pq_G);
    uint32_t ns    = (uint32_t)tb->pq_ns;
    uint32_t total = nc * ns;
    uint8_t *staging = (uint8_t *)b->gpu_drive_l2_idx_staging[slot];
    size_t done = 0;
    while (done < disk_bytes) {
        ssize_t r = pread(fd, staging + done, disk_bytes - done, off + (off_t)done);
        if (r <= 0) {
            if (r == -1 && errno == EINTR) continue;
            fprintf(stderr, "drive_load_pq_l2_idx_to_slot[8bit]: pread failed (off=%lld, want=%zu)\n",
                    (long long)off, disk_bytes - done);
            return -1;
        }
        done += (size_t)r;
    }
    uint8_t *dst = (uint8_t *)b->gpu_drive_l2_idx_scratch[slot];
    for (uint32_t m_ = 0; m_ < M; m_++) {
        uint8_t *row = dst + (size_t)m_ * total;
        for (uint32_t c = 0; c < nc; c++) {
            for (uint32_t s = 0; s < ns; s++) {
                row[c * ns + s] = staging[((size_t)c * ns + s) * M + m_];
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
                                const struct tensor_bufs *tb,
                                void **out_l2_idx)
{
    if (out_l2_idx) *out_l2_idx = tb ? tb->pq_idx_l2 : NULL;
    if (!tb || !tb->is_pq || tb->pq_drive_file_offset == 0) {
        return tb ? tb->pq_idx : NULL;
    }
    if (!b || b->gpu_drive_idx_n_slots < 1) return NULL;
    int n_sr  = b->gpu_drive_n_subrings;
    int sr_sz = b->gpu_drive_sr_size;
    /* Doc-35 feature 4 (proper): 2-sub-ring async commit. When the
     * current sub-ring is full, commit its CB asynchronously, swap to
     * the other sub-ring, and wait on its prior CB before reusing
     * those physical slots. Pipeline depth = 2 in-flight CBs; CPU
     * pread+transpose for the new sub-ring overlaps with GPU compute
     * on the just-committed CB.
     *
     * Single sub-ring (n_sr == 1) is the legacy sync-checkpoint path,
     * used when n_slots is odd or 1 (drive-RAM-min scenarios). */
    if (b->gpu_drive_idx_in_flight >= sr_sz) {
        if (n_sr > 1) {
            void *just_committed = ib_metal_recorder_commit_async(r);
            if (!just_committed) return NULL;
            /* Wait on the OTHER sub-ring's prior CB before reusing it. */
            if (b->gpu_drive_pending_cb) {
                int rc = ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
                b->gpu_drive_pending_cb = NULL;
                if (rc != 0) {
                    /* Drop the just-committed too so we don't leak it. */
                    (void)ib_metal_recorder_wait_committed(just_committed);
                    return NULL;
                }
            }
            b->gpu_drive_pending_cb = just_committed;
            b->gpu_drive_cur_sr = (b->gpu_drive_cur_sr + 1) % n_sr;
        } else {
            /* n_sr == 1: sync fallback. */
            if (ib_metal_recorder_checkpoint(r) != 0) return NULL;
        }
        b->gpu_drive_idx_in_flight = 0;
    }
    int slot = b->gpu_drive_cur_sr * sr_sz + b->gpu_drive_idx_in_flight;
    if (drive_load_pq_idx_to_slot(b, tb, slot) != 0) return NULL;
    /* PEAK-RAM fix: pyramid tensors also stream L2 indices into the
     * parallel L2 ring's SAME slot, in lockstep with L1. The ring
     * discipline (commit/checkpoint above) covers both buffers since the
     * GPU reads L1 and L2 of the same matmul together. */
    if (tb->pq_l2_drive_file_offset != 0) {
        if (drive_load_pq_l2_idx_to_slot(b, tb, slot) != 0) return NULL;
        if (out_l2_idx) *out_l2_idx = b->gpu_drive_l2_idx_scratch[slot];
    }
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
        void *idx_l2 = NULL;
        void *idx = drive_prepare_pq(r, b, tb, &idx_l2);
        if (!idx) return -1;
        /* Pyramid (l2_kind=2) path: route to the dedicated kernel that
         * does L1+L2 in one dispatch. Skips the optional simdmat-decode
         * branch entirely (that one only knows about flat PQv2).
         * idx_l2 is either the resident pq_idx_l2 (RAM mode) or the
         * just-filled L2 ring slot (drive mode). */
        if (tb->pq_K_l2 > 0 && tb->pq_cb_l2 && idx_l2) {
            return ib_metal_rec_matmul_pqv2_k256_half2_l2residual(r,
                tb->pq_rs, tb->pq_cb, idx,
                tb->pq_cb_l2, idx_l2,
                x_fp32, out,
                tb->pq_M, tb->pq_N, tb->pq_G, tb->pq_ns, tb->pq_K_l2,
                tb->pq_l2_idx_bits ? tb->pq_l2_idx_bits : 8);
        }
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
                   /* Fused PQv2 QKV kernel does not yet handle the L2
                    * residual stage — fall through to 3 separate
                    * l2residual decodes when any of Q/K/V is pyramid. */
                   && lb->q.pq_K_l2 == 0 && lb->k.pq_K_l2 == 0 && lb->v.pq_K_l2 == 0
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
                                                nh, nkh, hd, sl, pos, b->kv_window);
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
                       /* Fused PQv2 gate+up kernel does not yet handle
                        * the L2 residual stage — fall through to two
                        * separate l2residual decodes when either is
                        * pyramid. */
                       && lb->gate.pq_K_l2 == 0 && lb->up.pq_K_l2 == 0
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

/* Stage 5d — hybrid CPU/GPU one-shot matmul. See declaration in
 * metal_runtime.h. Resolves the (layer_idx, which) selector to the
 * tensor_bufs uploaded by ib_metal_upload_model, derives M/N from the
 * source IBF tensor metadata (or from the PQ descriptor when PQv2),
 * records ONE matmul through the existing recorder, commits, and
 * waits. Synchronous. */
extern "C" int
ib_metal_run_single_matmul(ib_metal_ctx *ctx,
                            void *model_bufs_opaque,
                            int layer_idx, int which,
                            const void *x_gpu,
                            void *y_gpu)
{
    if (!ctx || !model_bufs_opaque || !x_gpu || !y_gpu) return -1;
    ib_metal_model_buffers *b = (ib_metal_model_buffers *)model_bufs_opaque;
    if (!b->model) return -2;

    /* Resolve tensor_bufs slot + source meta for M/N. */
    const struct tensor_bufs *tb = NULL;
    const ib_tensor_meta     *tm = NULL;
    if (which == IB_METAL_TB_OUTPUT_HEAD) {
        tb = &b->output_head;
        tm = &b->model->output_head;
    } else {
        if (layer_idx < 0 || layer_idx >= b->num_layers) return -1;
        struct layer_bufs   *lb = &b->layers[layer_idx];
        const ib_layer_meta *lm = &b->model->layers[layer_idx];
        switch (which) {
            case IB_METAL_TB_Q_PROJ:    tb = &lb->q;    tm = &lm->q_proj;    break;
            case IB_METAL_TB_K_PROJ:    tb = &lb->k;    tm = &lm->k_proj;    break;
            case IB_METAL_TB_V_PROJ:    tb = &lb->v;    tm = &lm->v_proj;    break;
            case IB_METAL_TB_O_PROJ:    tb = &lb->o;    tm = &lm->o_proj;    break;
            case IB_METAL_TB_GATE_PROJ: tb = &lb->gate; tm = &lm->gate_proj; break;
            case IB_METAL_TB_UP_PROJ:   tb = &lb->up;   tm = &lm->up_proj;   break;
            case IB_METAL_TB_DOWN_PROJ: tb = &lb->down; tm = &lm->down_proj; break;
            default: return -1;
        }
    }
    if (!tb || !tm) return -1;

    /* M = output rows, N = input columns. PQ descriptor wins when present
     * (the PQ kernel reads pq_M/pq_N as authoritative); else pull from
     * the source tensor's shape, which is [M, N]. */
    int M, N;
    if (tb->is_pq) {
        M = tb->pq_M;
        N = tb->pq_N;
    } else {
        if (tm->ndim < 2) return -1;
        M = tm->shape[0];
        N = tm->shape[1];
    }
    if (M <= 0 || N <= 0) return -1;

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) return -1;

    /* Drive-mode ring counters reset on each fresh recorder. */
    b->gpu_drive_idx_in_flight = 0;
    b->gpu_drive_idx_slot = 0;
    b->gpu_drive_cur_sr = 0;
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }

    /* Reuse the per-model scratch buffers for INT4 W4A8 input quantization;
     * they're sized for max(hidden, intermediate) which covers every FFN
     * matmul. rec_matmul_tb ignores xq/xs for PQv2 and INT8 paths. */
    int rc = rec_matmul_tb(r, b, tb, x_gpu, y_gpu, b->xq, b->xs, M, N);
    if (rc != 0) {
        /* Best-effort: still commit the (empty) recorder so we don't leak it. */
        (void)ib_metal_recorder_commit(r);
        return rc;
    }

    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }
    return ib_metal_recorder_commit(r);
}

extern "C" int
ib_metal_forward_token(ib_metal_ctx *ctx,
                        ib_metal_model_buffers *b,
                        const float *cpu_embed_in,
                        int pos,
                        float *logits_out)
{
    if (!ctx || !b || !cpu_embed_in || !logits_out) return -1;
    if (pos < 0 || pos >= b->max_logical_pos) return -1;

    memcpy(b->x, cpu_embed_in, (size_t)b->hidden * sizeof(float));

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) return -1;
    /* Drive-mode ring counters start clean on each new recorder/CB. */
    b->gpu_drive_idx_in_flight = 0;
    b->gpu_drive_idx_slot = 0;
    b->gpu_drive_cur_sr = 0;
    /* Drain any pending CB from a previous forward (defensive — should
     * already be NULL after a clean exit). */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }

    record_single_forward_step(r, b, pos);

    /* Drain async-commit pending CB before the final commit. The final
     * commit's wait covers the current CB; this covers the in-flight one
     * (releasing its retained handle). */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }
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
    if (start_pos < 0 || start_pos + n_steps > b->max_logical_pos) return -1;
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

    /* Drain async-commit pending CB before the final commit. */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
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
    /* MPS-hybrid path: if we have fp16-dequanted weights for this
     * INT4-blk32 tensor, dispatch the simpler fp16-weight simdmat
     * kernel (no INT4 unpack, no per-element scale). */
    if (tb && !tb->is_pq && tb->w_fp16) {
        int rc = ib_metal_rec_matmul_fp16w_fp32x_batched_simdmat_k64_fp32_in(
            r, x_fp32, tb->w_fp16, out, B, M, N);
        if (rc == 0) return 0;
    }
    if (tb && tb->is_pq) {
        void *idx_l2 = NULL;
        void *idx = drive_prepare_pq(r, b, tb, &idx_l2);
        if (!idx) return -1;
        /* Pyramid (l2_kind=2) batched path: the simdmat-tiled fast
         * batched kernels don't know about the L2 residual, so loop
         * the per-token l2residual decode B times. Functionally
         * correct (matches CPU and per-token Metal); performance work
         * is a follow-up (a true batched L2 kernel would amortize
         * indices reads across B positions, mirroring the flat batched
         * design). idx_l2 is resident (RAM) or the L2 ring slot (drive);
         * the slot stays valid across the B-loop because the ring only
         * advances on the next drive_prepare_pq call. */
        if (tb->pq_K_l2 > 0 && tb->pq_cb_l2 && idx_l2) {
            for (int bb = 0; bb < B; bb++) {
                const void *xp = (const float *)x_fp32 + (size_t)bb * N;
                void       *op = (float *)out          + (size_t)bb * M;
                int rc = ib_metal_rec_matmul_pqv2_k256_half2_l2residual(r,
                    tb->pq_rs, tb->pq_cb, idx,
                    tb->pq_cb_l2, idx_l2,
                    xp, op,
                    tb->pq_M, tb->pq_N, tb->pq_G, tb->pq_ns, tb->pq_K_l2,
                    tb->pq_l2_idx_bits ? tb->pq_l2_idx_bits : 8);
                if (rc != 0) return rc;
            }
            return 0;
        }
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
    if (start_pos < 0 || start_pos + n_tokens > b->max_logical_pos) return -1;
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
    b->gpu_drive_cur_sr = 0;
    /* Drain any pending CB from a previous forward (defensive — should
     * already be NULL after a clean exit). */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }

    #define ROW_F(buf, n, dim) ((float*)(buf) + (size_t)(n) * (dim))

    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];

        /* Pre-attention RMSNorm (batched): x_b -> xb_b */
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->input_norm, b->xb_b, B, hidden, eps);

        /* Batched Q/K/V matmul. Try fused INT4-blk32 simdmat first
         * (one Metal dispatch instead of three); fall back to per-tensor
         * paths for PQv2 / INT8 / per-row INT4. */
        int rc_qkv = -1;
        if (!lb->q.is_pq && !lb->k.is_pq && !lb->v.is_pq
            && lb->q.bits == 4 && lb->q.blk32
            && lb->k.bits == 4 && lb->k.blk32
            && lb->v.bits == 4 && lb->v.blk32
            && !lb->q.w_fp16 && !lb->k.w_fp16 && !lb->v.w_fp16  /* prefer fp16 individual path when available */
            && (qh % 32) == 0 && (kv_dim % 32) == 0
            && (hidden % 128) == 0 && (B % 32) == 0) {
            rc_qkv = ib_metal_rec_matmul_w4a8_blk32_batched_qkv_simdmat_k64_fp32_in(
                r, b->xb_b,
                lb->q.w, lb->q.s,
                lb->k.w, lb->k.s,
                lb->v.w, lb->v.s,
                b->q_b, b->k_b, b->v_b,
                b->xq_b, b->xs_b,
                B, qh, kv_dim, hidden);
        }
        if (rc_qkv != 0) {
            if (rec_matmul_batched_tb(r, b, &lb->q, b->xb_b, b->q_b, b->xq_b, b->xs_b, B, qh, hidden) != 0)
                rec_matmul_batched(r, lb->q.bits, lb->q.blk32, b->xb_b,
                    lb->q.w, lb->q.s, b->q_b, b->xq_b, b->xs_b, B, qh, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->k, b->xb_b, b->k_b, b->xq_b, b->xs_b, B, kv_dim, hidden) != 0)
                rec_matmul_batched(r, lb->k.bits, lb->k.blk32, b->xb_b,
                    lb->k.w, lb->k.s, b->k_b, b->xq_b, b->xs_b, B, kv_dim, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->v, b->xb_b, b->v_b, b->xq_b, b->xs_b, B, kv_dim, hidden) != 0)
                rec_matmul_batched(r, lb->v.bits, lb->v.blk32, b->xb_b,
                    lb->v.w, lb->v.s, b->v_b, b->xq_b, b->xs_b, B, kv_dim, hidden);
        }

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
                B, nh, nkh, hd, sl, start_pos, b->kv_window);
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

        /* Batched gate / up matmul. Try fused INT4-blk32 simdmat first
         * (one Metal dispatch instead of two); fall back to per-tensor
         * paths for PQv2 / INT8 / per-row INT4. */
        int rc_gu = -1;
        if (!lb->gate.is_pq && !lb->up.is_pq
            && lb->gate.bits == 4 && lb->gate.blk32
            && lb->up.bits == 4 && lb->up.blk32
            && !lb->gate.w_fp16 && !lb->up.w_fp16  /* prefer fp16 individual path */
            && (inter % 32) == 0 && (hidden % 128) == 0 && (B % 32) == 0) {
            rc_gu = ib_metal_rec_matmul_w4a8_blk32_batched_gateup_simdmat_k64_fp32_in(
                r, b->xb_b,
                lb->gate.w, lb->gate.s,
                lb->up.w,   lb->up.s,
                b->hb_b, b->hb2_b,
                b->xq_b, b->xs_b,
                B, inter, hidden);
        }
        if (rc_gu != 0) {
            if (rec_matmul_batched_tb(r, b, &lb->gate, b->xb_b, b->hb_b, b->xq_b, b->xs_b, B, inter, hidden) != 0)
                rec_matmul_batched(r, lb->gate.bits, lb->gate.blk32, b->xb_b,
                    lb->gate.w, lb->gate.s, b->hb_b,  b->xq_b, b->xs_b, B, inter, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->up, b->xb_b, b->hb2_b, b->xq_b, b->xs_b, B, inter, hidden) != 0)
                rec_matmul_batched(r, lb->up.bits, lb->up.blk32, b->xb_b,
                    lb->up.w,   lb->up.s,   b->hb2_b, b->xq_b, b->xs_b, B, inter, hidden);
        }

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

    /* Drain async-commit pending CB before the final commit. The final
     * commit's wait covers the current CB; this covers the in-flight one
     * (releasing its retained handle). */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }
    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) return rc;

    memcpy(last_logits_out, b->logits, (size_t)b->vocab * sizeof(float));
    return 0;
    #undef ROW_F
}

/* All-logits variant of forward_prefill: produces per-position logits
 * for all n_tokens (vs the standard last-token only). Used by
 * speculative decoding to verify each draft token. Same KV-cache
 * write semantics as forward_prefill; same shape constraints.
 *
 * Output layout: all_logits_out[i] is the vocab-length logit vector
 * for the (start_pos + i)-th position, i = 0..n_tokens-1. Caller
 * must allocate n_tokens * vocab * sizeof(float) bytes. */
/* Internal implementation shared by both forward_prefill_logits_all and
 * forward_prefill_logits_all_ex. When hidden_states_out is non-NULL,
 * captures each layer's post-residual hidden state at a small perf cost
 * (one checkpoint per layer to drain the CB before memcpy from shared
 * MTLBuffer). NULL = original fast path. */
static int
forward_prefill_logits_all_impl(ib_metal_ctx *ctx,
                                  ib_metal_model_buffers *b,
                                  const float *cpu_embeds_in,
                                  int n_tokens, int start_pos,
                                  float *all_logits_out,
                                  float **hidden_states_out)
{
    if (!ctx || !b || !cpu_embeds_in || !all_logits_out) return -1;
    if (n_tokens < 1) return -1;
    if (start_pos < 0 || start_pos + n_tokens > b->max_logical_pos) return -1;
    if (!model_supports_batched_prefill(b)) return -2;
    /* No auto-chunk: speculative draft length is small (K ≤ 8). */
    if (n_tokens > b->b_max) return -3;
    /* Use B = n_tokens throughout — DO NOT pad up to multiple of 32.
     * Padding would pollute the KV cache with zero embeddings at
     * positions [start_pos+n_tokens..start_pos+B_pad-1] which subsequent
     * iterations would read in attention and produce garbage. The
     * rec_matmul_batched_tb auto-split kernel handles small B (<32)
     * via the SIMD-coop fallback; slower than simdmat but correct.
     * Speculative-decoding throughput needs the KV correctness first. */

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

    /* Copy embeddings into x_b. Only the first B rows are populated. */
    memcpy(b->x_b, cpu_embeds_in, (size_t)B * hidden * sizeof(float));

    /* Allocate a GPU buffer for B × vocab logits. Freed after copy. */
    size_t all_bytes = (size_t)B * b->vocab * sizeof(float);
    void *all_logits_gpu = ib_metal_alloc(ctx, all_bytes, NULL);
    if (!all_logits_gpu) return -1;

    ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
    if (!r) { ib_metal_free(ctx, all_logits_gpu); return -1; }
    b->gpu_drive_idx_in_flight = 0;
    b->gpu_drive_idx_slot = 0;
    b->gpu_drive_cur_sr = 0;
    /* Drain any pending CB from a previous forward (defensive). */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }

    #define ROW_F(buf, n, dim) ((float*)(buf) + (size_t)(n) * (dim))

    /* Same per-layer body as forward_prefill, using B_pad as the
     * batched dim. For positions ≥ n_tokens (padding) the KV writes
     * write zeros at positions start_pos+pad which are also future
     * (overwritten on next call) — benign. */
    int B_use = B;
    for (int L = 0; L < b->num_layers; L++) {
        struct layer_bufs *lb = &b->layers[L];
        ib_metal_rec_rmsnorm_fp16_batched(r,
            b->x_b, lb->input_norm, b->xb_b, B_use, hidden, eps);

        int rc_qkv = -1;
        if (!lb->q.is_pq && !lb->k.is_pq && !lb->v.is_pq
            && lb->q.bits == 4 && lb->q.blk32
            && lb->k.bits == 4 && lb->k.blk32
            && lb->v.bits == 4 && lb->v.blk32
            && !lb->q.w_fp16 && !lb->k.w_fp16 && !lb->v.w_fp16
            && (qh % 32) == 0 && (kv_dim % 32) == 0
            && (hidden % 128) == 0 && (B_use % 32) == 0) {
            rc_qkv = ib_metal_rec_matmul_w4a8_blk32_batched_qkv_simdmat_k64_fp32_in(
                r, b->xb_b,
                lb->q.w, lb->q.s, lb->k.w, lb->k.s, lb->v.w, lb->v.s,
                b->q_b, b->k_b, b->v_b, b->xq_b, b->xs_b,
                B_use, qh, kv_dim, hidden);
        }
        if (rc_qkv != 0) {
            if (rec_matmul_batched_tb(r, b, &lb->q, b->xb_b, b->q_b, b->xq_b, b->xs_b, B_use, qh, hidden) != 0)
                rec_matmul_batched(r, lb->q.bits, lb->q.blk32, b->xb_b, lb->q.w, lb->q.s, b->q_b, b->xq_b, b->xs_b, B_use, qh, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->k, b->xb_b, b->k_b, b->xq_b, b->xs_b, B_use, kv_dim, hidden) != 0)
                rec_matmul_batched(r, lb->k.bits, lb->k.blk32, b->xb_b, lb->k.w, lb->k.s, b->k_b, b->xq_b, b->xs_b, B_use, kv_dim, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->v, b->xb_b, b->v_b, b->xq_b, b->xs_b, B_use, kv_dim, hidden) != 0)
                rec_matmul_batched(r, lb->v.bits, lb->v.blk32, b->xb_b, lb->v.w, lb->v.s, b->v_b, b->xq_b, b->xs_b, B_use, kv_dim, hidden);
        }
        ib_metal_rec_rope_inplace_batched(r, b->q_b, B_use, nh,  hd, start_pos, th);
        ib_metal_rec_rope_inplace_batched(r, b->k_b, B_use, nkh, hd, start_pos, th);

        if (b->kv_bits == 16) {
            ib_metal_rec_attention_block_fp16_batched(r,
                b->q_b, b->k_b, b->v_b, lb->k_cache, lb->v_cache,
                b->scores_b, b->attn_out_b,
                B_use, nh, nkh, hd, sl, start_pos, b->kv_window);
        } else {
            for (int bb = 0; bb < B_use; bb++) {
                int pos = start_pos + bb;
                float *q_row = ROW_F(b->q_b, bb, qh);
                float *k_row = ROW_F(b->k_b, bb, kv_dim);
                float *v_row = ROW_F(b->v_b, bb, kv_dim);
                float *attn_out_row = ROW_F(b->attn_out_b, bb, qh);
                ib_metal_rec_attention_block_int8(r,
                    q_row, k_row, v_row,
                    lb->k_cache, lb->v_cache, lb->k_scales, lb->v_scales,
                    b->scores, attn_out_row,
                    nh, nkh, hd, sl, pos);
            }
        }
        if (rec_matmul_batched_tb(r, b, &lb->o, b->attn_out_b, b->xb2_b, b->xq_b, b->xs_b, B_use, hidden, qh) != 0)
            rec_matmul_batched(r, lb->o.bits, lb->o.blk32, b->attn_out_b, lb->o.w, lb->o.s, b->xb2_b, b->xq_b, b->xs_b, B_use, hidden, qh);
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb2_b, B_use, hidden);

        ib_metal_rec_rmsnorm_fp16_batched(r, b->x_b, lb->post_norm, b->xb_b, B_use, hidden, eps);

        int rc_gu = -1;
        if (!lb->gate.is_pq && !lb->up.is_pq
            && lb->gate.bits == 4 && lb->gate.blk32
            && lb->up.bits == 4 && lb->up.blk32
            && !lb->gate.w_fp16 && !lb->up.w_fp16
            && (inter % 32) == 0 && (hidden % 128) == 0 && (B_use % 32) == 0) {
            rc_gu = ib_metal_rec_matmul_w4a8_blk32_batched_gateup_simdmat_k64_fp32_in(
                r, b->xb_b,
                lb->gate.w, lb->gate.s, lb->up.w, lb->up.s,
                b->hb_b, b->hb2_b, b->xq_b, b->xs_b,
                B_use, inter, hidden);
        }
        if (rc_gu != 0) {
            if (rec_matmul_batched_tb(r, b, &lb->gate, b->xb_b, b->hb_b, b->xq_b, b->xs_b, B_use, inter, hidden) != 0)
                rec_matmul_batched(r, lb->gate.bits, lb->gate.blk32, b->xb_b, lb->gate.w, lb->gate.s, b->hb_b, b->xq_b, b->xs_b, B_use, inter, hidden);
            if (rec_matmul_batched_tb(r, b, &lb->up, b->xb_b, b->hb2_b, b->xq_b, b->xs_b, B_use, inter, hidden) != 0)
                rec_matmul_batched(r, lb->up.bits, lb->up.blk32, b->xb_b, lb->up.w, lb->up.s, b->hb2_b, b->xq_b, b->xs_b, B_use, inter, hidden);
        }
        ib_metal_rec_silu_mul_batched(r, b->hb_b, b->hb2_b, b->hb_b, B_use, inter);

        if (rec_matmul_batched_tb(r, b, &lb->down, b->hb_b, b->xb_b, b->xq_b, b->xs_b, B_use, hidden, inter) != 0)
            rec_matmul_batched(r, lb->down.bits, lb->down.blk32, b->hb_b, lb->down.w, lb->down.s, b->xb_b, b->xq_b, b->xs_b, B_use, hidden, inter);
        ib_metal_rec_residual_add_batched(r, b->x_b, b->xb_b, B_use, hidden);

        /* Phase 3.1: capture per-layer hidden state for callers that need
         * it (e.g. DFlash hybrid orchestrator). Requires a CB drain at
         * each layer so the shared MTLBuffer's CPU view of b->x_b is
         * up-to-date; NULL skips this for the fast path. */
        if (hidden_states_out && hidden_states_out[L]) {
            if (b->gpu_drive_pending_cb) {
                (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
                b->gpu_drive_pending_cb = NULL;
            }
            if (ib_metal_recorder_checkpoint(r) != 0) {
                ib_metal_free(ctx, all_logits_gpu);
                return -1;
            }
            b->gpu_drive_idx_in_flight = 0;
            b->gpu_drive_cur_sr = 0;
            memcpy(hidden_states_out[L], b->x_b,
                   (size_t)B_use * hidden * sizeof(float));
        }
    }

    /* Final batched RMSNorm + batched lm_head — outputs B_use × vocab. */
    ib_metal_rec_rmsnorm_fp16_batched(r, b->x_b, b->output_norm, b->xb_b, B_use, hidden, eps);
    if (rec_matmul_batched_tb(r, b, &b->output_head, b->xb_b, all_logits_gpu, b->xq_b, b->xs_b, B_use, b->vocab, hidden) != 0)
        rec_matmul_batched(r, b->output_head.bits, b->output_head.blk32, b->xb_b,
                           b->output_head.w, b->output_head.s, all_logits_gpu, b->xq_b, b->xs_b, B_use, b->vocab, hidden);

    /* Drain async-commit pending CB before the final commit. */
    if (b->gpu_drive_pending_cb) {
        (void)ib_metal_recorder_wait_committed(b->gpu_drive_pending_cb);
        b->gpu_drive_pending_cb = NULL;
    }
    int rc = ib_metal_recorder_commit(r);
    if (rc != 0) { ib_metal_free(ctx, all_logits_gpu); return rc; }

    /* Copy only the first n_tokens × vocab back to host. */
    memcpy(all_logits_out, all_logits_gpu, (size_t)n_tokens * b->vocab * sizeof(float));
    ib_metal_free(ctx, all_logits_gpu);
    return 0;
    #undef ROW_F
}

/* Public API: original fast path (no hidden-state capture). Delegates to
 * the impl with NULL — exact same behavior as before. */
extern "C" int
ib_metal_forward_prefill_logits_all(ib_metal_ctx *ctx,
                                      ib_metal_model_buffers *b,
                                      const float *cpu_embeds_in,
                                      int n_tokens, int start_pos,
                                      float *all_logits_out)
{
    return forward_prefill_logits_all_impl(ctx, b, cpu_embeds_in,
                                             n_tokens, start_pos,
                                             all_logits_out, NULL);
}

/* Public API: same as forward_prefill_logits_all, but additionally
 * captures each layer's post-residual hidden state into the caller-
 * provided buffers. hidden_states_out must be an array of n_layers
 * float* pointers; each non-NULL pointer must point to at least
 * n_tokens * hidden floats. NULL entries are skipped (caller can
 * selectively capture only specific layers). hidden_states_out itself
 * being NULL takes the fast path.
 *
 * Use case: DFlash-style hybrid speculative decoding where the draft
 * model conditions on the target's mid-stack hidden states.
 *
 * Note: capture forces a CB drain per captured layer, so this is
 * substantially slower than the no-capture path (~2-3x measured on
 * 1B at B=8). Don't enable for fast-path inference. */
extern "C" int
ib_metal_forward_prefill_logits_all_ex(ib_metal_ctx *ctx,
                                         ib_metal_model_buffers *b,
                                         const float *cpu_embeds_in,
                                         int n_tokens, int start_pos,
                                         float *all_logits_out,
                                         float **hidden_states_out)
{
    return forward_prefill_logits_all_impl(ctx, b, cpu_embeds_in,
                                             n_tokens, start_pos,
                                             all_logits_out,
                                             hidden_states_out);
}
