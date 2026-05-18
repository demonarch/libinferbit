#ifndef PQV2_KERNEL_H
#define PQV2_KERNEL_H

#include <stdint.h>
#include <stddef.h>

/* PQ-v2 standalone kernel + on-disk format for the pq-v2-bench branch.
 *
 * Layout matches scripts/poc/pqv2_encode.py PQv2Tensor (no AWQ for bench):
 *   half = G / n_subchunks
 *   n_chunks = N / G
 *   indices    [M, n_chunks, n_subchunks]   uint8 (K <= 256)
 *   codebooks  [n_subchunks][K][half]       int8
 *   cb_scales  [n_subchunks][K]             fp16 (stored as uint16)
 *   row_scale  [M]                          fp16
 *   optional L2-PQ (same shape) at l2_K
 */

typedef struct {
    uint32_t M, N, G, K, n_subchunks, half;
    uint32_t l2_kind;   /* 0=none, 2=pq */
    uint32_t l2_K;
    /* L2 on-disk index bit-width. 8 = legacy uint8 (one byte per index).
     * 6 = bit-packed 4-indices-in-3-bytes layout (Stage 5h.1). Only valid
     * when l2_kind == 2 and l2_K <= 64. Defaults to 8 for files written
     * before the packing field existed. */
    uint32_t l2_idx_bits;

    /* Stage 5k — scale precision encoding. 0 = legacy (row_scale fp16,
     * cb_scale fp16). 2 = row_scale int8 + per-tensor fp16 row_max, and
     * cb_scale fp8 (E4M3). Modes 1 and 3 are reserved per the doc but
     * unimplemented in v1. The kernel ALWAYS reads `row_scale` and
     * `cb_scale` as uint16 fp16 — when mode != 0, the loader decodes
     * the on-disk int8/fp8 bytes into newly-allocated fp16 arrays at
     * parse time, so the hot inner loops stay byte-identical. Cost
     * absorbed in the load-time codebook prebuild. */
    uint32_t scale_precision;

    /* Stage 5j — codebook pool. When > 0 in the on-disk header, the file
     * holds `cb_pool_size` codebooks and a per-slot `pool_id[n_subchunks]`
     * mapping. The v1 loader EXPANDS the pool back into a per-slot
     * codebook (size n_subchunks) at parse time, so the kernel reads
     * `cb_q / cb_scale` as if pool_size == n_subchunks. These fields are
     * informational — they record what was on disk so callers (e.g.
     * inspect_ibf) can report dedup. A future kernel can switch to
     * pool_id lookup directly, at which point the loader-side expansion
     * can be skipped. v1 scaffolding ships pool_size == n_subchunks with
     * an identity pool_id mapping (no actual clustering); real
     * within-tensor clustering is a follow-up. */
    uint32_t cb_pool_size;            /* L1 pool size on disk; 0 = no pool */
    uint32_t l2_cb_pool_size;         /* L2 pool size on disk; 0 = no pool */

    /* Stage 5g.2 — L1 index on-disk layout selector.
     *   0 = chunk-major [n_chunks][n_subchunks][M] (legacy, NEON-friendly).
     *   1 = row-major   [M][n_chunks][n_subchunks] (Metal zero-copy
     *       friendly — the GPU SIMD kernel already reads row-major, so
     *       the upload-time transpose becomes a no-op and the staging
     *       malloc can be skipped via newBufferWithBytesNoCopy).
     * Selected at encode time via IB_PQV2_L1_ROWMAJOR=1 (opt-in for v1).
     * Defaults to 0 for files written before this field existed —
     * disambiguated by the same blob-size heuristic that handles every
     * other append-only header field. */
    uint32_t l1_idx_layout;           /* 0 = chunk-major, 1 = row-major */

    /* fp16 stored as raw uint16. After Stage 5k decode (mode != 0), this
     * points at a loader-allocated buffer rather than the mmap'd file. */
    const uint16_t *row_scale;        /* [M] */

    /* L1 codebooks. Sized by `cb_pool_size > 0 ? cb_pool_size : n_subchunks`. */
    const int8_t  *cb_q;              /* [rows * K * half] */
    const uint16_t *cb_scale;         /* [rows * K] fp16 */
    /* L1 indices. Logical extent is always M*(N/G)*n_subchunks bytes.
     * Layout selected by `l1_idx_layout`:
     *   0 → on-disk [n_chunks][n_subchunks][M], i.e. legacy chunk-major
     *       (NEON kernel reads `indices[(c*ns + s)*M + m]`).
     *   1 → on-disk [M][n_chunks][n_subchunks], i.e. row-major
     *       (Metal upload becomes zero-copy; NEON kernel reads
     *       `indices[m*total + c*ns + s]`, less optimal cache pattern). */
    const uint8_t *indices;

    /* L2-PQ codebooks (NULL if no L2). Sized analogously by
     * `l2_cb_pool_size > 0 ? l2_cb_pool_size : n_subchunks`. */
    const int8_t  *l2_cb_q;
    const uint16_t *l2_cb_scale;
    const uint8_t *l2_indices;

    /* Pre-decoded fp32 codebooks. NULL = compute on the fly per matvec
     * (legacy slow path). Set by the file loader once per tensor so the
     * hot kernel skips the int8→fp32 decode loop on every call. */
    const float *cb_fp32;        /* [n_subchunks * K * half] */
    const float *l2_cb_fp32;     /* [n_subchunks * l2_K * half], NULL if no L2 */

    /* Path D drive mode (doc 32): byte offset of this tensor's indices
     * region within the on-disk IBF file. Set at load time when the
     * model is in drive mode; otherwise 0. Used by forward.c to
     * pread() into the shared scratch buffer before each matmul. */
    size_t indices_file_offset;
    /* Sidecar offset for the pre-transposed [m][total] copy. Used by
     * GPU drive mode (doc 35 feature 3) to skip the per-matmul
     * transpose. 0 = no sidecar; pread from drive_fd at
     * indices_file_offset (legacy chunk-major path). */
    size_t indices_pretransposed_offset;
    /* Goal C3 — pyramid drive-mode L2 redirect. Byte offset of this
     * tensor's L2 indices region (l2_kind == 2) within the on-disk IBF
     * file. 0 = no L2 indices / not redirected (mmap path). Set at
     * load time alongside indices_file_offset when the model is in
     * drive mode; forward.c::drive_load_indices then preads L2 into
     * a sibling scratch buffer and repoints pq->l2_indices to it. */
    size_t l2_indices_file_offset;
} pqv2_t;

/* Load .pqv2 file into freshly-malloc'd buffers. Returns 0 on success.
 * Caller frees t->row_scale, t->cb_q, t->cb_scale, t->indices, and the
 * three l2_* fields when non-NULL. */
int pqv2_load(const char *path, pqv2_t *out, void **owned);
void pqv2_free(void *owned);

/* y[M] = W[M,N] @ x[N], reference (scalar) implementation. */
void pqv2_matvec_scalar(const pqv2_t *t, const float *x, float *y);

/* y[M] = W[M,N] @ x[N], LUT-based fast path (precompute cb[k]·x_slice). */
void pqv2_matvec_lut(const pqv2_t *t, const float *x, float *y);

/* NEON variant: 4-way unrolled gather across rows. */
void pqv2_matvec_lut_neon(const pqv2_t *t, const float *x, float *y);

/* NEON INT8-TBL variant. Per (chunk, sub-chunk) the LUT [K] is built in
 * fp32, quantized to int8 with one fp32 lut_scale, then gathered via
 * vqtbl{1,4}q_s8 over rows. Requires K ∈ {16, 32, 64}.  */
void pqv2_matvec_tbl_int8(const pqv2_t *t, const float *x, float *y);

/* NEON INT8-TBL variant for K=128 (2 banks of 64 entries each).
 * Uses the high bit of the index to bank-select via NEON. */
void pqv2_matvec_tbl_int8_k128(const pqv2_t *t, const float *x, float *y);

/* NEON INT8-TBL variant for K=256 (4 banks of 64 entries each).
 * Uses the top 2 bits of the index to select bank, low 6 bits as offset. */
void pqv2_matvec_tbl_int8_k256(const pqv2_t *t, const float *x, float *y);

/* DERISK: K=256 matvec with activation-aware (chunk, subchunk) skipping.
 * For each (c,s), if max|x[c*G+s*half .. c*G+(s+1)*half]| < skip_thresh,
 * the entire (c,s) accumulation step is bypassed. Quality cost depends
 * on threshold and input distribution. Returns the fraction of (c,s)
 * blocks actually skipped via *out_skip_frac (or pass NULL to ignore). */
void pqv2_matvec_tbl_int8_k256_skip(
    const pqv2_t *t, const float *x, float *y,
    float skip_thresh, double *out_skip_frac);

/* DERISK: K≤64 matvec with same activation-aware skipping. */
void pqv2_matvec_tbl_int8_skip(
    const pqv2_t *t, const float *x, float *y,
    float skip_thresh, double *out_skip_frac);

/* DERISK: fp16-accumulator variant of K=256 single-position matvec.
 * Halves acc memory traffic (M halfs vs M floats) and frees ~half the
 * NEON registers used for acc. fp32 multiply by lut_scale, narrow to
 * fp16 just before accumulation. Final output remains fp32. */
void pqv2_matvec_tbl_int8_k256_fp16acc(
    const pqv2_t *t, const float *x, float *y);

/* DERISK: GEMM-style row-tiled batched matvec for K=256, B=4.
 *
 * Outer loop tiles 16 rows; persistent acc state in NEON regs across
 * all (chunk, subchunk) iterations. Indices read ONCE per (c,s) and
 * shared across the 4 batch positions — saves ~3/4 of indices traffic
 * vs sequential B=4. Test whether memory-amortization beats the
 * unchanged 4× compute work.
 *
 * x_batch[B, N], y_batch[B, M]. B is hardcoded 4. */
void pqv2_matvec_tbl_int8_k256_gemm_b4(
    const pqv2_t *t, const float *x_batch, float *y_batch);

/* Batched K=256 matvec: process B input positions against the same W.
 *
 * x_batch:  [B, N]  row-major (per-position contiguous)
 * y_batch:  [B, M]  row-major
 *
 * Reads weight indices ONCE per row and produces B partial sums per row,
 * amortizing the dominant memory traffic across B positions. Used by
 * speculative decoding's batched verify pass (B = K_draft, e.g. 4). */
void pqv2_matvec_tbl_int8_k256_batch(
    const pqv2_t *t, const float *x_batch, int B, float *y_batch);

/* Per-chunk-range accumulation primitive for K=256.
 *
 * Accumulates contributions for chunks [c_start, c_end) into caller-owned
 * `acc` (and `acc_l2` if non-NULL). Does NOT zero acc, NOT apply row_scale,
 * NOT write to y. Caller provides cb and l2_cb as fp32 codebooks (or
 * passes NULL for l2_cb when the tensor has no L2 stage).
 *
 * Used for chunk-parallel threading: each worker gets a chunk slice and
 * accumulates into its own thread-local acc buffer; main thread reduces
 * across workers and applies row_scale at the end. */
void pqv2_acc_tbl_int8_k256_chunks(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end);

/* Skip-aware chunk-range accumulator (K=256). Same as the plain chunks
 * variant but bypasses (c,s) iterations whose input slice has
 * max|x[c*G+s*half .. c*G+(s+1)*half]| < skip_thresh. Pass 0 to disable
 * the skip check entirely. Caller still owns + zeroes acc. */
void pqv2_acc_tbl_int8_k256_chunks_skip(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end,
    float skip_thresh);

/* Batched chunk-range accumulator: B input positions, B output accumulators.
 * Same per-position summation order as the single-position chunks variant
 * (so a B=1 call is bitwise-equivalent to pqv2_acc_tbl_int8_k256_chunks).
 * acc_batch and acc_l2_batch are laid out [B, M]. Caller zeroes both once.
 * Pass l2_cb=NULL and acc_l2_batch=NULL when the tensor has no L2 stage. */
void pqv2_acc_tbl_int8_k256_chunks_batch(
    const pqv2_t *t, const float *x_batch, int B,
    const float *cb, const float *l2_cb,
    float *acc_batch, float *acc_l2_batch,
    uint32_t c_start, uint32_t c_end);

/* fp16 helpers (IEEE half) */
float pqv2_h2f(uint16_t h);
uint16_t pqv2_f2h(float f);

/* PQv2 decode profiling (IB_PQV2_PROFILE). When the env var is set, the
 * K=256 decode hot path accumulates a LUT-build / gather / total
 * wall-clock breakdown and auto-prints to stderr every N chunks-inner
 * calls. This getter prints the accumulated totals on demand (e.g. at
 * process exit). No-op when profiling was never enabled. */
void ib_pqv2_profile_dump(void);

#endif
