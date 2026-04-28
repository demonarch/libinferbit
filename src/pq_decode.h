/*
 * pq_decode.h — IBF v5 stacked 2D PQ tensor decoder.
 *
 * Spec: docs/26_IBF_V5_PQ_FORMAT.md.
 *
 * Two operating modes:
 *   - Materialize: rebuild the full FP16 [M,N] weight matrix from PQ blocks.
 *     Used as the correctness oracle; matches the Python reference bit-for-bit.
 *   - Fused matmul: compute out = W * x directly from PQ blocks without
 *     materializing the full matrix. The performance path.
 *
 * Loader is single-tensor for now (matches the Python single-tensor writer).
 * Multi-tensor IBF v5 layer-aware loading lives elsewhere once the format
 * lands in the main IBF loader.
 */
#ifndef IB_PQ_DECODE_H
#define IB_PQ_DECODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    IB_PQ_FMT_NONE = 0,
    IB_PQ_FMT_L1,        /* "pq2d_v1_l1"     — 1 level, optional outlier */
    IB_PQ_FMT_L2,        /* "pq2d_v1_l2"     — 2 level residual, no outlier */
    IB_PQ_FMT_L1_L2,     /* "pq2d_v1_l1_l2"  — 2 level + outlier */
} ib_pq_format;

typedef struct {
    /* Shape and config */
    int M;
    int N;
    int G;
    int K;
    int n_levels;
    int rotate;
    int n_outlier;
    ib_pq_format format;

    /* Codebook tables — K * (G/2) FP16 entries each.
     * Stored as raw uint16_t (FP16 bit pattern); decode via fp16_to_fp32. */
    uint16_t* codebook_l1_l;
    uint16_t* codebook_l1_r;
    uint16_t* codebook_l2_l;   /* NULL if n_levels == 1 */
    uint16_t* codebook_l2_r;   /* NULL if n_levels == 1 */

    /* Indices: M rows × C columns where C = (N - n_outlier) / G. */
    uint8_t* indices_l1_l;
    uint8_t* indices_l1_r;
    uint8_t* indices_l2_l;     /* NULL if n_levels == 1 */
    uint8_t* indices_l2_r;     /* NULL if n_levels == 1 */
    int C;                     /* chunks per row */

    /* Per-row scale */
    uint16_t* row_scale;       /* M FP16 entries */

    /* Outlier sidecar (NULL if n_outlier == 0) */
    int32_t*  outlier_cols;    /* n_outlier int32 column indices */
    int8_t*   outlier_sidecar; /* M × n_outlier int8 */
    uint16_t* outlier_scale;   /* n_outlier FP16 per-col scales */

    /* Backing storage. Exactly one of these is non-NULL:
     *   _arena: contiguous heap allocation (fread loaders)
     *   _mmap_base: mmap'd file region (Path D loaders) */
    void* _arena;
    size_t _arena_size;
    void* _mmap_base;          /* set on the FIRST tensor of an mmap'd file; */
    size_t _mmap_size;         /* others share, only owner munmaps */
    int   _owns_mmap;          /* 1 = munmap on free, 0 = shared reference */
} ib_pq_tensor;

/* Load a single-tensor IBF v5 file (matches the Python writer in
 * inferbit-py/inferbit/_pq_format.py). On success returns 0 and fills *out.
 * On failure returns < 0 and *out is left zeroed. */
int  ib_pq_load_single(const char* path, ib_pq_tensor* out);
void ib_pq_free(ib_pq_tensor* t);

/* Multi-tensor IBF v5 loader. Returns an array of tensors and their
 * names. Caller must free both via ib_pq_multi_free. */
typedef struct {
    int n;
    char** names;          /* heap-allocated strings */
    ib_pq_tensor* tensors; /* parallel array */
    void* _mmap_base;      /* non-NULL when mmap-backed (Path D) */
    size_t _mmap_size;
} ib_pq_multi;

int  ib_pq_load_multi(const char* path, ib_pq_multi* out);
void ib_pq_multi_free(ib_pq_multi* m);
const ib_pq_tensor* ib_pq_multi_find(const ib_pq_multi* m, const char* name);

/* Path D: mmap the file once and view tensors as zero-copy pointers
 * into the OS page cache. Compared to ib_pq_load_multi:
 *   - load wall is constant regardless of file size (no fread copy)
 *   - working-set RAM is OS-paged, not preloaded
 *   - eviction & prefetch can be advised explicitly (see below)
 *
 * On success returns 0 and fills *out. The mmap is owned by the
 * ib_pq_multi struct; ib_pq_multi_free will munmap it. */
int ib_pq_open_mmap(const char* path, ib_pq_multi* out);

/* Tell the OS we plan to use a tensor's bytes soon — issues
 * MADV_WILLNEED so pages are read in ahead. Cheap. */
void ib_pq_advise_willneed(const ib_pq_tensor* t);

/* Tell the OS we are done with a tensor and the pages can be
 * dropped — MADV_DONTNEED. Frees physical RAM under pressure. */
void ib_pq_advise_dontneed(const ib_pq_tensor* t);

/* Streaming-inference scheduler primitives.
 *
 * Typical use during a forward pass over a multi-layer model:
 *   for L in 0..n_layers:
 *       if L+1 < n_layers:
 *           ib_pq_advise_willneed_n(&next_layer_tensors, k);
 *       run_layer(L);
 *       ib_pq_advise_dontneed_n(&this_layer_tensors, k);
 *
 * The willneed hint lets the OS overlap layer L+1's page-in with
 * layer L's compute. The dontneed lets the OS reclaim layer L's
 * pages when it needs to. Both are advisory — no-ops on already-
 * resident pages.
 *
 * Quantitative wins:
 *   - Cold start (pages not yet faulted): willneed overlaps fault
 *     with compute, materially cuts time-to-first-token.
 *   - Memory-pressured (model > RAM): dontneed lets the OS evict
 *     finished layers so the next layer fits.
 *   - Warm + abundant RAM: both calls are no-ops; same throughput.
 */
void ib_pq_advise_willneed_n(const ib_pq_tensor* const* tensors, int n);
void ib_pq_advise_dontneed_n(const ib_pq_tensor* const* tensors, int n);

/* Materialize: write M*N FP16 values into out_fp16.
 * out_fp16 must have space for M*N uint16_t. */
int ib_pq_reconstruct_fp16(const ib_pq_tensor* t, uint16_t* out_fp16);

/* Materialize FP32 (helper for tests): write M*N FP32 values into out_fp32. */
int ib_pq_reconstruct_fp32(const ib_pq_tensor* t, float* out_fp32);

/* Fused matmul: compute out[M] = W * x[N] in FP32.
 * x is in the rotated basis if t->rotate (caller's responsibility — at
 * runtime the rotation merges into the prior projection). */
int ib_pq_matmul_fp32(const ib_pq_tensor* t, const float* x, float* out);

/* Threaded variant. `pool` is an opaque ib_thread_pool* (defined in
 * inferbit_internal.h). If pool is NULL, falls back to single-threaded.
 * The per-row gather is split across worker threads; LUT-table build
 * and outlier sidecar both run on the calling thread. */
int ib_pq_matmul_fp32_threaded(const ib_pq_tensor* t, const float* x,
                                float* out, void* pool);

/* Byte-quantised-LUT variant. Same math, but the per-chunk partial-
 * products tables are quantised to int8 with one fp32 scale per
 * chunk-side. Reduces L2 cache traffic per row by ~32x at the cost of
 * 1 fp32 multiply per chunk and ~1 fp16 ULP precision per chunk-side
 * (bounded; signed errors largely cancel across chunks).
 *
 * Numerical contract: output is allowed to differ from the fp32-LUT
 * path by up to ~1e-3 frobenius-relative on typical model shapes. Use
 * for inference; use ib_pq_matmul_fp32 for verification. */
int ib_pq_matmul_fp32_q8lut(const ib_pq_tensor* t, const float* x,
                             float* out, void* pool);

/* Float16 helpers (IEEE 754 binary16). Pure software, portable. */
float    ib_fp16_to_fp32(uint16_t h);
uint16_t ib_fp32_to_fp16(float f);

#ifdef __cplusplus
}
#endif
#endif
