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

    /* Scratch buffer used for FP16 conversion (owned by struct) */
    void* _arena;              /* internal: contiguous allocation backing all pointers */
    size_t _arena_size;
} ib_pq_tensor;

/* Load a single-tensor IBF v5 file (matches the Python writer in
 * inferbit-py/inferbit/_pq_format.py). On success returns 0 and fills *out.
 * On failure returns < 0 and *out is left zeroed. */
int  ib_pq_load_single(const char* path, ib_pq_tensor* out);
void ib_pq_free(ib_pq_tensor* t);

/* Materialize: write M*N FP16 values into out_fp16.
 * out_fp16 must have space for M*N uint16_t. */
int ib_pq_reconstruct_fp16(const ib_pq_tensor* t, uint16_t* out_fp16);

/* Materialize FP32 (helper for tests): write M*N FP32 values into out_fp32. */
int ib_pq_reconstruct_fp32(const ib_pq_tensor* t, float* out_fp32);

/* Fused matmul: compute out[M] = W * x[N] in FP32.
 * x is in the rotated basis if t->rotate (caller's responsibility — at
 * runtime the rotation merges into the prior projection). */
int ib_pq_matmul_fp32(const ib_pq_tensor* t, const float* x, float* out);

/* Float16 helpers (IEEE 754 binary16). Pure software, portable. */
float    ib_fp16_to_fp32(uint16_t h);
uint16_t ib_fp32_to_fp16(float f);

#ifdef __cplusplus
}
#endif
#endif
