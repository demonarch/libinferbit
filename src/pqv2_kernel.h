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

    /* fp16 stored as raw uint16 */
    const uint16_t *row_scale;        /* [M] */

    /* L1 codebooks */
    const int8_t  *cb_q;              /* [n_subchunks * K * half] */
    const uint16_t *cb_scale;         /* [n_subchunks * K] fp16 */
    const uint8_t *indices;           /* [M * (N/G) * n_subchunks] */

    /* L2-PQ codebooks (NULL if no L2) */
    const int8_t  *l2_cb_q;
    const uint16_t *l2_cb_scale;
    const uint8_t *l2_indices;
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

/* fp16 helpers (IEEE half) */
float pqv2_h2f(uint16_t h);
uint16_t pqv2_f2h(float f);

#endif
