/*
 * pqv2_encode.h — PQv2 encoder (Stage 1 of docs/v2/00_CORRECTION.md).
 *
 * Production C encoder for the PQv2 / pyramid quantization format
 * already supported by the loader (src/pqv2_format.c) and the Metal
 * kernels (src/metal/kernels.metal, matmul_pqv2_k256_half2_*).
 *
 * Architecture rule: this is the *only* place PQv2 codebook fitting and
 * IBF v6 writing happens. The Python and Node packages are thin FFI
 * consumers — they set inferbit_convert_config.format and call the
 * existing inferbit_convert(). The dispatcher in convert.c routes to
 * pqv2_convert() in pqv2_encode.c.
 *
 * Layout produced (one tensor):
 *   - codebook (cb_q):    int8 codewords of shape [n_subchunks][K][half]
 *   - codebook scale:     fp16 per-codeword scale of shape [n_subchunks][K]
 *   - row_scale:          fp16 per-output-row multiplier of shape [M]
 *   - indices:            uint8 of shape [n_chunks][n_subchunks][M]
 *                         (chunk-major, matches metal_model.mm:526-538
 *                          and pqv2_format.c::parse_pqv2_blob)
 *
 * Defaults: K=256, half=2, G=64, n_subchunks=G/half=32. Hard-coded in
 * the kernels (kernels.metal: matmul_pqv2_k256_half2_*; metal_model.mm
 * line ~43, 1146); the encoder MUST emit these or the GPU dispatcher
 * will not consume the tensors.
 *
 * Pyramid (n_levels=2) variant: an additive L2 PQ residual layer with
 * its own per-(chunk, sub-chunk) codebook fitted on (W - L1_recon).
 * Loader treats this as l2_kind=2 (see pqv2_format.h header comment
 * and pqv2_format.c lines 40-47).
 *
 * Format-string distinction (pq2d_v1_l1 vs pq2d_v1_pyramid) is carried
 * implicitly by the l2_kind field in the per-tensor blob header: 0 ⇒
 * flat (pq2d_v1_l1), 2 ⇒ pyramid (pq2d_v1_pyramid). Adding a separate
 * format_str manifest slot would require the read side to change in
 * lockstep; we instead overload the existing l2_kind which already
 * differentiates the two paths in pqv2_load_internal and the kernels.
 * If a future writer needs a richer marker, the four bytes at file
 * offset [20..23] (today "reserved (0)" in the file header — see
 * pqv2_format.h) are available for that purpose without touching the
 * blob layout.
 */
#ifndef IB_PQV2_ENCODE_H
#define IB_PQV2_ENCODE_H

#include <stdint.h>
#include "inferbit.h"  /* for inferbit_convert_config / inferbit_convert_format */

#ifdef __cplusplus
extern "C" {
#endif

/* Per-tensor flat-PQv2 encode.
 *
 * W       : [M][N] fp32 weight matrix, row-major.
 * M, N    : matrix dims. N must be a multiple of G; G must be a multiple
 *           of half.
 * G       : group size (default 64). n_chunks = N / G.
 * K       : codebook size per sub-chunk slot (default 256).
 * half    : sub-chunk dimensionality (default 2). n_subchunks = G / half.
 * seed    : deterministic k-means seed.
 *
 * Outputs (all caller-allocated; sizes documented in the header above):
 *   cb_int8_out       : [n_subchunks * K * half]      int8
 *   cb_scale_fp16_out : [n_subchunks * K]             uint16 (fp16 bits)
 *   row_scale_fp16_out: [M]                            uint16 (fp16 bits)
 *   indices_out       : [n_chunks * n_subchunks * M]   uint8 (chunk-major)
 *
 * Returns 0 on success, < 0 on error.
 */
int pqv2_encode_flat(const float *W, int M, int N,
                     int G, int K, int half,
                     int8_t   *cb_int8_out,
                     uint16_t *cb_scale_fp16_out,
                     uint16_t *row_scale_fp16_out,
                     uint8_t  *indices_out,
                     uint32_t  seed);

/* Per-tensor pyramid (n_levels=2) encode. Same inputs/outputs as the
 * flat version, plus an additive L2 residual codebook fitted on the
 * post-row-scale L1 reconstruction error.
 *
 * K_outer is a (currently unused) coarse-cluster hint; the production
 * loader-compatible layout uses one inner codebook of size K per
 * sub-chunk slot at L2 (l2_K == K_inner). K_l2 == K_inner; uint8
 * indices. Keeping K_outer in the signature so the Python/Node
 * wrappers can later sweep it; passing 0 disables coarse pre-clustering
 * and is the production default.
 *
 * K_inner range: [1, 64]. The L1 codebook is fixed at K=256 (production
 * K=256 NEON kernel); the L2 codebook is bound by the same NEON kernel's
 * vqtbl4q_s8 lookup table (4×16 = 64 entries). Larger K_inner would be
 * silently dropped by the forward dispatch — both pqv2_threaded_matvec_k256
 * (forward.c:399) and pqv2_matvec_tbl_int8_k256 (pqv2_kernel.c:1010)
 * guard on `t->l2_K <= 64` before applying L2.
 *
 *   cb_int8_l1, cb_scale_fp16_l1, row_scale, indices_l1: same as flat
 *                                                       (sized to K=256).
 *   cb_int8_l2_out       : [n_subchunks * K_inner * half] int8
 *   cb_scale_fp16_l2_out : [n_subchunks * K_inner]        uint16 fp16
 *   indices_l2_out       : [n_chunks * n_subchunks * M]   uint8
 */
int pqv2_encode_pyramid(const float *W, int M, int N,
                        int G, int K_outer, int K_inner, int half,
                        int8_t   *cb_int8_l1_out,
                        uint16_t *cb_scale_fp16_l1_out,
                        int8_t   *cb_int8_l2_out,
                        uint16_t *cb_scale_fp16_l2_out,
                        uint16_t *row_scale_fp16_out,
                        uint8_t  *indices_l1_out,
                        uint8_t  *indices_l2_out,
                        uint32_t  seed);

/* IBF v6 PQv2 writer entrypoint — convert an HF safetensors directory
 * (or single .safetensors / .gguf file) into an IBF v6 PQv2 .ibf file.
 *
 * Routing: called by inferbit_convert() in convert.c whenever
 * cfg->format != INFERBIT_CONVERT_INT4. The INT4 path is untouched.
 *
 * FFN tensors (gate_proj, up_proj, down_proj) and attention output
 * (o_proj) are encoded as PQv2; attention QKV + embeddings + lm_head
 * stay INT8 (because the production GPU kernels for those are not
 * PQv2-aware yet); norms stay FP16. Mirrors the v0.4.1 INT4 path's
 * bit-allocation philosophy.
 *
 * Returns INFERBIT_OK on success, an INFERBIT_ERROR_* code on failure
 * (use inferbit_last_error() for details).
 */
int pqv2_convert(const char *input_path,
                 const char *output_path,
                 const inferbit_convert_config *cfg);

#ifdef __cplusplus
}
#endif
#endif /* IB_PQV2_ENCODE_H */
