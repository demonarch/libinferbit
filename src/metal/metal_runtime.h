/* Metal runtime — C interface for libinferbit's Apple Metal GPU backend.
 *
 * All functions are no-ops / return NULL on non-Apple builds. Apple Silicon
 * has unified memory: ib_metal_alloc returns a pointer that's accessible
 * from BOTH CPU and GPU (storage mode shared), with no explicit copy
 * needed before/after a kernel dispatch.
 *
 * Usage:
 *   ib_metal_ctx *ctx = ib_metal_create();
 *   if (!ctx) { fallback to CPU path; }
 *   float *gpu_in  = ib_metal_alloc(ctx, n * sizeof(float), host_data);
 *   float *gpu_out = ib_metal_alloc(ctx, n * sizeof(float), NULL);
 *   ib_metal_vec_mul2(ctx, gpu_in, gpu_out, n);
 *   // gpu_out contents are visible directly (unified memory)
 *   ib_metal_free(ctx, gpu_in);
 *   ib_metal_free(ctx, gpu_out);
 *   ib_metal_destroy(ctx);
 */
#ifndef IB_METAL_RUNTIME_H
#define IB_METAL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context — wraps MTLDevice + MTLCommandQueue + MTLLibrary. */
typedef struct ib_metal_ctx ib_metal_ctx;

/* Create a Metal context using the system default device. Returns NULL
 * if Metal is unavailable, no compatible device is present, or kernel
 * library load fails. */
ib_metal_ctx *ib_metal_create(void);
void ib_metal_destroy(ib_metal_ctx *ctx);

/* Returns 1 if Metal is available + working, 0 otherwise. */
int ib_metal_available(void);

/* Returns a human-readable device name (e.g. "Apple M2"). NULL on error.
 * Caller does not free; lifetime tied to ctx. */
const char *ib_metal_device_name(ib_metal_ctx *ctx);

/* Allocate a unified-memory buffer (MTLResourceStorageModeShared). The
 * returned pointer is BOTH a host-visible pointer AND backed by an
 * MTLBuffer the kernels can use. Caller frees with ib_metal_free.
 *
 * If init is non-NULL, copies `bytes` from init into the buffer.
 * On failure returns NULL. */
void *ib_metal_alloc(ib_metal_ctx *ctx, size_t bytes, const void *init);
void  ib_metal_free(ib_metal_ctx *ctx, void *buf);

/* Hello-world test kernel: out[i] = in[i] * 2.0f. Synchronous. */
int ib_metal_vec_mul2(ib_metal_ctx *ctx,
                       const void *gpu_in, void *gpu_out, int n);

/* INT4-weight × INT8-activation matmul. Mirrors the CPU `matmul_w4a8`
 * kernel exactly — same packing, same scale model, same quality.
 *
 *   weights   [M, N/2] uint8  — packed nibbles (low nibble first), bias 8.
 *   w_scales  [M]      half   — per-row weight scale (fp16).
 *   x_q       [N]      int8   — quantized activation values.
 *   x_scales  [N/IB_W4A8_GROUP] float — per-128-group activation scales.
 *   out       [M]      float  — output: out[m] = sum_n W[m,n] * x[n].
 *
 * All buffer pointers must come from ib_metal_alloc on this context.
 * Synchronous: command buffer waits for completion before returning.
 * Returns 0 on success, -1 on error. */
int ib_metal_matmul_w4a8(ib_metal_ctx *ctx,
                          const void *weights,
                          const void *w_scales,
                          const void *x_q,
                          const void *x_scales,
                          void *out,
                          int M, int N);

/* fp32 → int8 quantization with per-128-group scale. Mirrors the CPU
 * `ib_quantize_input_int8_g128` exactly so chained Metal matmuls can
 * use the quantized output without CPU round-trip.
 *
 *   x        [N]        float — fp32 activations
 *   x_q      [N]        int8  — quantized output
 *   x_scales [N/128]    float — per-group scale (max|x| / 127)
 *
 * N must be a multiple of 128 for clean tiling (else last group is short).
 * Synchronous. Returns 0 on success. */
int ib_metal_quantize_input_int8_g128(ib_metal_ctx *ctx,
                                        const void *x,
                                        void *x_q, void *x_scales,
                                        int N);

/* Fused fp32-input matmul: quantize → matmul, both in ONE command buffer.
 *
 * This is the drop-in replacement for `tensor_matmul` on the GPU side.
 * The intermediate x_q + x_scales buffers are allocated once (or reused)
 * via ctx and released back to the buffer pool afterward.
 *
 * Eliminates the per-kernel dispatch overhead that Phase 2's microbench
 * identified as the bottleneck for TinyLlama-class workloads — both
 * kernels submit as ONE GPU command, paying only one cb commit/wait.
 *
 *   x        [N]      fp32 input  (GPU buffer)
 *   weights  [M, N/2] uint8       (GPU buffer)
 *   w_scales [M]      half        (GPU buffer)
 *   out      [M]      fp32 output (GPU buffer)
 *
 * `scratch_x_q` and `scratch_x_scales` may be NULL — the function will
 * allocate them internally and release. Pass non-NULL caller-owned
 * buffers for repeated calls to avoid allocation overhead. */
int ib_metal_matmul_w4a8_fp32_in(ib_metal_ctx *ctx,
                                  const void *x_fp32,
                                  const void *weights,
                                  const void *w_scales,
                                  void *out,
                                  void *scratch_x_q,        /* int8[N], or NULL */
                                  void *scratch_x_scales,   /* float[N/128], or NULL */
                                  int M, int N);

/* W4A8 with per-32-element block weight scales. w_scales is fp16 with
 * length M*(N/32). Same input quantization (per-128 INT8) as the
 * per-row variant. Closes the per-row outlier-clipping PPL gap on
 * Llama-3-class models. */
int ib_metal_matmul_w4a8_blk32_fp32_in(ib_metal_ctx *ctx,
                                         const void *x_fp32,
                                         const void *weights,
                                         const void *w_scales_blk32,
                                         void *out,
                                         void *scratch_x_q,
                                         void *scratch_x_scales,
                                         int M, int N);

/* INT8-weight matmul: out[m] = scale[m] * sum_n (W[m,n] * x[n]).
 *
 * Mirrors the CPU `matmul_int8` kernel exactly.
 *
 *   weights   [M, N] int8  — full bytes, no packing
 *   w_scales  [M]    half  — per-row weight scale (fp16)
 *   x         [N]    float — fp32 activations (NOT quantized)
 *   out       [M]    float
 *
 * Synchronous. Returns 0 on success, -1 on error. */
int ib_metal_matmul_int8_fp32_in(ib_metal_ctx *ctx,
                                   const void *x_fp32,
                                   const void *weights,
                                   const void *w_scales,
                                   void *out,
                                   int M, int N);

/* RMSNorm with fp16 weight. Mirrors the CPU kernel exactly:
 *   out[i] = x[i] * weight[i] / sqrt(mean(x[i]^2) + eps)
 *
 *   x       [N]  fp32 input
 *   weight  [N]  fp16 weight (matches IBF on-disk layout)
 *   out     [N]  fp32 output (may equal x for in-place — they alias safely
 *                because pass-2 reads x[i] and writes out[i] one-to-one).
 *
 * Single threadgroup per call. Synchronous. Returns 0 on success. */
int ib_metal_rmsnorm_fp16(ib_metal_ctx *ctx,
                           const void *x_fp32,
                           const void *weight_fp16,
                           void *out_fp32,
                           int N, float eps);

/* SiLU-gated multiply: out[i] = silu(gate[i]) * up[i].
 *   gate, up, out  [N]  fp32
 * Buffers may alias (out can equal gate or up). Synchronous. */
int ib_metal_silu_mul(ib_metal_ctx *ctx,
                       const void *gate_fp32,
                       const void *up_fp32,
                       void *out_fp32,
                       int N);

/* In-place Llama-style interleaved RoPE applied to one tensor laid out
 * as [n_heads, head_dim]. Caller dispatches twice (Q, K) for GQA.
 *
 *   tensor   fp32 buffer of size n_heads*head_dim, modified in place.
 *   pos      token position.
 *   theta    rope_theta (10000.0 for Llama). */
int ib_metal_rope_inplace(ib_metal_ctx *ctx,
                           void *tensor_fp32,
                           int n_heads, int head_dim,
                           int pos, float theta);

/* In-place softmax over n_rows rows of length row_len. Numerically
 * stable (subtracts row max). One threadgroup per row. */
int ib_metal_softmax_rows(ib_metal_ctx *ctx,
                            void *data_fp32,
                            int n_rows, int row_len);

/* Embedding lookup: out[i] = embeddings[token, i] (fp16→fp32). One
 * dispatch per token; for prefill, caller loops or batches in cb. */
int ib_metal_embed_lookup_fp16(ib_metal_ctx *ctx,
                                 const void *embeddings_fp16,
                                 int token,
                                 int hidden,
                                 void *out_fp32);

/* Full attention block on GPU. KV cache is fp16 layout [seq_len, kv_dim]
 * for both keys and values. Writes K, V into the cache at `pos`, then
 * computes scores, in-place softmax, and attention-weighted V — all
 * inside ONE command buffer (one commit/wait).
 *
 *   q         [n_heads * head_dim]      fp32   pre-RoPE'd query
 *   k         [n_kv_heads * head_dim]   fp32   pre-RoPE'd key
 *   v         [n_kv_heads * head_dim]   fp32   value
 *   k_cache   [seq_len, kv_dim]         fp16   in/out
 *   v_cache   [seq_len, kv_dim]         fp16   in/out
 *   scores    [n_heads, seq_len]        fp32   scratch (only [:, :pos+1] used)
 *   attn_out  [n_heads * head_dim]      fp32   output
 *
 * `seq_len` is the cache stride; `pos` is the current token position
 * (so we read/write rows 0..pos and the new row goes at pos).
 *
 * Returns 0 on success. Synchronous. */
int ib_metal_attention_block_fp16(ib_metal_ctx *ctx,
                                    const void *q_fp32,
                                    const void *k_fp32,
                                    const void *v_fp32,
                                    void *k_cache_fp16,
                                    void *v_cache_fp16,
                                    void *scores_fp32,
                                    void *attn_out_fp32,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int seq_len, int pos);

/* INT8 KV variant: same semantics as fp16 attention block but the KV
 * cache is stored as int8 weights with per-(pos, kv_head) fp32 scales,
 * matching libinferbit's kv_bits=8 layout. The new K/V row at `pos` is
 * quantized inside the kernel (using max(|x|)/127 per head). */
int ib_metal_attention_block_int8(ib_metal_ctx *ctx,
                                    const void *q_fp32,
                                    const void *k_fp32,
                                    const void *v_fp32,
                                    void *k_cache_int8,
                                    void *v_cache_int8,
                                    void *k_scales_fp32,
                                    void *v_scales_fp32,
                                    void *scores_fp32,
                                    void *attn_out_fp32,
                                    int n_heads, int n_kv_heads,
                                    int head_dim, int seq_len, int pos);

/* Element-wise residual: a[i] += b[i].  N-element fp32. */
int ib_metal_residual_add(ib_metal_ctx *ctx,
                            void *a_fp32, const void *b_fp32, int N);

/* ── Command recorder ─────────────────────────────────────────────────
 *
 * The standalone ib_metal_* dispatchers each open a command buffer,
 * encode one operation, commit, and waitUntilCompleted. Every commit/wait
 * pays ~100-200µs of dispatch overhead on Apple Silicon; for tiny ops
 * (rmsnorm at 2048, residual at 2048) this dominates total cost.
 *
 * The recorder lets callers assemble many kernels into ONE command
 * buffer, paying the dispatch overhead exactly once. Pattern:
 *
 *   ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
 *   ib_metal_rec_rmsnorm_fp16(r, ...);
 *   ib_metal_rec_matmul_w4a8_fp32_in(r, ...);
 *   ib_metal_rec_residual_add(r, ...);
 *   ib_metal_recorder_commit(r);  // commits + waits + frees recorder
 *
 * Each rec_* call records one encoder into the shared cb and returns
 * immediately. Encode order is the GPU dispatch order. */
typedef struct ib_metal_recorder ib_metal_recorder;
ib_metal_recorder *ib_metal_recorder_begin(ib_metal_ctx *ctx);
int ib_metal_recorder_commit(ib_metal_recorder *rec);

int ib_metal_rec_rmsnorm_fp16(ib_metal_recorder *rec,
                                const void *x_fp32,
                                const void *weight_fp16,
                                void *out_fp32,
                                int N, float eps);

int ib_metal_rec_residual_add(ib_metal_recorder *rec,
                                void *a_fp32, const void *b_fp32, int N);

int ib_metal_rec_matmul_w4a8_fp32_in(ib_metal_recorder *rec,
                                       const void *x_fp32,
                                       const void *weights,
                                       const void *w_scales,
                                       void *out,
                                       void *scratch_x_q,
                                       void *scratch_x_scales,
                                       int M, int N);

int ib_metal_rec_matmul_w4a8_blk32_fp32_in(ib_metal_recorder *rec,
                                             const void *x_fp32,
                                             const void *weights,
                                             const void *w_scales_blk32,
                                             void *out,
                                             void *scratch_x_q,
                                             void *scratch_x_scales,
                                             int M, int N);

/* Batched W4A8 blk32 matmul. x_fp32 is fp32[B][N], out is fp32[B][M].
 * scratch_x_q is char[B][N], scratch_x_scales is fp32[B][N/128]. Same
 * weight + scale layout as the unbatched variant. Single-pipeline
 * dispatch quantizes all B rows then runs the batched matmul. */
int ib_metal_rec_matmul_w4a8_blk32_batched_fp32_in(ib_metal_recorder *rec,
                                                     const void *x_fp32,
                                                     const void *weights,
                                                     const void *w_scales_blk32,
                                                     void *out,
                                                     void *scratch_x_q,
                                                     void *scratch_x_scales,
                                                     int B, int M, int N);

/* Tiled variant: weights for each output row are loaded into
 * threadgroup memory once and shared across TILE_B (=16) SIMD groups
 * within the threadgroup. ~TILE_B× weight bandwidth reduction over
 * the non-tiled batched kernel. Same arg shapes. */
int ib_metal_rec_matmul_w4a8_blk32_batched_tiled_fp32_in(ib_metal_recorder *rec,
                                                           const void *x_fp32,
                                                           const void *weights,
                                                           const void *w_scales_blk32,
                                                           void *out,
                                                           void *scratch_x_q,
                                                           void *scratch_x_scales,
                                                           int B, int M, int N);

/* simdgroup_matrix variant: uses Apple Silicon's 8x8 fp16 matrix
 * multiply intrinsic. Cooperatively dequants W (INT4 → fp16) and A
 * (INT8 → fp16) tiles into threadgroup memory, then runs hardware
 * matrix multiplies with fp32 accumulators. Requires B%8==0, M%8==0,
 * N%128==0. Returns -2 if shape doesn't fit. */
int ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_fp32_in(ib_metal_recorder *rec,
                                                            const void *x_fp32,
                                                            const void *weights,
                                                            const void *w_scales_blk32,
                                                            void *out,
                                                            void *scratch_x_q,
                                                            void *scratch_x_scales,
                                                            int B, int M, int N);

/* 4-SIMDgroup variant of simdmat: 16x16 output tile per threadgroup
 * with cooperative W/A dequant shared by 4 SIMD groups. Requires
 * B%16==0, M%16==0, N%128==0. */
int ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg_fp32_in(ib_metal_recorder *rec,
                                                                const void *x_fp32,
                                                                const void *weights,
                                                                const void *w_scales_blk32,
                                                                void *out,
                                                                void *scratch_x_q,
                                                                void *scratch_x_scales,
                                                                int B, int M, int N);

/* 16-SIMDgroup variant: 32x32 output tile per threadgroup, 4x4
 * sub-tile grid. Requires B%32==0, M%32==0, N%128==0. */
int ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg32_fp32_in(ib_metal_recorder *rec,
                                                                  const void *x_fp32,
                                                                  const void *weights,
                                                                  const void *w_scales_blk32,
                                                                  void *out,
                                                                  void *scratch_x_q,
                                                                  void *scratch_x_scales,
                                                                  int B, int M, int N);

/* Batched fused QKV (K=64 pipelined simdmat): Q+K+V matmuls in one
 * Metal compute-encoder dispatch. M_Q, M_KV all multiples of 32;
 * N % 128 == 0; B % 32 == 0. The 3-output dispatcher reuses the same
 * x_q / x_scales scratch buffers prepared once by the caller. */
int ib_metal_rec_matmul_w4a8_blk32_batched_qkv_simdmat_k64_fp32_in(
    ib_metal_recorder *rec,
    const void *x_fp32,
    const void *q_w, const void *q_s,
    const void *k_w, const void *k_s,
    const void *v_w, const void *v_s,
    void *q_out, void *k_out, void *v_out,
    void *scratch_x_q, void *scratch_x_scales,
    int B, int M_Q, int M_KV, int N);

/* Batched fused gate+up (K=64 pipelined simdmat): gate and up matmuls
 * in one Metal compute-encoder dispatch. M, B multiples of 32, N % 128. */
int ib_metal_rec_matmul_w4a8_blk32_batched_gateup_simdmat_k64_fp32_in(
    ib_metal_recorder *rec,
    const void *x_fp32,
    const void *gate_w, const void *gate_s,
    const void *up_w,   const void *up_s,
    void *gate_out, void *up_out,
    void *scratch_x_q, void *scratch_x_scales,
    int B, int M, int N);

/* MPS-hybrid prefill: fp16 weights × fp32 inputs batched matmul via
 * K=64 pipelined simdmat. Skips the INT4 unpack + per-element scale
 * multiply by reading pre-dequanted fp16 weights directly. Caller is
 * responsible for the upload-time dequant.
 * Shape constraints: B % 32 == 0, M % 32 == 0, N % 64 == 0. */
int ib_metal_rec_matmul_fp16w_fp32x_batched_simdmat_k64_fp32_in(
    ib_metal_recorder *rec,
    const void *x_fp32,
    const void *weights_fp16,
    void *out,
    int B, int M, int N);

/* Fused fp32-input variant of tg32: skips the separate INT8 quantize
 * pass + scratch round-trip. Reads activations directly as fp32 and
 * converts to fp16 in the cooperative load. Requires B%32==0, M%32==0. */
int ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg32_a16_fp32_in(
    ib_metal_recorder *rec,
    const void *x_fp32,
    const void *weights,
    const void *w_scales_blk32,
    void *out,
    int B, int M, int N);

/* 32-SIMDgroup variant: 64x32 output tile per threadgroup, 8x4
 * sub-tile grid, 1024 threads/TG (M4 max). Requires B%32==0, M%64==0. */
int ib_metal_rec_matmul_w4a8_blk32_batched_simdmat_tg64_fp32_in(ib_metal_recorder *rec,
                                                                  const void *x_fp32,
                                                                  const void *weights,
                                                                  const void *w_scales_blk32,
                                                                  void *out,
                                                                  void *scratch_x_q,
                                                                  void *scratch_x_scales,
                                                                  int B, int M, int N);

int ib_metal_rec_matmul_int8_fp32_in(ib_metal_recorder *rec,
                                       const void *x_fp32,
                                       const void *weights,
                                       const void *w_scales,
                                       void *out,
                                       int M, int N);

/* INT8 matmul via simdgroup_matrix (B=1 with internal 8x padding).
 * Targets the per-token output_head where M is huge. Requires
 * M%8==0, N%128==0; returns -2 on shape mismatch. */
int ib_metal_rec_matmul_int8_fp32_in_simdmat(ib_metal_recorder *rec,
                                               const void *x_fp32,
                                               const void *weights,
                                               const void *w_scales,
                                               void *out,
                                               int M, int N);

/* Batched INT8 matmul. x is fp32[B][N], out is fp32[B][M]. */
int ib_metal_rec_matmul_int8_fp32_in_batched(ib_metal_recorder *rec,
                                               const void *x_fp32,
                                               const void *weights,
                                               const void *w_scales,
                                               void *out,
                                               int B, int M, int N);

/* Batched per-row ops: B rows, length N (or kv_dim/intermediate). */
int ib_metal_rec_rmsnorm_fp16_batched(ib_metal_recorder *rec,
                                        const void *x_fp32,
                                        const void *weight_fp16,
                                        void *out_fp32,
                                        int B, int N, float eps);

int ib_metal_rec_residual_add_batched(ib_metal_recorder *rec,
                                        void *a_fp32, const void *b_fp32,
                                        int B, int N);

int ib_metal_rec_silu_mul_batched(ib_metal_recorder *rec,
                                    const void *gate_fp32,
                                    const void *up_fp32,
                                    void *out_fp32,
                                    int B, int N);

/* Batched RoPE: row b uses position (start_pos + b). */
int ib_metal_rec_rope_inplace_batched(ib_metal_recorder *rec,
                                        void *tensor_fp32,
                                        int B, int n_heads, int head_dim,
                                        int start_pos, float theta);

int ib_metal_rec_rope_inplace(ib_metal_recorder *rec,
                                void *tensor_fp32,
                                int n_heads, int head_dim,
                                int pos, float theta);

/* Fused Q+K RoPE: applies the in-place rotation to both Q and K in
 * one Metal dispatch. Saves 1 dispatch per layer. */
int ib_metal_rec_rope_inplace_qk(ib_metal_recorder *rec,
                                   void *q_fp32, void *k_fp32,
                                   int n_q_heads, int n_kv_heads,
                                   int head_dim,
                                   int pos, float theta);

int ib_metal_rec_silu_mul(ib_metal_recorder *rec,
                            const void *gate_fp32,
                            const void *up_fp32,
                            void *out_fp32,
                            int N);

/* Fused silu_mul + W4A8 blk32 matmul (for decode down_proj). Reads gate
 * and up activations directly (fp32), computes silu(gate[k]) * up[k]
 * inline and dots against the INT4 weight row. Saves one Metal dispatch
 * per layer. M/N constraints same as the unfused matmul. */
int ib_metal_rec_matmul_w4a8_blk32_dr_a32_silu_fp32_in(ib_metal_recorder *rec,
                                                         const void *gate_fp32,
                                                         const void *up_fp32,
                                                         const void *weights,
                                                         const void *w_scales,
                                                         void *out,
                                                         int M, int N);

/* Fused residual-add variant of W4A8 blk32 matmul: writes
 * out[m] = out[m] + dot_product instead of overwriting. Used to fuse
 * the post-matmul residual_add. Same buffer layout as _a32 but `out`
 * must already hold the residual value. */
int ib_metal_rec_matmul_w4a8_blk32_dr_a32_add_fp32_in(ib_metal_recorder *rec,
                                                       const void *x_fp32,
                                                       const void *weights,
                                                       const void *w_scales,
                                                       void *out,
                                                       int M, int N);

/* Q/K/V fused matmul: one Metal dispatch computes all three projections.
 * INT4 blk32 variant. Output partitioning by row: [Q M_Q rows][K M_KV
 * rows][V M_KV rows]. */
int ib_metal_rec_matmul_w4a8_blk32_dr_a32_qkv_fp32_in(ib_metal_recorder *rec,
                                                       const void *x_fp32,
                                                       const void *q_w,
                                                       const void *q_s,
                                                       const void *k_w,
                                                       const void *k_s,
                                                       const void *v_w,
                                                       const void *v_s,
                                                       void *q_out,
                                                       void *k_out,
                                                       void *v_out,
                                                       int M_Q, int M_KV, int N);

/* INT8 variant of the Q/K/V fused matmul (for mixed-precision IBFs
 * with INT8 q/k/v). Same arg order, weights/scales as INT8 + fp16. */
int ib_metal_rec_matmul_int8_fp32_in_qkv(ib_metal_recorder *rec,
                                           const void *x_fp32,
                                           const void *q_w,
                                           const void *q_s,
                                           const void *k_w,
                                           const void *k_s,
                                           const void *v_w,
                                           const void *v_s,
                                           void *q_out,
                                           void *k_out,
                                           void *v_out,
                                           int M_Q, int M_KV, int N);

/* Gate + Up fused matmul (INT4 blk32). 2 matmuls with same input and
 * same M into one dispatch. */
int ib_metal_rec_matmul_w4a8_blk32_dr_a32_gateup_fp32_in(ib_metal_recorder *rec,
                                                          const void *x_fp32,
                                                          const void *gate_w,
                                                          const void *gate_s,
                                                          const void *up_w,
                                                          const void *up_s,
                                                          void *gate_out,
                                                          void *up_out,
                                                          int M, int N);

/* RMSNorm + Gate + Up fused: each TG cooperatively computes the
 * inverse-RMS scalar from x_in, then uses (x_in * inv_rms * rms_weight)
 * as the activation for both gate and up matmuls. Saves one explicit
 * RMSNorm dispatch per layer. */
int ib_metal_rec_matmul_w4a8_blk32_dr_a32_rmsnorm_gateup_fp32_in(
    ib_metal_recorder *rec,
    const void *x_in_fp32,
    const void *rms_weight_fp16,
    const void *gate_w, const void *gate_s,
    const void *up_w,   const void *up_s,
    void *gate_out, void *up_out,
    int M, int N, float eps);

int ib_metal_rec_softmax_rows(ib_metal_recorder *rec,
                                void *data_fp32,
                                int n_rows, int row_len);

int ib_metal_rec_embed_lookup_fp16(ib_metal_recorder *rec,
                                     const void *embeddings_fp16,
                                     int token, int hidden,
                                     void *out_fp32);

/* Records the full attention block (kv_write + scores + softmax + weighted_v). */
int ib_metal_rec_attention_block_fp16(ib_metal_recorder *rec,
                                        const void *q_fp32,
                                        const void *k_fp32,
                                        const void *v_fp32,
                                        void *k_cache_fp16,
                                        void *v_cache_fp16,
                                        void *scores_fp32,
                                        void *attn_out_fp32,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int seq_len, int pos);

/* Batched fp16-KV attention: runs B prefill positions in 4 dispatches
 * (vs. 4×B in the per-position variant). Causal mask handled internally.
 * q/k/v are [B][...]; scores is [B][n_heads][start_pos+B]; attn_out is
 * [B][n_heads*head_dim]. */
int ib_metal_rec_attention_block_fp16_batched(ib_metal_recorder *rec,
                                                const void *q_fp32,
                                                const void *k_fp32,
                                                const void *v_fp32,
                                                void *k_cache_fp16,
                                                void *v_cache_fp16,
                                                void *scores_fp32,
                                                void *attn_out_fp32,
                                                int B, int n_heads, int n_kv_heads,
                                                int head_dim, int seq_len, int start_pos);

/* INT8 KV variant of the attention block. */
int ib_metal_rec_attention_block_int8(ib_metal_recorder *rec,
                                        const void *q_fp32,
                                        const void *k_fp32,
                                        const void *v_fp32,
                                        void *k_cache_int8,
                                        void *v_cache_int8,
                                        void *k_scales_fp32,
                                        void *v_scales_fp32,
                                        void *scores_fp32,
                                        void *attn_out_fp32,
                                        int n_heads, int n_kv_heads,
                                        int head_dim, int seq_len, int pos);

/* ── Real-model integration (Phase 7) ───────────────────────────────
 *
 * Upload an IBF-loaded model's weights into GPU buffers, then run the
 * per-token forward (all layers + final norm + lm_head) inside ONE
 * MTLCommandBuffer per call.
 *
 * Currently requires a uniformly-INT4 IBF: all matmul tensors bits=4
 * (q/k/v/o_proj, gate/up/down_proj, output_head), norms bits=16,
 * kv_bits=16. Embedding can be any bits (decoded on CPU per token).
 *
 * `model` is a libinferbit-loaded `inferbit_model*` (opaque here to
 * keep the header free of internal types). */
typedef struct ib_metal_model_buffers ib_metal_model_buffers;

ib_metal_model_buffers *ib_metal_upload_model(ib_metal_ctx *ctx,
                                                const void *inferbit_model);
void ib_metal_release_model(ib_metal_ctx *ctx, ib_metal_model_buffers *bufs);

/* After ib_metal_upload_model has placed every weight tensor on the GPU,
 * the only thing the CPU still needs from the IBF mmap is the token
 * embedding (used by cpu_embed_lookup). This function copies the embedding
 * bytes into a small malloc'd buffer, rebinds model->weight_data, and
 * munmaps the original IBF — capping CPU peak RSS at the embedding size.
 * Returns the number of bytes copied (>0) on success, 0 if not mmap'd
 * (no-op), or -1 on failure. Caller must NOT use any non-embedding tensor
 * via model->weight_data after this call. */
int ib_metal_strip_cpu_mmap(void *inferbit_model);

/* PQv2 K=256 half=2 matvec recorder (the stacked 2D codebook pyramid
 * differentiator). out[M] = row_scale[m] * sum_{c,s} cb[s][idx[c,s,m]] · x[c*G+s*half : ...].
 *
 * Inputs (all GPU buffers obtained from ib_metal_alloc):
 *   row_scale_fp16: [M] fp16
 *   cb_fp16:        [n_subchunks * K * half] fp16 (pre-decoded codebooks)
 *   indices_u8:     [n_chunks * n_subchunks * M] u8 (transposed for coalesced reads)
 *   x_fp32:         [N] fp32 activation
 *   out_fp32:       [M] fp32 result
 *
 * Requires K=256, half=2 (current production config). G and n_subchunks
 * are passed as constants. Returns 0 on success. */
int ib_metal_rec_matmul_pqv2_k256_half2(ib_metal_recorder *rec,
                                          const void *row_scale_fp16,
                                          const void *cb_fp16,
                                          const void *indices_u8,
                                          const void *x_fp32,
                                          void *out_fp32,
                                          int M, int N, int G, int n_subchunks);

/* fp16 weights × fp32 input → fp32 output. Plain matmul for lm_head
 * when stored as raw fp16 (PQv2 IBFs leave the head un-quantized). */
int ib_metal_rec_matmul_fp16w_fp32x(ib_metal_recorder *rec,
                                      const void *x_fp32,
                                      const void *w_fp16,
                                      void *out_fp32,
                                      int M, int N);

/* Fused PQv2 Q+K+V: one dispatch handles three projections sharing the
 * same fp32 input x. Returns 0 on success. */
int ib_metal_rec_matmul_pqv2_qkv_k256_half2(ib_metal_recorder *rec,
    const void *x_fp32,
    const void *q_rs, const void *q_cb, const void *q_idx, void *q_out,
    const void *k_rs, const void *k_cb, const void *k_idx, void *k_out,
    const void *v_rs, const void *v_cb, const void *v_idx, void *v_out,
    int M_q, int M_kv, int N, int G, int n_subchunks);

/* Fused PQv2 gate+up: one dispatch handles both projections sharing the
 * same fp32 input x. M_io is the shared output dimension. */
int ib_metal_rec_matmul_pqv2_gateup_k256_half2(ib_metal_recorder *rec,
    const void *x_fp32,
    const void *g_rs, const void *g_cb, const void *g_idx, void *g_out,
    const void *u_rs, const void *u_cb, const void *u_idx, void *u_out,
    int M_io, int N, int G, int n_subchunks);

/* PQv2 batched simdmat — uses Apple's simdgroup_matrix HW for the
 * prefill matmul, gathering PQv2 weights into a fp16 tile per K-chunk
 * before each 32×32 tile matmul. Returns -2 when shape constraints
 * (M%32, B%32, N%64) aren't met; caller falls back. */
int ib_metal_rec_matmul_pqv2_batched_simdmat(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int B, int M, int N, int G, int n_subchunks);

/* PQv2 batched simdmat with B_BLOCK=8 (4 SIMDgroups per TG, 32×8 output
 * tile). Closes the prefill speed cliff for prompt sizes that aren't a
 * multiple of 32 — works at any B%8==0. Returns -2 when M%32, B%8, or
 * N%64 not satisfied. */
int ib_metal_rec_matmul_pqv2_batched_simdmat_b8(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int B, int M, int N, int G, int n_subchunks);

/* PQv2 batched simdmat B_BLOCK=16: 32×16 output tile, 8 SGs/TG, 4 KB+8 KB
 * TG memory. Halves W loads vs b8 for any B%16. Smooths the prompt-size
 * speed curve at B in {16, 48, 80, ...}. */
int ib_metal_rec_matmul_pqv2_batched_simdmat_b16(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int B, int M, int N, int G, int n_subchunks);

/* PQv2 batched simdmat tg64: 64×32 output tile, 32 SIMDgroups per TG.
 * Better W-load amortization. Requires M%64 + B%32 + N%64. */
int ib_metal_rec_matmul_pqv2_batched_simdmat_tg64(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int B, int M, int N, int G, int n_subchunks);

/* PQv2 batched simdmat tg16: 16×32 output tile, 8 SIMDgroups per TG,
 * 6 KB TG memory. Targets higher per-shader-core TG occupancy for
 * gather-bound throughput. Requires M%16 + B%32 + N%64. */
int ib_metal_rec_matmul_pqv2_batched_simdmat_tg16(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int B, int M, int N, int G, int n_subchunks);

/* Greedy argmax over logits → int32 token id (single TG, 32 lanes). */
int ib_metal_rec_argmax_logits(ib_metal_recorder *rec,
    const void *logits_fp32, void *out_token_i32, int vocab);

/* PQv2 embedding lookup on GPU: token id → fp32 embedding row. */
int ib_metal_rec_embed_lookup_pqv2(ib_metal_recorder *rec,
    const void *in_token_i32,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    void *out_fp32,
    int M_vocab, int N, int G, int n_subchunks);

/* PQv2 decode (B=1) via Apple simdgroup_matrix. X-broadcast trick:
 * fills an 8x8 X tile with replicated x[k..k+7] rows so the matrix
 * unit can do 8-row matvec partials per instruction. Returns -2 if
 * M%8 or N%64 not satisfied; caller falls back to SIMD-coop. */
int ib_metal_rec_matmul_pqv2_simdmat_decode(ib_metal_recorder *rec,
    const void *row_scale_fp16, const void *cb_fp16, const void *indices_u8,
    const void *x_fp32, void *out_fp32,
    int M, int N, int G, int n_subchunks);

/* Batched variant for prefill: x is [B][N], out is [B][M]. Each TG
 * dispatch handles one (m_block, b) pair so LUT-build is per-token. */
int ib_metal_rec_matmul_pqv2_k256_half2_batched(ib_metal_recorder *rec,
                                                  const void *row_scale_fp16,
                                                  const void *cb_fp16,
                                                  const void *indices_u8,
                                                  const void *x_fp32,
                                                  void *out_fp32,
                                                  int B, int M, int N,
                                                  int G, int n_subchunks);

/* Forward one token. `cpu_embed_in` is fp32[hidden] — the result of
 * the CPU-side embedding lookup for the current token. `pos` is the
 * absolute token position. `logits_out` receives fp32[vocab].
 * Returns 0 on success. */
int ib_metal_forward_token(ib_metal_ctx *ctx,
                            ib_metal_model_buffers *bufs,
                            const float *cpu_embed_in,
                            int pos,
                            float *logits_out);

/* Batched prefill forward: runs B tokens through all layers in a single
 * command buffer, batching the per-layer matmuls (Q/K/V/O/gate/up/down)
 * across all B tokens to amortize weight-load bandwidth. Per-token ops
 * (RMSNorm/RoPE/attention/residual/silu) still run in a recorded loop
 * — they're cheap per call and need per-position state.
 *
 * Inputs:
 *   cpu_embeds_in : float[B][hidden] — CPU-side embedding lookups
 *   start_pos     : absolute position of token 0 (KV cache slot)
 *   last_logits_out: float[vocab] — only the last token's logits are
 *                    materialized (typical prefill use case)
 *
 * Constraints (current MVP):
 *   - All q/k/v/o/gate/up/down + output_head must be INT4 blk32. Returns
 *     -2 if any tensor is per-row INT4 or INT8 — caller should fall back
 *     to ib_metal_forward_token.
 *   - B must be ≤ bufs->b_max (set via IB_PREFILL_BMAX env at upload).
 *
 * Returns 0 on success, -1 on hard error, -2 if model layout incompatible. */
int ib_metal_forward_prefill(ib_metal_ctx *ctx,
                              ib_metal_model_buffers *bufs,
                              const float *cpu_embeds_in,
                              int n_tokens, int start_pos,
                              float *last_logits_out);

/* Resets every layer's KV cache write position back to 0 — used between
 * generation runs that don't share a prefix. */
void ib_metal_reset_kv(ib_metal_model_buffers *bufs);

/* GPU drive mode (doc 32): commit + wait the recorder's current
 * command buffer, then allocate a fresh one. Same recorder handle
 * remains valid for subsequent rec_* calls. Used to serialize a
 * per-matmul streaming pattern where CPU preads weight pages into a
 * shared MTLBuffer scratch between dispatches. */
int ib_metal_recorder_checkpoint(ib_metal_recorder *rec);

/* Async-commit variant (doc-35 feature 4): commit without waiting,
 * allocate fresh CB, return a handle for later wait. Used to overlap
 * CPU pread/transpose with GPU compute on the prior CB. */
void *ib_metal_recorder_commit_async(ib_metal_recorder *rec);
int   ib_metal_recorder_wait_committed(void *cb_handle);

/* All-logits variant of forward_prefill: outputs per-position logits
 * for all n_tokens (vs the standard last-token only). Used by
 * speculative decoding to verify each draft token in one batched
 * forward pass. n_tokens is rounded up to a multiple of 32 internally
 * for the batched lm_head simdmat; padding rows are computed but
 * discarded. all_logits_out must be sized for n_tokens × vocab fp32. */
int ib_metal_forward_prefill_logits_all(ib_metal_ctx *ctx,
                                          ib_metal_model_buffers *b,
                                          const float *cpu_embeds_in,
                                          int n_tokens, int start_pos,
                                          float *all_logits_out);

/* Phase 3.1: variant of forward_prefill_logits_all that ALSO captures
 * each layer's post-residual hidden state to caller-provided buffers.
 * hidden_states_out is an array of num_layers float pointers; each
 * non-NULL pointer must hold at least n_tokens * hidden floats. NULL
 * entries (including hidden_states_out itself == NULL) are skipped.
 *
 * Capture forces a CB drain per captured layer — slower than the
 * no-capture path. Intended for Phase 4 DFlash hybrid speculative
 * orchestration, not fast-path inference. */
int ib_metal_forward_prefill_logits_all_ex(ib_metal_ctx *ctx,
                                             ib_metal_model_buffers *b,
                                             const float *cpu_embeds_in,
                                             int n_tokens, int start_pos,
                                             float *all_logits_out,
                                             float **hidden_states_out);

/* Paginated greedy decode: runs `n_steps` autoregressive forwards on the
 * GPU in a single command buffer, with argmax + embedding-lookup also
 * on the GPU. CPU pays only one waitUntilCompleted (instead of n_steps).
 *
 * `init_input_embed_fp32`: caller pre-decodes the first input token's
 * fp32 embedding (so the first iteration is identical to forward_token).
 * For step i>0 the kernel feeds back the previous-step argmax via the
 * GPU embed-lookup kernel — requires the model's token_embedding to be
 * PQ-encoded (which the v6/PQv2 IBF is).
 *
 * `start_pos`: position of the first step's KV write. KV cache will be
 * extended to start_pos + n_steps - 1.
 *
 * `out_tokens[n_steps]`: filled with the int32 argmax tokens.
 *
 * Returns 0 on success; -2 if token_embedding isn't PQ-encoded; -1 on
 * other errors. */
int ib_metal_forward_decode_n(ib_metal_ctx *ctx,
                               ib_metal_model_buffers *bufs,
                               const float *init_input_embed_fp32,
                               int start_pos, int n_steps,
                               int *out_tokens);

#ifdef __cplusplus
}
#endif

#endif /* IB_METAL_RUNTIME_H */
