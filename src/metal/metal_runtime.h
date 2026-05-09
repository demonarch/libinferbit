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

#ifdef __cplusplus
}
#endif

#endif /* IB_METAL_RUNTIME_H */
