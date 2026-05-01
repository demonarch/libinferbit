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
    IB_PQ_FMT_PYRAMID,   /* "pq2d_v1_pyramid" — conditional pyramid:
                          * L2 codebook flattened as [K_outer * K_inner, G/2];
                          * indices i2 are combined as i1*K_inner + i2_local.
                          * K_l2 unrestricted (vs L1_L2 which requires {16,64,K}).
                          * Reconstruct path is identical to L1_L2 — only the
                          * validation differs. */
} ib_pq_format;

typedef struct {
    /* Shape and config */
    int M;
    int N;
    int G;
    int K;
    int K_l2;        /* L2 codebook size; 0 = same as K (for n_levels==1) */
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

    /* Indices: M rows × C columns where C = (N - n_outlier) / G.
     * L2 indices may be 1 byte (K_l2 ≤ 256) or 2 bytes (PYRAMID with
     * K_l2 > 256). l2_idx_bytes encodes which. Packed 4-bit (K_l2==16)
     * uses 1 byte per chunk pair regardless. */
    uint8_t* indices_l1_l;
    uint8_t* indices_l1_r;
    uint8_t* indices_l2_l;     /* NULL if n_levels == 1; raw bytes — cast per l2_idx_bytes */
    uint8_t* indices_l2_r;     /* NULL if n_levels == 1; raw bytes — cast per l2_idx_bytes */
    int C;                     /* chunks per row */
    int l2_idx_bytes;          /* 1 = uint8 (default); 2 = uint16 (PYRAMID K_l2>256) */
    int l2_packed_bits;        /* 0 = unpacked (default); >0 = N-bit packed on
                                * disk, unpacked to uint16 in RAM at load. */

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

/* Raw (non-PQ) tensor stored alongside PQ tensors in the IBF v5
 * file. Used for token embeddings, RMSNorm weights, and any other
 * model state needed by inferbit_pq_forward that isn't PQ-quantized. */
typedef enum {
    IB_RAW_F32 = 0,
    IB_RAW_F16 = 1,
    IB_RAW_I32 = 2,
    IB_RAW_I16 = 3,
    IB_RAW_I8  = 4,
    IB_RAW_U8  = 5,
} ib_raw_dtype;

typedef struct {
    char*  name;          /* heap-allocated */
    void*  data;          /* heap-allocated copy (or mmap pointer) */
    int    dtype;         /* ib_raw_dtype */
    int    ndim;
    int    shape[4];
    size_t size_bytes;
    int    _owns_data;    /* 1 if heap-allocated and we should free it */
} ib_pq_raw_tensor;

/* Multi-tensor IBF v5 loader. Returns an array of tensors and their
 * names. Caller must free both via ib_pq_multi_free. */
typedef struct {
    int n;
    char** names;          /* heap-allocated strings */
    ib_pq_tensor* tensors; /* parallel array */
    void* _mmap_base;      /* non-NULL when mmap-backed (Path D) */
    size_t _mmap_size;
    /* Phase 9: raw tensors + free-form JSON config. NULL/0/n_raw=0 if
     * the file was written without them (backward compatible). */
    int    n_raw;
    ib_pq_raw_tensor* raw_tensors;
    char*  config_json;    /* heap-allocated NUL-terminated, NULL if absent */
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

/* Phase 2: cache-resident streaming-precompute matmul.
 *
 * Same math as ib_pq_matmul_fp32 but restructured so codebook ↔ activation
 * dot-product tables are computed PER CHUNK and stay in L1/L2 cache for
 * the whole-row accumulation pass over that chunk. Replaces the existing
 * "all-chunks-at-once" precompute that blows cache on big-K tensors.
 *
 * Per-chunk working set: (K + K_l2) × 2 floats ≈ a few KB; fits in L1
 * even on Raspberry Pi (64 KB L1). Index streams are sequential reads.
 *
 * Numerically equivalent to ib_pq_matmul_fp32 (FP32 throughout). Use
 * for performance; the existing entry point remains for verification. */
int ib_pq_matmul_fp32_streaming(const ib_pq_tensor* t, const float* x, float* out);

/* Phase 5 helpers — top-K lm_head two-stage.
 *
 * L1-only matmul: skips the L2 codebook contribution. Cheap pass over
 * full M output rows; used as the candidate filter. Output is a coarse
 * approximation to the full pyramid logits but preserves rank structure
 * well (validated test 35: 100% argmax coverage at top-32 on 32K vocab).
 */
int ib_pq_matmul_fp32_l1_only(const ib_pq_tensor* t, const float* x, float* out);

/* Subset matmul: full-pyramid logits computed only for the selected
 * row indices. Used as the refinement pass after top-K extraction.
 * row_indices: array of n_rows int32 row indices; out: float[n_rows]. */
int ib_pq_matmul_fp32_subset(const ib_pq_tensor* t, const float* x,
                              const int32_t* row_indices, int n_rows,
                              float* out);

/* Top-K orchestrator: cheap L1-only pass → top-K extract → full pyramid
 * on the K candidates → fill out_logits[K] and out_token_ids[K] sorted
 * descending by refined logit. Caller-allocated buffers must hold K
 * elements each. K ≤ M. Returns 0 on success.
 *
 * For sampling: caller applies temperature/top-P over out_logits[K]
 * and indexes into out_token_ids[K]. */
int ib_pq_lm_head_topk(const ib_pq_tensor* t, const float* x, int K_top,
                        float* out_logits, int32_t* out_token_ids);

/* Phase 4: activation-sparse streaming matmul.
 *
 * Skips chunks whose input activation values (in x) are all below
 * `act_threshold` in absolute value. Designed for the down_proj
 * matmul whose input (post-SwiGLU intermediate) is ~50% near-zero
 * per token (test 33). Skipping zero-contributing chunks delivers
 * ~2× MLP compute reduction with no quality loss.
 *
 * act_threshold ≤ 0 disables the sparsity check (equivalent to
 * ib_pq_matmul_fp32_streaming).
 */
int ib_pq_matmul_fp32_streaming_sparse(const ib_pq_tensor* t, const float* x,
                                        float* out, float act_threshold);

/* Phase 8.F: variance-bounded L2 skip.
 *
 * Per (output_row, chunk) pair, decides whether to apply L2 based on
 * a precomputed per-cluster ||C2[k]||_max bound vs the L1 contribution
 * magnitude. Skip L2 when the L1 contribution dominates the L2 bound.
 *
 * skip_threshold ≥ 0: ratio above which L2 is skipped. 0 = always
 * apply L2 (equivalent to streaming matmul). Larger = more aggressive
 * skip, lower compute, higher quality cost. Test 41b found ~1pp PPL
 * edge over random skip at low rates; mechanism is marginal but
 * cheap once the bound table is precomputed.
 */
int ib_pq_matmul_fp32_streaming_l2skip(const ib_pq_tensor* t, const float* x,
                                        float* out, float skip_threshold);

/* Phase 6: persistent decode cache for cross-token reuse.
 *
 * The fp16→fp32 codebook decode and inner_cols build happen once on
 * cache creation; cached matmul variants skip that prelude on every
 * call. Intended use: build one cache per (tensor) at model load,
 * reuse across all token forward passes.
 */
typedef struct ib_pq_lut_cache ib_pq_lut_cache;

int  ib_pq_lut_cache_create(const ib_pq_tensor* t, ib_pq_lut_cache** out);
void ib_pq_lut_cache_free(ib_pq_lut_cache* c);

int ib_pq_matmul_fp32_streaming_cached(const ib_pq_tensor* t,
                                        const ib_pq_lut_cache* cache,
                                        const float* x, float* out);
int ib_pq_matmul_fp32_streaming_l2skip_cached(const ib_pq_tensor* t,
                                                const ib_pq_lut_cache* cache,
                                                const float* x, float* out,
                                                float skip_threshold);

/* Phase 3: INT8 activations on top of cache.
 *
 * Quantizes the cached fp32 codebooks to int8 per-row scales and exposes
 * a streaming matmul that quantizes x_chunk to int8 per chunk and uses
 * NEON dotprod (ARM_FEATURE_DOTPROD) / AVX2 cascade for the inner dot.
 * Accuracy budget: ~10-12 effective bits per product after scaling.
 *
 * Call once on an existing cache before using the int8 matmul; idempotent.
 */
int ib_pq_lut_cache_quantize_int8(ib_pq_lut_cache* cache);
int ib_pq_matmul_fp32_streaming_int8_cached(const ib_pq_tensor* t,
                                              const ib_pq_lut_cache* cache,
                                              const float* x, float* out);

/* Phase 9: per-tensor cache fleet for a multi-tensor IBF.
 * Builds one ib_pq_lut_cache per tensor; lookup-by-name in O(1).
 * Owns and frees the underlying caches.
 */
typedef struct ib_pq_multi_caches ib_pq_multi_caches;

int  ib_pq_multi_caches_create(const ib_pq_multi* multi, ib_pq_multi_caches** out);
void ib_pq_multi_caches_free(ib_pq_multi_caches* mc);
const ib_pq_lut_cache* ib_pq_multi_caches_get(const ib_pq_multi_caches* mc, const char* name);
int  ib_pq_multi_caches_quantize_all_int8(ib_pq_multi_caches* mc);

/* ── Session: owns IBF + cache fleet + per-tensor policy. ──
 *
 * Design goal: dispatch decisions live in C, not in the language wrapper.
 * The wrapper opens a session, optionally tunes per-tensor policies, and
 * issues matmul-by-name calls. The session picks the right kernel.
 */

typedef enum {
    IB_PQ_VARIANT_STREAMING = 0,  /* default, full pyramid */
    IB_PQ_VARIANT_L1_ONLY,        /* skip L2 entirely (cheap, lossy) */
    IB_PQ_VARIANT_L2SKIP,         /* variance-bounded L2 skip */
    IB_PQ_VARIANT_SPARSE,         /* skip near-zero activation chunks */
    IB_PQ_VARIANT_INT8,           /* int8 codebook + activations */
} ib_pq_variant;

typedef struct {
    int   variant;            /* ib_pq_variant */
    float skip_threshold;     /* for L2SKIP */
    float act_threshold;      /* for SPARSE */
} ib_pq_policy;

typedef struct ib_pq_session ib_pq_session;

int  ib_pq_session_open(const char* ibf_path, ib_pq_session** out);
void ib_pq_session_close(ib_pq_session* s);

int  ib_pq_session_set_default_policy(ib_pq_session* s, ib_pq_policy p);
int  ib_pq_session_set_policy(ib_pq_session* s, const char* name, ib_pq_policy p);

/* Matmul by tensor name. Variant comes from the per-tensor policy
 * (or the session default if none set). Out is M floats.
 */
int  ib_pq_session_matmul(ib_pq_session* s, const char* name,
                           const float* x, float* out);

/* Top-K lm_head two-stage. K_top values in out_logits + ids, sorted desc. */
int  ib_pq_session_lm_head_topk(ib_pq_session* s, const char* name,
                                 const float* x, int K_top,
                                 float* out_logits, int32_t* out_ids);

/* Read-only metadata for the wrapper (so it doesn't need to redeclare
 * the IbPqTensor struct just to know shape). */
int  ib_pq_session_tensor_shape(const ib_pq_session* s, const char* name,
                                 int* out_M, int* out_N);
int  ib_pq_session_tensor_count(const ib_pq_session* s);
const char* ib_pq_session_tensor_name(const ib_pq_session* s, int i);

/* Phase 9: raw (non-PQ) tensor access. Returns 0 on success.
 * out_data borrows from the session — do not free. */
int  ib_pq_session_raw_count(const ib_pq_session* s);
const char* ib_pq_session_raw_name(const ib_pq_session* s, int i);
int  ib_pq_session_raw_get(const ib_pq_session* s, const char* name,
                            const void** out_data, int* out_dtype,
                            int* out_shape, int* out_ndim);

/* Phase 9: free-form JSON config string (NUL-terminated). NULL if absent. */
const char* ib_pq_session_config_json(const ib_pq_session* s);

/* ── Phase 9: forward-pass primitives (no PQ inside) ──
 * Used to assemble inferbit_pq_forward: matmuls go through the session,
 * everything else (norm, rotary, activation, residual, attention) uses
 * these. Pure C, scalar-then-SIMD where it matters.
 */

/* RMSNorm: out[i] = x[i] * weight[i] / sqrt(mean(x*x) + eps).
 * Layer-by-layer in float; weight is fp32. */
void ib_rmsnorm_f32(float* out, const float* x, const float* weight,
                     int hidden, float eps);

/* SwiGLU activation in place (or to out): out[i] = silu(gate[i]) * up[i]
 *   silu(x) = x / (1 + exp(-x))
 */
void ib_silu_gate_f32(float* out, const float* gate, const float* up, int n);

/* Residual add: x[i] += delta[i]. */
void ib_residual_add_f32(float* x, const float* delta, int n);

/* RoPE (rotary positional embedding). Rotates x in pairs (x[2i], x[2i+1])
 * by angle pos * theta^(-2i/head_dim). x is shaped [n_heads * head_dim],
 * head_dim must be even. theta is the rope base (typically 10000.0).
 * For a single position. For batched / cached positions call once per pos. */
void ib_rope_f32(float* x, int n_heads, int head_dim, int pos, float theta);

/* Softmax in place over n elements. Numerically stable (subtracts max). */
void ib_softmax_f32(float* x, int n);

/* Float16 helpers (IEEE 754 binary16). Pure software, portable. */
float    ib_fp16_to_fp32(uint16_t h);
uint16_t ib_fp32_to_fp16(float f);

#ifdef __cplusplus
}
#endif
#endif
