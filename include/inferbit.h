/*
 * inferbit.h — Public C API for libinferbit
 *
 * This is the sole public header. All interaction with libinferbit
 * goes through the functions declared here.
 */

#ifndef INFERBIT_H
#define INFERBIT_H

#include <stddef.h>
#include <stdint.h>

/* DLL export/import for Windows */
#ifdef _WIN32
  #ifdef INFERBIT_BUILD_DLL
    #define IB_API __declspec(dllexport)
  #else
    #define IB_API __declspec(dllimport)
  #endif
#else
  #define IB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ────────────────────────────────────────────────── */

#define INFERBIT_VERSION_MAJOR 0
#define INFERBIT_VERSION_MINOR 2
#define INFERBIT_VERSION_PATCH 3
#define INFERBIT_VERSION_STRING "0.2.3"

IB_API const char* inferbit_version(void);
IB_API int         inferbit_version_major(void);
IB_API int         inferbit_version_minor(void);
IB_API int         inferbit_version_patch(void);

/* ── Error codes ────────────────────────────────────────────── */

#define INFERBIT_OK              0
#define INFERBIT_ERROR_LOAD     -1
#define INFERBIT_ERROR_FORMAT   -2
#define INFERBIT_ERROR_MEMORY   -3
#define INFERBIT_ERROR_CONTEXT  -4
#define INFERBIT_ERROR_PARAM    -5
#define INFERBIT_ERROR_INTERNAL -6

IB_API const char* inferbit_last_error(void);

/* ── Opaque types ───────────────────────────────────────────── */

typedef struct inferbit_model  inferbit_model;
typedef struct inferbit_config inferbit_config;

/* Backend dispatch hint for a single matmul (Stage 5d of
 * docs/v2/00_CORRECTION.md). Tagged per-tensor via
 * `ib_tensor_meta.preferred_backend` at load time; at forward time the
 * dispatcher reads this hint and routes the matmul to the CPU kernel
 * (default) or to the Metal recorder for one-shot synchronous dispatch.
 *
 * AUTO  (0): no hint — run on whichever backend the surrounding forward
 *            path is using (CPU forward → CPU; Metal forward → GPU).
 *            v1 default for every tensor, so the unmodified behaviour
 *            is bit-identical to pre-hybrid.
 * CPU   (1): force the CPU kernel even when a GPU context is available.
 * METAL (2): route this matmul through Metal — the surrounding forward
 *            stays where it is. The dispatcher lazily creates the Metal
 *            context the first time it sees a METAL-tagged tensor.
 *
 * v1 knob: set `IB_HYBRID_FFN_GPU=1` at load time and the loader tags
 * every FFN matmul (gate_proj / up_proj / down_proj of every layer)
 * METAL. Attention stays AUTO. */
typedef enum {
    INFERBIT_BACKEND_AUTO  = 0,
    INFERBIT_BACKEND_CPU   = 1,
    INFERBIT_BACKEND_METAL = 2,
} inferbit_backend;

/* ── Configuration ──────────────────────────────────────────── */

IB_API inferbit_config* inferbit_config_create(void);
IB_API void             inferbit_config_free(inferbit_config* config);

IB_API void inferbit_config_set_threads(inferbit_config* config, int threads);
IB_API void inferbit_config_set_context_length(inferbit_config* config, int length);
IB_API void inferbit_config_set_kv_cache_dynamic(inferbit_config* config, int dynamic);

/* Rotating KV-cache window. 0 = full causal cache (default). >0 = each
 * layer's KV cache is a ring buffer of `window` token slots; logical
 * position p maps to physical slot p % window, and attention only
 * attends to the most recent `window` positions. Peak KV RAM becomes
 * O(window) instead of O(context_length) — the long-context bounded-RAM
 * lever. Trade-off: attention horizon is capped at `window` tokens. */
IB_API void inferbit_config_set_kv_window(inferbit_config* config, int window);

/* Native parse mode (dev/debug only) */
IB_API void inferbit_config_set_native_parse(inferbit_config* config, int enabled);
IB_API void inferbit_config_set_native_bits(inferbit_config* config, int bits);

/* KV-cache storage format (Stage 3b of docs/v2/00_CORRECTION.md).
 *
 * Selects the on-cache representation of K/V activations. Layers over
 * the legacy integer `kv_bits` knob carried in the .ibf header: when
 * set explicitly, this format choice wins; otherwise FP16 is the
 * default and the loader maps it to whatever the header asked for
 * (fp32/fp16 → FP16 here, 8 → INT8).
 *
 *   INFERBIT_KV_FP16 — raw fp32 / fp16 K/V backing. Default.
 *   INFERBIT_KV_INT8 — per-head int8 + fp32 scale. 2× compression vs FP16.
 *   INFERBIT_KV_PQ8  — PQ-encoded K/V indices, decoded inline on read.
 *                     v1: falls back to INT8 with a one-shot stderr
 *                     warning. On-line codebook fitting at write time
 *                     is a future enhancement; see src/forward.c for
 *                     the write/read sites that will host the encode/
 *                     decode hooks. */
typedef enum {
    INFERBIT_KV_FP16 = 0,
    INFERBIT_KV_INT8 = 1,
    INFERBIT_KV_PQ8  = 2,
} inferbit_kv_format;

IB_API void inferbit_config_set_kv_format(inferbit_config* config, inferbit_kv_format format);

/* ── Model lifecycle ────────────────────────────────────────── */

IB_API inferbit_model* inferbit_load(const char* path, const inferbit_config* config);
IB_API void            inferbit_free(inferbit_model* model);

/* ── Sampling parameters ────────────────────────────────────── */

typedef struct {
    float temperature;
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   max_tokens;
    int   seed;
} inferbit_sample_params;

IB_API inferbit_sample_params inferbit_default_sample_params(void);

/* ── Generation ─────────────────────────────────────────────── */

IB_API int inferbit_generate(
    inferbit_model*        model,
    const int32_t*         input_tokens,
    int                    num_input_tokens,
    int32_t*               out_tokens,
    int                    max_out_tokens,
    inferbit_sample_params params
);

typedef int (*inferbit_stream_callback)(int32_t token, void* ctx);

IB_API int inferbit_generate_stream(
    inferbit_model*        model,
    const int32_t*         input_tokens,
    int                    num_input_tokens,
    inferbit_stream_callback callback,
    void*                  ctx,
    inferbit_sample_params params
);

IB_API int inferbit_forward(
    inferbit_model*  model,
    const int32_t*   tokens,
    int              num_tokens,
    float*           out_logits,
    int              vocab_size
);

/* ── Hidden-state capture (doc 36 phase 4.1) ────────────────────
 *
 * Prefill forward that ALSO returns per-layer post-residual hidden
 * states — the hook the DFlash hybrid orchestrator needs to condition
 * a draft model on the target's mid-stack activations.
 *
 *   tokens / n_tokens : input token IDs.
 *   layer_ids         : which layer outputs to capture (0-based).
 *   n_layer_ids       : length of layer_ids.
 *   hiddens_out       : caller-allocated, laid out
 *                       [n_layer_ids][n_tokens][hidden_size] fp32.
 *                       Pass NULL (with n_layer_ids 0) for logits only.
 *   logits_out        : caller-allocated [n_tokens][vocab_size] fp32.
 *
 * Runs on the Metal backend; the GPU context is created lazily on the
 * first call and cached on the model. Returns INFERBIT_OK or an error
 * code. */
IB_API int inferbit_forward_with_hiddens(
    inferbit_model*  model,
    const int32_t*   tokens,
    int              n_tokens,
    const int*       layer_ids,
    int              n_layer_ids,
    float*           hiddens_out,
    float*           logits_out
);

/* Evenly-spaced target-layer selection (mirrors dflash's
 * build_target_layer_ids). Picks n_draft_layers indices spread across
 * the target's depth, skipping the first/last 3 layers. Writes ascending
 * indices into out_ids (caller provides n_draft_layers ints of space).
 * Returns the count actually written. */
IB_API int inferbit_build_target_layer_ids(
    int  n_target_layers,
    int  n_draft_layers,
    int* out_ids
);

/* ── KV-cache control ───────────────────────────────────────── */

IB_API void inferbit_kv_clear(inferbit_model* model);
IB_API void inferbit_kv_truncate(inferbit_model* model, int length);
IB_API int  inferbit_kv_length(const inferbit_model* model);

/* ── Model info ─────────────────────────────────────────────── */

IB_API const char* inferbit_model_architecture(const inferbit_model* model);
IB_API int         inferbit_model_num_layers(const inferbit_model* model);
IB_API int         inferbit_model_hidden_size(const inferbit_model* model);
IB_API int         inferbit_model_vocab_size(const inferbit_model* model);
IB_API int         inferbit_model_max_context(const inferbit_model* model);
IB_API int         inferbit_model_default_bits(const inferbit_model* model);
IB_API size_t      inferbit_model_weight_memory(const inferbit_model* model);
IB_API size_t      inferbit_model_kv_memory(const inferbit_model* model);
IB_API size_t      inferbit_model_total_memory(const inferbit_model* model);

/* Force the backend routing decision now (uploads the model to the GPU if
 * Metal is available + the model is GPU-capable). Without this the upload
 * happens lazily on the first forward, spiking time-to-first-token. Safe
 * to call repeatedly — a no-op once resolved. Returns 0 always. */
IB_API int         inferbit_model_warmup(inferbit_model* model);
/* Returns the backend the model's forward path will use: "metal" (GPU-
 * routed) or "cpu". Resolves the (otherwise lazy) routing decision, same
 * as inferbit_model_warmup. */
IB_API const char* inferbit_model_backend(inferbit_model* model);

/* ── Speculative decoding ───────────────────────────────────── */

IB_API void inferbit_set_draft_model(inferbit_model* model, inferbit_model* draft, int draft_tokens);
IB_API void inferbit_unset_draft_model(inferbit_model* model);

/* Prompt-lookup speculation: no external draft model required.
 *
 * Each decode step, the last `ngram` output tokens are searched against the
 * running history (prompt + generated tokens). On a match, the `k` tokens
 * following the earliest match are used as draft candidates, verified in a
 * single forward pass over the main model. Cheapest form of speculation —
 * useful for workloads with repeated structure (code completion, retrieval-
 * augmented generation, translation, repeated templated output).
 *
 * Set ngram = 0 to disable. Typical values: ngram = 2 or 3, k = 4 to 10.
 * Works with greedy decoding (temperature < 0.01). Mutually exclusive with
 * external-draft speculation — an external draft, if set, takes precedence. */
IB_API void inferbit_set_prompt_lookup(inferbit_model* model, int ngram, int k);

/* ── Conversion ─────────────────────────────────────────────── */

/* Output quantization family for inferbit_convert(). v0.4.1 ships only
 * INT4; PQv2 / pyramid are Stage 1 of docs/v2/00_CORRECTION.md and
 * implemented in src/pqv2_encode.c. New values append at the end so
 * the ABI stays stable. */
typedef enum {
    INFERBIT_CONVERT_INT4         = 0,  /* INT4-blk32 + INT8 (v0.4.1 default) */
    INFERBIT_CONVERT_PQV2_FLAT    = 1,  /* PQv2 flat (pq2d_v1_l1) */
    INFERBIT_CONVERT_PQV2_PYRAMID = 2,  /* PQv2 pyramid (pq2d_v1_pyramid) */
} inferbit_convert_format;

/* ── Stage 5b — per-tensor format policy (docs/v2/00_CORRECTION.md) ──
 *
 * The encoder accepts a per-tensor-class override for the output format.
 * When the relevant entry in `inferbit_convert_config::per_class_format`
 * is INFERBIT_CONVERT_INT4 (= 0, the zero-init default), the global
 * `cfg->format` value is used for that class — preserving v0.4.1 behaviour
 * for callers that don't touch the new field. When non-zero, the per-class
 * value wins for tensors of that class.
 *
 * Class taxonomy mirrors the per-projection emission in
 * src/pqv2_encode.c::pqv2_convert: one entry per attention projection
 * (Q/K/V/O), one per FFN projection (gate/up/down), plus embedding and
 * lm_head slots. */
typedef enum {
    INFERBIT_TENSOR_CLASS_FFN_GATE = 0,
    INFERBIT_TENSOR_CLASS_FFN_UP   = 1,
    INFERBIT_TENSOR_CLASS_FFN_DOWN = 2,
    INFERBIT_TENSOR_CLASS_ATTN_Q   = 3,
    INFERBIT_TENSOR_CLASS_ATTN_K   = 4,
    INFERBIT_TENSOR_CLASS_ATTN_V   = 5,
    INFERBIT_TENSOR_CLASS_ATTN_O   = 6,
    INFERBIT_TENSOR_CLASS_EMBED    = 7,
    INFERBIT_TENSOR_CLASS_LM_HEAD  = 8,
    INFERBIT_TENSOR_CLASS_COUNT    = 9,
} inferbit_tensor_class;

/* ── Stage 5c — per-tensor residency hint ────────────────────────────
 *
 * Loader-honored hint stored per tensor in the IBFv6 blob header. The
 * `AUTO` default lets the loader pick based on `IB_RESIDENCY_MODE` (and
 * the `IB_RESIDENCY_RAM_LAYERS=N` override that promotes the first N
 * layers from DRIVE → RAM). RAM-tagged tensors stay resident even in
 * drive mode; DRIVE-tagged tensors stream from disk even in RAM mode.
 *
 * v1 scope: the encoder writes the hint, the loader parses it into
 * `ib_tensor_meta::residency_hint`, and emits a single diagnostic line
 * gated on `IB_PQV2_TRACE=1`. Actual policy enforcement (mlock on RAM,
 * MADV_DONTNEED on DRIVE) is wired in a follow-up. */
typedef enum {
    INFERBIT_RESIDENCY_AUTO  = 0,
    INFERBIT_RESIDENCY_RAM   = 1,
    INFERBIT_RESIDENCY_DRIVE = 2,
} inferbit_residency;

typedef struct {
    int   default_bits;       /* Quantization bits for MLP layers (2, 4, 8). Default: 4 */
    int   sensitive_bits;     /* Bits for attention/embeddings (4, 8). Default: 8 */
    float sparsity;           /* Target structured sparsity 0.0-0.6. Default: 0.0 */
    int   block_size;         /* Sparsity block size. Default: 8 */
    int   kv_bits;            /* KV cache quantization bits. Default: 8 */
    int   threads;            /* Threads for quantization. Default: 0 (auto) */
    void (*progress)(float pct, const char* stage, void* ctx);  /* Progress callback */
    void* progress_ctx;
    /* Output format family. Default INFERBIT_CONVERT_INT4 keeps the
     * v0.4.1 path bit-for-bit. Setting PQV2_FLAT / PQV2_PYRAMID routes
     * the entire convert call through src/pqv2_encode.c.
     * Placed at the end so older callers that zero-init the struct
     * still get INT4 behaviour. */
    inferbit_convert_format format;
    /* Stage 3a (docs/v2/00_CORRECTION.md): post-hoc Mixture-of-Mini-
     * Experts (MoME). When > 1, the encoder splits each FFN tensor
     * (gate_proj / up_proj / down_proj) into `mome_experts` equal sub-
     * matrices along the rows-axis (M / K rows each), emits them as
     * separate IBF v6 tensors named `Lk.mlp.<proj>.expert{e}`, and
     * writes a zero-initialised `[K, hidden]` raw fp16 router tensor
     * named `Lk.mlp.router`. The runtime forward path detects the
     * non-NULL expert array, runs the router matmul, picks top-N
     * experts via softmax + top-k, and dispatches only those experts.
     *
     * v1 (scaffolding) semantics:
     *   - mome_experts == 1 (or zero): no behaviour change.
     *   - mome_experts >  1 with zero router: bit-identical to the
     *     non-MoME case — running ALL experts on a trivial row-split
     *     is the same as running the full FFN matrix.
     *   - mome_experts >  1 with calibrated non-zero router: future
     *     work; the runtime uses softmax-weighted dispatch over the
     *     top-N selected experts.
     *
     * Valid values for v1: 1, 2, 4, 8, 16. Larger values are accepted
     * but capped at IB_MOME_MAX_EXPERTS in the encoder. Trivial row-
     * split requires M divisible by mome_experts; otherwise the
     * encoder falls back to a single expert for that tensor and emits
     * a warning. Placed at the end of the struct for ABI stability —
     * older callers that zero-init get the no-MoME default. */
    int mome_experts;

    /* Stage 5b — per-tensor format policy. When all entries are zero
     * (= INFERBIT_CONVERT_INT4, the default-init for a zero'd config),
     * the encoder uses the global `cfg->format` for every tensor —
     * preserving v0.4.1 behavior. When an entry is non-zero, that
     * tensor class uses that format regardless of the global flag.
     * Index by `inferbit_tensor_class`. ABI-stable: appended at end. */
    inferbit_convert_format per_class_format[INFERBIT_TENSOR_CLASS_COUNT];

    /* Stage 5c — per-tensor residency hint policy. AUTO (= 0, the
     * zero-init default) lets the encoder's heuristic decide: embed,
     * lm_head, norms, and the first 2 layers' tensors are tagged RAM
     * (hot path); everything else is left AUTO so the loader picks at
     * load time. Non-zero entries override the heuristic for that
     * class. Index by `inferbit_tensor_class`. */
    inferbit_residency per_class_residency[INFERBIT_TENSOR_CLASS_COUNT];

    /* Stage 5k — lower-precision scales (docs/v2/00_CORRECTION.md).
     *   0 = legacy: row_scale fp16, codebook_scale fp16 (default).
     *   2 = row_scale int8 + per-tensor fp16 row_max; codebook_scale
     *       fp8 (E4M3). Free at kernel time — decoded into fp16 by the
     *       loader so the hot inner loops are byte-identical to legacy.
     * Modes 1 and 3 are reserved per the doc but not implemented in v1
     * (incremental gains, not worth a separate header field). Zero-init
     * stays bit-identical to v0.4.1. */
    int scale_precision;

    /* Stage 5j — codebook + scale dedup (docs/v2/00_CORRECTION.md).
     * v1 scaffolding only:
     *   0 = no codebook pool (default — legacy behavior).
     *   1 = emit codebook pool (pool_size == n_subchunks, identity
     *       pool_id mapping). Validates the loader/kernel pooled path
     *       end-to-end; quality bit-identical because the pool is a
     *       redundant copy of the per-slot codebooks. Real clustering
     *       (k-medoids over slot codebooks within a tensor) is a
     *       follow-up; this field exists so the format extension can
     *       roll out incrementally. */
    int codebook_dedup;
} inferbit_convert_config;

/* Per-class config-setter helper. Convenience wrapper around the
 * `per_class_format[]` array — keeps callers that only know about
 * one or two classes from having to know the enum width. */
IB_API void inferbit_config_set_class_format(inferbit_convert_config *cfg,
                                              inferbit_tensor_class cls,
                                              inferbit_convert_format fmt);

/* Same for residency hints (Stage 5c). */
IB_API void inferbit_config_set_class_residency(inferbit_convert_config *cfg,
                                                 inferbit_tensor_class cls,
                                                 inferbit_residency hint);

IB_API inferbit_convert_config inferbit_default_convert_config(void);

/* Detect input format from file contents */
typedef enum {
    INFERBIT_FORMAT_UNKNOWN     = 0,
    INFERBIT_FORMAT_SAFETENSORS = 1,
    INFERBIT_FORMAT_GGUF        = 2,
    INFERBIT_FORMAT_IBF         = 3,
} inferbit_format;

IB_API inferbit_format inferbit_detect_format(const char* path);

/*
 * Convert a local model file to .ibf format.
 *
 * input_path:  Path to .safetensors or .gguf file (or directory with multiple .safetensors)
 * output_path: Path for the output .ibf file
 * config:      Conversion parameters (NULL for defaults)
 *
 * Returns INFERBIT_OK on success, error code on failure.
 * Use inferbit_last_error() for details.
 */
IB_API int inferbit_convert(
    const char* input_path,
    const char* output_path,
    const inferbit_convert_config* config
);

/* ── Evaluation ─────────────────────────────────────────────── */

/*
 * Compute perplexity over tokenized samples (teacher forcing).
 * Returns perplexity value, or -1.0 on error.
 */
IB_API double inferbit_perplexity(
    inferbit_model* model,
    const int32_t* const* samples,
    const int* sample_lengths,
    int num_samples
);

/* ── Calibration ────────────────────────────────────────────── */

typedef struct {
    int    bits;
    int    sensitive_bits;
    int    selected;
    char   ibf_path[512];
    double perplexity;
    double tokens_per_sec;
    double ms_per_token;
    double memory_mb;
    int    passes;
    char   failed[512];
} inferbit_profile_result;

/*
 * Search quantization profiles (INT2 → INT4 → INT8), pick first passing gates.
 * results must point to an array of 3 inferbit_profile_result.
 * selected_index receives the index of the chosen profile (0-2).
 */
IB_API int inferbit_calibrate(
    const char* input_path,
    const char* output_dir,
    const int32_t* const* samples,
    const int* sample_lengths,
    int num_samples,
    int output_tokens,
    int warmup_runs,
    int measured_runs,
    double max_perplexity,
    double min_tokens_per_sec,
    double max_memory_mb,
    int threads,
    void (*progress)(const char* stage, void* ctx),
    void* progress_ctx,
    inferbit_profile_result* results,
    int* selected_index
);

/* ── Distribution-time compression (Stage 5i) ───────────────────
 *
 * Pack/unpack .ibf <-> .ibf.zst using zstd in long-range mode. These
 * are out-of-band utilities: the runtime loader never decompresses
 * on the fly (per the file-size rule in docs/v2/00_CORRECTION.md).
 * The intended flow is:
 *
 *   developer: inferbit_pack(model.ibf, model.ibf.zst, 19)
 *   user:      inferbit_unpack(model.ibf.zst, model.ibf)
 *   runtime:   inferbit_load(model.ibf)   <- plain mmap, zero-copy
 *
 * Returns INFERBIT_OK on success, -1 on failure
 * (inferbit_last_error() set). When libinferbit was built without
 * zstd both functions return -1 immediately with a clear error. */
IB_API int inferbit_pack(const char *ibf_in, const char *zst_out, int level);
IB_API int inferbit_unpack(const char *zst_in, const char *ibf_out);

/* ── Phase 4 — DFlash hybrid orchestrator ────────────────────────
 *
 * Confidence-gated early-exit decode. Captures the hidden state at
 * `early_exit_layer` during each forward pass. When the L2-norm of that
 * hidden state falls below `confidence_threshold`, the model is considered
 * "confident" and decode short-circuits — it projects the captured
 * early-layer state through the output head instead of running the
 * remaining layers. Otherwise it runs the full forward.
 *
 * Both paths share KV-cache state. Quality vs. throughput is governed
 * entirely by the threshold: 0 = always full forward (no early exit ever),
 * +inf = always early exit (lowest quality, highest throughput).
 *
 * v1 — CPU only. Metal-routed models fall back transparently to the
 * standard full forward (the attach call succeeds; early-exit is simply
 * never taken). Prefill (num_tokens > 1) is always full forward — DFlash
 * only applies to per-token decode. */
typedef struct inferbit_dflash_config {
    int   early_exit_layer;          /* 0-indexed; e.g. num_layers / 2 */
    float confidence_threshold;      /* L2-norm threshold; tune empirically */
    int   warmup_tokens;             /* run full forward for this many decode steps before allowing early exit (lets KV settle) */
} inferbit_dflash_config;

IB_API int inferbit_dflash_attach(inferbit_model* model, const inferbit_dflash_config* cfg);
IB_API int inferbit_dflash_detach(inferbit_model* model);

/* Stats from the most recent generate() — how many decode steps went the
 * early-exit path vs. the full path. Useful for tuning the threshold. */
IB_API int inferbit_dflash_last_full_count(const inferbit_model* model);
IB_API int inferbit_dflash_last_early_count(const inferbit_model* model);

#ifdef __cplusplus
}
#endif

#endif /* INFERBIT_H */
