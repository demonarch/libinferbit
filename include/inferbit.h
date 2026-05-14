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

typedef struct {
    int   default_bits;       /* Quantization bits for MLP layers (2, 4, 8). Default: 4 */
    int   sensitive_bits;     /* Bits for attention/embeddings (4, 8). Default: 8 */
    float sparsity;           /* Target structured sparsity 0.0-0.6. Default: 0.0 */
    int   block_size;         /* Sparsity block size. Default: 8 */
    int   kv_bits;            /* KV cache quantization bits. Default: 8 */
    int   threads;            /* Threads for quantization. Default: 0 (auto) */
    void (*progress)(float pct, const char* stage, void* ctx);  /* Progress callback */
    void* progress_ctx;
} inferbit_convert_config;

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

#ifdef __cplusplus
}
#endif

#endif /* INFERBIT_H */
