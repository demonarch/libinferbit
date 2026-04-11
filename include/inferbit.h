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

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ────────────────────────────────────────────────── */

#define INFERBIT_VERSION_MAJOR 0
#define INFERBIT_VERSION_MINOR 1
#define INFERBIT_VERSION_PATCH 0
#define INFERBIT_VERSION_STRING "0.1.0"

const char* inferbit_version(void);
int         inferbit_version_major(void);
int         inferbit_version_minor(void);
int         inferbit_version_patch(void);

/* ── Error codes ────────────────────────────────────────────── */

#define INFERBIT_OK              0
#define INFERBIT_ERROR_LOAD     -1
#define INFERBIT_ERROR_FORMAT   -2
#define INFERBIT_ERROR_MEMORY   -3
#define INFERBIT_ERROR_CONTEXT  -4
#define INFERBIT_ERROR_PARAM    -5
#define INFERBIT_ERROR_INTERNAL -6

const char* inferbit_last_error(void);

/* ── Opaque types ───────────────────────────────────────────── */

typedef struct inferbit_model  inferbit_model;
typedef struct inferbit_config inferbit_config;

/* ── Configuration ──────────────────────────────────────────── */

inferbit_config* inferbit_config_create(void);
void             inferbit_config_free(inferbit_config* config);

void inferbit_config_set_threads(inferbit_config* config, int threads);
void inferbit_config_set_context_length(inferbit_config* config, int length);
void inferbit_config_set_kv_cache_dynamic(inferbit_config* config, int dynamic);

/* Native parse mode (dev/debug only) */
void inferbit_config_set_native_parse(inferbit_config* config, int enabled);
void inferbit_config_set_native_bits(inferbit_config* config, int bits);

/* ── Model lifecycle ────────────────────────────────────────── */

inferbit_model* inferbit_load(const char* path, const inferbit_config* config);
void            inferbit_free(inferbit_model* model);

/* ── Sampling parameters ────────────────────────────────────── */

typedef struct {
    float temperature;
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   max_tokens;
    int   seed;
} inferbit_sample_params;

inferbit_sample_params inferbit_default_sample_params(void);

/* ── Generation ─────────────────────────────────────────────── */

int inferbit_generate(
    inferbit_model*        model,
    const int32_t*         input_tokens,
    int                    num_input_tokens,
    int32_t*               out_tokens,
    int                    max_out_tokens,
    inferbit_sample_params params
);

typedef int (*inferbit_stream_callback)(int32_t token, void* ctx);

int inferbit_generate_stream(
    inferbit_model*        model,
    const int32_t*         input_tokens,
    int                    num_input_tokens,
    inferbit_stream_callback callback,
    void*                  ctx,
    inferbit_sample_params params
);

int inferbit_forward(
    inferbit_model*  model,
    const int32_t*   tokens,
    int              num_tokens,
    float*           out_logits,
    int              vocab_size
);

/* ── KV-cache control ───────────────────────────────────────── */

void inferbit_kv_clear(inferbit_model* model);
void inferbit_kv_truncate(inferbit_model* model, int length);
int  inferbit_kv_length(const inferbit_model* model);

/* ── Model info ─────────────────────────────────────────────── */

const char* inferbit_model_architecture(const inferbit_model* model);
int         inferbit_model_num_layers(const inferbit_model* model);
int         inferbit_model_hidden_size(const inferbit_model* model);
int         inferbit_model_vocab_size(const inferbit_model* model);
int         inferbit_model_max_context(const inferbit_model* model);
int         inferbit_model_default_bits(const inferbit_model* model);
size_t      inferbit_model_weight_memory(const inferbit_model* model);
size_t      inferbit_model_kv_memory(const inferbit_model* model);
size_t      inferbit_model_total_memory(const inferbit_model* model);

/* ── Speculative decoding ───────────────────────────────────── */

void inferbit_set_draft_model(inferbit_model* model, inferbit_model* draft, int draft_tokens);
void inferbit_unset_draft_model(inferbit_model* model);

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

inferbit_convert_config inferbit_default_convert_config(void);

/* Detect input format from file contents */
typedef enum {
    INFERBIT_FORMAT_UNKNOWN     = 0,
    INFERBIT_FORMAT_SAFETENSORS = 1,
    INFERBIT_FORMAT_GGUF        = 2,
    INFERBIT_FORMAT_IBF         = 3,
} inferbit_format;

inferbit_format inferbit_detect_format(const char* path);

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
int inferbit_convert(
    const char* input_path,
    const char* output_path,
    const inferbit_convert_config* config
);

/* ── Evaluation ─────────────────────────────────────────────── */

/*
 * Compute perplexity over tokenized samples (teacher forcing).
 * Returns perplexity value, or -1.0 on error.
 */
double inferbit_perplexity(
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
int inferbit_calibrate(
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
