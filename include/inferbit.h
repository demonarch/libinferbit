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

#ifdef __cplusplus
}
#endif

#endif /* INFERBIT_H */
