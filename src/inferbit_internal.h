/*
 * inferbit_internal.h — Internal types and helpers
 *
 * Not part of the public API. Shared across source files within libinferbit.
 */

#ifndef INFERBIT_INTERNAL_H
#define INFERBIT_INTERNAL_H

#include "inferbit.h"
#include <stdbool.h>

/* ── Thread-local error ─────────────────────────────────────── */

void ib_set_error(const char* fmt, ...);

/* ── IBF header (parsed from JSON) ──────────────────────────── */

typedef struct {
    /* Model identity */
    char architecture[64];
    char name[128];

    /* Architecture params */
    int num_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    int max_context_length;

    /* RoPE */
    float rope_theta;

    /* Normalization */
    float norm_epsilon;
    char  norm_type[16];      /* "rmsnorm" or "layernorm" */

    /* Activation */
    char activation[16];      /* "silu", "gelu", "gelu_tanh" */

    /* Flags */
    bool tie_word_embeddings;
    bool attention_bias;
    bool mlp_bias;

    /* Special tokens */
    int bos_token_id;
    int eos_token_id;

    /* Quantization */
    int   default_bits;
    int   sensitive_bits;
    float sparsity;
    int   block_size;

    /* KV cache */
    int kv_bits;

    /* Data section */
    size_t weight_data_offset;
    size_t weight_data_size;
    int    alignment;
} ib_ibf_header;

/* ── Per-tensor metadata ────────────────────────────────────── */

typedef struct {
    size_t offset;        /* Offset within weight data section */
    size_t size;          /* Size in bytes */
    int    shape[4];      /* Dimensions (unused dims = 0) */
    int    ndim;          /* Number of dimensions */
    int    bits;          /* Quantization bits (2, 4, 8, 16) */
    size_t scale_offset;  /* Offset of scale factors (0 = none) */
    size_t scale_size;    /* Size of scale data */
    bool   has_bias;
} ib_tensor_meta;

/* ── Per-layer metadata ─────────────────────────────────────── */

typedef struct {
    ib_tensor_meta q_proj;
    ib_tensor_meta k_proj;
    ib_tensor_meta v_proj;
    ib_tensor_meta o_proj;
    ib_tensor_meta gate_proj;
    ib_tensor_meta up_proj;
    ib_tensor_meta down_proj;
    ib_tensor_meta input_norm;
    ib_tensor_meta post_attn_norm;

    /* Sparsity mask */
    size_t sparsity_mask_offset;
    size_t sparsity_mask_size;
} ib_layer_meta;

/* ── KV cache ───────────────────────────────────────────────── */

typedef struct {
    void*  key_data;       /* Quantized key cache */
    void*  value_data;     /* Quantized value cache */
    float* key_scales;     /* Scale factors for keys */
    float* value_scales;   /* Scale factors for values */
    int    length;         /* Current number of tokens in cache */
    int    capacity;       /* Max tokens allocated */
    bool   dynamic;        /* Whether cache grows dynamically */
} ib_kv_cache;

/* ── Model struct ───────────────────────────────────────────── */

struct inferbit_model {
    /* IBF metadata */
    ib_ibf_header header;

    /* Layer metadata */
    ib_layer_meta* layers;       /* Array of num_layers */

    /* Embedding and output head metadata */
    ib_tensor_meta token_embedding;
    ib_tensor_meta output_norm;
    ib_tensor_meta output_head;

    /* Weight data (mmap'd or allocated) */
    void*  weight_data;
    size_t weight_data_size;
    bool   weight_data_mmap;     /* True if mmap'd, false if malloc'd */
    int    mmap_fd;              /* File descriptor if mmap'd */

    /* KV cache (one per layer) */
    ib_kv_cache* kv_caches;

    /* Activation buffers (reused across layers) */
    float* buf_residual;         /* [hidden_size] */
    float* buf_hidden;           /* [hidden_size] */
    float* buf_attn;             /* [hidden_size] */
    float* buf_mlp;              /* [intermediate_size] */
    float* buf_mlp2;             /* [intermediate_size] */
    float* buf_logits;           /* [vocab_size] */
    float* buf_qkv;              /* Scratch for Q, K, V projections */

    /* Speculative decoding */
    inferbit_model* draft_model;
    int             draft_tokens;

    /* Threading */
    int num_threads;
    struct ib_thread_pool* thread_pool;
};

/* ── Config struct ──────────────────────────────────────────── */

struct inferbit_config {
    int  threads;
    int  context_length;
    bool kv_dynamic;
    bool native_parse;
    int  native_bits;
};

/* ── SIMD dispatch ──────────────────────────────────────────── */

typedef enum {
    IB_SIMD_NONE   = 0,
    IB_SIMD_AVX2   = 1,
    IB_SIMD_AVX512 = 2,
    IB_SIMD_NEON   = 3,
} ib_simd_level;

ib_simd_level ib_detect_simd(void);

/* ── Kernel function pointers (set at init based on SIMD) ──── */

typedef struct {
    /* INT4 matmul: out[M] = weights[M, N] @ input[N] */
    void (*matmul_int4)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* INT8 matmul */
    void (*matmul_int8)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* INT2 ternary matmul: weights are {-1, 0, +1}, 4 per byte */
    void (*matmul_int2)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* RMSNorm: out[N] = rmsnorm(input[N], weight[N], eps) */
    void (*rmsnorm)(
        float* out, const float* input, const float* weight,
        float eps, int N
    );

    /* RoPE: apply rotary position encoding in-place */
    void (*rope)(
        float* q, float* k, int head_dim, int pos, float theta
    );

    /* Softmax: in-place softmax over N elements */
    void (*softmax)(float* data, int N);

    /* Element-wise: out = a * b (SiLU gate) */
    void (*silu_mul)(float* out, const float* gate, const float* up, int N);

    /* W4A8 matmul: INT4 weights × INT8 activation, grouped activation scale.
     *
     * Activation is quantized in groups of IB_W4A8_GROUP=128 elements, each
     * with its own FP32 scale. Output[i] = sum over groups g of
     *   (weights[i,g] · input[g]) * scales_a[g] * scales_w[i].
     *
     * N must be a multiple of IB_W4A8_GROUP (128). Uses ARM sdot / x86 VNNI
     * when available. */
    void (*matmul_w4a8)(
        float* out, const void* weights, const float* scales_w,
        const int8_t* input, const float* scales_a, int M, int N
    );
} ib_kernels;

/* Activation quantization group size for W4A8. Chosen to fit all transformer
 * hidden dims used in practice (multiples of 128). */
#define IB_W4A8_GROUP 128

/* Global kernel dispatch table */
extern ib_kernels ib_kern;

void ib_init_kernels(ib_simd_level level);

/* ── Prefix cache (on-disk KV reuse for repeated prompts) ───── */

/* Try to restore KV state for tokens[0..n_tokens-2] from on-disk cache.
 * Returns restored prefix length (> 0) on hit, 0 on miss, -1 on I/O error
 * (non-fatal; caller should fall back to normal prefill). */
int ib_prefix_cache_try_restore(inferbit_model* model,
                                const int32_t* tokens, int n_tokens);

/* Save KV state for tokens[0..n_tokens-2] after a successful prefill.
 * Returns 1 on success, 0 on skip/error. */
int ib_prefix_cache_save(const inferbit_model* model,
                         const int32_t* tokens, int n_tokens);

/* ── Safetensors parser ─────────────────────────────────────── */

typedef struct ib_safetensors ib_safetensors;

ib_safetensors* ib_st_open(const char* path);
void            ib_st_close(ib_safetensors* sf);
const void*     ib_st_tensor_data(const ib_safetensors* sf, int index);
size_t          ib_st_tensor_size(const ib_safetensors* sf, int index);
int             ib_st_find(const ib_safetensors* sf, const char* name);
int             ib_st_find_suffix(const ib_safetensors* sf, const char* suffix);
int             ib_st_num_tensors(const ib_safetensors* sf);
const char*     ib_st_tensor_name_at(const ib_safetensors* sf, int index);
const char*     ib_st_tensor_dtype_at(const ib_safetensors* sf, int index);
int             ib_st_tensor_ndim_at(const ib_safetensors* sf, int index);
int             ib_st_tensor_shape_at(const ib_safetensors* sf, int index, int dim);

/* ── Multi-shard safetensors ─────────────────────────────────── */

typedef struct ib_safetensors_multi ib_safetensors_multi;

ib_safetensors_multi* ib_st_multi_open(const char* dir_path);
void                  ib_st_multi_close(ib_safetensors_multi* multi);
int  ib_st_multi_find(const ib_safetensors_multi* multi, const char* name,
                      int* out_shard, int* out_tensor);
int  ib_st_multi_find_suffix(const ib_safetensors_multi* multi, const char* suffix,
                             int* out_shard, int* out_tensor);
const void* ib_st_multi_tensor_data(const ib_safetensors_multi* multi, int shard, int tensor);
const char* ib_st_multi_tensor_dtype(const ib_safetensors_multi* multi, int shard, int tensor);
int         ib_st_multi_tensor_shape(const ib_safetensors_multi* multi, int shard, int tensor, int dim);
int         ib_st_multi_num_shards(const ib_safetensors_multi* multi);

/* ── GGUF parser ────────────────────────────────────────────── */

typedef struct ib_gguf ib_gguf;

ib_gguf*    ib_gguf_open(const char* path);
void        ib_gguf_close(ib_gguf* gg);
int         ib_gguf_num_tensors(const ib_gguf* gg);
int         ib_gguf_find(const ib_gguf* gg, const char* name);
int         ib_gguf_find_suffix(const ib_gguf* gg, const char* suffix);
const void* ib_gguf_tensor_data(const ib_gguf* gg, int index);
size_t      ib_gguf_tensor_size(const ib_gguf* gg, int index);
int         ib_gguf_tensor_type(const ib_gguf* gg, int index);
int         ib_gguf_tensor_shape(const ib_gguf* gg, int index, int dim);
int         ib_gguf_tensor_ndim(const ib_gguf* gg, int index);
const char* ib_gguf_tensor_name(const ib_gguf* gg, int index);
int         ib_gguf_meta_int(const ib_gguf* gg, const char* key, int def);
float       ib_gguf_meta_float(const ib_gguf* gg, const char* key, float def);
const char* ib_gguf_meta_string(const ib_gguf* gg, const char* key);
/* ib_gguf_get_config declared after ib_model_config below */

/* ── Tensor source (unified single/multi-shard access) ──────── */

typedef struct ib_tensor_source ib_tensor_source;

ib_tensor_source* ib_ts_open(const char* path);  /* file or directory */
void              ib_ts_close(ib_tensor_source* ts);
int  ib_ts_find(const ib_tensor_source* ts, const char* name,
                int* out_shard, int* out_tensor);
int  ib_ts_find_suffix(const ib_tensor_source* ts, const char* suffix,
                       int* out_shard, int* out_tensor);
const void* ib_ts_tensor_data(const ib_tensor_source* ts, int shard, int tensor);
const char* ib_ts_tensor_dtype(const ib_tensor_source* ts, int shard, int tensor);
int         ib_ts_tensor_shape(const ib_tensor_source* ts, int shard, int tensor, int dim);

/* ── Config JSON parser ─────────────────────────────────────── */

typedef struct {
    char  arch[64];
    int   num_layers;
    int   hidden_size;
    int   num_heads;
    int   num_kv_heads;
    int   head_dim;
    int   intermediate_size;
    int   vocab_size;
    int   max_context_length;
    float rope_theta;
    float norm_epsilon;
    char  norm_type[16];
    char  activation[16];
    int   tie_word_embeddings;
    int   bos_token_id;
    int   eos_token_id;
} ib_model_config;

int ib_parse_config_json(const char* path, ib_model_config* cfg);
int ib_gguf_get_config(const ib_gguf* gg, ib_model_config* cfg);

/* ── GGUF converter ─────────────────────────────────────────── */

int ib_convert_gguf(const char* input_path, const char* output_path,
                    const inferbit_convert_config* cfg);

/* ── Quantization ───────────────────────────────────────────── */

void ib_quantize_int8(int8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
void ib_quantize_int4(uint8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
void ib_quantize_int2(uint8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
void ib_copy_norm_fp16(uint16_t* out, const void* src, const char* dtype, int size);

/* ── Forward pass ───────────────────────────────────────────── */

int ib_forward(inferbit_model* model, const int32_t* tokens, int num_tokens, float* out_logits);

/* ── Threading ──────────────────────────────────────────────── */

typedef struct ib_thread_pool ib_thread_pool;

ib_thread_pool* ib_pool_create(int n_threads);
void            ib_pool_destroy(ib_thread_pool* tp);
void            ib_pool_run(ib_thread_pool* tp,
                            void (*fn)(void* arg, int thread_id, int start, int end),
                            void* arg, int total, int chunk_size);
void            ib_parallel_matmul(ib_thread_pool* tp, float* out, const void* weights,
                                   const float* scales, const float* input,
                                   int M, int N, int bits);
void            ib_parallel_matmul_w4a8(ib_thread_pool* tp, float* out,
                                        const void* weights, const float* scales_w,
                                        const int8_t* input, const float* scales_a,
                                        int M, int N);
/* Per-group symmetric INT8 quantization. Writes N INT8 values and
 * ceil(N/IB_W4A8_GROUP) FP32 scales. Returns the number of groups written. */
int             ib_quantize_input_int8_g128(const float* input, int8_t* out_q,
                                            float* out_scales, int N);

#endif /* INFERBIT_INTERNAL_H */
