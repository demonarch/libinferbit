/*
 * convert.c — Convert safetensors/GGUF to .ibf format
 *
 * Reads source model, quantizes weights, packs into aligned layout,
 * writes .ibf file with JSON header + binary weight data.
 */

#include "inferbit_internal.h"
#include "cJSON.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

#define IBF_MAGIC      "INFERBIT"
#define IBF_ALIGNMENT  64

/* ── Helpers ────────────────────────────────────────────────── */

static size_t align_up(size_t val, size_t align) {
    return (val + align - 1) & ~(align - 1);
}

static void progress_noop(float pct, const char* stage, void* ctx) {
    (void)pct; (void)stage; (void)ctx;
}

/* ── Detect tensor naming convention ────────────────────────── */

/*
 * LLaMA-style: model.layers.0.self_attn.q_proj.weight
 * Mistral:     model.layers.0.self_attn.q_proj.weight  (same)
 * Falcon:      transformer.h.0.self_attention.query_key_value.weight
 *
 * We detect the pattern from what tensors exist.
 */

typedef struct {
    char prefix[64];         /* "model." or "transformer." or "" */
    char layer_fmt[64];      /* "layers.%d" or "h.%d" */
    char q_proj[64];         /* "self_attn.q_proj.weight" */
    char k_proj[64];
    char v_proj[64];
    char o_proj[64];
    char gate_proj[64];
    char up_proj[64];
    char down_proj[64];
    char input_norm[64];
    char post_norm[64];
    char embed[128];
    char final_norm[128];
    char lm_head[128];
} ib_tensor_names;

static int detect_naming_ts(const ib_tensor_source* ts, ib_tensor_names* names) {
    int s, t;
    if (ib_ts_find_suffix(ts, "model.layers.0.self_attn.q_proj.weight", &s, &t) == 0) {
        strcpy(names->prefix, "model.");
        strcpy(names->layer_fmt, "layers.%d.");
        strcpy(names->q_proj, "self_attn.q_proj.weight");
        strcpy(names->k_proj, "self_attn.k_proj.weight");
        strcpy(names->v_proj, "self_attn.v_proj.weight");
        strcpy(names->o_proj, "self_attn.o_proj.weight");
        strcpy(names->gate_proj, "mlp.gate_proj.weight");
        strcpy(names->up_proj, "mlp.up_proj.weight");
        strcpy(names->down_proj, "mlp.down_proj.weight");
        strcpy(names->input_norm, "input_layernorm.weight");
        strcpy(names->post_norm, "post_attention_layernorm.weight");
        strcpy(names->embed, "model.embed_tokens.weight");
        strcpy(names->final_norm, "model.norm.weight");
        strcpy(names->lm_head, "lm_head.weight");
        return 0;
    }
    if (ib_ts_find_suffix(ts, "layers.0.self_attn.q_proj.weight", &s, &t) == 0) {
        strcpy(names->prefix, "");
        strcpy(names->layer_fmt, "layers.%d.");
        strcpy(names->q_proj, "self_attn.q_proj.weight");
        strcpy(names->k_proj, "self_attn.k_proj.weight");
        strcpy(names->v_proj, "self_attn.v_proj.weight");
        strcpy(names->o_proj, "self_attn.o_proj.weight");
        strcpy(names->gate_proj, "mlp.gate_proj.weight");
        strcpy(names->up_proj, "mlp.up_proj.weight");
        strcpy(names->down_proj, "mlp.down_proj.weight");
        strcpy(names->input_norm, "input_layernorm.weight");
        strcpy(names->post_norm, "post_attention_layernorm.weight");
        strcpy(names->embed, "embed_tokens.weight");
        strcpy(names->final_norm, "norm.weight");
        strcpy(names->lm_head, "lm_head.weight");
        return 0;
    }
    ib_set_error("unrecognized tensor naming convention");
    return -1;
}

/* Legacy version for single-file (kept for existing tests) */
static int detect_naming(const ib_safetensors* sf, ib_tensor_names* names) {
    if (ib_st_find_suffix(sf, "model.layers.0.self_attn.q_proj.weight") >= 0) {
        strcpy(names->prefix, "model.");
        strcpy(names->layer_fmt, "layers.%d.");
        strcpy(names->q_proj, "self_attn.q_proj.weight");
        strcpy(names->k_proj, "self_attn.k_proj.weight");
        strcpy(names->v_proj, "self_attn.v_proj.weight");
        strcpy(names->o_proj, "self_attn.o_proj.weight");
        strcpy(names->gate_proj, "mlp.gate_proj.weight");
        strcpy(names->up_proj, "mlp.up_proj.weight");
        strcpy(names->down_proj, "mlp.down_proj.weight");
        strcpy(names->input_norm, "input_layernorm.weight");
        strcpy(names->post_norm, "post_attention_layernorm.weight");
        strcpy(names->embed, "model.embed_tokens.weight");
        strcpy(names->final_norm, "model.norm.weight");
        strcpy(names->lm_head, "lm_head.weight");
        return 0;
    }

    /* Try without "model." prefix */
    if (ib_st_find_suffix(sf, "layers.0.self_attn.q_proj.weight") >= 0) {
        strcpy(names->prefix, "");
        strcpy(names->layer_fmt, "layers.%d.");
        strcpy(names->q_proj, "self_attn.q_proj.weight");
        strcpy(names->k_proj, "self_attn.k_proj.weight");
        strcpy(names->v_proj, "self_attn.v_proj.weight");
        strcpy(names->o_proj, "self_attn.o_proj.weight");
        strcpy(names->gate_proj, "mlp.gate_proj.weight");
        strcpy(names->up_proj, "mlp.up_proj.weight");
        strcpy(names->down_proj, "mlp.down_proj.weight");
        strcpy(names->input_norm, "input_layernorm.weight");
        strcpy(names->post_norm, "post_attention_layernorm.weight");
        strcpy(names->embed, "embed_tokens.weight");
        strcpy(names->final_norm, "norm.weight");
        strcpy(names->lm_head, "lm_head.weight");
        return 0;
    }

    ib_set_error("unrecognized tensor naming convention — cannot identify model architecture");
    return -1;
}

/* Find layer tensor via tensor_source. Sets shard+tensor indices. Returns 0 on found. */
static int find_layer_tensor_ts(const ib_tensor_source* ts, const ib_tensor_names* names,
                                 int layer, const char* suffix, int* shard, int* tensor) {
    char full_name[512];
    char layer_part[64];
    snprintf(layer_part, sizeof(layer_part), names->layer_fmt, layer);
    snprintf(full_name, sizeof(full_name), "%s%s%s", names->prefix, layer_part, suffix);
    return ib_ts_find(ts, full_name, shard, tensor);
}

/* Legacy single-file version for detect functions */
static int find_layer_tensor(const ib_safetensors* sf, const ib_tensor_names* names, int layer, const char* suffix) {
    char full_name[512];
    char layer_part[64];
    snprintf(layer_part, sizeof(layer_part), names->layer_fmt, layer);
    snprintf(full_name, sizeof(full_name), "%s%s%s", names->prefix, layer_part, suffix);
    return ib_st_find(sf, full_name);
}

/* ── Detect architecture from tensor shapes ─────────────────── */

typedef struct {
    char arch[64];
    int  num_layers;
    int  hidden_size;
    int  num_heads;
    int  num_kv_heads;
    int  head_dim;
    int  intermediate_size;
    int  vocab_size;
} ib_detected_arch;

static int detect_arch(const ib_safetensors* sf, const ib_tensor_names* names, ib_detected_arch* arch) {
    /* Find embedding to get vocab_size and hidden_size */
    int emb_idx = ib_st_find(sf, names->embed);
    if (emb_idx < 0) {
        ib_set_error("cannot find embedding tensor: %s", names->embed);
        return -1;
    }
    arch->vocab_size  = ib_st_tensor_shape_at(sf, emb_idx, 0);
    arch->hidden_size = ib_st_tensor_shape_at(sf, emb_idx, 1);

    /* Find q_proj layer 0 to get num_heads */
    int q_idx = find_layer_tensor(sf, names, 0, names->q_proj);
    if (q_idx < 0) {
        ib_set_error("cannot find layer 0 q_proj");
        return -1;
    }
    int q_out = ib_st_tensor_shape_at(sf, q_idx, 0);

    /* Find k_proj to detect GQA */
    int k_idx = find_layer_tensor(sf, names, 0, names->k_proj);
    int k_out = k_idx >= 0 ? ib_st_tensor_shape_at(sf, k_idx, 0) : q_out;

    /* Find gate_proj to get intermediate_size */
    int g_idx = find_layer_tensor(sf, names, 0, names->gate_proj);
    arch->intermediate_size = g_idx >= 0 ? ib_st_tensor_shape_at(sf, g_idx, 0) : arch->hidden_size * 4;

    /* Count layers */
    arch->num_layers = 0;
    for (int i = 0; i < 1000; i++) {
        if (find_layer_tensor(sf, names, i, names->q_proj) < 0) break;
        arch->num_layers++;
    }

    /* Derive head dimensions */
    /* Assume head_dim = hidden_size / num_heads, and q_out = num_heads * head_dim */
    /* Common head_dims: 64, 128 */
    int head_dim = 128;
    if (arch->hidden_size <= 2048) head_dim = 64;
    arch->head_dim = head_dim;
    arch->num_heads = q_out / head_dim;
    arch->num_kv_heads = k_out / head_dim;

    strcpy(arch->arch, "llama");  /* Default; caller can override */

    return 0;
}

/* ── Write quantized tensor to file ─────────────────────────── */

typedef struct {
    size_t weight_offset;
    size_t weight_size;
    size_t scale_offset;
    size_t scale_size;
    int    bits;
    int    rows;
    int    cols;
} ib_written_tensor;

static size_t write_aligned(FILE* f, size_t current_offset) {
    size_t aligned = align_up(current_offset, IBF_ALIGNMENT);
    size_t pad = aligned - current_offset;
    if (pad > 0) {
        uint8_t zeros[64] = {0};
        fwrite(zeros, 1, pad, f);
    }
    return aligned;
}

static ib_written_tensor write_quantized_tensor_ts(
    FILE* f, size_t* offset,
    const ib_tensor_source* ts, int shard, int tensor_idx,
    int bits
) {
    ib_written_tensor result = {0};
    if (tensor_idx < 0) return result;

    const void* data = ib_ts_tensor_data(ts, shard, tensor_idx);
    const char* dtype = ib_ts_tensor_dtype(ts, shard, tensor_idx);
    int rows = ib_ts_tensor_shape(ts, shard, tensor_idx, 0);
    int cols = ib_ts_tensor_shape(ts, shard, tensor_idx, 1);
    if (cols == 0) cols = 1;

    result.rows = rows;
    result.cols = cols;
    result.bits = bits;

    /* Align offset */
    *offset = write_aligned(f, *offset);
    result.weight_offset = *offset;

    if (bits == 8) {
        size_t w_size = (size_t)rows * cols;
        int8_t* qw = malloc(w_size);
        uint16_t* scales = malloc(rows * sizeof(uint16_t));

        ib_quantize_int8(qw, scales, data, dtype, rows, cols);

        fwrite(qw, 1, w_size, f);
        result.weight_size = w_size;
        *offset += w_size;

        /* Write scales (align first) */
        *offset = write_aligned(f, *offset);
        result.scale_offset = *offset;
        result.scale_size = rows * 2;
        fwrite(scales, 2, rows, f);
        *offset += result.scale_size;

        free(qw);
        free(scales);
    } else if (bits == 4) {
        size_t w_size = (size_t)rows * cols / 2;
        uint8_t* qw = malloc(w_size);
        uint16_t* scales = malloc(rows * sizeof(uint16_t));

        ib_quantize_int4(qw, scales, data, dtype, rows, cols);

        fwrite(qw, 1, w_size, f);
        result.weight_size = w_size;
        *offset += w_size;

        *offset = write_aligned(f, *offset);
        result.scale_offset = *offset;
        result.scale_size = rows * 2;
        fwrite(scales, 2, rows, f);
        *offset += result.scale_size;

        free(qw);
        free(scales);
    } else if (bits == 2) {
        size_t w_size = (size_t)rows * cols / 4;
        uint8_t* qw = malloc(w_size);
        uint16_t* scales = malloc(rows * sizeof(uint16_t));

        ib_quantize_int2(qw, scales, data, dtype, rows, cols);

        fwrite(qw, 1, w_size, f);
        result.weight_size = w_size;
        *offset += w_size;

        *offset = write_aligned(f, *offset);
        result.scale_offset = *offset;
        result.scale_size = rows * 2;
        fwrite(scales, 2, rows, f);
        *offset += result.scale_size;

        free(qw);
        free(scales);
    } else if (bits == 16) {
        /* FP16 norm — just copy/convert */
        size_t size = (size_t)rows * (cols > 1 ? cols : 1);
        uint16_t* fp16 = malloc(size * 2);
        ib_copy_norm_fp16(fp16, data, dtype, (int)size);
        fwrite(fp16, 2, size, f);
        result.weight_size = size * 2;
        *offset += result.weight_size;
        result.scale_offset = 0;
        result.scale_size = 0;
        free(fp16);
    }

    return result;
}

/* ── Public API ─────────────────────────────────────────────── */

inferbit_convert_config inferbit_default_convert_config(void) {
    inferbit_convert_config c = {0};
    c.default_bits   = 4;
    c.sensitive_bits  = 8;
    c.sparsity        = 0.0f;
    c.block_size      = 8;
    c.kv_bits         = 8;
    c.threads         = 0;
    c.progress        = NULL;
    c.progress_ctx    = NULL;
    return c;
}

inferbit_format inferbit_detect_format(const char* path) {
    if (!path) return INFERBIT_FORMAT_UNKNOWN;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return INFERBIT_FORMAT_UNKNOWN;

    uint8_t magic[8];
    ssize_t n = read(fd, magic, 8);
    close(fd);
    if (n < 8) return INFERBIT_FORMAT_UNKNOWN;

    if (memcmp(magic, "INFERBIT", 8) == 0) return INFERBIT_FORMAT_IBF;

    /* GGUF magic: "GGUF" at offset 0 */
    if (memcmp(magic, "GGUF", 4) == 0) return INFERBIT_FORMAT_GGUF;

    /* Safetensors: first 8 bytes are uint64 LE header size (typically < 10MB) */
    uint64_t header_size;
    memcpy(&header_size, magic, 8);
    if (header_size > 0 && header_size < 100 * 1024 * 1024) {
        /* Likely safetensors — check if file extension confirms */
        size_t len = strlen(path);
        if (len > 12 && strcmp(path + len - 12, ".safetensors") == 0) {
            return INFERBIT_FORMAT_SAFETENSORS;
        }
        /* Still probably safetensors if header size is reasonable */
        return INFERBIT_FORMAT_SAFETENSORS;
    }

    return INFERBIT_FORMAT_UNKNOWN;
}

int inferbit_convert(
    const char* input_path,
    const char* output_path,
    const inferbit_convert_config* config
) {
    if (!input_path || !output_path) {
        ib_set_error("NULL path argument");
        return INFERBIT_ERROR_PARAM;
    }

    inferbit_convert_config cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = inferbit_default_convert_config();
    }
    if (!cfg.progress) cfg.progress = progress_noop;

    /* Detect if input is a directory or single file */
    struct stat input_stat;
    if (stat(input_path, &input_stat) != 0) {
        ib_set_error("cannot stat input: %s: %s", input_path, strerror(errno));
        return INFERBIT_ERROR_LOAD;
    }
    int is_dir = S_ISDIR(input_stat.st_mode);

    inferbit_format fmt = INFERBIT_FORMAT_UNKNOWN;
    if (!is_dir) {
        fmt = inferbit_detect_format(input_path);
        if (fmt == INFERBIT_FORMAT_GGUF) {
            return ib_convert_gguf(input_path, output_path, &cfg);
        }
        if (fmt != INFERBIT_FORMAT_SAFETENSORS) {
            ib_set_error("unrecognized input format: %s", input_path);
            return INFERBIT_ERROR_FORMAT;
        }
    }

    cfg.progress(0.0f, "opening", cfg.progress_ctx);

    /* Open tensor source (handles both single-file and directory/multi-shard) */
    ib_tensor_source* ts = ib_ts_open(input_path);
    if (!ts) return INFERBIT_ERROR_LOAD;

    cfg.progress(0.05f, "detecting architecture", cfg.progress_ctx);

    /* Try to read config.json for exact architecture params */
    ib_model_config model_cfg;
    int has_config = 0;
    char config_path[1024];

    if (is_dir) {
        snprintf(config_path, sizeof(config_path), "%s/config.json", input_path);
    } else {
        /* Look for config.json in same directory as the file */
        const char* last_slash = strrchr(input_path, '/');
        if (last_slash) {
            size_t dir_len = last_slash - input_path;
            snprintf(config_path, sizeof(config_path), "%.*s/config.json", (int)dir_len, input_path);
        } else {
            snprintf(config_path, sizeof(config_path), "config.json");
        }
    }
    has_config = (ib_parse_config_json(config_path, &model_cfg) == 0);

    /* Detect naming convention */
    ib_tensor_names names;
    if (detect_naming_ts(ts, &names) != 0) {
        ib_ts_close(ts);
        return INFERBIT_ERROR_FORMAT;
    }

    /* Determine architecture:
     * - If config.json found, use its values (authoritative)
     * - Otherwise, detect from tensor shapes (fallback) */
    ib_detected_arch arch;
    if (has_config) {
        strncpy(arch.arch, model_cfg.arch, sizeof(arch.arch) - 1);
        arch.num_layers        = model_cfg.num_layers;
        arch.hidden_size       = model_cfg.hidden_size;
        arch.num_heads         = model_cfg.num_heads;
        arch.num_kv_heads      = model_cfg.num_kv_heads;
        arch.head_dim          = model_cfg.head_dim;
        arch.intermediate_size = model_cfg.intermediate_size;
        arch.vocab_size        = model_cfg.vocab_size;
    } else {
        /* Fallback: we need a single-shard sf for detect_arch.
         * For multi-shard without config.json, we open the first shard. */
        ib_safetensors* probe = ib_st_open(input_path);
        if (!probe && is_dir) {
            /* Just detect from tensor source by looking at shapes */
            int s, t;
            if (ib_ts_find(ts, names.embed, &s, &t) == 0) {
                arch.vocab_size  = ib_ts_tensor_shape(ts, s, t, 0);
                arch.hidden_size = ib_ts_tensor_shape(ts, s, t, 1);
            }
            if (ib_ts_find_suffix(ts, names.q_proj, &s, &t) == 0) {
                int q_out = ib_ts_tensor_shape(ts, s, t, 0);
                arch.head_dim = 128;
                if (arch.hidden_size <= 2048) arch.head_dim = 64;
                arch.num_heads = q_out / arch.head_dim;
            }
            if (ib_ts_find_suffix(ts, names.k_proj, &s, &t) == 0) {
                int k_out = ib_ts_tensor_shape(ts, s, t, 0);
                arch.num_kv_heads = k_out / arch.head_dim;
            }
            if (ib_ts_find_suffix(ts, names.gate_proj, &s, &t) == 0) {
                arch.intermediate_size = ib_ts_tensor_shape(ts, s, t, 0);
            }
            /* Count layers */
            arch.num_layers = 0;
            for (int i = 0; i < 1000; i++) {
                if (find_layer_tensor_ts(ts, &names, i, names.q_proj, &s, &t) != 0) break;
                arch.num_layers++;
            }
            strcpy(arch.arch, "llama");
        } else if (probe) {
            detect_arch(probe, &names, &arch);
            ib_st_close(probe);
        } else {
            ib_ts_close(ts);
            ib_set_error("cannot detect architecture without config.json");
            return INFERBIT_ERROR_FORMAT;
        }
    }

    cfg.progress(0.1f, "quantizing", cfg.progress_ctx);

    /* Open output file */
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        ib_set_error("failed to open output: %s: %s", output_path, strerror(errno));
        ib_ts_close(ts);
        return INFERBIT_ERROR_LOAD;
    }

    /* Write placeholder preamble (32 bytes) */
    uint8_t preamble[32] = {0};
    fwrite(preamble, 1, 32, out);

    /* Reserve space for JSON header */
    size_t json_reserve = 128 * 1024;  /* 128KB for large models with many layers */
    uint8_t* json_pad = calloc(json_reserve, 1);
    fwrite(json_pad, 1, json_reserve, out);
    free(json_pad);

    size_t weight_data_start = align_up(32 + json_reserve, IBF_ALIGNMENT);
    fseek(out, (long)weight_data_start, SEEK_SET);
    size_t offset = weight_data_start;

    /* ── Quantize and write embedding ───────────────────────── */
    int emb_shard, emb_tensor;
    ib_written_tensor emb_wt = {0};
    if (ib_ts_find(ts, names.embed, &emb_shard, &emb_tensor) == 0) {
        emb_wt = write_quantized_tensor_ts(out, &offset, ts, emb_shard, emb_tensor, cfg.sensitive_bits);
        emb_wt.weight_offset -= weight_data_start;
        if (emb_wt.scale_size > 0) emb_wt.scale_offset -= weight_data_start;
    }

    /* ── Quantize and write layers ──────────────────────────── */
    typedef struct {
        ib_written_tensor q, k, v, o, gate, up, down, in_norm, post_norm;
    } layer_tensors;

    layer_tensors* lt = calloc(arch.num_layers, sizeof(layer_tensors));

    for (int l = 0; l < arch.num_layers; l++) {
        float pct = 0.1f + 0.8f * ((float)l / (float)arch.num_layers);
        cfg.progress(pct, "quantizing layers", cfg.progress_ctx);

        int sens = cfg.sensitive_bits;
        int def  = cfg.default_bits;
        int s, t;

        #define CONVERT_TENSOR(dst, name_suffix, bits_val) do { \
            if (find_layer_tensor_ts(ts, &names, l, name_suffix, &s, &t) == 0) { \
                dst = write_quantized_tensor_ts(out, &offset, ts, s, t, bits_val); \
                if (dst.weight_size > 0) dst.weight_offset -= weight_data_start; \
                if (dst.scale_size > 0) dst.scale_offset -= weight_data_start; \
            } \
        } while(0)

        CONVERT_TENSOR(lt[l].q, names.q_proj, sens);
        CONVERT_TENSOR(lt[l].k, names.k_proj, sens);
        CONVERT_TENSOR(lt[l].v, names.v_proj, sens);
        CONVERT_TENSOR(lt[l].o, names.o_proj, def);
        CONVERT_TENSOR(lt[l].gate, names.gate_proj, def);
        CONVERT_TENSOR(lt[l].up, names.up_proj, def);
        CONVERT_TENSOR(lt[l].down, names.down_proj, def);
        CONVERT_TENSOR(lt[l].in_norm, names.input_norm, 16);
        CONVERT_TENSOR(lt[l].post_norm, names.post_norm, 16);

        #undef CONVERT_TENSOR
    }

    cfg.progress(0.9f, "writing output head", cfg.progress_ctx);

    /* ── Output norm and head ───────────────────────────────── */
    int ns, nt_idx;
    ib_written_tensor out_norm_wt = {0};
    if (ib_ts_find(ts, names.final_norm, &ns, &nt_idx) == 0) {
        out_norm_wt = write_quantized_tensor_ts(out, &offset, ts, ns, nt_idx, 16);
        out_norm_wt.weight_offset -= weight_data_start;
    }

    int head_shard = emb_shard, head_tensor = emb_tensor;
    int tie_embeddings = 1;
    if (ib_ts_find(ts, names.lm_head, &head_shard, &head_tensor) == 0) {
        tie_embeddings = 0;
    }
    if (has_config) tie_embeddings = model_cfg.tie_word_embeddings;

    ib_written_tensor out_head_wt;
    if (tie_embeddings) {
        /* Re-use embedding data */
        out_head_wt = emb_wt;
    } else {
        out_head_wt = write_quantized_tensor_ts(out, &offset, ts, head_shard, head_tensor, cfg.sensitive_bits);
        out_head_wt.weight_offset -= weight_data_start;
        if (out_head_wt.scale_size > 0) out_head_wt.scale_offset -= weight_data_start;
    }

    size_t total_weight_size = offset - weight_data_start;

    cfg.progress(0.95f, "writing header", cfg.progress_ctx);

    /* ── Build JSON header ──────────────────────────────────── */
    cJSON* root = cJSON_CreateObject();

    cJSON_AddNumberToObject(root, "version", 1);

    cJSON* model_obj = cJSON_AddObjectToObject(root, "model");
    cJSON_AddStringToObject(model_obj, "architecture", arch.arch);

    cJSON* arch_obj = cJSON_AddObjectToObject(root, "architecture");
    cJSON_AddNumberToObject(arch_obj, "num_layers", arch.num_layers);
    cJSON_AddNumberToObject(arch_obj, "hidden_size", arch.hidden_size);
    cJSON_AddNumberToObject(arch_obj, "num_heads", arch.num_heads);
    cJSON_AddNumberToObject(arch_obj, "num_kv_heads", arch.num_kv_heads);
    cJSON_AddNumberToObject(arch_obj, "head_dim", arch.head_dim);
    cJSON_AddNumberToObject(arch_obj, "intermediate_size", arch.intermediate_size);
    cJSON_AddNumberToObject(arch_obj, "vocab_size", arch.vocab_size);
    cJSON_AddNumberToObject(arch_obj, "max_context_length",
        has_config ? model_cfg.max_context_length : 4096);
    cJSON_AddNumberToObject(arch_obj, "rope_theta",
        has_config ? (double)model_cfg.rope_theta : 10000.0);
    cJSON_AddNumberToObject(arch_obj, "norm_epsilon",
        has_config ? (double)model_cfg.norm_epsilon : 1e-5);
    cJSON_AddStringToObject(arch_obj, "norm_type",
        has_config ? model_cfg.norm_type : "rmsnorm");
    cJSON_AddStringToObject(arch_obj, "activation",
        has_config ? model_cfg.activation : "silu");
    cJSON_AddBoolToObject(arch_obj, "tie_word_embeddings", tie_embeddings);
    cJSON_AddNumberToObject(arch_obj, "bos_token_id",
        has_config ? model_cfg.bos_token_id : 1);
    cJSON_AddNumberToObject(arch_obj, "eos_token_id",
        has_config ? model_cfg.eos_token_id : 2);

    cJSON* quant_obj = cJSON_AddObjectToObject(root, "quantization");
    cJSON_AddNumberToObject(quant_obj, "default_bits", cfg.default_bits);
    cJSON_AddNumberToObject(quant_obj, "sensitive_bits", cfg.sensitive_bits);
    cJSON_AddNumberToObject(quant_obj, "sparsity", cfg.sparsity);
    cJSON_AddNumberToObject(quant_obj, "block_size", cfg.block_size);

    cJSON* kv_obj = cJSON_AddObjectToObject(root, "kv_cache");
    cJSON_AddNumberToObject(kv_obj, "bits", cfg.kv_bits);

    cJSON* data_obj = cJSON_AddObjectToObject(root, "data");
    cJSON_AddNumberToObject(data_obj, "weight_data_offset", (double)weight_data_start);
    cJSON_AddNumberToObject(data_obj, "weight_data_size", (double)total_weight_size);
    cJSON_AddNumberToObject(data_obj, "alignment", IBF_ALIGNMENT);

    /* Layers */
    cJSON* layers_arr = cJSON_AddArrayToObject(root, "layers");
    for (int l = 0; l < arch.num_layers; l++) {
        cJSON* layer = cJSON_CreateObject();
        cJSON_AddNumberToObject(layer, "index", l);
        cJSON* weights = cJSON_AddObjectToObject(layer, "weights");

        /* Helper macro for tensor JSON */
        #define ADD_TENSOR(name_str, wt) do { \
            cJSON* t = cJSON_CreateObject(); \
            cJSON_AddNumberToObject(t, "offset", (double)(wt).weight_offset); \
            cJSON_AddNumberToObject(t, "size", (double)(wt).weight_size); \
            cJSON* sh = cJSON_AddArrayToObject(t, "shape"); \
            if ((wt).cols > 1) { cJSON_AddItemToArray(sh, cJSON_CreateNumber((wt).rows)); cJSON_AddItemToArray(sh, cJSON_CreateNumber((wt).cols)); } \
            else { cJSON_AddItemToArray(sh, cJSON_CreateNumber((wt).rows)); } \
            cJSON_AddNumberToObject(t, "bits", (wt).bits); \
            cJSON_AddNumberToObject(t, "scale_offset", (double)(wt).scale_offset); \
            cJSON_AddNumberToObject(t, "scale_size", (double)(wt).scale_size); \
            cJSON_AddBoolToObject(t, "has_bias", 0); \
            cJSON_AddItemToObject(weights, name_str, t); \
        } while(0)

        ADD_TENSOR("q_proj", lt[l].q);
        ADD_TENSOR("k_proj", lt[l].k);
        ADD_TENSOR("v_proj", lt[l].v);
        ADD_TENSOR("o_proj", lt[l].o);
        ADD_TENSOR("gate_proj", lt[l].gate);
        ADD_TENSOR("up_proj", lt[l].up);
        ADD_TENSOR("down_proj", lt[l].down);
        ADD_TENSOR("input_norm", lt[l].in_norm);
        ADD_TENSOR("post_attn_norm", lt[l].post_norm);

        #undef ADD_TENSOR

        cJSON* sp = cJSON_AddObjectToObject(layer, "sparsity_mask");
        cJSON_AddNumberToObject(sp, "offset", 0);
        cJSON_AddNumberToObject(sp, "size", 0);

        cJSON_AddItemToArray(layers_arr, layer);
    }

    /* Embeddings */
    cJSON* emb_obj = cJSON_AddObjectToObject(root, "embeddings");
    cJSON* emb_t = cJSON_CreateObject();
    cJSON_AddNumberToObject(emb_t, "offset", (double)emb_wt.weight_offset);
    cJSON_AddNumberToObject(emb_t, "size", (double)emb_wt.weight_size);
    cJSON* emb_sh = cJSON_AddArrayToObject(emb_t, "shape");
    cJSON_AddItemToArray(emb_sh, cJSON_CreateNumber(arch.vocab_size));
    cJSON_AddItemToArray(emb_sh, cJSON_CreateNumber(arch.hidden_size));
    cJSON_AddNumberToObject(emb_t, "bits", emb_wt.bits);
    cJSON_AddNumberToObject(emb_t, "scale_offset", (double)emb_wt.scale_offset);
    cJSON_AddNumberToObject(emb_t, "scale_size", (double)emb_wt.scale_size);
    cJSON_AddBoolToObject(emb_t, "has_bias", 0);
    cJSON_AddItemToObject(emb_obj, "token_embedding", emb_t);

    /* Output */
    cJSON* out_obj = cJSON_AddObjectToObject(root, "output");

    cJSON* on = cJSON_CreateObject();
    cJSON_AddNumberToObject(on, "offset", (double)out_norm_wt.weight_offset);
    cJSON_AddNumberToObject(on, "size", (double)out_norm_wt.weight_size);
    cJSON* on_sh = cJSON_AddArrayToObject(on, "shape");
    cJSON_AddItemToArray(on_sh, cJSON_CreateNumber(arch.hidden_size));
    cJSON_AddNumberToObject(on, "bits", 16);
    cJSON_AddNumberToObject(on, "scale_offset", 0);
    cJSON_AddNumberToObject(on, "scale_size", 0);
    cJSON_AddBoolToObject(on, "has_bias", 0);
    cJSON_AddItemToObject(out_obj, "norm", on);

    cJSON* oh = cJSON_CreateObject();
    cJSON_AddNumberToObject(oh, "offset", (double)out_head_wt.weight_offset);
    cJSON_AddNumberToObject(oh, "size", (double)out_head_wt.weight_size);
    cJSON* oh_sh = cJSON_AddArrayToObject(oh, "shape");
    cJSON_AddItemToArray(oh_sh, cJSON_CreateNumber(arch.vocab_size));
    cJSON_AddItemToArray(oh_sh, cJSON_CreateNumber(arch.hidden_size));
    cJSON_AddNumberToObject(oh, "bits", out_head_wt.bits);
    cJSON_AddNumberToObject(oh, "scale_offset", (double)out_head_wt.scale_offset);
    cJSON_AddNumberToObject(oh, "scale_size", (double)out_head_wt.scale_size);
    cJSON_AddBoolToObject(oh, "has_bias", 0);
    cJSON_AddItemToObject(out_obj, "head", oh);

    /* Serialize JSON */
    char* json_str = cJSON_PrintUnformatted(root);
    size_t json_len = strlen(json_str);
    cJSON_Delete(root);

    if (json_len >= json_reserve) {
        ib_set_error("JSON header too large: %zu bytes", json_len);
        free(json_str);
        free(lt);
        fclose(out);
        ib_ts_close(ts);
        return INFERBIT_ERROR_INTERNAL;
    }

    /* Write preamble and JSON header at the beginning of the file */
    fseek(out, 0, SEEK_SET);
    fwrite(IBF_MAGIC, 1, 8, out);
    uint32_t version = 1;
    fwrite(&version, 4, 1, out);
    uint32_t hsize = (uint32_t)json_len;
    fwrite(&hsize, 4, 1, out);
    uint32_t flags = 0;
    fwrite(&flags, 4, 1, out);
    uint8_t reserved[12] = {0};
    fwrite(reserved, 1, 12, out);
    fwrite(json_str, 1, json_len, out);

    free(json_str);
    free(lt);
    fclose(out);
    ib_ts_close(ts);

    cfg.progress(1.0f, "done", cfg.progress_ctx);
    return INFERBIT_OK;
}
