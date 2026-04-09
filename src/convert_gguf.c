/*
 * convert_gguf.c — Convert GGUF models to .ibf
 *
 * GGUF tensors may already be quantized (Q4_0, Q8_0, etc.).
 * We dequantize them to FP32, then re-quantize to our INT4/INT8 format.
 * For F16/F32 GGUF tensors, we quantize directly.
 */

#include "inferbit_internal.h"
#include "cJSON.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IBF_MAGIC     "INFERBIT"
#define IBF_ALIGNMENT 64

/* ── GGML dequantization ────────────────────────────────────── */

/* Q4_0: 32 values per block, 18 bytes per block (2 bytes scale + 16 bytes data) */
static void dequant_q4_0(float* out, const void* data, int64_t elems) {
    const uint8_t* p = (const uint8_t*)data;
    int64_t blocks = elems / 32;
    for (int64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, p, 2);
        /* FP16 scale */
        float scale;
        uint32_t sign = (uint32_t)(scale_bits >> 15) << 31;
        uint32_t exp = (scale_bits >> 10) & 0x1F;
        uint32_t mant = scale_bits & 0x3FF;
        if (exp == 0 && mant == 0) { scale = 0; }
        else if (exp == 0) { while(!(mant&0x400)){mant<<=1;exp--;} exp++; mant&=~0x400;
            uint32_t bits = sign|((exp+112)<<23)|(mant<<13); memcpy(&scale,&bits,4); }
        else if (exp == 31) { scale = sign ? -INFINITY : INFINITY; }
        else { uint32_t bits = sign|((exp+112)<<23)|(mant<<13); memcpy(&scale,&bits,4); }
        p += 2;

        for (int i = 0; i < 16; i++) {
            uint8_t byte = p[i];
            int8_t v0 = (int8_t)(byte & 0x0F) - 8;
            int8_t v1 = (int8_t)((byte >> 4) & 0x0F) - 8;
            out[b * 32 + i * 2]     = (float)v0 * scale;
            out[b * 32 + i * 2 + 1] = (float)v1 * scale;
        }
        p += 16;
    }
}

/* Q8_0: 32 values per block, 34 bytes per block (2 bytes scale + 32 bytes data) */
static void dequant_q8_0(float* out, const void* data, int64_t elems) {
    const uint8_t* p = (const uint8_t*)data;
    int64_t blocks = elems / 32;
    for (int64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, p, 2);
        float scale;
        uint32_t sign = (uint32_t)(scale_bits >> 15) << 31;
        uint32_t exp = (scale_bits >> 10) & 0x1F;
        uint32_t mant = scale_bits & 0x3FF;
        if (exp == 0 && mant == 0) { scale = 0; }
        else if (exp == 0) { while(!(mant&0x400)){mant<<=1;exp--;} exp++; mant&=~0x400;
            uint32_t bits = sign|((exp+112)<<23)|(mant<<13); memcpy(&scale,&bits,4); }
        else if (exp == 31) { scale = sign ? -INFINITY : INFINITY; }
        else { uint32_t bits = sign|((exp+112)<<23)|(mant<<13); memcpy(&scale,&bits,4); }
        p += 2;

        for (int i = 0; i < 32; i++) {
            out[b * 32 + i] = (float)(int8_t)p[i] * scale;
        }
        p += 32;
    }
}

/* F16 */
static void dequant_f16(float* out, const void* data, int64_t elems) {
    const uint8_t* p = (const uint8_t*)data;
    for (int64_t i = 0; i < elems; i++) {
        uint16_t h;
        memcpy(&h, p + i * 2, 2);
        uint32_t sign = (uint32_t)(h >> 15) << 31;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        float f;
        if (exp == 0 && mant == 0) { uint32_t b = sign; memcpy(&f,&b,4); }
        else if (exp == 0) { while(!(mant&0x400)){mant<<=1;exp--;} exp++; mant&=~0x400;
            uint32_t b = sign|((exp+112)<<23)|(mant<<13); memcpy(&f,&b,4); }
        else if (exp == 31) { uint32_t b = sign|0x7F800000|(mant<<13); memcpy(&f,&b,4); }
        else { uint32_t b = sign|((exp+112)<<23)|(mant<<13); memcpy(&f,&b,4); }
        out[i] = f;
    }
}

/* BF16 */
static void dequant_bf16(float* out, const void* data, int64_t elems) {
    const uint8_t* p = (const uint8_t*)data;
    for (int64_t i = 0; i < elems; i++) {
        uint16_t h;
        memcpy(&h, p + i * 2, 2);
        uint32_t b = (uint32_t)h << 16;
        memcpy(&out[i], &b, 4);
    }
}

/* F32 — just copy */
static void dequant_f32(float* out, const void* data, int64_t elems) {
    memcpy(out, data, elems * sizeof(float));
}

/* Dequantize any GGML type to FP32 */
static int dequant_tensor(float* out, const void* data, int type, int64_t elems) {
    switch (type) {
        case 0: dequant_f32(out, data, elems); return 0;
        case 1: dequant_f16(out, data, elems); return 0;
        case 2: dequant_q4_0(out, data, elems); return 0;
        case 8: dequant_q8_0(out, data, elems); return 0;
        case 30: dequant_bf16(out, data, elems); return 0;
        default:
            ib_set_error("unsupported GGML tensor type: %d", type);
            return -1;
    }
}

/* ── Helpers ────────────────────────────────────────────────── */

static size_t align_up(size_t val, size_t align) {
    return (val + align - 1) & ~(align - 1);
}

static uint16_t f32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000);
    int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = b & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

/* ── Tensor naming for GGUF ─────────────────────────────────── */

/* GGUF uses llama.cpp naming: "blk.0.attn_q.weight" */

static int gguf_find_layer(const ib_gguf* gg, int layer, const char* suffix) {
    char name[256];
    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);
    return ib_gguf_find(gg, name);
}

/* ── Write quantized from FP32 buffer ───────────────────────── */

typedef struct {
    size_t weight_offset;
    size_t weight_size;
    size_t scale_offset;
    size_t scale_size;
    int    bits;
    int    rows;
    int    cols;
} ib_wt;

static size_t write_aligned_g(FILE* f, size_t off) {
    size_t a = align_up(off, IBF_ALIGNMENT);
    size_t pad = a - off;
    if (pad > 0) { uint8_t z[64] = {0}; fwrite(z, 1, pad, f); }
    return a;
}

static ib_wt write_from_fp32(FILE* f, size_t* offset, const float* data,
                              int rows, int cols, int bits) {
    ib_wt result = {0};
    result.rows = rows;
    result.cols = cols;
    result.bits = bits;

    *offset = write_aligned_g(f, *offset);
    result.weight_offset = *offset;

    if (bits == 8) {
        size_t w_size = (size_t)rows * cols;
        int8_t* qw = malloc(w_size);
        uint16_t* scales = malloc(rows * 2);

        for (int r = 0; r < rows; r++) {
            float max_abs = 0;
            for (int c = 0; c < cols; c++) {
                float a = fabsf(data[r * cols + c]);
                if (a > max_abs) max_abs = a;
            }
            float scale = max_abs / 127.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[r] = f32_to_fp16(scale);
            float inv = 1.0f / scale;
            for (int c = 0; c < cols; c++) {
                int32_t q = (int32_t)roundf(data[r * cols + c] * inv);
                if (q < -127) q = -127; if (q > 127) q = 127;
                qw[r * cols + c] = (int8_t)q;
            }
        }

        fwrite(qw, 1, w_size, f);
        result.weight_size = w_size;
        *offset += w_size;
        *offset = write_aligned_g(f, *offset);
        result.scale_offset = *offset;
        result.scale_size = rows * 2;
        fwrite(scales, 2, rows, f);
        *offset += result.scale_size;
        free(qw); free(scales);
    } else if (bits == 4) {
        size_t w_size = (size_t)rows * cols / 2;
        uint8_t* qw = malloc(w_size);
        uint16_t* scales = malloc(rows * 2);

        for (int r = 0; r < rows; r++) {
            float max_abs = 0;
            for (int c = 0; c < cols; c++) {
                float a = fabsf(data[r * cols + c]);
                if (a > max_abs) max_abs = a;
            }
            float scale = max_abs / 7.0f;
            if (scale < 1e-10f) scale = 1e-10f;
            scales[r] = f32_to_fp16(scale);
            float inv = 1.0f / scale;
            for (int c = 0; c < cols; c += 2) {
                int32_t q0 = (int32_t)roundf(data[r * cols + c] * inv);
                int32_t q1 = (c+1 < cols) ? (int32_t)roundf(data[r * cols + c + 1] * inv) : 0;
                if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
                if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;
                qw[r * (cols/2) + c/2] = ((uint8_t)(q0+8) & 0x0F) | (((uint8_t)(q1+8) & 0x0F) << 4);
            }
        }

        fwrite(qw, 1, w_size, f);
        result.weight_size = w_size;
        *offset += w_size;
        *offset = write_aligned_g(f, *offset);
        result.scale_offset = *offset;
        result.scale_size = rows * 2;
        fwrite(scales, 2, rows, f);
        *offset += result.scale_size;
        free(qw); free(scales);
    } else if (bits == 16) {
        size_t count = (size_t)rows * (cols > 1 ? cols : 1);
        uint16_t* fp16 = malloc(count * 2);
        for (size_t i = 0; i < count; i++) fp16[i] = f32_to_fp16(data[i]);
        fwrite(fp16, 2, count, f);
        result.weight_size = count * 2;
        *offset += result.weight_size;
        free(fp16);
    }

    return result;
}

/* Dequant a GGUF tensor and write as our format */
static ib_wt convert_gguf_tensor(FILE* f, size_t* offset,
                                  const ib_gguf* gg, int idx, int bits) {
    ib_wt result = {0};
    if (idx < 0) return result;

    int ndim = ib_gguf_tensor_ndim(gg, idx);
    /* GGUF stores shapes in reverse: [cols, rows] for 2D */
    int rows, cols;
    if (ndim >= 2) {
        cols = ib_gguf_tensor_shape(gg, idx, 0);
        rows = ib_gguf_tensor_shape(gg, idx, 1);
    } else {
        rows = ib_gguf_tensor_shape(gg, idx, 0);
        cols = 1;
    }

    int64_t elems = (int64_t)rows * cols;
    float* fp32 = malloc(elems * sizeof(float));
    if (!fp32) return result;

    const void* data = ib_gguf_tensor_data(gg, idx);
    int type = ib_gguf_tensor_type(gg, idx);

    if (dequant_tensor(fp32, data, type, elems) != 0) {
        free(fp32);
        return result;
    }

    result = write_from_fp32(f, offset, fp32, rows, cols, bits);
    free(fp32);
    return result;
}

/* ── Main GGUF conversion ───────────────────────────────────── */

int ib_convert_gguf(const char* input_path, const char* output_path,
                    const inferbit_convert_config* cfg) {
    cfg->progress(0.0f, "opening GGUF", cfg->progress_ctx);

    ib_gguf* gg = ib_gguf_open(input_path);
    if (!gg) return INFERBIT_ERROR_LOAD;

    cfg->progress(0.05f, "reading metadata", cfg->progress_ctx);

    /* Get architecture from GGUF metadata */
    ib_model_config model_cfg;
    if (ib_gguf_get_config(gg, &model_cfg) != 0) {
        ib_set_error("failed to extract architecture from GGUF metadata");
        ib_gguf_close(gg);
        return INFERBIT_ERROR_FORMAT;
    }

    cfg->progress(0.1f, "quantizing", cfg->progress_ctx);

    /* Open output */
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        ib_set_error("failed to open output: %s", output_path);
        ib_gguf_close(gg);
        return INFERBIT_ERROR_LOAD;
    }

    /* Preamble placeholder */
    uint8_t preamble[32] = {0};
    fwrite(preamble, 1, 32, out);

    size_t json_reserve = 128 * 1024;
    uint8_t* pad = calloc(json_reserve, 1);
    fwrite(pad, 1, json_reserve, out);
    free(pad);

    size_t weight_start = align_up(32 + json_reserve, IBF_ALIGNMENT);
    fseek(out, (long)weight_start, SEEK_SET);
    size_t offset = weight_start;

    int sens = cfg->sensitive_bits;
    int def  = cfg->default_bits;

    /* Embedding */
    int emb_idx = ib_gguf_find_suffix(gg, "token_embd.weight");
    ib_wt emb = convert_gguf_tensor(out, &offset, gg, emb_idx, sens);
    emb.weight_offset -= weight_start;
    if (emb.scale_size > 0) emb.scale_offset -= weight_start;

    /* Layers */
    typedef struct {
        ib_wt q, k, v, o, gate, up, down, in_norm, post_norm;
    } lt;
    lt* layers = calloc(model_cfg.num_layers, sizeof(lt));

    for (int l = 0; l < model_cfg.num_layers; l++) {
        float pct = 0.1f + 0.8f * ((float)l / model_cfg.num_layers);
        cfg->progress(pct, "quantizing layers", cfg->progress_ctx);

        #define CVT(dst, suffix, b) do { \
            int idx = gguf_find_layer(gg, l, suffix); \
            if (idx >= 0) { \
                dst = convert_gguf_tensor(out, &offset, gg, idx, b); \
                if (dst.weight_size > 0) dst.weight_offset -= weight_start; \
                if (dst.scale_size > 0) dst.scale_offset -= weight_start; \
            } \
        } while(0)

        CVT(layers[l].q, "attn_q.weight", sens);
        CVT(layers[l].k, "attn_k.weight", sens);
        CVT(layers[l].v, "attn_v.weight", sens);
        CVT(layers[l].o, "attn_output.weight", def);
        CVT(layers[l].gate, "ffn_gate.weight", def);
        CVT(layers[l].up, "ffn_up.weight", def);
        CVT(layers[l].down, "ffn_down.weight", def);
        CVT(layers[l].in_norm, "attn_norm.weight", 16);
        CVT(layers[l].post_norm, "ffn_norm.weight", 16);

        #undef CVT
    }

    cfg->progress(0.9f, "writing output", cfg->progress_ctx);

    /* Output norm and head */
    int norm_idx = ib_gguf_find(gg, "output_norm.weight");
    ib_wt out_norm = convert_gguf_tensor(out, &offset, gg, norm_idx, 16);
    out_norm.weight_offset -= weight_start;

    int head_idx = ib_gguf_find(gg, "output.weight");
    int tie = (head_idx < 0);
    ib_wt out_head;
    if (tie) {
        out_head = emb;
    } else {
        out_head = convert_gguf_tensor(out, &offset, gg, head_idx, sens);
        out_head.weight_offset -= weight_start;
        if (out_head.scale_size > 0) out_head.scale_offset -= weight_start;
    }

    size_t total_weight = offset - weight_start;

    cfg->progress(0.95f, "writing header", cfg->progress_ctx);

    /* Build JSON header — same structure as safetensors conversion */
    cJSON* root = cJSON_CreateObject();
    cJSON_AddNumberToObject(root, "version", 1);

    cJSON* m = cJSON_AddObjectToObject(root, "model");
    cJSON_AddStringToObject(m, "architecture", model_cfg.arch);

    cJSON* a = cJSON_AddObjectToObject(root, "architecture");
    cJSON_AddNumberToObject(a, "num_layers", model_cfg.num_layers);
    cJSON_AddNumberToObject(a, "hidden_size", model_cfg.hidden_size);
    cJSON_AddNumberToObject(a, "num_heads", model_cfg.num_heads);
    cJSON_AddNumberToObject(a, "num_kv_heads", model_cfg.num_kv_heads);
    cJSON_AddNumberToObject(a, "head_dim", model_cfg.head_dim);
    cJSON_AddNumberToObject(a, "intermediate_size", model_cfg.intermediate_size);
    cJSON_AddNumberToObject(a, "vocab_size", model_cfg.vocab_size);
    cJSON_AddNumberToObject(a, "max_context_length", model_cfg.max_context_length);
    cJSON_AddNumberToObject(a, "rope_theta", (double)model_cfg.rope_theta);
    cJSON_AddNumberToObject(a, "norm_epsilon", (double)model_cfg.norm_epsilon);
    cJSON_AddStringToObject(a, "norm_type", model_cfg.norm_type);
    cJSON_AddStringToObject(a, "activation", model_cfg.activation);
    cJSON_AddBoolToObject(a, "tie_word_embeddings", tie);
    cJSON_AddNumberToObject(a, "bos_token_id", model_cfg.bos_token_id);
    cJSON_AddNumberToObject(a, "eos_token_id", model_cfg.eos_token_id);

    cJSON* q = cJSON_AddObjectToObject(root, "quantization");
    cJSON_AddNumberToObject(q, "default_bits", cfg->default_bits);
    cJSON_AddNumberToObject(q, "sensitive_bits", cfg->sensitive_bits);
    cJSON_AddNumberToObject(q, "sparsity", cfg->sparsity);
    cJSON_AddNumberToObject(q, "block_size", cfg->block_size);

    cJSON* kv = cJSON_AddObjectToObject(root, "kv_cache");
    cJSON_AddNumberToObject(kv, "bits", cfg->kv_bits);

    cJSON* d = cJSON_AddObjectToObject(root, "data");
    cJSON_AddNumberToObject(d, "weight_data_offset", (double)weight_start);
    cJSON_AddNumberToObject(d, "weight_data_size", (double)total_weight);
    cJSON_AddNumberToObject(d, "alignment", IBF_ALIGNMENT);

    /* Layers array */
    cJSON* la = cJSON_AddArrayToObject(root, "layers");
    for (int l = 0; l < model_cfg.num_layers; l++) {
        cJSON* layer = cJSON_CreateObject();
        cJSON_AddNumberToObject(layer, "index", l);
        cJSON* w = cJSON_AddObjectToObject(layer, "weights");

        #define ADD(name, wt) do { \
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
            cJSON_AddItemToObject(w, name, t); \
        } while(0)

        ADD("q_proj", layers[l].q);
        ADD("k_proj", layers[l].k);
        ADD("v_proj", layers[l].v);
        ADD("o_proj", layers[l].o);
        ADD("gate_proj", layers[l].gate);
        ADD("up_proj", layers[l].up);
        ADD("down_proj", layers[l].down);
        ADD("input_norm", layers[l].in_norm);
        ADD("post_attn_norm", layers[l].post_norm);
        #undef ADD

        cJSON* sp = cJSON_AddObjectToObject(layer, "sparsity_mask");
        cJSON_AddNumberToObject(sp, "offset", 0);
        cJSON_AddNumberToObject(sp, "size", 0);
        cJSON_AddItemToArray(la, layer);
    }

    /* Embeddings */
    cJSON* eo = cJSON_AddObjectToObject(root, "embeddings");
    cJSON* et = cJSON_CreateObject();
    cJSON_AddNumberToObject(et, "offset", (double)emb.weight_offset);
    cJSON_AddNumberToObject(et, "size", (double)emb.weight_size);
    cJSON* es = cJSON_AddArrayToObject(et, "shape");
    cJSON_AddItemToArray(es, cJSON_CreateNumber(model_cfg.vocab_size));
    cJSON_AddItemToArray(es, cJSON_CreateNumber(model_cfg.hidden_size));
    cJSON_AddNumberToObject(et, "bits", emb.bits);
    cJSON_AddNumberToObject(et, "scale_offset", (double)emb.scale_offset);
    cJSON_AddNumberToObject(et, "scale_size", (double)emb.scale_size);
    cJSON_AddBoolToObject(et, "has_bias", 0);
    cJSON_AddItemToObject(eo, "token_embedding", et);

    /* Output */
    cJSON* oo = cJSON_AddObjectToObject(root, "output");
    cJSON* on = cJSON_CreateObject();
    cJSON_AddNumberToObject(on, "offset", (double)out_norm.weight_offset);
    cJSON_AddNumberToObject(on, "size", (double)out_norm.weight_size);
    cJSON* ons = cJSON_AddArrayToObject(on, "shape");
    cJSON_AddItemToArray(ons, cJSON_CreateNumber(model_cfg.hidden_size));
    cJSON_AddNumberToObject(on, "bits", 16);
    cJSON_AddNumberToObject(on, "scale_offset", 0);
    cJSON_AddNumberToObject(on, "scale_size", 0);
    cJSON_AddBoolToObject(on, "has_bias", 0);
    cJSON_AddItemToObject(oo, "norm", on);

    cJSON* oh = cJSON_CreateObject();
    cJSON_AddNumberToObject(oh, "offset", (double)out_head.weight_offset);
    cJSON_AddNumberToObject(oh, "size", (double)out_head.weight_size);
    cJSON* ohs = cJSON_AddArrayToObject(oh, "shape");
    cJSON_AddItemToArray(ohs, cJSON_CreateNumber(model_cfg.vocab_size));
    cJSON_AddItemToArray(ohs, cJSON_CreateNumber(model_cfg.hidden_size));
    cJSON_AddNumberToObject(oh, "bits", out_head.bits);
    cJSON_AddNumberToObject(oh, "scale_offset", (double)out_head.scale_offset);
    cJSON_AddNumberToObject(oh, "scale_size", (double)out_head.scale_size);
    cJSON_AddBoolToObject(oh, "has_bias", 0);
    cJSON_AddItemToObject(oo, "head", oh);

    char* json_str = cJSON_PrintUnformatted(root);
    size_t json_len = strlen(json_str);
    cJSON_Delete(root);

    /* Write header */
    fseek(out, 0, SEEK_SET);
    fwrite(IBF_MAGIC, 1, 8, out);
    uint32_t ver = 1;
    fwrite(&ver, 4, 1, out);
    uint32_t hs = (uint32_t)json_len;
    fwrite(&hs, 4, 1, out);
    uint32_t flags = 0;
    fwrite(&flags, 4, 1, out);
    uint8_t reserved[12] = {0};
    fwrite(reserved, 1, 12, out);
    fwrite(json_str, 1, json_len, out);

    free(json_str);
    free(layers);
    fclose(out);
    ib_gguf_close(gg);

    cfg->progress(1.0f, "done", cfg->progress_ctx);
    return INFERBIT_OK;
}
