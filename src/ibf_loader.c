/*
 * ibf_loader.c — Parse and load .ibf files
 *
 * File layout:
 *   [0..7]    Magic "INFERBIT"
 *   [8..11]   Format version (uint32 LE)
 *   [12..15]  Header size in bytes (uint32 LE)
 *   [16..19]  Flags (uint32 bitfield)
 *   [20..31]  Reserved (zeros)
 *   [32..32+H) JSON header (UTF-8)
 *   [A..)     Weight data (64-byte aligned)
 */

#include "inferbit_internal.h"
#include "platform.h"
#include "cJSON.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IBF_MAGIC      "INFERBIT"
#define IBF_MAGIC_SIZE 8
#define IBF_PREAMBLE   32
#define IBF_ALIGNMENT  64
#define IBF_VERSION    1

/* ── Helpers ────────────────────────────────────────────────── */

static size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static const char* json_str(const cJSON* obj, const char* key, const char* def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsString(item) && item->valuestring) return item->valuestring;
    return def;
}

static int json_int(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return item->valueint;
    return def;
}

static double json_double(const cJSON* obj, const char* key, double def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return item->valuedouble;
    return def;
}

static int json_bool(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsBool(item)) return cJSON_IsTrue(item);
    return def;
}

static size_t json_size(const cJSON* obj, const char* key, size_t def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return (size_t)item->valuedouble;
    return def;
}

/* ── Parse tensor metadata from JSON ────────────────────────── */

static ib_tensor_meta parse_tensor(const cJSON* obj) {
    ib_tensor_meta t = {0};
    if (!obj || !cJSON_IsObject(obj)) return t;

    t.offset       = json_size(obj, "offset", 0);
    t.size         = json_size(obj, "size", 0);
    t.bits         = json_int(obj, "bits", 16);
    t.scale_offset = json_size(obj, "scale_offset", 0);
    t.scale_size   = json_size(obj, "scale_size", 0);
    t.has_bias     = json_bool(obj, "has_bias", 0);

    cJSON* shape = cJSON_GetObjectItemCaseSensitive(obj, "shape");
    if (cJSON_IsArray(shape)) {
        t.ndim = cJSON_GetArraySize(shape);
        if (t.ndim > 4) t.ndim = 4;
        for (int i = 0; i < t.ndim; i++) {
            cJSON* dim = cJSON_GetArrayItem(shape, i);
            t.shape[i] = cJSON_IsNumber(dim) ? dim->valueint : 0;
        }
    }

    return t;
}

/* ── Parse layer metadata ───────────────────────────────────── */

static ib_layer_meta parse_layer(const cJSON* obj) {
    ib_layer_meta layer = {0};
    if (!obj) return layer;

    cJSON* weights = cJSON_GetObjectItemCaseSensitive(obj, "weights");
    if (weights) {
        layer.q_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "q_proj"));
        layer.k_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "k_proj"));
        layer.v_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "v_proj"));
        layer.o_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "o_proj"));
        layer.gate_proj      = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "gate_proj"));
        layer.up_proj        = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "up_proj"));
        layer.down_proj      = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "down_proj"));
        layer.input_norm     = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "input_norm"));
        layer.post_attn_norm = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "post_attn_norm"));
    }

    cJSON* sparsity = cJSON_GetObjectItemCaseSensitive(obj, "sparsity_mask");
    if (sparsity) {
        layer.sparsity_mask_offset = json_size(sparsity, "offset", 0);
        layer.sparsity_mask_size   = json_size(sparsity, "size", 0);
    }

    return layer;
}

/* ── Parse IBF header JSON ──────────────────────────────────── */

static int parse_header_json(const char* json_str_buf, size_t json_len,
                             ib_ibf_header* header, ib_layer_meta** out_layers,
                             ib_tensor_meta* token_emb, ib_tensor_meta* out_norm,
                             ib_tensor_meta* out_head) {
    cJSON* root = cJSON_ParseWithLength(json_str_buf, json_len);
    if (!root) {
        ib_set_error("failed to parse IBF JSON header: %s", cJSON_GetErrorPtr());
        return -1;
    }

    /* Model info */
    cJSON* model = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (model) {
        const char* arch = json_str(model, "architecture", "unknown");
        strncpy(header->architecture, arch, sizeof(header->architecture) - 1);
        const char* name = json_str(model, "name", "");
        strncpy(header->name, name, sizeof(header->name) - 1);
    }

    /* Architecture */
    cJSON* arch = cJSON_GetObjectItemCaseSensitive(root, "architecture");
    if (arch) {
        header->num_layers         = json_int(arch, "num_layers", 0);
        header->hidden_size        = json_int(arch, "hidden_size", 0);
        header->num_heads          = json_int(arch, "num_heads", 0);
        header->num_kv_heads       = json_int(arch, "num_kv_heads", header->num_heads);
        header->head_dim           = json_int(arch, "head_dim", header->hidden_size / header->num_heads);
        header->intermediate_size  = json_int(arch, "intermediate_size", 0);
        header->vocab_size         = json_int(arch, "vocab_size", 0);
        header->max_context_length = json_int(arch, "max_context_length", 2048);
        header->rope_theta         = (float)json_double(arch, "rope_theta", 10000.0);
        header->norm_epsilon       = (float)json_double(arch, "norm_epsilon", 1e-5);

        const char* nt = json_str(arch, "norm_type", "rmsnorm");
        strncpy(header->norm_type, nt, sizeof(header->norm_type) - 1);

        const char* act = json_str(arch, "activation", "silu");
        strncpy(header->activation, act, sizeof(header->activation) - 1);

        header->tie_word_embeddings = json_bool(arch, "tie_word_embeddings", 0);
        header->attention_bias      = json_bool(arch, "attention_bias", 0);
        header->mlp_bias            = json_bool(arch, "mlp_bias", 0);
        header->bos_token_id        = json_int(arch, "bos_token_id", 1);
        header->eos_token_id        = json_int(arch, "eos_token_id", 2);
    }

    /* Quantization */
    cJSON* quant = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (quant) {
        header->default_bits   = json_int(quant, "default_bits", 4);
        header->sensitive_bits = json_int(quant, "sensitive_bits", 8);
        header->sparsity       = (float)json_double(quant, "sparsity", 0.0);
        header->block_size     = json_int(quant, "block_size", 8);
    }

    /* KV cache */
    cJSON* kv = cJSON_GetObjectItemCaseSensitive(root, "kv_cache");
    if (kv) {
        header->kv_bits = json_int(kv, "bits", 8);
    }

    /* Data section */
    cJSON* data = cJSON_GetObjectItemCaseSensitive(root, "data");
    if (data) {
        header->weight_data_offset = json_size(data, "weight_data_offset", 0);
        header->weight_data_size   = json_size(data, "weight_data_size", 0);
        header->alignment          = json_int(data, "alignment", 64);
    }

    /* Layers */
    cJSON* layers_arr = cJSON_GetObjectItemCaseSensitive(root, "layers");
    if (cJSON_IsArray(layers_arr) && header->num_layers > 0) {
        *out_layers = calloc(header->num_layers, sizeof(ib_layer_meta));
        if (!*out_layers) {
            ib_set_error("failed to allocate layer metadata");
            cJSON_Delete(root);
            return -1;
        }
        int n = cJSON_GetArraySize(layers_arr);
        if (n > header->num_layers) n = header->num_layers;
        for (int i = 0; i < n; i++) {
            (*out_layers)[i] = parse_layer(cJSON_GetArrayItem(layers_arr, i));
        }
    }

    /* Embeddings */
    cJSON* emb = cJSON_GetObjectItemCaseSensitive(root, "embeddings");
    if (emb) {
        *token_emb = parse_tensor(cJSON_GetObjectItemCaseSensitive(emb, "token_embedding"));
    }

    /* Output */
    cJSON* output = cJSON_GetObjectItemCaseSensitive(root, "output");
    if (output) {
        *out_norm = parse_tensor(cJSON_GetObjectItemCaseSensitive(output, "norm"));
        *out_head = parse_tensor(cJSON_GetObjectItemCaseSensitive(output, "head"));
    }

    cJSON_Delete(root);
    return 0;
}

/* ── Allocate KV caches ─────────────────────────────────────── */

static int alloc_kv_caches(inferbit_model* model, int context_length, int dynamic) {
    int num_layers  = model->header.num_layers;
    int num_kv_heads = model->header.num_kv_heads;
    int head_dim    = model->header.head_dim;
    int kv_bits     = model->header.kv_bits;
    int capacity    = context_length > 0 ? context_length : model->header.max_context_length;

    model->kv_caches = calloc(num_layers, sizeof(ib_kv_cache));
    if (!model->kv_caches) return -1;

    /* Bytes per token for one layer's K or V cache. */
    size_t kv_dim = (size_t)num_kv_heads * head_dim;
    size_t bytes_per_token;
    if (kv_bits >= 16) {
        bytes_per_token = kv_dim * sizeof(float);
    } else if (kv_bits == 8) {
        bytes_per_token = kv_dim;
    } else if (kv_bits == 4) {
        bytes_per_token = (kv_dim + 1) / 2;
    } else if (kv_bits == 2) {
        bytes_per_token = (kv_dim + 3) / 4;
    } else {
        ib_set_error("unsupported kv_bits=%d", kv_bits);
        return -1;
    }

    for (int i = 0; i < num_layers; i++) {
        ib_kv_cache* kv = &model->kv_caches[i];
        kv->length   = 0;
        kv->dynamic  = (dynamic != 0);

        if (dynamic) {
            /* Start empty, grow later */
            kv->capacity   = 0;
            kv->key_data   = NULL;
            kv->value_data = NULL;
        } else {
            kv->capacity   = capacity;
            kv->key_data   = calloc(capacity, bytes_per_token);
            kv->value_data = calloc(capacity, bytes_per_token);
            if (!kv->key_data || !kv->value_data) return -1;
        }

        /* Scale factors (FP32) for quantized KV */
        if (kv_bits < 16) {
            size_t scale_count = (size_t)capacity * num_kv_heads;
            kv->key_scales   = calloc(scale_count, sizeof(float));
            kv->value_scales = calloc(scale_count, sizeof(float));
            if (!kv->key_scales || !kv->value_scales) return -1;
        }
    }

    return 0;
}

/* ── Allocate activation buffers ────────────────────────────── */

static int alloc_buffers(inferbit_model* model) {
    int h = model->header.hidden_size;
    int inter = model->header.intermediate_size;
    int vocab = model->header.vocab_size;
    int num_heads = model->header.num_heads;
    int head_dim = model->header.head_dim;
    int max_ctx = model->header.max_context_length;

    model->buf_residual = calloc(h, sizeof(float));
    model->buf_hidden   = calloc(h, sizeof(float));
    model->buf_attn     = calloc(h, sizeof(float));
    model->buf_mlp      = calloc(inter, sizeof(float));
    model->buf_mlp2     = calloc(inter, sizeof(float));
    model->buf_logits   = calloc(vocab, sizeof(float));

    int num_kv_heads = model->header.num_kv_heads;
    int kv_dim = num_kv_heads * head_dim;

    /* QKV scratch layout (must match forward.c):
     * q:         [hidden]
     * k:         [kv_dim]
     * v:         [kv_dim]
     * att:       [num_heads * max_ctx]
     * scale_buf: [max(hidden, inter, vocab)]
     */
    int scale_sz = h > inter ? h : inter;
    if (vocab > scale_sz) scale_sz = vocab;
    size_t qkv_size = (size_t)h + kv_dim + kv_dim + (size_t)num_heads * max_ctx + scale_sz;
    model->buf_qkv = calloc(qkv_size, sizeof(float));

    if (!model->buf_residual || !model->buf_hidden || !model->buf_attn ||
        !model->buf_mlp || !model->buf_mlp2 || !model->buf_logits || !model->buf_qkv) {
        return -1;
    }

    return 0;
}

/* ── Public: load .ibf file ─────────────────────────────────── */

inferbit_model* ibf_load(const char* path, const inferbit_config* config) {
    /* Open file */
    int fd = ib_open(path, O_RDONLY);
    if (fd < 0) {
        ib_set_error("failed to open %s: %s", path, strerror(errno));
        return NULL;
    }

    /* Get file size */
    ib_struct_stat st;
    if (ib_fstat(fd, &st) < 0) {
        ib_set_error("failed to stat %s: %s", path, strerror(errno));
        ib_close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    if (file_size < IBF_PREAMBLE) {
        ib_set_error("file too small to be IBF: %zu bytes", file_size);
        ib_close(fd);
        return NULL;
    }

    /* Read preamble (32 bytes) */
    uint8_t preamble[IBF_PREAMBLE];
    if (ib_read(fd, preamble, IBF_PREAMBLE) != IBF_PREAMBLE) {
        ib_set_error("failed to read IBF preamble");
        ib_close(fd);
        return NULL;
    }

    /* Validate magic */
    if (memcmp(preamble, IBF_MAGIC, IBF_MAGIC_SIZE) != 0) {
        ib_set_error("invalid IBF magic number");
        ib_close(fd);
        return NULL;
    }

    /* Read version (uint32 LE) */
    uint32_t version;
    memcpy(&version, preamble + 8, 4);
    if (version > IBF_VERSION) {
        ib_set_error("unsupported IBF version: %u (max supported: %u)", version, IBF_VERSION);
        ib_close(fd);
        return NULL;
    }

    /* Read header size */
    uint32_t header_size;
    memcpy(&header_size, preamble + 12, 4);

    if (IBF_PREAMBLE + header_size > file_size) {
        ib_set_error("IBF header size exceeds file size");
        ib_close(fd);
        return NULL;
    }

    /* Read flags */
    uint32_t flags;
    memcpy(&flags, preamble + 16, 4);
    (void)flags;  /* TODO: use flags for sparsity_masks, calibration_data */

    /* Read JSON header */
    char* json_buf = malloc(header_size + 1);
    if (!json_buf) {
        ib_set_error("failed to allocate JSON header buffer");
        ib_close(fd);
        return NULL;
    }
    int nread = (int)ib_read(fd, json_buf, (unsigned int)header_size);
    if (nread < 0 || (unsigned int)nread != header_size) {
        ib_set_error("failed to read IBF JSON header");
        free(json_buf);
        ib_close(fd);
        return NULL;
    }
    json_buf[header_size] = '\0';

    /* Allocate model */
    inferbit_model* model = calloc(1, sizeof(inferbit_model));
    if (!model) {
        ib_set_error("failed to allocate model");
        free(json_buf);
        ib_close(fd);
        return NULL;
    }

    /* Parse JSON header */
    if (parse_header_json(json_buf, header_size, &model->header,
                          &model->layers, &model->token_embedding,
                          &model->output_norm, &model->output_head) != 0) {
        free(json_buf);
        free(model);
        ib_close(fd);
        return NULL;
    }
    free(json_buf);

    /* Validate required fields */
    if (model->header.num_layers <= 0 || model->header.hidden_size <= 0 ||
        model->header.vocab_size <= 0) {
        ib_set_error("IBF header missing required architecture fields");
        free(model->layers);
        free(model);
        ib_close(fd);
        return NULL;
    }

    /* Compute weight data offset (aligned) */
    size_t weight_offset = align_up(IBF_PREAMBLE + header_size, IBF_ALIGNMENT);
    if (model->header.weight_data_offset > 0) {
        weight_offset = model->header.weight_data_offset;
    }

    size_t weight_size = file_size - weight_offset;
    if (model->header.weight_data_size > 0 && model->header.weight_data_size < weight_size) {
        weight_size = model->header.weight_data_size;
    }

    /* Memory-map weight data */
    void* mapped = ib_mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ib_set_error("failed to mmap %s: %s", path, strerror(errno));
        free(model->layers);
        free(model);
        ib_close(fd);
        return NULL;
    }

    /* Ask the kernel to back this region with huge pages when possible.
     *
     * Linux: MADV_HUGEPAGE promotes to transparent 2MB pages (if THP is
     * enabled) and cuts the TLB miss rate dramatically on multi-GB weight
     * sets. MADV_WILLNEED + MADV_SEQUENTIAL tell the prefetcher that the
     * full region will be walked forward.
     *
     * macOS: MADV_HUGEPAGE isn't supported, but the kernel automatically
     * promotes to 16KB pages (which is already 4× the Linux default). We
     * still hint WILLNEED so the first-touch faults don't stall decode.
     *
     * Failures here are benign — they just mean we run at default page
     * size. We don't check the return. */
#ifdef MADV_HUGEPAGE
    madvise(mapped, file_size, MADV_HUGEPAGE);
#endif
#ifdef MADV_WILLNEED
    madvise(mapped, file_size, MADV_WILLNEED);
#endif

    model->weight_data      = (uint8_t*)mapped + weight_offset;
    model->weight_data_size = weight_size;
    model->weight_data_mmap = true;
    model->mmap_fd          = fd;
    /* Note: we keep fd open while mmap is active (some systems need it) */

    /* Apply config */
    int ctx_len = 0;
    int kv_dynamic = 0;
    int threads = 4;
    if (config) {
        ctx_len    = config->context_length;
        kv_dynamic = config->kv_dynamic;
        threads    = config->threads > 0 ? config->threads : 4;
    }
    model->num_threads = threads;

    /* Allocate KV caches */
    if (alloc_kv_caches(model, ctx_len, kv_dynamic) != 0) {
        ib_set_error("failed to allocate KV caches");
        ib_munmap(mapped, file_size);
        ib_close(fd);
        free(model->layers);
        free(model);
        return NULL;
    }

    /* Allocate activation buffers */
    if (alloc_buffers(model) != 0) {
        ib_set_error("failed to allocate activation buffers");
        /* TODO: proper cleanup of kv_caches */
        ib_munmap(mapped, file_size);
        ib_close(fd);
        free(model->layers);
        free(model);
        return NULL;
    }

    /* Initialize kernel dispatch */
    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);

    /* Create thread pool (NULL if single-threaded) */
    model->thread_pool = ib_pool_create(threads);

    return model;
}
