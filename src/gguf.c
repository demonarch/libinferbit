/*
 * gguf.c — Parse GGUF format (used by llama.cpp and Ollama)
 *
 * GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * File layout:
 *   Magic "GGUF" (4 bytes)
 *   Version (uint32)
 *   Tensor count (uint64)
 *   Metadata KV count (uint64)
 *   Metadata KV pairs...
 *   Tensor info entries...
 *   Padding to alignment (default 32 bytes)
 *   Tensor data...
 */

#include "inferbit_internal.h"
#include "platform.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── GGUF types ─────────────────────────────────────────────── */

#define GGUF_MAGIC 0x46554747  /* "GGUF" little-endian */

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

/* GGML tensor types we care about */
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_BF16    = 30,
};

/* Bytes per element for each type (for block-quantized types, this is average) */
static float ggml_type_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4.0f;
        case GGML_TYPE_F16:  return 2.0f;
        case GGML_TYPE_BF16: return 2.0f;
        case GGML_TYPE_Q4_0: return 0.5f + 2.0f/32;  /* 32 elements per block: 16 bytes data + 2 bytes scale */
        case GGML_TYPE_Q4_1: return 0.5f + 4.0f/32;
        case GGML_TYPE_Q5_0: return 0.625f + 2.0f/32;
        case GGML_TYPE_Q5_1: return 0.625f + 4.0f/32;
        case GGML_TYPE_Q8_0: return 1.0f + 2.0f/32;
        case GGML_TYPE_Q8_1: return 1.0f + 4.0f/32;
        case GGML_TYPE_Q2_K: return 2.5625f/8;
        case GGML_TYPE_Q3_K: return 3.4375f/8;
        case GGML_TYPE_Q4_K: return 4.5f/8;
        case GGML_TYPE_Q5_K: return 5.5f/8;
        case GGML_TYPE_Q6_K: return 6.5625f/8;
        case GGML_TYPE_Q8_K: return 8.5f/8;
        default: return 0;
    }
}

static int ggml_block_size(int type) {
    switch (type) {
        case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_BF16:
            return 1;
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1:
            return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_K:
            return 256;
        default: return 1;
    }
}

/* ── GGUF tensor entry ──────────────────────────────────────── */

typedef struct {
    char    name[256];
    int     ndim;
    int64_t shape[4];
    int     type;           /* ggml_type */
    size_t  offset;         /* Offset from start of tensor data section */
    size_t  abs_offset;     /* Absolute offset in file */
    size_t  size;           /* Size in bytes */
} ib_gguf_tensor;

/* ── GGUF metadata entry ────────────────────────────────────── */

typedef struct {
    char    key[256];
    int     type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        uint64_t u64;
        int64_t  i64;
        float    f32;
        double   f64;
        int      b;
        char     str[512];
    } value;
} ib_gguf_meta;

/* ── GGUF file handle ───────────────────────────────────────── */

struct ib_gguf {
    void*          mmap_base;
    size_t         mmap_size;
    int            fd;
    uint32_t       version;

    ib_gguf_tensor* tensors;
    int             num_tensors;

    ib_gguf_meta*   metadata;
    int             num_metadata;

    size_t          data_offset;  /* Where tensor data starts */
};

/* ── Binary reader ──────────────────────────────────────────── */

typedef struct {
    const uint8_t* data;
    size_t         size;
    size_t         pos;
} ib_reader;

static int read_u8(ib_reader* r, uint8_t* out) {
    if (r->pos + 1 > r->size) return -1;
    *out = r->data[r->pos++];
    return 0;
}

static int read_u32(ib_reader* r, uint32_t* out) {
    if (r->pos + 4 > r->size) return -1;
    memcpy(out, r->data + r->pos, 4);
    r->pos += 4;
    return 0;
}

static int read_u64(ib_reader* r, uint64_t* out) {
    if (r->pos + 8 > r->size) return -1;
    memcpy(out, r->data + r->pos, 8);
    r->pos += 8;
    return 0;
}

static int read_i32(ib_reader* r, int32_t* out) {
    if (r->pos + 4 > r->size) return -1;
    memcpy(out, r->data + r->pos, 4);
    r->pos += 4;
    return 0;
}

static int read_i64(ib_reader* r, int64_t* out) {
    if (r->pos + 8 > r->size) return -1;
    memcpy(out, r->data + r->pos, 8);
    r->pos += 8;
    return 0;
}

static int read_f32(ib_reader* r, float* out) {
    if (r->pos + 4 > r->size) return -1;
    memcpy(out, r->data + r->pos, 4);
    r->pos += 4;
    return 0;
}

static int read_f64(ib_reader* r, double* out) {
    if (r->pos + 8 > r->size) return -1;
    memcpy(out, r->data + r->pos, 8);
    r->pos += 8;
    return 0;
}

static int read_string(ib_reader* r, char* out, int max_len) {
    uint64_t len;
    if (read_u64(r, &len) != 0) return -1;
    if (r->pos + len > r->size) return -1;
    int copy = (int)len < max_len - 1 ? (int)len : max_len - 1;
    memcpy(out, r->data + r->pos, copy);
    out[copy] = '\0';
    r->pos += len;
    return 0;
}

/* Skip a metadata value of given type */
static int skip_value(ib_reader* r, int type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL:
            r->pos += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
            r->pos += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
            r->pos += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
            r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            uint64_t len;
            if (read_u64(r, &len) != 0) return -1;
            r->pos += len;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t elem_type;
            uint64_t count;
            if (read_u32(r, &elem_type) != 0) return -1;
            if (read_u64(r, &count) != 0) return -1;
            for (uint64_t i = 0; i < count; i++) {
                if (skip_value(r, elem_type) != 0) return -1;
            }
            break;
        }
        default: return -1;
    }
    if (r->pos > r->size) return -1;
    return 0;
}

/* Read a metadata value */
static int read_meta_value(ib_reader* r, ib_gguf_meta* meta) {
    uint32_t type;
    if (read_u32(r, &type) != 0) return -1;
    meta->type = type;

    switch (type) {
        case GGUF_TYPE_UINT8:  return read_u8(r, &meta->value.u8);
        case GGUF_TYPE_INT8:   { uint8_t v; int rc = read_u8(r, &v); meta->value.i8 = (int8_t)v; return rc; }
        case GGUF_TYPE_UINT32: return read_u32(r, &meta->value.u32);
        case GGUF_TYPE_INT32:  return read_i32(r, &meta->value.i32);
        case GGUF_TYPE_UINT64: return read_u64(r, &meta->value.u64);
        case GGUF_TYPE_INT64:  return read_i64(r, &meta->value.i64);
        case GGUF_TYPE_FLOAT32: return read_f32(r, &meta->value.f32);
        case GGUF_TYPE_FLOAT64: return read_f64(r, &meta->value.f64);
        case GGUF_TYPE_BOOL:   { uint8_t v; int rc = read_u8(r, &v); meta->value.b = v; return rc; }
        case GGUF_TYPE_STRING: return read_string(r, meta->value.str, sizeof(meta->value.str));
        case GGUF_TYPE_UINT16: { r->pos += 2; return 0; }  /* TODO: proper read */
        case GGUF_TYPE_INT16:  { r->pos += 2; return 0; }
        case GGUF_TYPE_ARRAY:  return skip_value(r, GGUF_TYPE_ARRAY);
        default: return -1;
    }
}

/* ── Open GGUF file ─────────────────────────────────────────── */

ib_gguf* ib_gguf_open(const char* path) {
    int fd = ib_open(path, O_RDONLY);
    if (fd < 0) {
        ib_set_error("failed to open %s: %s", path, strerror(errno));
        return NULL;
    }

    ib_struct_stat st;
    if (ib_fstat(fd, &st) < 0) {
        ib_set_error("failed to stat %s", path);
        ib_close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    void* base = ib_mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ib_set_error("failed to mmap %s", path);
        ib_close(fd);
        return NULL;
    }

    ib_reader r = { .data = (const uint8_t*)base, .size = file_size, .pos = 0 };

    /* Read header */
    uint32_t magic;
    if (read_u32(&r, &magic) != 0 || magic != GGUF_MAGIC) {
        ib_set_error("invalid GGUF magic");
        ib_munmap(base, file_size);
        ib_close(fd);
        return NULL;
    }

    uint32_t version;
    if (read_u32(&r, &version) != 0) goto fail;

    uint64_t tensor_count, meta_count;
    if (read_u64(&r, &tensor_count) != 0) goto fail;
    if (read_u64(&r, &meta_count) != 0) goto fail;

    ib_gguf* gg = calloc(1, sizeof(ib_gguf));
    gg->mmap_base = base;
    gg->mmap_size = file_size;
    gg->fd = fd;
    gg->version = version;

    /* Read metadata */
    gg->num_metadata = (int)meta_count;
    gg->metadata = calloc(meta_count, sizeof(ib_gguf_meta));

    for (uint64_t i = 0; i < meta_count; i++) {
        if (read_string(&r, gg->metadata[i].key, sizeof(gg->metadata[i].key)) != 0) goto fail2;
        if (read_meta_value(&r, &gg->metadata[i]) != 0) goto fail2;
    }

    /* Read tensor info */
    gg->num_tensors = (int)tensor_count;
    gg->tensors = calloc(tensor_count, sizeof(ib_gguf_tensor));

    for (uint64_t i = 0; i < tensor_count; i++) {
        ib_gguf_tensor* t = &gg->tensors[i];
        if (read_string(&r, t->name, sizeof(t->name)) != 0) goto fail2;

        uint32_t ndim;
        if (read_u32(&r, &ndim) != 0) goto fail2;
        t->ndim = (int)ndim;

        for (int d = 0; d < t->ndim && d < 4; d++) {
            uint64_t dim;
            if (read_u64(&r, &dim) != 0) goto fail2;
            t->shape[d] = (int64_t)dim;
        }
        /* Skip extra dims if ndim > 4 */
        for (int d = 4; d < t->ndim; d++) {
            uint64_t dummy;
            if (read_u64(&r, &dummy) != 0) goto fail2;
        }

        uint32_t type;
        if (read_u32(&r, &type) != 0) goto fail2;
        t->type = (int)type;

        uint64_t offset;
        if (read_u64(&r, &offset) != 0) goto fail2;
        t->offset = (size_t)offset;
    }

    /* Tensor data starts after header, aligned to 32 bytes (GGUF default) */
    int alignment = 32;
    /* Check for custom alignment in metadata */
    for (int i = 0; i < gg->num_metadata; i++) {
        if (strcmp(gg->metadata[i].key, "general.alignment") == 0) {
            if (gg->metadata[i].type == GGUF_TYPE_UINT32)
                alignment = (int)gg->metadata[i].value.u32;
            break;
        }
    }

    size_t header_end = r.pos;
    gg->data_offset = (header_end + alignment - 1) & ~((size_t)alignment - 1);

    /* Compute absolute offsets and sizes */
    for (int i = 0; i < gg->num_tensors; i++) {
        ib_gguf_tensor* t = &gg->tensors[i];
        t->abs_offset = gg->data_offset + t->offset;

        /* Compute size from shape and type */
        int64_t elems = 1;
        for (int d = 0; d < t->ndim; d++) elems *= t->shape[d];
        int bs = ggml_block_size(t->type);
        int64_t blocks = (elems + bs - 1) / bs;
        t->size = (size_t)(blocks * ggml_type_size(t->type) * bs);
    }

    return gg;

fail2:
    free(gg->tensors);
    free(gg->metadata);
    free(gg);
fail:
    ib_set_error("failed to parse GGUF header");
    ib_munmap(base, file_size);
    ib_close(fd);
    return NULL;
}

void ib_gguf_close(ib_gguf* gg) {
    if (!gg) return;
    if (gg->mmap_base) ib_munmap(gg->mmap_base, gg->mmap_size);
    if (gg->fd >= 0) ib_close(gg->fd);
    free(gg->tensors);
    free(gg->metadata);
    free(gg);
}

/* ── Accessors ──────────────────────────────────────────────── */

int ib_gguf_num_tensors(const ib_gguf* gg) {
    return gg ? gg->num_tensors : 0;
}

int ib_gguf_find(const ib_gguf* gg, const char* name) {
    if (!gg || !name) return -1;
    for (int i = 0; i < gg->num_tensors; i++) {
        if (strcmp(gg->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

int ib_gguf_find_suffix(const ib_gguf* gg, const char* suffix) {
    if (!gg || !suffix) return -1;
    size_t slen = strlen(suffix);
    for (int i = 0; i < gg->num_tensors; i++) {
        size_t nlen = strlen(gg->tensors[i].name);
        if (nlen >= slen && strcmp(gg->tensors[i].name + nlen - slen, suffix) == 0)
            return i;
    }
    return -1;
}

const void* ib_gguf_tensor_data(const ib_gguf* gg, int index) {
    if (!gg || index < 0 || index >= gg->num_tensors) return NULL;
    return (const uint8_t*)gg->mmap_base + gg->tensors[index].abs_offset;
}

size_t ib_gguf_tensor_size(const ib_gguf* gg, int index) {
    if (!gg || index < 0 || index >= gg->num_tensors) return 0;
    return gg->tensors[index].size;
}

int ib_gguf_tensor_type(const ib_gguf* gg, int index) {
    if (!gg || index < 0 || index >= gg->num_tensors) return -1;
    return gg->tensors[index].type;
}

int ib_gguf_tensor_shape(const ib_gguf* gg, int index, int dim) {
    if (!gg || index < 0 || index >= gg->num_tensors) return 0;
    if (dim < 0 || dim >= gg->tensors[index].ndim) return 0;
    return (int)gg->tensors[index].shape[dim];
}

int ib_gguf_tensor_ndim(const ib_gguf* gg, int index) {
    if (!gg || index < 0 || index >= gg->num_tensors) return 0;
    return gg->tensors[index].ndim;
}

const char* ib_gguf_tensor_name(const ib_gguf* gg, int index) {
    if (!gg || index < 0 || index >= gg->num_tensors) return NULL;
    return gg->tensors[index].name;
}

/* ── Metadata access ────────────────────────────────────────── */

int ib_gguf_meta_find(const ib_gguf* gg, const char* key) {
    if (!gg || !key) return -1;
    for (int i = 0; i < gg->num_metadata; i++) {
        if (strcmp(gg->metadata[i].key, key) == 0) return i;
    }
    return -1;
}

int ib_gguf_meta_int(const ib_gguf* gg, const char* key, int def) {
    int idx = ib_gguf_meta_find(gg, key);
    if (idx < 0) return def;
    ib_gguf_meta* m = &gg->metadata[idx];
    switch (m->type) {
        case GGUF_TYPE_UINT8:  return m->value.u8;
        case GGUF_TYPE_INT8:   return m->value.i8;
        case GGUF_TYPE_UINT32: return (int)m->value.u32;
        case GGUF_TYPE_INT32:  return m->value.i32;
        case GGUF_TYPE_UINT64: return (int)m->value.u64;
        case GGUF_TYPE_INT64:  return (int)m->value.i64;
        default: return def;
    }
}

float ib_gguf_meta_float(const ib_gguf* gg, const char* key, float def) {
    int idx = ib_gguf_meta_find(gg, key);
    if (idx < 0) return def;
    ib_gguf_meta* m = &gg->metadata[idx];
    if (m->type == GGUF_TYPE_FLOAT32) return m->value.f32;
    if (m->type == GGUF_TYPE_FLOAT64) return (float)m->value.f64;
    return def;
}

const char* ib_gguf_meta_string(const ib_gguf* gg, const char* key) {
    int idx = ib_gguf_meta_find(gg, key);
    if (idx < 0) return NULL;
    if (gg->metadata[idx].type != GGUF_TYPE_STRING) return NULL;
    return gg->metadata[idx].value.str;
}

/* ── Extract architecture config from GGUF metadata ─────────── */

int ib_gguf_get_config(const ib_gguf* gg, ib_model_config* cfg) {
    if (!gg || !cfg) return -1;
    memset(cfg, 0, sizeof(*cfg));

    const char* arch = ib_gguf_meta_string(gg, "general.architecture");
    strncpy(cfg->arch, arch ? arch : "llama", sizeof(cfg->arch) - 1);

    /* Architecture-prefixed keys: e.g., "llama.embedding_length" */
    char key[256];

    #define META_INT(field, suffix, def) do { \
        snprintf(key, sizeof(key), "%s." suffix, cfg->arch); \
        cfg->field = ib_gguf_meta_int(gg, key, def); \
    } while(0)

    #define META_FLOAT(field, suffix, def) do { \
        snprintf(key, sizeof(key), "%s." suffix, cfg->arch); \
        cfg->field = ib_gguf_meta_float(gg, key, def); \
    } while(0)

    META_INT(hidden_size, "embedding_length", 0);
    META_INT(num_layers, "block_count", 0);
    META_INT(num_heads, "attention.head_count", 0);
    META_INT(num_kv_heads, "attention.head_count_kv", cfg->num_heads);
    META_INT(vocab_size, "vocab_size", 0);
    META_INT(max_context_length, "context_length", 4096);
    META_INT(intermediate_size, "feed_forward_length", cfg->hidden_size * 4);
    META_FLOAT(rope_theta, "rope.freq_base", 10000.0f);
    META_FLOAT(norm_epsilon, "attention.layer_norm_rms_epsilon", 1e-5f);

    #undef META_INT
    #undef META_FLOAT

    cfg->head_dim = cfg->num_heads > 0 ? cfg->hidden_size / cfg->num_heads : 128;

    /* Vocab size may also come from tokenizer metadata */
    if (cfg->vocab_size == 0) {
        cfg->vocab_size = ib_gguf_meta_int(gg, "tokenizer.ggml.tokens.length", 0);
    }
    /* Or count from token embedding tensor shape */
    if (cfg->vocab_size == 0) {
        int idx = ib_gguf_find_suffix(gg, "token_embd.weight");
        if (idx >= 0) cfg->vocab_size = ib_gguf_tensor_shape(gg, idx, 1);
    }

    strncpy(cfg->norm_type, "rmsnorm", sizeof(cfg->norm_type) - 1);
    strncpy(cfg->activation, "silu", sizeof(cfg->activation) - 1);

    return (cfg->hidden_size > 0 && cfg->num_layers > 0) ? 0 : -1;
}
