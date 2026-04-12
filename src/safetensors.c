/*
 * safetensors.c — Parse safetensors format
 *
 * Format:
 *   [0..7]   Header size (uint64 LE)
 *   [8..8+H) JSON header: { "tensor_name": { "dtype": "F16", "shape": [M,N], "data_offsets": [start, end] }, ... }
 *   [8+H..)  Raw tensor data
 */

#include "inferbit_internal.h"
#include "platform.h"
#include "cJSON.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Tensor entry from safetensors ──────────────────────────── */

typedef struct {
    char    name[256];
    char    dtype[16];     /* "F16", "F32", "BF16", "I8", etc. */
    int     shape[4];
    int     ndim;
    size_t  data_start;    /* Offset within data section */
    size_t  data_end;
    size_t  abs_offset;    /* Absolute offset in file = 8 + header_size + data_start */
} ib_st_tensor;

struct ib_safetensors {
    ib_st_tensor* tensors;
    int           num_tensors;
    void*         mmap_base;
    size_t        mmap_size;
    int           fd;
    size_t        data_offset;  /* Absolute offset where tensor data begins */
};

/* ── Parse safetensors file ─────────────────────────────────── */

ib_safetensors* ib_st_open(const char* path) {
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

    if (file_size < 8) {
        ib_set_error("file too small for safetensors: %zu", file_size);
        ib_close(fd);
        return NULL;
    }

    /* mmap entire file */
    void* base = ib_mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) {
        ib_set_error("failed to mmap %s", path);
        ib_close(fd);
        return NULL;
    }

    /* Read header size (uint64 LE) */
    uint64_t header_size;
    memcpy(&header_size, base, 8);

    if (8 + header_size > file_size) {
        ib_set_error("safetensors header size exceeds file");
        ib_munmap(base, file_size);
        ib_close(fd);
        return NULL;
    }

    /* Parse JSON header */
    const char* json_str = (const char*)base + 8;
    cJSON* root = cJSON_ParseWithLength(json_str, (size_t)header_size);
    if (!root) {
        ib_set_error("failed to parse safetensors header JSON");
        ib_munmap(base, file_size);
        ib_close(fd);
        return NULL;
    }

    /* Count tensors (skip __metadata__) */
    int count = 0;
    cJSON* item;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") == 0) continue;
        count++;
    }

    ib_safetensors* sf = calloc(1, sizeof(ib_safetensors));
    sf->tensors = calloc(count, sizeof(ib_st_tensor));
    sf->num_tensors = count;
    sf->mmap_base = base;
    sf->mmap_size = file_size;
    sf->fd = fd;
    sf->data_offset = 8 + (size_t)header_size;

    int idx = 0;
    cJSON_ArrayForEach(item, root) {
        if (strcmp(item->string, "__metadata__") == 0) continue;
        if (idx >= count) break;

        ib_st_tensor* t = &sf->tensors[idx];
        strncpy(t->name, item->string, sizeof(t->name) - 1);

        cJSON* dtype = cJSON_GetObjectItemCaseSensitive(item, "dtype");
        if (cJSON_IsString(dtype)) {
            strncpy(t->dtype, dtype->valuestring, sizeof(t->dtype) - 1);
        }

        cJSON* shape = cJSON_GetObjectItemCaseSensitive(item, "shape");
        if (cJSON_IsArray(shape)) {
            t->ndim = cJSON_GetArraySize(shape);
            if (t->ndim > 4) t->ndim = 4;
            for (int i = 0; i < t->ndim; i++) {
                cJSON* dim = cJSON_GetArrayItem(shape, i);
                t->shape[i] = cJSON_IsNumber(dim) ? dim->valueint : 0;
            }
        }

        cJSON* offsets = cJSON_GetObjectItemCaseSensitive(item, "data_offsets");
        if (cJSON_IsArray(offsets) && cJSON_GetArraySize(offsets) == 2) {
            t->data_start = (size_t)cJSON_GetArrayItem(offsets, 0)->valuedouble;
            t->data_end   = (size_t)cJSON_GetArrayItem(offsets, 1)->valuedouble;
        }

        t->abs_offset = sf->data_offset + t->data_start;
        idx++;
    }

    cJSON_Delete(root);
    return sf;
}

void ib_st_close(ib_safetensors* sf) {
    if (!sf) return;
    if (sf->mmap_base) ib_munmap(sf->mmap_base, sf->mmap_size);
    if (sf->fd >= 0) ib_close(sf->fd);
    free(sf->tensors);
    free(sf);
}

/* Get raw pointer to tensor data */
const void* ib_st_tensor_data(const ib_safetensors* sf, int index) {
    if (!sf || index < 0 || index >= sf->num_tensors) return NULL;
    return (const uint8_t*)sf->mmap_base + sf->tensors[index].abs_offset;
}

size_t ib_st_tensor_size(const ib_safetensors* sf, int index) {
    if (!sf || index < 0 || index >= sf->num_tensors) return 0;
    return sf->tensors[index].data_end - sf->tensors[index].data_start;
}

/* Find tensor by name */
int ib_st_find(const ib_safetensors* sf, const char* name) {
    if (!sf || !name) return -1;
    for (int i = 0; i < sf->num_tensors; i++) {
        if (strcmp(sf->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

/* Find tensor by suffix (e.g., "layers.0.self_attn.q_proj.weight") */
int ib_st_find_suffix(const ib_safetensors* sf, const char* suffix) {
    if (!sf || !suffix) return -1;
    size_t slen = strlen(suffix);
    for (int i = 0; i < sf->num_tensors; i++) {
        size_t nlen = strlen(sf->tensors[i].name);
        if (nlen >= slen && strcmp(sf->tensors[i].name + nlen - slen, suffix) == 0) {
            return i;
        }
    }
    return -1;
}

int ib_st_num_tensors(const ib_safetensors* sf) {
    return sf ? sf->num_tensors : 0;
}

const char* ib_st_tensor_name_at(const ib_safetensors* sf, int index) {
    if (!sf || index < 0 || index >= sf->num_tensors) return NULL;
    return sf->tensors[index].name;
}

const char* ib_st_tensor_dtype_at(const ib_safetensors* sf, int index) {
    if (!sf || index < 0 || index >= sf->num_tensors) return NULL;
    return sf->tensors[index].dtype;
}

int ib_st_tensor_ndim_at(const ib_safetensors* sf, int index) {
    if (!sf || index < 0 || index >= sf->num_tensors) return 0;
    return sf->tensors[index].ndim;
}

int ib_st_tensor_shape_at(const ib_safetensors* sf, int index, int dim) {
    if (!sf || index < 0 || index >= sf->num_tensors) return 0;
    if (dim < 0 || dim >= sf->tensors[index].ndim) return 0;
    return sf->tensors[index].shape[dim];
}
