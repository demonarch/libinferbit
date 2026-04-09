/*
 * safetensors_multi.c — Multi-file safetensors support
 *
 * Handles sharded models (model-00001-of-00003.safetensors, etc.)
 * by opening all shards and providing a unified tensor lookup.
 */

#include "inferbit_internal.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Multi-shard container ──────────────────────────────────── */

struct ib_safetensors_multi {
    ib_safetensors** shards;
    int              num_shards;
};

/* Sort helper for filenames */
static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

ib_safetensors_multi* ib_st_multi_open(const char* dir_path) {
    DIR* dir = opendir(dir_path);
    if (!dir) {
        ib_set_error("failed to open directory: %s", dir_path);
        return NULL;
    }

    /* Collect .safetensors filenames */
    char* files[256];
    int count = 0;

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL && count < 256) {
        size_t len = strlen(entry->d_name);
        if (len > 12 && strcmp(entry->d_name + len - 12, ".safetensors") == 0) {
            char full[1024];
            snprintf(full, sizeof(full), "%s/%s", dir_path, entry->d_name);
            files[count] = strdup(full);
            count++;
        }
    }
    closedir(dir);

    if (count == 0) {
        ib_set_error("no .safetensors files found in %s", dir_path);
        return NULL;
    }

    /* Sort for deterministic order */
    qsort(files, count, sizeof(char*), cmp_str);

    /* Open all shards */
    ib_safetensors_multi* multi = calloc(1, sizeof(ib_safetensors_multi));
    multi->shards = calloc(count, sizeof(ib_safetensors*));
    multi->num_shards = 0;

    for (int i = 0; i < count; i++) {
        ib_safetensors* sf = ib_st_open(files[i]);
        if (sf) {
            multi->shards[multi->num_shards++] = sf;
        }
        free(files[i]);
    }

    if (multi->num_shards == 0) {
        ib_set_error("failed to open any safetensors shards in %s", dir_path);
        free(multi->shards);
        free(multi);
        return NULL;
    }

    return multi;
}

void ib_st_multi_close(ib_safetensors_multi* multi) {
    if (!multi) return;
    for (int i = 0; i < multi->num_shards; i++) {
        ib_st_close(multi->shards[i]);
    }
    free(multi->shards);
    free(multi);
}

/* Find a tensor across all shards. Returns shard index and tensor index. */
int ib_st_multi_find(const ib_safetensors_multi* multi, const char* name,
                     int* out_shard, int* out_tensor) {
    if (!multi || !name) return -1;
    for (int s = 0; s < multi->num_shards; s++) {
        int t = ib_st_find(multi->shards[s], name);
        if (t >= 0) {
            if (out_shard) *out_shard = s;
            if (out_tensor) *out_tensor = t;
            return 0;
        }
    }
    return -1;
}

int ib_st_multi_find_suffix(const ib_safetensors_multi* multi, const char* suffix,
                            int* out_shard, int* out_tensor) {
    if (!multi || !suffix) return -1;
    for (int s = 0; s < multi->num_shards; s++) {
        int t = ib_st_find_suffix(multi->shards[s], suffix);
        if (t >= 0) {
            if (out_shard) *out_shard = s;
            if (out_tensor) *out_tensor = t;
            return 0;
        }
    }
    return -1;
}

/* Get tensor data from a shard */
const void* ib_st_multi_tensor_data(const ib_safetensors_multi* multi, int shard, int tensor) {
    if (!multi || shard < 0 || shard >= multi->num_shards) return NULL;
    return ib_st_tensor_data(multi->shards[shard], tensor);
}

const char* ib_st_multi_tensor_dtype(const ib_safetensors_multi* multi, int shard, int tensor) {
    if (!multi || shard < 0 || shard >= multi->num_shards) return NULL;
    return ib_st_tensor_dtype_at(multi->shards[shard], tensor);
}

int ib_st_multi_tensor_shape(const ib_safetensors_multi* multi, int shard, int tensor, int dim) {
    if (!multi || shard < 0 || shard >= multi->num_shards) return 0;
    return ib_st_tensor_shape_at(multi->shards[shard], tensor, dim);
}

int ib_st_multi_num_shards(const ib_safetensors_multi* multi) {
    return multi ? multi->num_shards : 0;
}
