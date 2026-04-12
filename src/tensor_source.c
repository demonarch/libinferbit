/*
 * tensor_source.c — Unified tensor access for single-file and multi-shard safetensors
 *
 * Provides a single interface that the converter uses regardless of
 * whether the source is one file or many shards.
 */

#include "inferbit_internal.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>

struct ib_tensor_source {
    /* Exactly one of these is non-NULL */
    ib_safetensors*       single;
    ib_safetensors_multi* multi;
};

/* ── Open ───────────────────────────────────────────────────── */

static int is_directory(const char* path) {
    ib_struct_stat st;
    if (ib_stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode);
}

ib_tensor_source* ib_ts_open(const char* path) {
    ib_tensor_source* ts = calloc(1, sizeof(ib_tensor_source));
    if (!ts) return NULL;

    if (is_directory(path)) {
        ts->multi = ib_st_multi_open(path);
        if (!ts->multi) { free(ts); return NULL; }
    } else {
        ts->single = ib_st_open(path);
        if (!ts->single) { free(ts); return NULL; }
    }

    return ts;
}

void ib_ts_close(ib_tensor_source* ts) {
    if (!ts) return;
    if (ts->single) ib_st_close(ts->single);
    if (ts->multi) ib_st_multi_close(ts->multi);
    free(ts);
}

/* ── Find tensor ────────────────────────────────────────────── */

int ib_ts_find(const ib_tensor_source* ts, const char* name,
               int* out_shard, int* out_tensor) {
    if (!ts) return -1;

    if (ts->single) {
        int idx = ib_st_find(ts->single, name);
        if (idx >= 0) {
            if (out_shard) *out_shard = 0;
            if (out_tensor) *out_tensor = idx;
            return 0;
        }
        return -1;
    }

    return ib_st_multi_find(ts->multi, name, out_shard, out_tensor);
}

int ib_ts_find_suffix(const ib_tensor_source* ts, const char* suffix,
                      int* out_shard, int* out_tensor) {
    if (!ts) return -1;

    if (ts->single) {
        int idx = ib_st_find_suffix(ts->single, suffix);
        if (idx >= 0) {
            if (out_shard) *out_shard = 0;
            if (out_tensor) *out_tensor = idx;
            return 0;
        }
        return -1;
    }

    return ib_st_multi_find_suffix(ts->multi, suffix, out_shard, out_tensor);
}

/* ── Access tensor data ─────────────────────────────────────── */

const void* ib_ts_tensor_data(const ib_tensor_source* ts, int shard, int tensor) {
    if (!ts) return NULL;
    if (ts->single) return ib_st_tensor_data(ts->single, tensor);
    return ib_st_multi_tensor_data(ts->multi, shard, tensor);
}

const char* ib_ts_tensor_dtype(const ib_tensor_source* ts, int shard, int tensor) {
    if (!ts) return NULL;
    if (ts->single) return ib_st_tensor_dtype_at(ts->single, tensor);
    return ib_st_multi_tensor_dtype(ts->multi, shard, tensor);
}

int ib_ts_tensor_shape(const ib_tensor_source* ts, int shard, int tensor, int dim) {
    if (!ts) return 0;
    if (ts->single) return ib_st_tensor_shape_at(ts->single, tensor, dim);
    return ib_st_multi_tensor_shape(ts->multi, shard, tensor, dim);
}
