#include "inferbit_internal.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* Defined in ibf_loader.c */
inferbit_model* ibf_load(const char* path, const inferbit_config* config);

/* ── Check file extension ───────────────────────────────────── */

static int ends_with(const char* str, const char* suffix) {
    size_t slen = strlen(str);
    size_t xlen = strlen(suffix);
    if (xlen > slen) return 0;
    return strcmp(str + slen - xlen, suffix) == 0;
}

/* ── Load ───────────────────────────────────────────────────── */

inferbit_model* inferbit_load(const char* path, const inferbit_config* config) {
    if (!path) {
        ib_set_error("path is NULL");
        return NULL;
    }

    if (config && config->native_parse) {
        /* TODO: native parse mode for safetensors/GGUF */
        ib_set_error("native parse mode not yet implemented");
        return NULL;
    }

    if (!ends_with(path, ".ibf")) {
        ib_set_error("expected .ibf file (use native_parse mode for safetensors/GGUF)");
        return NULL;
    }

    return ibf_load(path, config);
}

/* ── Free ───────────────────────────────────────────────────── */

void inferbit_free(inferbit_model* model) {
    if (!model) return;

    /* Unmap weight data */
    if (model->weight_data_mmap && model->weight_data) {
        /*
         * weight_data points into the mmap'd region (with offset).
         * We need the original mmap base and full file size to munmap.
         * For now, we stored the fd — re-stat to get file size.
         */
        if (model->mmap_fd >= 0) {
            struct stat st;
            if (fstat(model->mmap_fd, &st) == 0) {
                /* Compute mmap base: weight_data minus the weight offset */
                size_t weight_offset = model->header.weight_data_offset;
                void* base = (uint8_t*)model->weight_data - weight_offset;
                munmap(base, (size_t)st.st_size);
            }
            close(model->mmap_fd);
        }
    }

    /* Destroy thread pool */
    ib_pool_destroy(model->thread_pool);

    /* Free KV caches */
    if (model->kv_caches) {
        for (int i = 0; i < model->header.num_layers; i++) {
            ib_kv_cache* kv = &model->kv_caches[i];
            free(kv->key_data);
            free(kv->value_data);
            free(kv->key_scales);
            free(kv->value_scales);
        }
        free(model->kv_caches);
    }

    /* Free activation buffers */
    free(model->buf_residual);
    free(model->buf_hidden);
    free(model->buf_attn);
    free(model->buf_mlp);
    free(model->buf_mlp2);
    free(model->buf_logits);
    free(model->buf_qkv);

    /* Free layer metadata */
    free(model->layers);

    free(model);
}

/* ── Model info ─────────────────────────────────────────────── */

const char* inferbit_model_architecture(const inferbit_model* m) {
    return m ? m->header.architecture : "unknown";
}

int inferbit_model_num_layers(const inferbit_model* m) {
    return m ? m->header.num_layers : 0;
}

int inferbit_model_hidden_size(const inferbit_model* m) {
    return m ? m->header.hidden_size : 0;
}

int inferbit_model_vocab_size(const inferbit_model* m) {
    return m ? m->header.vocab_size : 0;
}

int inferbit_model_max_context(const inferbit_model* m) {
    return m ? m->header.max_context_length : 0;
}

int inferbit_model_default_bits(const inferbit_model* m) {
    return m ? m->header.default_bits : 0;
}

size_t inferbit_model_weight_memory(const inferbit_model* m) {
    return m ? m->weight_data_size : 0;
}

size_t inferbit_model_kv_memory(const inferbit_model* m) {
    if (!m || !m->kv_caches) return 0;
    size_t total = 0;
    for (int i = 0; i < m->header.num_layers; i++) {
        ib_kv_cache* kv = &m->kv_caches[i];
        if (kv->key_data) {
            /* Estimate from capacity and head dimensions */
            int kv_heads = m->header.num_kv_heads;
            int head_dim = m->header.head_dim;
            int bits = m->header.kv_bits;
            size_t bytes_per_token = (size_t)kv_heads * head_dim * bits / 8;
            total += kv->capacity * bytes_per_token * 2;  /* keys + values */
        }
    }
    return total;
}

size_t inferbit_model_total_memory(const inferbit_model* m) {
    return inferbit_model_weight_memory(m) + inferbit_model_kv_memory(m);
}
