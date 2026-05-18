#include "inferbit_internal.h"
#include "pqv2_format.h"
#include "platform.h"   /* ib_close, cross-platform I/O */
#include <stdlib.h>
#include <string.h>

#ifdef IB_HAS_METAL
#include "metal/metal_runtime.h"   /* lazy Metal ctx cleanup (phase 4.1) */
#endif

/* Defined in ibf_loader.c */
inferbit_model* ibf_load(const char* path, const inferbit_config* config);
/* Defined in pqv2_model.c — detects IBF v6 magic, falls back to v5 */
inferbit_model* pqv2_or_legacy_load(const char* path, const inferbit_config* config);

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

    return pqv2_or_legacy_load(path, config);
}

/* ── Free ───────────────────────────────────────────────────── */

void inferbit_free(inferbit_model* model) {
    if (!model) return;

#ifdef IB_HAS_METAL
    /* Stage 5d hybrid staging buffers — must be freed BEFORE metal_ctx
     * (they were allocated via ib_metal_alloc against that ctx). */
    if (model->metal_ctx) {
        if (model->hybrid_x_buf) {
            ib_metal_free((ib_metal_ctx*)model->metal_ctx, model->hybrid_x_buf);
            model->hybrid_x_buf = NULL;
            model->hybrid_x_buf_floats = 0;
        }
        if (model->hybrid_y_buf) {
            ib_metal_free((ib_metal_ctx*)model->metal_ctx, model->hybrid_y_buf);
            model->hybrid_y_buf = NULL;
            model->hybrid_y_buf_floats = 0;
        }
    }
    /* Lazily-created Metal context + buffers (doc 36 phase 4.1). */
    if (model->metal_bufs) {
        ib_metal_release_model((ib_metal_ctx*)model->metal_ctx,
                               (ib_metal_model_buffers*)model->metal_bufs);
        model->metal_bufs = NULL;
    }
    if (model->metal_ctx) {
        ib_metal_destroy((ib_metal_ctx*)model->metal_ctx);
        model->metal_ctx = NULL;
    }
#endif

    /* IBF v6 backing — release before clearing weight_data so we don't
     * double-free the mmap region (which is owned by the pqv2_file). */
    if (model->pqv2_file_backing) {
        ib_pqv2_file_free(model->pqv2_file_backing);
        free(model->pqv2_file_backing);
        model->pqv2_file_backing = NULL;
        /* The IBF v6 path owns the mmap; skip the legacy unmap below. */
        model->weight_data = NULL;
        model->weight_data_mmap = false;
        model->mmap_fd = -1;
    }
    if (model->pqv2_thread_acc_pool) {
        free(model->pqv2_thread_acc_pool);
        model->pqv2_thread_acc_pool = NULL;
    }
    if (model->drive_indices_scratch) {
        free(model->drive_indices_scratch);
        model->drive_indices_scratch = NULL;
        model->drive_indices_scratch_size = 0;
    }
    /* model->drive_fd is owned by pqv2_file_backing — don't close here. */
    /* model->drive_fd_pretransposed IS owned here (unlinked tmpfile). */
    if (model->drive_fd_pretransposed >= 0) {
        ib_close(model->drive_fd_pretransposed);
        model->drive_fd_pretransposed = -1;
    }

    /* Unmap weight data */
    if (model->weight_data_mmap && model->weight_data) {
        /*
         * weight_data points into the mmap'd region (with offset).
         * We need the original mmap base and full file size to munmap.
         * For now, we stored the fd — re-stat to get file size.
         */
        if (model->mmap_fd >= 0) {
            ib_struct_stat st;
            if (ib_fstat(model->mmap_fd, &st) == 0) {
                /* Compute mmap base: weight_data minus the weight offset */
                size_t weight_offset = model->header.weight_data_offset;
                void* base = (uint8_t*)model->weight_data - weight_offset;
                ib_munmap(base, (size_t)st.st_size);
            }
            ib_close(model->mmap_fd);
        }
    }

    /* Stripped-mmap path: weight_data is now an offset into a malloc'd
     * embedding-only buffer (set by ib_metal_strip_cpu_mmap). Free the
     * buffer rather than munmap'ing. */
    if (model->embed_strip_buffer) {
        free(model->embed_strip_buffer);
        model->embed_strip_buffer = NULL;
        model->weight_data = NULL;
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

    /* Free batched-forward scratch */
    free(model->bb_x);
    free(model->bb_xb);
    free(model->bb_xb2);
    free(model->bb_q);
    free(model->bb_k);
    free(model->bb_v);
    free(model->bb_hb);
    free(model->bb_hb2);
    free(model->bb_scale);
    free(model->bb_att);
    free(model->bb_qscratch);
    free(model->bb_sa);
    free(model->bb_positions);

    /* Free MoME per-layer expert arrays (Stage 3a, docs/v2/00_CORRECTION.md).
     * Each *_proj_experts is either NULL (mome_experts == 1) or a
     * calloc'd array of K ib_tensor_meta — its members' pq pointers
     * index into the IBF v6 file backing and are NOT owned here. */
    if (model->layers) {
        for (int li = 0; li < model->header.num_layers; li++) {
            ib_layer_meta *L = &model->layers[li];
            free(L->gate_proj_experts);
            free(L->up_proj_experts);
            free(L->down_proj_experts);
        }
    }

    /* Free layer metadata */
    free(model->layers);

    /* DFlash orchestrator state (Phase 4). free(NULL) is a no-op, so this
     * is safe whether or not inferbit_dflash_attach was ever called. */
    free(model->dflash_cfg);
    free(model->dflash_capture_buf);

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
