#include "inferbit_internal.h"

#ifdef IB_HAS_METAL
#include "metal/metal_runtime.h"
#endif

/* TODO: Milestone 4 — implement KV cache allocation and management */

void inferbit_kv_clear(inferbit_model* model) {
    if (!model || !model->kv_caches) return;
    for (int i = 0; i < model->header.num_layers; i++) {
        model->kv_caches[i].length = 0;
    }
#ifdef IB_HAS_METAL
    /* When Metal-routed, KV state lives in the GPU buffers — reset it too. */
    if (model->metal_bufs)
        ib_metal_reset_kv((ib_metal_model_buffers*)model->metal_bufs);
#endif
}

void inferbit_kv_truncate(inferbit_model* model, int length) {
    if (!model || !model->kv_caches) return;
    if (length < 0) length = 0;
    for (int i = 0; i < model->header.num_layers; i++) {
        if (model->kv_caches[i].length > length) {
            model->kv_caches[i].length = length;
        }
    }
    /* No Metal-specific work needed: the Metal forwards derive start_pos
     * from kv_caches[].length, so lowering that counter above already
     * truncates the Metal KV path correctly. */
}

int inferbit_kv_length(const inferbit_model* model) {
    if (!model || !model->kv_caches) return 0;
    /* All layers have the same KV length */
    return model->kv_caches[0].length;
}
