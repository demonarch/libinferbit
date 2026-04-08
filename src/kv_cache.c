#include "inferbit_internal.h"

/* TODO: Milestone 4 — implement KV cache allocation and management */

void inferbit_kv_clear(inferbit_model* model) {
    if (!model || !model->kv_caches) return;
    for (int i = 0; i < model->header.num_layers; i++) {
        model->kv_caches[i].length = 0;
    }
}

void inferbit_kv_truncate(inferbit_model* model, int length) {
    if (!model || !model->kv_caches) return;
    if (length < 0) length = 0;
    for (int i = 0; i < model->header.num_layers; i++) {
        if (model->kv_caches[i].length > length) {
            model->kv_caches[i].length = length;
        }
    }
}

int inferbit_kv_length(const inferbit_model* model) {
    if (!model || !model->kv_caches) return 0;
    /* All layers have the same KV length */
    return model->kv_caches[0].length;
}
