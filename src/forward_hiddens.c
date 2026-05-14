/*
 * forward_hiddens.c — doc 36 phase 4.1
 *
 * inferbit_forward_with_hiddens: a prefill forward pass that ALSO returns
 * each requested layer's post-residual hidden state. This is the C-side
 * hook the DFlash hybrid orchestrator (phase 4.3) needs — the draft model
 * conditions its K-token predictions on the target's mid-stack hidden
 * states.
 *
 * The target runs through libinferbit's Metal kernels (including drive
 * mode). The Metal context + uploaded buffers are created lazily on the
 * first call and cached on the model, then freed in inferbit_free.
 */
#include "inferbit_internal.h"
#include <stdlib.h>

#ifdef IB_HAS_METAL
#include "metal/metal_runtime.h"
#endif

/* Defined in forward.c — decodes one token ID into an fp32 embedding row. */
void ib_embedding_lookup(const inferbit_model* m, int token_id, float* out);

/* Evenly-spaced target-layer selection, mirroring dflash's
 * build_target_layer_ids: pick n_draft_layers indices spread across the
 * target's depth, skipping the first and last 3 layers (those are too
 * token-local / too output-specialised to be useful draft conditioning).
 * Writes ascending indices into out_ids (caller supplies n_draft_layers
 * ints of space). Returns the count actually written. */
int inferbit_build_target_layer_ids(int n_target_layers, int n_draft_layers,
                                    int* out_ids) {
    if (!out_ids || n_target_layers <= 0 || n_draft_layers <= 0) return 0;
    int lo = 3, hi = n_target_layers - 4;          /* inclusive usable range */
    if (hi < lo) { lo = 0; hi = n_target_layers - 1; }  /* tiny-model fallback */
    int span = hi - lo;
    int n = n_draft_layers;
    if (n > span + 1) n = span + 1;                /* range can't hold more */
    for (int i = 0; i < n; i++) {
        out_ids[i] = (n == 1)
            ? (lo + span / 2)
            : (lo + (int)((long)i * span / (n - 1)));
    }
    return n;
}

#ifndef IB_HAS_METAL

/* Metal backend not compiled in — the hidden-state capture path runs on
 * the GPU prefill kernels, so there is nothing to fall back to. */
int inferbit_forward_with_hiddens(inferbit_model* model,
                                  const int32_t* tokens, int n_tokens,
                                  const int* layer_ids, int n_layer_ids,
                                  float* hiddens_out, float* logits_out) {
    (void)model; (void)tokens; (void)n_tokens;
    (void)layer_ids; (void)n_layer_ids; (void)hiddens_out; (void)logits_out;
    ib_set_error("inferbit_forward_with_hiddens requires the Metal backend "
                 "(build with -DIB_ENABLE_METAL=ON)");
    return INFERBIT_ERROR_INTERNAL;
}

#else  /* IB_HAS_METAL */

int inferbit_forward_with_hiddens(inferbit_model* model,
                                  const int32_t* tokens, int n_tokens,
                                  const int* layer_ids, int n_layer_ids,
                                  float* hiddens_out, float* logits_out) {
    if (!model || !tokens || !logits_out) {
        ib_set_error("NULL argument to inferbit_forward_with_hiddens");
        return INFERBIT_ERROR_PARAM;
    }
    if (n_tokens <= 0) {
        ib_set_error("n_tokens must be > 0");
        return INFERBIT_ERROR_PARAM;
    }

    int num_layers = model->header.num_layers;
    int hidden     = model->header.hidden_size;

    /* Lazy GPU upload, cached on the model for subsequent calls. */
    if (!model->metal_ctx) {
        ib_metal_ctx* ctx = ib_metal_create();
        if (!ctx) {
            ib_set_error("ib_metal_create failed");
            return INFERBIT_ERROR_INTERNAL;
        }
        ib_metal_model_buffers* bufs = ib_metal_upload_model(ctx, model);
        if (!bufs) {
            ib_metal_destroy(ctx);
            ib_set_error("ib_metal_upload_model failed");
            return INFERBIT_ERROR_INTERNAL;
        }
        model->metal_ctx  = ctx;
        model->metal_bufs = bufs;
    }
    ib_metal_ctx*           ctx  = (ib_metal_ctx*)model->metal_ctx;
    ib_metal_model_buffers* bufs = (ib_metal_model_buffers*)model->metal_bufs;

    /* Token IDs -> fp32 embeddings [n_tokens][hidden]. */
    float* embeds = (float*)malloc((size_t)n_tokens * hidden * sizeof(float));
    if (!embeds) {
        ib_set_error("oom: embeddings buffer");
        return INFERBIT_ERROR_MEMORY;
    }
    for (int t = 0; t < n_tokens; t++) {
        ib_embedding_lookup(model, tokens[t], embeds + (size_t)t * hidden);
    }

    /* Per-layer capture pointers. hiddens_out is laid out
     * [n_layer_ids][n_tokens][hidden]; hs[L] points at the slab for a
     * requested layer, NULL for the rest. NULL hs = logits-only fast path. */
    float** hs = NULL;
    if (hiddens_out && layer_ids && n_layer_ids > 0) {
        hs = (float**)calloc((size_t)num_layers, sizeof(float*));
        if (!hs) {
            free(embeds);
            ib_set_error("oom: hidden-state pointer array");
            return INFERBIT_ERROR_MEMORY;
        }
        for (int i = 0; i < n_layer_ids; i++) {
            int L = layer_ids[i];
            if (L < 0 || L >= num_layers) {
                free(hs);
                free(embeds);
                ib_set_error("layer_id %d out of range [0,%d)", L, num_layers);
                return INFERBIT_ERROR_PARAM;
            }
            hs[L] = hiddens_out + (size_t)i * n_tokens * hidden;
        }
    }

    ib_metal_reset_kv(bufs);
    int rc = ib_metal_forward_prefill_logits_all_ex(ctx, bufs, embeds,
                                                    n_tokens, 0,
                                                    logits_out, hs);
    free(hs);
    free(embeds);
    if (rc != 0) {
        ib_set_error("ib_metal_forward_prefill_logits_all_ex rc=%d", rc);
        return INFERBIT_ERROR_INTERNAL;
    }
    return INFERBIT_OK;
}

#endif  /* IB_HAS_METAL */
