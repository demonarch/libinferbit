/*
 * dflash_orchestrator.c — Phase 4 hybrid orchestrator (doc 36)
 *
 * Confidence-gated early-exit decoder. Reads the post-residual hidden
 * state at a configurable early layer (captured during every full
 * forward via the 3-line hook in forward_single_ex) and uses its
 * L2-norm as a "confidence" signal. When the previous step's signal
 * indicates the model is confident — and we're past warmup — the next
 * decode step short-circuits: instead of running the remaining N - L
 * layers, we project the captured early-layer hidden state directly
 * through the final RMSNorm + LM head to produce logits.
 *
 * v1 — CPU only.
 *   - Prefill (num_tokens > 1) always full forward. Per-token early exit
 *     only kicks in during decode.
 *   - Metal-routed models bypass DFlash entirely. The orchestrator
 *     declines to handle the request (sets *handled = 0) and ib_forward
 *     falls through to the existing Metal path. A future extension would
 *     wire DFlash into the Metal prefill/decode pipeline as well — see
 *     TODO at the bottom of this file.
 *   - The early-exit projection path does NOT advance the KV cache for
 *     the skipped layers. This is intentional: the captured hidden was
 *     written AT the early-exit layer during a previous FULL forward, at
 *     which point the KV cache was fully populated for all N layers.
 *     The early-exit path of the CURRENT step uses a captured hidden
 *     from the PREVIOUS step, so the current step never produces a
 *     post-stack hidden at all — which means the KV cache for the
 *     current step's layer-l+1..N never gets a new row written.
 *
 *     Implication: on early-exit steps the KV cache lags by one
 *     position for layers > early_exit_layer. The simplest sound way
 *     to handle this in v1 is: when we take an early exit, we still
 *     need to update the KV cache for layers up to early_exit_layer
 *     (so the next step's attention has the K/V for the current
 *     token). The cleanest implementation is to run a full forward
 *     anyway when we *would* early-exit, but discard its top half —
 *     that defeats the purpose of early exit. The alternative is a
 *     partial forward through early_exit_layer that writes KV only
 *     for those layers and projects out via lm_head; layers above
 *     early_exit_layer will not see this token at all, so the next
 *     full forward's attention in those layers will skip a position.
 *
 *     Quality cost of the latter approach is bounded: only the top
 *     N - L layers' attention misses one historical token per
 *     early-exit step. For sliding-window / kv-window models that's
 *     in-line with the existing horizon truncation. For full-cache
 *     models it's a real but small lossy approximation. This is the
 *     DFlash quality-vs-throughput trade-off the threshold knob
 *     controls.
 *
 *     For v1 we ship the SIMPLER variant: when we *would* early
 *     exit, we route to ib_forward_single anyway (full forward) so
 *     the KV cache stays consistent, and we *still* count this as
 *     an early-exit step in the stats AS IF we'd taken it — i.e.
 *     v1 measures the decision rate, not the actual cycles saved.
 *     The actual lm_head-from-early-hidden projection is wired up
 *     and unit-testable, but to avoid KV-state corruption we don't
 *     skip the layer loop yet. The next session can flip the switch
 *     once the KV-handling story is decided. See TODO below.
 */

#include "inferbit_internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Public API ─────────────────────────────────────────────── */

int inferbit_dflash_attach(inferbit_model* model,
                           const inferbit_dflash_config* cfg) {
    if (!model || !cfg) {
        ib_set_error("inferbit_dflash_attach: NULL argument");
        return INFERBIT_ERROR_PARAM;
    }
    int n_layers = model->header.num_layers;
    int hidden   = model->header.hidden_size;
    if (cfg->early_exit_layer < 0 || cfg->early_exit_layer >= n_layers) {
        ib_set_error("inferbit_dflash_attach: early_exit_layer %d out of range [0, %d)",
                     cfg->early_exit_layer, n_layers);
        return INFERBIT_ERROR_PARAM;
    }
    if (!(cfg->confidence_threshold >= 0.0f)) {
        /* Catches NaN as well as negatives. 0 = always full forward. */
        ib_set_error("inferbit_dflash_attach: confidence_threshold must be >= 0");
        return INFERBIT_ERROR_PARAM;
    }
    if (cfg->warmup_tokens < 0) {
        ib_set_error("inferbit_dflash_attach: warmup_tokens must be >= 0");
        return INFERBIT_ERROR_PARAM;
    }

    /* If already attached, replace cleanly. */
    if (model->dflash_cfg) {
        (void)inferbit_dflash_detach(model);
    }

    inferbit_dflash_config* owned = malloc(sizeof(*owned));
    if (!owned) {
        ib_set_error("inferbit_dflash_attach: out of memory (cfg)");
        return INFERBIT_ERROR_MEMORY;
    }
    *owned = *cfg;

    float* capbuf = calloc((size_t)hidden, sizeof(float));
    if (!capbuf) {
        free(owned);
        ib_set_error("inferbit_dflash_attach: out of memory (capture buffer)");
        return INFERBIT_ERROR_MEMORY;
    }

    model->dflash_cfg          = owned;
    model->dflash_capture_buf  = capbuf;
    model->dflash_last_norm    = 0.0f;
    model->dflash_decode_step  = 0;
    model->dflash_full_count   = 0;
    model->dflash_early_count  = 0;
    return INFERBIT_OK;
}

int inferbit_dflash_detach(inferbit_model* model) {
    if (!model) {
        ib_set_error("inferbit_dflash_detach: NULL model");
        return INFERBIT_ERROR_PARAM;
    }
    free(model->dflash_cfg);
    free(model->dflash_capture_buf);
    model->dflash_cfg          = NULL;
    model->dflash_capture_buf  = NULL;
    model->dflash_last_norm    = 0.0f;
    model->dflash_decode_step  = 0;
    model->dflash_full_count   = 0;
    model->dflash_early_count  = 0;
    return INFERBIT_OK;
}

int inferbit_dflash_last_full_count(const inferbit_model* model) {
    return model ? model->dflash_full_count : 0;
}

int inferbit_dflash_last_early_count(const inferbit_model* model) {
    return model ? model->dflash_early_count : 0;
}

/* ── Internal: numerics helper ───────────────────────────────── */

static float vec_l2_norm(const float* v, int n) {
    /* Plain double-accumulator L2. The captured hidden is hidden_size ≤
     * 8192 in practice; a SIMD kernel isn't load-bearing here — this
     * runs once per decode step. */
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        s += (double)v[i] * (double)v[i];
    }
    return (float)sqrt(s);
}

/* ── Internal: routing entrypoint ────────────────────────────── */

/* Decide whether the orchestrator handles this forward call. If yes,
 * dispatch and write logits; if no, set *handled = 0 and let ib_forward
 * fall through to the existing routing. */
int ib_dflash_try_route(inferbit_model* model,
                        const int32_t* tokens,
                        int num_tokens,
                        float* out_logits,
                        int* handled) {
    *handled = 0;

    if (!model || !model->dflash_cfg) return INFERBIT_OK;

    /* Prefill: not DFlash-applicable. Decline and let the standard
     * batched prefill run. The capture hook in forward_single_ex won't
     * fire on the batched path (forward_batch doesn't capture), so the
     * confidence signal will simply not refresh during prefill — which
     * is fine: the first decode step will run as a warmup full forward
     * and seed dflash_last_norm. */
    if (num_tokens != 1) return INFERBIT_OK;

    /* Metal-routed calls never reach the orchestrator — the routing
     * hook in ib_forward() runs the Metal check first and returns
     * before reaching us. (See forward.c, near the top of ib_forward.)
     * TODO: wire DFlash into the Metal forward (capture the hidden at
     * early_exit_layer on the GPU, project via the existing Metal
     * output-head kernel). */

    /* From here on: single-token CPU decode with a DFlash config. We
     * own the call. */
    *handled = 1;

    const inferbit_dflash_config* cfg = model->dflash_cfg;
    int hidden = model->header.hidden_size;
    int pos    = inferbit_kv_length(model);
    int tok    = tokens[0];

    /* Decision: based on the PREVIOUS step's captured norm. We can only
     * early-exit once we have a captured signal AND we're past warmup. */
    int can_consider_early_exit =
        (model->dflash_decode_step >= cfg->warmup_tokens) &&
        (model->dflash_last_norm > 0.0f);

    int take_early_exit =
        can_consider_early_exit &&
        (cfg->confidence_threshold > 0.0f) &&
        (model->dflash_last_norm < cfg->confidence_threshold);

    /* Run the forward. In v1 we ALWAYS run the full forward, even when
     * take_early_exit is true — see file header for the KV-consistency
     * reasoning. The full forward populates dflash_capture_buf via the
     * hook in forward_single_ex. */
    int rc = ib_forward_single(model, tok, pos, out_logits);
    if (rc != INFERBIT_OK) return rc;

    /* Refresh the confidence signal from this step's captured hidden. */
    if (model->dflash_capture_buf) {
        model->dflash_last_norm =
            vec_l2_norm(model->dflash_capture_buf, hidden);
    }

    if (take_early_exit) {
        /* Project the captured early-layer hidden state through the LM
         * head. The result OVERWRITES out_logits — the caller wants the
         * early-exit logits, by definition.
         *
         * We need a scratch buffer for the matmul. buf_logits is
         * vocab-sized and conveniently free at this point (we already
         * wrote the full-path logits into out_logits). But the LM-head
         * helper also takes a scale_buf sized >= max(hidden, inter,
         * vocab). buf_qkv has the appropriate sizing at model-load (see
         * the partition in forward_single_ex). Use it.
         *
         * IMPORTANT: ib_apply_lm_head_finalize clobbers its hidden_io
         * arg in place (RMSNorm), so we run on a temp copy — leaving
         * dflash_capture_buf intact for inspection / norm calc above. */
        float* hidden_io = calloc((size_t)hidden, sizeof(float));
        if (hidden_io) {
            memcpy(hidden_io, model->dflash_capture_buf,
                   (size_t)hidden * sizeof(float));
            /* buf_qkv is sized to accommodate the q+k+v+att+scale layout in
             * forward_single_ex; the scale_buf slice at its tail is the
             * canonical scratch for the LM-head matmul. For simplicity in
             * v1 we just hand the front of buf_qkv as scratch — it's
             * already vocab+ floats in capacity. */
            ib_apply_lm_head_finalize(model, hidden_io, out_logits,
                                      model->buf_qkv);
            free(hidden_io);
        }
        model->dflash_early_count++;
    } else {
        model->dflash_full_count++;
    }

    model->dflash_decode_step++;
    return INFERBIT_OK;
}

/* ── TODOs for the next session ──────────────────────────────────
 *
 * 1. Decide the KV-cache story for true early-exit. Two options:
 *    (a) On an early-exit decision, run a TRUNCATED forward — embed +
 *        layers [0 .. early_exit_layer] only — writing KV for those
 *        layers but skipping the rest. Cheap. Cost: layers >
 *        early_exit_layer permanently miss the early-exit token in
 *        their attention horizon. Compatible with the kv_window
 *        rotating cache; less so with the full causal cache.
 *    (b) Hybrid: on early-exit, run the truncated forward AND inject
 *        the captured early-layer hidden as an approximation of the
 *        post-stack hidden into layers > early_exit_layer's KV cache.
 *        Higher quality, more code.
 *    (c) (current v1) Always run full forward; measure decision rate
 *        only. No speedup, but the API + decision logic is testable.
 *
 *    Once decided, the truncated-forward variant should be a small
 *    static function in forward.c (NOT here — it needs the static
 *    tensor_matmul/kv_cache_write/rmsnorm_fp16 helpers), exposed via
 *    a single ib_forward_single_truncated(model, token, pos, max_layer)
 *    declaration in inferbit_internal.h.
 *
 * 2. Metal extension. The capture hook would need a Metal-side
 *    counterpart (a kernel that snapshots the residual buffer at the
 *    end of layer L into a CPU-readable buffer). The orchestrator's
 *    decision logic is backend-agnostic and would route to a
 *    new ib_dflash_decode_metal(...).
 *
 * 3. Calibration. The user is expected to run their own offline
 *    sweep using inferbit_forward_with_hiddens + the *_count getters.
 *    A small Python helper in inferbit-py would be a natural
 *    follow-up (NOT in this library — keep the C surface lean).
 */
