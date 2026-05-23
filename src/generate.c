#include "inferbit_internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Sampling ───────────────────────────────────────────────── */

inferbit_sample_params inferbit_default_sample_params(void) {
    inferbit_sample_params p;
    p.temperature    = 1.0f;
    p.top_k          = 40;
    p.top_p          = 0.9f;
    p.repeat_penalty = 1.0f;
    p.max_tokens     = 256;
    p.seed           = -1;
    return p;
}

/* Simple xorshift RNG */
static uint32_t rng_state = 0;

static void rng_seed(int seed) {
    if (seed < 0) {
        /* Use address of local variable as entropy source */
        uint32_t s;
        rng_state = (uint32_t)(uintptr_t)&s ^ 0xDEADBEEF;
    } else {
        rng_state = (uint32_t)seed;
    }
    if (rng_state == 0) rng_state = 1;
}

static uint32_t rng_next(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static float rng_float(void) {
    return (float)(rng_next() >> 8) / (float)(1 << 24);
}

/* ── Sampling strategies ────────────────────────────────────── */

static int sample_argmax(const float* logits, int vocab_size) {
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

static void apply_temperature(float* logits, int vocab_size, float temp) {
    if (temp <= 0.0f || temp == 1.0f) return;
    float inv_temp = 1.0f / temp;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv_temp;
    }
}

static void apply_repeat_penalty(float* logits, const int32_t* recent, int recent_len, float penalty) {
    if (penalty <= 1.0f) return;
    for (int i = 0; i < recent_len; i++) {
        int tok = recent[i];
        if (logits[tok] > 0) {
            logits[tok] /= penalty;
        } else {
            logits[tok] *= penalty;
        }
    }
}

/* Convert logits to a probability distribution in place. */
static void apply_softmax(float* logits, int vocab_size) {
    float max_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] *= inv;
    }
}

/* Max-heap sift-down over an index array `idx` of `n` elements rooted at
 * `i`, keyed by probs[idx[.]] (largest probability at the root). */
static void heap_sift_down_idx(int* idx, int n, int i, const float* probs) {
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, largest = i;
        if (l < n && probs[idx[l]] > probs[idx[largest]]) largest = l;
        if (r < n && probs[idx[r]] > probs[idx[largest]]) largest = r;
        if (largest == i) break;
        int t = idx[i]; idx[i] = idx[largest]; idx[largest] = t;
        i = largest;
    }
}

/* Fused top-k + top-p nucleus sample. Selects at most `top_k` highest-
 * probability tokens (k<=0 or k>=vocab means "no top-k limit"), walks
 * them in descending probability accumulating until cumulative >= top_p,
 * and samples one from that nucleus. Replaces the separate apply_top_k +
 * sample_top_p two-heap sequence with a single partial-selection pass.
 *
 * Build one max-heap of indices keyed by probability (heapify is
 * O(vocab)), then pop the largest repeatedly — recording nucleus
 * indices and accumulating cumulative probability — until EITHER we have
 * popped `top_k` entries (when top_k is a positive limit < vocab) OR
 * cumulative >= top_p, whichever comes first. O(vocab + nucleus*log vocab),
 * a single heap pass instead of two. Folding top-k into the post-softmax
 * selection is equivalent to the old pre-softmax apply_top_k: softmax is
 * monotonic, so the k highest probabilities are the k highest logits. */
static int sample_top_k_top_p(float* probs, int vocab_size, int top_k, float top_p) {
    /* Whether top_k acts as a real limit on the nucleus size. */
    int k_limited = (top_k > 0 && top_k < vocab_size);

    /* Two working arrays: the heap of indices and the nucleus list.
     * Stack-allocate when the vocab is small, else one combined malloc. */
    int stack_buf[2 * 4096];
    int* heap;
    int* nucleus;
    int* alloc = NULL;
    if (vocab_size <= 4096) {
        heap = stack_buf;
        nucleus = stack_buf + vocab_size;
    } else {
        alloc = malloc((size_t)vocab_size * 2 * sizeof(int));
        if (!alloc) return sample_argmax(probs, vocab_size);
        heap = alloc;
        nucleus = alloc + vocab_size;
    }

    /* Build the index heap and heapify into a max-heap by probability. */
    int heap_n = vocab_size;
    for (int i = 0; i < vocab_size; i++) heap[i] = i;
    for (int i = heap_n / 2 - 1; i >= 0; i--) {
        heap_sift_down_idx(heap, heap_n, i, probs);
    }

    /* Pop largest-probability indices until the nucleus covers top_p, or
     * until we have collected top_k entries (whichever comes first). */
    float cumulative = 0.0f;
    int nucleus_n = 0;
    while (heap_n > 0) {
        int top = heap[0];
        nucleus[nucleus_n++] = top;
        cumulative += probs[top];
        /* Remove root: move last element up and sift down. */
        heap[0] = heap[--heap_n];
        if (heap_n > 0) heap_sift_down_idx(heap, heap_n, 0, probs);
        if (cumulative >= top_p) break;
        if (k_limited && nucleus_n >= top_k) break;
    }

    /* Sample from the recorded nucleus (same arithmetic as before). */
    float r = rng_float() * cumulative;
    float running = 0.0f;
    int result = nucleus[0];
    for (int i = 0; i < nucleus_n; i++) {
        running += probs[nucleus[i]];
        if (running >= r) {
            result = nucleus[i];
            break;
        }
    }

    if (alloc) free(alloc);
    return result;
}

static int sample_token(float* logits, int vocab_size, inferbit_sample_params params,
                        const int32_t* recent_tokens, int recent_len) {
    /* Greedy if temperature ~0 */
    if (params.temperature < 0.01f) {
        return sample_argmax(logits, vocab_size);
    }

    /* Apply repeat penalty */
    apply_repeat_penalty(logits, recent_tokens, recent_len, params.repeat_penalty);

    /* Apply temperature */
    apply_temperature(logits, vocab_size, params.temperature);

    /* Convert to probabilities */
    apply_softmax(logits, vocab_size);

    /* Fused top-K + top-P selection and sample */
    return sample_top_k_top_p(logits, vocab_size, params.top_k, params.top_p);
}

/* ── Spec amortization telemetry (IB_SPEC_LOG=1) ────────────────
 *
 * One line per generation. Fields:
 *   rounds    : speculative verify rounds executed
 *   drafted   : candidate tokens drafted (denominator of accept rate)
 *   accepted  : drafted tokens the exact model agreed with
 *   rate      : accepted / drafted (per-token accept rate)
 *   mean_run  : accepted / rounds (mean accepted-run-length per round)
 *   amort     : generated / forwards (tokens emitted per expensive forward —
 *               the amortization factor; higher = more tokens per weight read)
 *   fwd       : count of expensive (COOLDOWN/full-precision) main-model forwards
 *               — the batched verify passes plus single-token refresh forwards.
 */
static void ib_spec_log_line(const char* tag, long long rounds, long long drafted,
                             long long accepted, long long generated,
                             long long forwards) {
    fprintf(stderr,
            "[%s] rounds=%lld drafted=%lld accepted=%lld rate=%.3f "
            "mean_run=%.2f amort=%.2f (fwd=%lld)\n",
            tag, rounds, drafted, accepted,
            drafted   ? (double)accepted / drafted   : 0.0,
            rounds    ? (double)accepted / rounds     : 0.0,
            forwards  ? (double)generated / forwards  : 0.0,
            forwards);
}

/* ── Public API ─────────────────────────────────────────────── */

int inferbit_forward(
    inferbit_model* model,
    const int32_t*  tokens,
    int             num_tokens,
    float*          out_logits,
    int             vocab_size
) {
    if (!model || !tokens || !out_logits) {
        ib_set_error("NULL argument to inferbit_forward");
        return INFERBIT_ERROR_PARAM;
    }
    if (vocab_size != model->header.vocab_size) {
        ib_set_error("vocab_size mismatch: %d vs %d", vocab_size, model->header.vocab_size);
        return INFERBIT_ERROR_PARAM;
    }
    return ib_forward(model, tokens, num_tokens, out_logits);
}

int inferbit_generate(
    inferbit_model*        model,
    const int32_t*         input_tokens,
    int                    num_input_tokens,
    int32_t*               out_tokens,
    int                    max_out_tokens,
    inferbit_sample_params params
) {
    if (!model || !input_tokens || !out_tokens) {
        ib_set_error("NULL argument to inferbit_generate");
        return INFERBIT_ERROR_PARAM;
    }
    if (num_input_tokens <= 0 || max_out_tokens <= 0) {
        ib_set_error("invalid token counts");
        return INFERBIT_ERROR_PARAM;
    }

    rng_seed(params.seed);

    int vocab = model->header.vocab_size;
    float* logits = model->buf_logits;

    int rc = ib_forward(model, input_tokens, num_input_tokens, logits);
    if (rc != INFERBIT_OK) return rc;

    int generated = 0;
    int eos = model->header.eos_token_id;
    int greedy = (params.temperature < 0.01f);

    int use_spec = (model->draft_model != NULL && greedy);
    if (use_spec) {
        inferbit_model* draft = model->draft_model;
        if (draft->header.vocab_size != vocab) {
            use_spec = 0;
        } else {
            inferbit_kv_clear(draft);
            rc = ib_forward(draft, input_tokens, num_input_tokens, draft->buf_logits);
            if (rc != INFERBIT_OK) use_spec = 0;
        }
    }

    /* Prompt-lookup speculation: no external draft, draft candidates come
     * from n-gram match over the running history. Greedy-only. Disabled
     * automatically when an external draft is in use. */
    int use_lookup = (!use_spec && greedy &&
                      model->lookup_ngram > 0 && model->lookup_k > 0);

    /* Self-speculative burst: no external draft, no n-gram lookup, but burst
     * is enabled and we're greedy. The MAIN model drafts cheaply (BURST:
     * L1-only and/or early-exit), then a full-precision (COOLDOWN) batched
     * verify accepts the prefix the exact model agrees with and rewrites KV.
     * Output is therefore TOKEN-IDENTICAL to exact greedy decoding — the
     * cool-down verify re-anchors state, so the harder squeeze costs zero
     * quality. This supersedes the old per-token profile toggle, which lost
     * quality because it never re-anchored the coarse KV the burst wrote. */
    int use_self_spec = (!use_spec && !use_lookup && greedy &&
                         model->burst.cfg.enabled);

    if (!use_spec && !use_lookup && !use_self_spec) {
        int32_t next_token = sample_token(logits, vocab, params, input_tokens, num_input_tokens);
        out_tokens[generated++] = next_token;
        if (next_token == eos) { ib_burst_log_summary(model); return generated; }

        while (generated < max_out_tokens) {
            /* ── Standalone per-step duty cycle: pick BURST/COOLDOWN for this
             * MAIN-model decode step. Returns IB_PROFILE_EXACT (no-op,
             * byte-identical to today) when burst is disabled. This is the
             * loop that exercises main-model L1-only + expert-sparsity for
             * PPL/tok-s measurement (spec verify alone only runs COOLDOWN). */
            ib_burst_step_decide(model);
            rc = ib_forward(model, &next_token, 1, logits);
            if (rc != INFERBIT_OK) return rc;
            next_token = sample_token(logits, vocab, params, out_tokens, generated);
            out_tokens[generated++] = next_token;
            if (next_token == eos) break;
        }
        ib_burst_log_summary(model);
        return generated;
    }

    if (use_self_spec) {
        int spec_k = model->burst.cfg.cooldown_period > 0
                         ? model->burst.cfg.cooldown_period : 8;
        if (spec_k < 2) spec_k = 2;
        if (spec_k > 32) spec_k = 32;
        /* IB_SPEC_K overrides the draft length (clamped to IB_BATCH_MAX). Larger
         * k amortizes the verify weight reads over more tokens when acceptance
         * is high. Unset → unchanged. */
        spec_k = ib_spec_k_override(spec_k);
        if (spec_k < 2) spec_k = 2;
        int32_t candidates[32];
        float* logits_batch = (float*)malloc((size_t)spec_k * (size_t)vocab * sizeof(float));
        float* draft_logits = (float*)malloc((size_t)vocab * sizeof(float));
        if (!logits_batch || !draft_logits) {
            free(logits_batch); free(draft_logits);
            ib_set_error("alloc self-spec buffers");
            return INFERBIT_ERROR_MEMORY;
        }

        int spec_log = ib_spec_log_enabled();
        long long stat_drafted = 0, stat_accepted = 0, stat_rounds = 0;
        long long stat_forwards = 0;   /* expensive main-model forwards */

        while (generated < max_out_tokens) {
            int k = spec_k;
            if (k > (max_out_tokens - generated)) k = max_out_tokens - generated;
            if (k <= 0) break;

            int base = inferbit_kv_length(model);

            /* ── DRAFT under BURST (cheap). candidates[0] is the exact next
             * token (argmax of the exact `logits` we already hold) so it is
             * always accepted and generation always progresses; the remaining
             * k-1 come from cheap burst forwards (L1-only and/or early-exit).
             * Burst KV written here is DISCARDED before the verify. The exact
             * `logits` buffer is preserved (draft writes draft_logits). */
            inferbit_set_compute_profile(model, IB_PROFILE_BURST);
            int dml = ib_active_max_layer(model);  /* burst early-exit depth, -1=full */
            candidates[0] = sample_argmax(logits, vocab);
            int32_t dtok = candidates[0];
            for (int i = 1; i < k; i++) {
                int pos = inferbit_kv_length(model);
                if (dml >= 0 && dml < model->header.num_layers)
                    rc = inferbit_forward_truncated(model, dtok, pos, dml, draft_logits);
                else
                    rc = ib_forward(model, &dtok, 1, draft_logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(draft_logits); return rc; }
                dtok = sample_argmax(draft_logits, vocab);
                candidates[i] = dtok;
            }

            /* Discard burst KV, then verify all k positions at full precision
             * (COOLDOWN). forward_positions rewrites exact KV for base..base+k-1. */
            inferbit_kv_truncate(model, base);
            inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
            rc = ib_forward_positions(model, candidates, k, logits_batch);
            if (rc != INFERBIT_OK) { free(logits_batch); free(draft_logits); return rc; }
            if (spec_log) stat_forwards++;   /* one batched verify */

            /* Accept the prefix the EXACT model agrees with → output is
             * token-identical to exact greedy. Position 0 vs the pre-round
             * exact `logits`; position i (>=1) vs logits_batch[i-1]. */
            int accepted = 0, mismatch = 0; int32_t mismatch_tok = -1; int match_eos = 0;
            for (int i = 0; i < k && generated < max_out_tokens; i++) {
                const float* cur = (i == 0) ? logits
                                            : (logits_batch + (size_t)(i - 1) * (size_t)vocab);
                int32_t mtok = sample_argmax(cur, vocab);
                if (mtok == candidates[i]) {
                    out_tokens[generated++] = mtok;
                    accepted++;
                    if (mtok == eos) { match_eos = 1; break; }
                } else {
                    mismatch = 1; mismatch_tok = mtok; break;
                }
            }
            ib_burst_feed_accept(model, accepted, k);
            if (spec_log) { stat_drafted += k; stat_accepted += accepted; stat_rounds++; }

            /* Roll back KV to the accepted length (drops the rejected suffix). */
            if (accepted < k) inferbit_kv_truncate(model, base + accepted);
            if (match_eos) goto self_spec_done;

            if (mismatch) {
                /* The exact correct token at the first divergent position is the
                 * argmax we just computed. Emit it and refresh `logits`. */
                out_tokens[generated++] = mismatch_tok;
                if (mismatch_tok == eos) goto self_spec_done;
                inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
                rc = ib_forward(model, &mismatch_tok, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(draft_logits); return rc; }
                if (spec_log) stat_forwards++;   /* mismatch refresh */
                continue;
            }
            if (generated >= max_out_tokens) break;

            /* All k accepted: bonus token from position k-1, then refresh. */
            const float* last = logits_batch + (size_t)(k - 1) * (size_t)vocab;
            int32_t extra = sample_argmax(last, vocab);
            out_tokens[generated++] = extra;
            if (extra == eos) goto self_spec_done;
            inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
            rc = ib_forward(model, &extra, 1, logits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(draft_logits); return rc; }
            if (spec_log) stat_forwards++;   /* bonus-token refresh */
        }
    self_spec_done:
        if (spec_log) ib_spec_log_line("ib-self-spec", stat_rounds, stat_drafted,
                                       stat_accepted, generated, stat_forwards);
        ib_burst_log_summary(model);
        free(logits_batch); free(draft_logits);
        return generated;
    }

    if (use_lookup) {
        int hist_cap = num_input_tokens + max_out_tokens + 8;
        int32_t* history = (int32_t*)malloc((size_t)hist_cap * sizeof(int32_t));
        if (!history) { ib_set_error("alloc history"); return INFERBIT_ERROR_MEMORY; }
        memcpy(history, input_tokens, (size_t)num_input_tokens * sizeof(int32_t));
        int hist_n = num_input_tokens;

        int ngram      = model->lookup_ngram;
        int lookup_k   = model->lookup_k;
        if (lookup_k > 32) lookup_k = 32;
        /* IB_SPEC_K overrides the draft length (clamped to IB_BATCH_MAX). */
        lookup_k = ib_spec_k_override(lookup_k);
        int32_t candidates[IB_BATCH_MAX];

        /* Tree / multi-candidate lookup (IB_SPEC_TREE=1): gather several n-gram
         * continuations and verify them together in one batched forward. The
         * total positions across all branches share the single IB_BATCH_MAX
         * verify budget, so per-branch length is bounded. Lossless: the accept
         * rule below compares against the EXACT verify logits. */
        int spec_tree = ib_spec_tree_enabled();
        int tree_per_branch = lookup_k;
        if (spec_tree) {
            /* Keep branches short enough that a few fit one verify. */
            if (tree_per_branch > IB_BATCH_MAX / 2) tree_per_branch = IB_BATCH_MAX / 2;
            if (tree_per_branch < 1) tree_per_branch = 1;
        }
        int32_t tree_tokens[IB_BATCH_MAX];
        int     tree_off[IB_BATCH_MAX];
        int     tree_len[IB_BATCH_MAX];

        /* Batched verify scratch: room for IB_BATCH_MAX per-position logits. */
        float* logits_batch = (float*)malloc((size_t)IB_BATCH_MAX * (size_t)vocab * sizeof(float));
        if (!logits_batch) { free(history); ib_set_error("alloc logits_batch"); return INFERBIT_ERROR_MEMORY; }

        int spec_log = ib_spec_log_enabled();
        long long stat_drafted = 0, stat_accepted = 0, stat_rounds = 0;
        long long stat_forwards = 0;   /* expensive main-model forwards */

        while (generated < max_out_tokens) {
            /* ── Burst/cool-down: decide profile for this round.
             * Returns IB_PROFILE_EXACT (no-op) when burst is disabled. */
            ib_burst_step_decide(model);

            /* ── Tree branch: gather several candidate continuations, verify
             * them in one batched forward, accept the branch the EXACT model
             * follows. Lossless because every emitted token is argmax of exact
             * logits at the correct KV position. */
            if (spec_tree) {
                int nb = ib_prompt_lookup_search_tree(history, hist_n, ngram,
                                                      tree_per_branch, IB_BATCH_MAX,
                                                      IB_BATCH_MAX, tree_tokens,
                                                      tree_off, tree_len);
                if (nb == 0) {
                    /* Miss: one standard decode step. */
                    int32_t t = sample_argmax(logits, vocab);
                    out_tokens[generated++] = t;
                    history[hist_n++] = t;
                    if (t == eos) goto lookup_done;
                    rc = ib_forward(model, &t, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                int total = tree_off[nb - 1] + tree_len[nb - 1];

                inferbit_set_compute_profile(model, IB_PROFILE_BURST);
                int base_main_len = inferbit_kv_length(model);
                inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
                rc = ib_forward_positions(model, tree_tokens, total, logits_batch);
                if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                if (spec_log) stat_forwards++;   /* one batched verify */

                /* Accept rule: the exact greedy first token is argmax of the
                 * pre-round `logits` (independent of any candidate). Find the
                 * one branch whose first token equals it; if none, it's a
                 * position-0 mismatch identical to a linear lookup miss. */
                int32_t gold0 = sample_argmax(logits, vocab);
                int chosen = -1;
                for (int b = 0; b < nb; b++) {
                    if (tree_tokens[tree_off[b]] == gold0) { chosen = b; break; }
                }

                /* Position-0 margin telemetry (same as linear path). */
                if (model->burst.cfg.enabled) {
                    float top1 = -1e38f, top2 = -1e38f;
                    for (int vi = 0; vi < vocab; vi++) {
                        float v = logits[vi];
                        if (v > top1) { top2 = top1; top1 = v; }
                        else if (v > top2) { top2 = v; }
                    }
                    model->burst.last_margin = top1 - top2;
                }

                if (chosen < 0) {
                    /* No branch matches the exact first token: emit gold0, drop
                     * the whole speculated tree, refresh `logits`. */
                    inferbit_kv_truncate(model, base_main_len);
                    out_tokens[generated++] = gold0;
                    history[hist_n++] = gold0;
                    ib_burst_feed_accept(model, 0, total);
                    if (spec_log) { stat_drafted += total; stat_rounds++; }
                    if (gold0 == eos) goto lookup_done;
                    rc = ib_forward(model, &gold0, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                /* Walk the chosen branch, accepting the prefix the exact model
                 * agrees with. Position 0 vs pre-round `logits`; position j>0 vs
                 * the verify logits at this branch's KV slot (off + j - 1). */
                int blen = tree_len[chosen];
                int boff = tree_off[chosen];
                int accepted = 0, mismatch = 0, match_eos = 0;
                int32_t mismatch_tok = -1;
                for (int j = 0; j < blen && generated < max_out_tokens; j++) {
                    const float* cur = (j == 0)
                        ? logits
                        : (logits_batch + (size_t)(boff + j - 1) * (size_t)vocab);
                    int32_t mtok = sample_argmax(cur, vocab);
                    if (mtok == tree_tokens[boff + j]) {
                        out_tokens[generated++] = mtok;
                        history[hist_n++] = mtok;
                        accepted++;
                        if (mtok == eos) { match_eos = 1; break; }
                    } else {
                        mismatch = 1; mismatch_tok = mtok; break;
                    }
                }

                ib_burst_feed_accept(model, accepted, blen);
                if (spec_log) { stat_drafted += total; stat_accepted += accepted; stat_rounds++; }

                /* Commit: the verify wrote KV for ALL branches in pass order, so
                 * the accepted branch's tokens are NOT a contiguous front prefix.
                 * Truncate to base and replay exactly the accepted tokens (one
                 * batched forward) so KV holds only the committed sequence and
                 * `logits` is refreshed to the last committed position. */
                inferbit_kv_truncate(model, base_main_len);
                if (match_eos) goto lookup_done;

                if (mismatch) {
                    /* Replay the accepted prefix (if any), then emit the exact
                     * divergent token and refresh through it. */
                    if (accepted > 0) {
                        rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                        if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                        if (spec_log) stat_forwards++;
                    }
                    out_tokens[generated++] = mismatch_tok;
                    history[hist_n++] = mismatch_tok;
                    if (mismatch_tok == eos) goto lookup_done;
                    rc = ib_forward(model, &mismatch_tok, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                if (generated >= max_out_tokens) {
                    /* Replay accepted to keep KV consistent before exiting. */
                    if (accepted > 0) {
                        rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                        if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                        if (spec_log) stat_forwards++;
                    }
                    break;
                }

                /* Whole branch accepted. Replay it (one batched forward); the
                 * last position's logits give the bonus token. */
                rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                if (spec_log) stat_forwards++;
                const float* tlast = logits_batch + (size_t)(accepted - 1) * (size_t)vocab;
                int32_t extra = sample_argmax(tlast, vocab);
                out_tokens[generated++] = extra;
                history[hist_n++] = extra;
                if (extra == eos) goto lookup_done;
                rc = ib_forward(model, &extra, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            int k = ib_prompt_lookup_search(history, hist_n, ngram, lookup_k, candidates);
            if (k > (max_out_tokens - generated)) k = max_out_tokens - generated;

            if (k == 0) {
                /* Miss: one standard decode step (no draft phase, skip burst). */
                int32_t t = sample_argmax(logits, vocab);
                out_tokens[generated++] = t;
                history[hist_n++] = t;
                if (t == eos) goto lookup_done;
                rc = ib_forward(model, &t, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            /* Draft phase runs under BURST profile (cheap/coarse).
             * For lookup, the "draft" is the n-gram match — no model forward
             * happens, but we set the profile to document the intent and to
             * ensure that any auxiliary work in this window is profiled. */
            inferbit_set_compute_profile(model, IB_PROFILE_BURST);

            /* Batched verify: one forward over all k candidates, producing k
             * per-position logit vectors. Switch to COOLDOWN before verify so
             * the verify forward runs as the precise anchor. Position 0 is
             * verified against the pre-round `logits` (unchanged); position
             * i (1..k-1) against logits_batch[(i-1)*vocab]. */
            int base_main_len = inferbit_kv_length(model);
            inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
            rc = ib_forward_positions(model, candidates, k, logits_batch);
            if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
            if (spec_log) stat_forwards++;   /* one batched verify */

            int accepted = 0, mismatch = 0;
            int32_t mismatch_tok = -1;
            int match_eos = 0;
            for (int i = 0; i < k && generated < max_out_tokens; i++) {
                const float* cur = (i == 0) ? logits
                                            : (logits_batch + (size_t)(i - 1) * (size_t)vocab);
                int32_t mtok = sample_argmax(cur, vocab);
                if (mtok == candidates[i]) {
                    out_tokens[generated++] = mtok;
                    history[hist_n++] = mtok;
                    accepted++;
                    if (mtok == eos) { match_eos = 1; break; }
                } else {
                    mismatch = 1;
                    mismatch_tok = mtok;
                    break;
                }
            }

            /* Feed acceptance result into the burst controller's EMA. */
            ib_burst_feed_accept(model, accepted, k);

            if (spec_log) { stat_drafted += k; stat_accepted += accepted; stat_rounds++; }

            /* ── Compute top-1/top-2 logit margin for the committed token.
             * "Committed token" is the first accepted position (position 0),
             * whose logits are in `logits` (the pre-round main-model output).
             * Guard behind cfg.enabled to avoid overhead on the default path. */
            if (model->burst.cfg.enabled) {
                const float* margin_src = logits; /* position-0 verify logits */
                float top1 = -1e38f, top2 = -1e38f;
                for (int vi = 0; vi < vocab; vi++) {
                    float v = margin_src[vi];
                    if (v > top1) { top2 = top1; top1 = v; }
                    else if (v > top2) { top2 = v; }
                }
                model->burst.last_margin = top1 - top2;
            }

            /* Roll back main KV if not all k got committed. */
            if (accepted < k) inferbit_kv_truncate(model, base_main_len + accepted);

            if (match_eos) {
                /* Accepted mtok that is EOS — we're done. */
                goto lookup_done;
            }

            if (mismatch) {
                out_tokens[generated++] = mismatch_tok;
                history[hist_n++] = mismatch_tok;
                if (mismatch_tok == eos) goto lookup_done;
                rc = ib_forward(model, &mismatch_tok, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            if (generated >= max_out_tokens) break;

            /* All k accepted. The "next-step" logits are the k-th position's
             * output from the batched forward. Sample the bonus token from
             * there, emit it, then advance main by 1 to refresh `logits`. */
            const float* last = logits_batch + (size_t)(k - 1) * (size_t)vocab;
            int32_t extra = sample_argmax(last, vocab);
            out_tokens[generated++] = extra;
            history[hist_n++] = extra;
            if (extra == eos) goto lookup_done;
            rc = ib_forward(model, &extra, 1, logits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(history); return rc; }
            if (spec_log) stat_forwards++;
        }

    lookup_done:
        free(logits_batch);
        if (spec_log) ib_spec_log_line(spec_tree ? "ib-spec-lookup-tree" : "ib-spec-lookup",
                                       stat_rounds, stat_drafted, stat_accepted,
                                       generated, stat_forwards);
        ib_burst_log_summary(model);
        free(history);
        return generated;
    }

    inferbit_model* draft = model->draft_model;
    float* dlogits = draft->buf_logits;
    int draft_k = model->draft_tokens > 0 ? model->draft_tokens : 4;
    if (draft_k > 32) draft_k = 32;
    /* IB_SPEC_K overrides the draft length (clamped to IB_BATCH_MAX). Tree mode
     * does NOT apply to external-draft (lookup-only). */
    draft_k = ib_spec_k_override(draft_k);
    int32_t candidates[IB_BATCH_MAX];

    float* logits_batch = (float*)malloc((size_t)draft_k * (size_t)vocab * sizeof(float));
    if (!logits_batch) { ib_set_error("alloc logits_batch"); return INFERBIT_ERROR_MEMORY; }

    /* Optional accept-rate telemetry — off unless IB_SPEC_LOG is set. */
    int spec_log = ib_spec_log_enabled();
    long long stat_drafted = 0, stat_accepted = 0, stat_rounds = 0;
    long long stat_forwards = 0;   /* expensive main-model forwards */

    while (generated < max_out_tokens) {
        int base_draft_len = inferbit_kv_length(draft);

        /* ── Burst/cool-down: decide profile for this round.
         * Returns IB_PROFILE_EXACT (no-op) when burst is disabled. */
        ib_burst_step_decide(model);

        int k = draft_k;
        if (k > (max_out_tokens - generated)) k = max_out_tokens - generated;

        /* Draft phase: run under BURST profile (cheap/coarse). */
        inferbit_set_compute_profile(model, IB_PROFILE_BURST);

        int32_t dtok = sample_argmax(dlogits, vocab);
        for (int i = 0; i < k; i++) {
            candidates[i] = dtok;
            rc = ib_forward(draft, &dtok, 1, dlogits);
            if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
            dtok = sample_argmax(dlogits, vocab);
        }

        /* Verify phase: switch to COOLDOWN so the batched verify is the precise
         * anchor. Batched verify over main: one forward producing k per-position
         * logits. Position 0 checks against pre-round `logits`; 1..k-1 against
         * logits_batch. */
        int base_main_len = inferbit_kv_length(model);
        inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
        rc = ib_forward_positions(model, candidates, k, logits_batch);
        if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
        if (spec_log) stat_forwards++;   /* one batched verify */

        int accepted = 0;
        int mismatch = 0;
        int match_eos = 0;
        int32_t mismatch_tok = -1;

        for (int i = 0; i < k && generated < max_out_tokens; i++) {
            const float* cur = (i == 0) ? logits
                                        : (logits_batch + (size_t)(i - 1) * (size_t)vocab);
            int32_t mtok = sample_argmax(cur, vocab);
            if (mtok == candidates[i]) {
                out_tokens[generated++] = mtok;
                accepted++;
                if (mtok == eos) { match_eos = 1; break; }
            } else {
                mismatch = 1;
                mismatch_tok = mtok;
                break;
            }
        }

        /* Feed acceptance result into the burst controller's EMA. */
        ib_burst_feed_accept(model, accepted, k);

        if (spec_log) { stat_drafted += k; stat_accepted += accepted; stat_rounds++; }

        /* ── Compute top-1/top-2 logit margin for the committed token.
         * Use the position-0 logits (pre-round `logits`) as the main-model
         * output for the first committed token. Guard behind cfg.enabled to
         * avoid overhead on the default path. */
        if (model->burst.cfg.enabled) {
            const float* margin_src = logits; /* position-0 verify logits */
            float top1 = -1e38f, top2 = -1e38f;
            for (int vi = 0; vi < vocab; vi++) {
                float v = margin_src[vi];
                if (v > top1) { top2 = top1; top1 = v; }
                else if (v > top2) { top2 = v; }
            }
            model->burst.last_margin = top1 - top2;
        }

        /* Roll back main KV if we did not commit all k positions. */
        if (accepted < k) inferbit_kv_truncate(model, base_main_len + accepted);

        if (match_eos) {
            if (spec_log) ib_spec_log_line("ib-spec", stat_rounds, stat_drafted,
                                           stat_accepted, generated, stat_forwards);
            free(logits_batch);
            return generated;
        }

        if (mismatch) {
            out_tokens[generated++] = mismatch_tok;
            if (mismatch_tok == eos) {
                if (spec_log) ib_spec_log_line("ib-spec", stat_rounds, stat_drafted,
                                               stat_accepted, generated, stat_forwards);
                free(logits_batch);
                return generated;
            }
            rc = ib_forward(model, &mismatch_tok, 1, logits);
            if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
            if (spec_log) stat_forwards++;   /* mismatch refresh (main) */

            inferbit_kv_truncate(draft, base_draft_len + accepted);
            rc = ib_forward(draft, &mismatch_tok, 1, dlogits);
            if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
            continue;
        }

        if (generated >= max_out_tokens) break;

        /* All k accepted — emit the bonus token, then advance main and draft. */
        const float* last = logits_batch + (size_t)(k - 1) * (size_t)vocab;
        int32_t extra = sample_argmax(last, vocab);
        out_tokens[generated++] = extra;
        if (extra == eos) {
            if (spec_log) ib_spec_log_line("ib-spec", stat_rounds, stat_drafted,
                                           stat_accepted, generated, stat_forwards);
            free(logits_batch);
            return generated;
        }

        rc = ib_forward(model, &extra, 1, logits);
        if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
        if (spec_log) stat_forwards++;   /* bonus-token refresh (main) */
        rc = ib_forward(draft, &extra, 1, dlogits);
        if (rc != INFERBIT_OK) { free(logits_batch); return rc; }
    }

    if (spec_log) ib_spec_log_line("ib-spec", stat_rounds, stat_drafted,
                                   stat_accepted, generated, stat_forwards);
    ib_burst_log_summary(model);
    free(logits_batch);
    return generated;
}

int inferbit_generate_stream(
    inferbit_model*          model,
    const int32_t*           input_tokens,
    int                      num_input_tokens,
    inferbit_stream_callback callback,
    void*                    ctx,
    inferbit_sample_params   params
) {
    if (!model || !input_tokens || !callback) {
        ib_set_error("NULL argument to inferbit_generate_stream");
        return INFERBIT_ERROR_PARAM;
    }
    if (num_input_tokens <= 0) {
        ib_set_error("invalid token count");
        return INFERBIT_ERROR_PARAM;
    }

    rng_seed(params.seed);

    int vocab = model->header.vocab_size;
    float* logits = model->buf_logits;

    /* Prefill */
    int rc = ib_forward(model, input_tokens, num_input_tokens, logits);
    if (rc != INFERBIT_OK) return rc;

    /* Track recent tokens for repeat penalty */
    int max_recent = 64;
    int32_t* recent = calloc(params.max_tokens + num_input_tokens, sizeof(int32_t));
    if (!recent) {
        ib_set_error("failed to allocate recent tokens buffer");
        return INFERBIT_ERROR_MEMORY;
    }
    memcpy(recent, input_tokens, num_input_tokens * sizeof(int32_t));
    int recent_len = num_input_tokens;

    /* Sample and stream */
    int generated = 0;
    int eos = model->header.eos_token_id;

    /* Check for speculative decoding (greedy only) */
    int greedy = (params.temperature < 0.01f);
    int use_spec = (model->draft_model != NULL && greedy);
    if (use_spec) {
        inferbit_model* draft = model->draft_model;
        if (draft->header.vocab_size != vocab) {
            use_spec = 0;
        } else {
            inferbit_kv_clear(draft);
            rc = ib_forward(draft, input_tokens, num_input_tokens, draft->buf_logits);
            if (rc != INFERBIT_OK) use_spec = 0;
        }
    }

    int use_lookup = (!use_spec && greedy &&
                      model->lookup_ngram > 0 && model->lookup_k > 0);

    if (!use_spec && !use_lookup) {
        /* Standard streaming path */
        int32_t next_token = sample_token(logits, vocab, params, recent,
                                          recent_len > max_recent ? max_recent : recent_len);
        recent[recent_len++] = next_token;
        generated++;

        if (next_token == eos || callback(next_token, ctx) == 0) {
            ib_burst_log_summary(model);
            free(recent);
            return generated;
        }

        while (generated < params.max_tokens) {
            /* ── Standalone per-step duty cycle (see inferbit_generate).
             * EXACT no-op when burst is disabled. */
            ib_burst_step_decide(model);
            rc = ib_forward(model, &next_token, 1, logits);
            if (rc != INFERBIT_OK) { free(recent); return rc; }

            int pen_start = recent_len > max_recent ? recent_len - max_recent : 0;
            next_token = sample_token(logits, vocab, params, recent + pen_start, recent_len - pen_start);
            recent[recent_len++] = next_token;
            generated++;

            if (next_token == eos || callback(next_token, ctx) == 0) break;
        }

        ib_burst_log_summary(model);
        free(recent);
        return generated;
    }

    if (use_lookup) {
        /* Prompt-lookup streaming path: draft from history n-gram match,
         * verify batched with ib_forward_positions. */
        int ngram    = model->lookup_ngram;
        int lookup_k = model->lookup_k;
        if (lookup_k > 32) lookup_k = 32;
        /* IB_SPEC_K overrides the draft length (clamped to IB_BATCH_MAX). */
        lookup_k = ib_spec_k_override(lookup_k);
        int32_t candidates[IB_BATCH_MAX];
        int stopped = 0;

        /* Tree / multi-candidate lookup (IB_SPEC_TREE=1) — see inferbit_generate
         * for the accept-rule rationale; identical here with callback/stop flow. */
        int spec_tree = ib_spec_tree_enabled();
        int tree_per_branch = lookup_k;
        if (spec_tree) {
            if (tree_per_branch > IB_BATCH_MAX / 2) tree_per_branch = IB_BATCH_MAX / 2;
            if (tree_per_branch < 1) tree_per_branch = 1;
        }
        int32_t tree_tokens[IB_BATCH_MAX];
        int     tree_off[IB_BATCH_MAX];
        int     tree_len[IB_BATCH_MAX];

        float* logits_batch = (float*)malloc((size_t)IB_BATCH_MAX * (size_t)vocab * sizeof(float));
        if (!logits_batch) { free(recent); ib_set_error("alloc logits_batch"); return INFERBIT_ERROR_MEMORY; }

        int spec_log = ib_spec_log_enabled();
        long long stat_drafted = 0, stat_accepted = 0, stat_rounds = 0;
        long long stat_forwards = 0;   /* expensive main-model forwards */

        while (generated < params.max_tokens && !stopped) {
            /* ── Burst/cool-down: decide profile for this round.
             * Returns IB_PROFILE_EXACT (no-op) when burst is disabled. */
            ib_burst_step_decide(model);

            /* ── Tree branch: gather several continuations, verify together,
             * accept the branch the EXACT model follows. Lossless: every
             * emitted token is argmax of exact logits at the right KV slot. */
            if (spec_tree) {
                int nb = ib_prompt_lookup_search_tree(recent, recent_len, ngram,
                                                      tree_per_branch, IB_BATCH_MAX,
                                                      IB_BATCH_MAX, tree_tokens,
                                                      tree_off, tree_len);
                if (nb == 0) {
                    int32_t t = sample_argmax(logits, vocab);
                    recent[recent_len++] = t; generated++;
                    if (t == eos || callback(t, ctx) == 0) { stopped = 1; break; }
                    rc = ib_forward(model, &t, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                int total = tree_off[nb - 1] + tree_len[nb - 1];

                inferbit_set_compute_profile(model, IB_PROFILE_BURST);
                int base_main_len = inferbit_kv_length(model);
                inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
                rc = ib_forward_positions(model, tree_tokens, total, logits_batch);
                if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                if (spec_log) stat_forwards++;   /* one batched verify */

                int32_t gold0 = sample_argmax(logits, vocab);
                int chosen = -1;
                for (int b = 0; b < nb; b++) {
                    if (tree_tokens[tree_off[b]] == gold0) { chosen = b; break; }
                }

                if (model->burst.cfg.enabled) {
                    float top1 = -1e38f, top2 = -1e38f;
                    for (int vi = 0; vi < vocab; vi++) {
                        float v = logits[vi];
                        if (v > top1) { top2 = top1; top1 = v; }
                        else if (v > top2) { top2 = v; }
                    }
                    model->burst.last_margin = top1 - top2;
                }

                if (chosen < 0) {
                    inferbit_kv_truncate(model, base_main_len);
                    recent[recent_len++] = gold0; generated++;
                    ib_burst_feed_accept(model, 0, total);
                    if (spec_log) { stat_drafted += total; stat_rounds++; }
                    if (gold0 == eos || callback(gold0, ctx) == 0) { stopped = 1; break; }
                    rc = ib_forward(model, &gold0, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                int blen = tree_len[chosen];
                int boff = tree_off[chosen];
                int accepted = 0, mismatch = 0, match_eos_or_stop = 0;
                int32_t mismatch_tok = -1;
                for (int j = 0; j < blen && generated < params.max_tokens; j++) {
                    const float* cur = (j == 0)
                        ? logits
                        : (logits_batch + (size_t)(boff + j - 1) * (size_t)vocab);
                    int32_t mtok = sample_argmax(cur, vocab);
                    if (mtok == tree_tokens[boff + j]) {
                        recent[recent_len++] = mtok; generated++;
                        accepted++;
                        if (mtok == eos || callback(mtok, ctx) == 0) {
                            stopped = 1; match_eos_or_stop = 1; break;
                        }
                    } else {
                        mismatch = 1; mismatch_tok = mtok; break;
                    }
                }

                ib_burst_feed_accept(model, accepted, blen);
                if (spec_log) { stat_drafted += total; stat_accepted += accepted; stat_rounds++; }

                /* Commit: truncate to base, replay only the accepted branch
                 * tokens (the verify wrote ALL branches, non-contiguously). */
                inferbit_kv_truncate(model, base_main_len);
                if (match_eos_or_stop) continue;   /* loop exits via stopped */

                if (mismatch) {
                    if (accepted > 0) {
                        rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                        if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                        if (spec_log) stat_forwards++;
                    }
                    recent[recent_len++] = mismatch_tok; generated++;
                    if (mismatch_tok == eos || callback(mismatch_tok, ctx) == 0) { stopped = 1; continue; }
                    rc = ib_forward(model, &mismatch_tok, 1, logits);
                    if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                    if (spec_log) stat_forwards++;
                    continue;
                }

                if (generated >= params.max_tokens) {
                    if (accepted > 0) {
                        rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                        if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                        if (spec_log) stat_forwards++;
                    }
                    break;
                }

                /* Whole branch accepted — replay it, bonus from last position. */
                rc = ib_forward_positions(model, tree_tokens + boff, accepted, logits_batch);
                if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                if (spec_log) stat_forwards++;
                const float* tlast = logits_batch + (size_t)(accepted - 1) * (size_t)vocab;
                int32_t extra = sample_argmax(tlast, vocab);
                recent[recent_len++] = extra; generated++;
                if (extra == eos || callback(extra, ctx) == 0) { stopped = 1; break; }
                rc = ib_forward(model, &extra, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            int k = ib_prompt_lookup_search(recent, recent_len, ngram, lookup_k, candidates);
            if (k > (params.max_tokens - generated)) k = params.max_tokens - generated;

            if (k == 0) {
                /* Miss: one standard decode step (no draft phase, skip burst). */
                int32_t t = sample_argmax(logits, vocab);
                recent[recent_len++] = t; generated++;
                if (t == eos || callback(t, ctx) == 0) { stopped = 1; break; }
                rc = ib_forward(model, &t, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            /* Draft phase: run under BURST profile (cheap/coarse).
             * For lookup, the draft is the n-gram match — no model forward here,
             * but we set the profile to document the phase boundary. */
            inferbit_set_compute_profile(model, IB_PROFILE_BURST);

            /* Verify phase: switch to COOLDOWN before the batched verify so that
             * the verify forward runs as the precise anchor. */
            int base_main_len = inferbit_kv_length(model);
            inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
            rc = ib_forward_positions(model, candidates, k, logits_batch);
            if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
            if (spec_log) stat_forwards++;   /* one batched verify */

            int accepted = 0, mismatch = 0;
            int32_t mismatch_tok = -1;
            int match_eos_or_stop = 0;
            for (int i = 0; i < k && generated < params.max_tokens; i++) {
                const float* cur = (i == 0) ? logits
                                            : (logits_batch + (size_t)(i - 1) * (size_t)vocab);
                int32_t mtok = sample_argmax(cur, vocab);
                if (mtok == candidates[i]) {
                    recent[recent_len++] = mtok; generated++;
                    accepted++;
                    if (mtok == eos || callback(mtok, ctx) == 0) {
                        stopped = 1; match_eos_or_stop = 1; break;
                    }
                } else {
                    mismatch = 1;
                    mismatch_tok = mtok;
                    break;
                }
            }

            /* Feed acceptance result into the burst controller's EMA. */
            ib_burst_feed_accept(model, accepted, k);

            if (spec_log) { stat_drafted += k; stat_accepted += accepted; stat_rounds++; }

            /* ── Compute top-1/top-2 logit margin for the committed token.
             * Guard behind cfg.enabled to avoid overhead on the default path. */
            if (model->burst.cfg.enabled) {
                const float* margin_src = logits; /* position-0 verify logits */
                float top1 = -1e38f, top2 = -1e38f;
                for (int vi = 0; vi < vocab; vi++) {
                    float v = margin_src[vi];
                    if (v > top1) { top2 = top1; top1 = v; }
                    else if (v > top2) { top2 = v; }
                }
                model->burst.last_margin = top1 - top2;
            }

            if (accepted < k) inferbit_kv_truncate(model, base_main_len + accepted);

            if (match_eos_or_stop) continue;   /* loop exits via stopped */

            if (mismatch) {
                recent[recent_len++] = mismatch_tok; generated++;
                if (mismatch_tok == eos || callback(mismatch_tok, ctx) == 0) { stopped = 1; continue; }
                rc = ib_forward(model, &mismatch_tok, 1, logits);
                if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
                if (spec_log) stat_forwards++;
                continue;
            }

            if (generated >= params.max_tokens) break;

            /* All k accepted — bonus token from last batched position. */
            const float* last = logits_batch + (size_t)(k - 1) * (size_t)vocab;
            int32_t extra = sample_argmax(last, vocab);
            recent[recent_len++] = extra; generated++;
            if (extra == eos || callback(extra, ctx) == 0) { stopped = 1; break; }
            rc = ib_forward(model, &extra, 1, logits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
            if (spec_log) stat_forwards++;
        }

        if (spec_log) ib_spec_log_line(spec_tree ? "ib-spec-lookup-tree" : "ib-spec-lookup",
                                       stat_rounds, stat_drafted, stat_accepted,
                                       generated, stat_forwards);
        ib_burst_log_summary(model);
        free(logits_batch);
        free(recent);
        return generated;
    }

    /* Speculative streaming path (external draft, batched verify). */
    inferbit_model* draft = model->draft_model;
    float* dlogits = draft->buf_logits;
    int draft_k = model->draft_tokens > 0 ? model->draft_tokens : 4;
    if (draft_k > 32) draft_k = 32;
    /* IB_SPEC_K overrides the draft length (clamped to IB_BATCH_MAX). Tree mode
     * does NOT apply to external-draft (lookup-only). */
    draft_k = ib_spec_k_override(draft_k);
    int32_t candidates[IB_BATCH_MAX];
    int stopped = 0;

    float* logits_batch = (float*)malloc((size_t)draft_k * (size_t)vocab * sizeof(float));
    if (!logits_batch) { free(recent); ib_set_error("alloc logits_batch"); return INFERBIT_ERROR_MEMORY; }

    int spec_log = ib_spec_log_enabled();
    long long stat_drafted = 0, stat_accepted = 0, stat_rounds = 0;
    long long stat_forwards = 0;   /* expensive main-model forwards */

    while (generated < params.max_tokens && !stopped) {
        int base_draft_len = inferbit_kv_length(draft);

        /* ── Burst/cool-down: decide profile for this round.
         * Returns IB_PROFILE_EXACT (no-op) when burst is disabled. */
        ib_burst_step_decide(model);

        int k = draft_k;
        if (k > (params.max_tokens - generated)) k = params.max_tokens - generated;

        /* Draft phase: run under BURST profile (cheap/coarse). */
        inferbit_set_compute_profile(model, IB_PROFILE_BURST);

        int32_t dtok = sample_argmax(dlogits, vocab);
        for (int i = 0; i < k; i++) {
            candidates[i] = dtok;
            rc = ib_forward(draft, &dtok, 1, dlogits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
            dtok = sample_argmax(dlogits, vocab);
        }

        /* Verify phase: switch to COOLDOWN so the batched verify is the precise
         * anchor. Batched verify over main: one forward producing k per-position
         * logits. */
        int base_main_len  = inferbit_kv_length(model);
        inferbit_set_compute_profile(model, IB_PROFILE_COOLDOWN);
        rc = ib_forward_positions(model, candidates, k, logits_batch);
        if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
        if (spec_log) stat_forwards++;   /* one batched verify */

        int accepted = 0;
        int mismatch = 0;
        int32_t mismatch_tok = -1;
        int match_eos_or_stop = 0;
        for (int i = 0; i < k && generated < params.max_tokens; i++) {
            const float* cur = (i == 0) ? logits
                                        : (logits_batch + (size_t)(i - 1) * (size_t)vocab);
            int32_t mtok = sample_argmax(cur, vocab);
            if (mtok == candidates[i]) {
                generated++;
                accepted++;
                if (mtok == eos || callback(mtok, ctx) == 0) {
                    stopped = 1; match_eos_or_stop = 1; break;
                }
            } else {
                mismatch = 1;
                mismatch_tok = mtok;
                break;
            }
        }

        /* Feed acceptance result into the burst controller's EMA. */
        ib_burst_feed_accept(model, accepted, k);

        if (spec_log) { stat_drafted += k; stat_accepted += accepted; stat_rounds++; }

        /* ── Compute top-1/top-2 logit margin for the committed token.
         * Guard behind cfg.enabled to avoid overhead on the default path. */
        if (model->burst.cfg.enabled) {
            const float* margin_src = logits; /* position-0 verify logits */
            float top1 = -1e38f, top2 = -1e38f;
            for (int vi = 0; vi < vocab; vi++) {
                float v = margin_src[vi];
                if (v > top1) { top2 = top1; top1 = v; }
                else if (v > top2) { top2 = v; }
            }
            model->burst.last_margin = top1 - top2;
        }

        if (accepted < k) inferbit_kv_truncate(model, base_main_len + accepted);

        if (match_eos_or_stop) continue;   /* loop exits via stopped */

        if (mismatch) {
            generated++;
            if (mismatch_tok == eos || callback(mismatch_tok, ctx) == 0) { stopped = 1; continue; }
            rc = ib_forward(model, &mismatch_tok, 1, logits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
            if (spec_log) stat_forwards++;   /* mismatch refresh (main) */
            inferbit_kv_truncate(draft, base_draft_len + accepted);
            rc = ib_forward(draft, &mismatch_tok, 1, dlogits);
            if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
            continue;
        }

        /* All k accepted — bonus token from last batched position. */
        if (generated >= params.max_tokens) break;
        const float* last = logits_batch + (size_t)(k - 1) * (size_t)vocab;
        int32_t extra = sample_argmax(last, vocab);
        generated++;
        if (extra == eos || callback(extra, ctx) == 0) { stopped = 1; break; }
        rc = ib_forward(model, &extra, 1, logits);
        if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
        if (spec_log) stat_forwards++;   /* bonus-token refresh (main) */
        rc = ib_forward(draft, &extra, 1, dlogits);
        if (rc != INFERBIT_OK) { free(logits_batch); free(recent); return rc; }
    }
    if (spec_log) ib_spec_log_line("ib-spec", stat_rounds, stat_drafted,
                                   stat_accepted, generated, stat_forwards);
    ib_burst_log_summary(model);
    free(logits_batch);

    free(recent);
    return generated;
}
