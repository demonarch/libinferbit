#include "inferbit_internal.h"
#include <math.h>
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

/* Top-K: zero out everything except top K logits */
static void apply_top_k(float* logits, int vocab_size, int k) {
    if (k <= 0 || k >= vocab_size) return;

    /* Find the k-th largest value (simple partial sort) */
    /* For correctness over speed — this is O(vocab * k) but vocab is small enough */
    float threshold = -INFINITY;
    for (int round = 0; round < k; round++) {
        float best = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] > best && (round == 0 || logits[i] <= threshold || (logits[i] == threshold))) {
                /* We need a more careful approach */
                (void)0;
            }
        }
        (void)best;
    }

    /* Simpler approach: sort indices, keep top-k */
    /* Allocate temp array of (value, index) pairs */
    /* For now, use a simpler threshold-finding method */

    /* Find k-th largest by repeated scanning */
    float kth = -INFINITY;
    float prev_min = INFINITY;
    for (int round = 0; round < k; round++) {
        float best = -INFINITY;
        for (int i = 0; i < vocab_size; i++) {
            if (logits[i] < prev_min && logits[i] > best) {
                best = logits[i];
            } else if (logits[i] == prev_min && best < prev_min) {
                best = logits[i];
            }
        }
        prev_min = best;
        kth = best;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] < kth) {
            logits[i] = -INFINITY;
        }
    }
}

/* Top-P (nucleus sampling): zero out tokens outside the nucleus */
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

static int sample_top_p(float* probs, int vocab_size, float top_p) {
    /* Build (prob, index) pairs sorted descending by prob */
    /* Simple insertion into a temp buffer */
    /* For small vocabs this is fine; for large vocabs we'd use a heap */

    /* Cumulative sum approach: walk sorted probs until we exceed top_p */
    /* We need sorted indices — do a simple O(n^2) sort for correctness */

    /* Allocate index array on stack if small enough, else heap */
    int* indices = NULL;
    int stack_buf[4096];
    if (vocab_size <= 4096) {
        indices = stack_buf;
    } else {
        indices = malloc(vocab_size * sizeof(int));
        if (!indices) return sample_argmax(probs, vocab_size);
    }

    for (int i = 0; i < vocab_size; i++) indices[i] = i;

    /* Selection sort (top elements only until we hit top_p) */
    float cumulative = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        /* Find max in remaining */
        int best = i;
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[indices[j]] > probs[indices[best]]) {
                best = j;
            }
        }
        /* Swap */
        int tmp = indices[i];
        indices[i] = indices[best];
        indices[best] = tmp;

        cumulative += probs[indices[i]];
        if (cumulative >= top_p) {
            cutoff = i + 1;
            break;
        }
    }

    /* Sample from the nucleus */
    float r = rng_float() * cumulative;
    float running = 0.0f;
    int result = indices[0];
    for (int i = 0; i < cutoff; i++) {
        running += probs[indices[i]];
        if (running >= r) {
            result = indices[i];
            break;
        }
    }

    if (indices != stack_buf) free(indices);
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

    /* Apply top-K */
    apply_top_k(logits, vocab_size, params.top_k);

    /* Convert to probabilities */
    apply_softmax(logits, vocab_size);

    /* Apply top-P and sample */
    return sample_top_p(logits, vocab_size, params.top_p);
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

    int greedy    = (params.temperature < 0.01f);
    int use_draft = greedy && model->draft_model != NULL &&
                    model->draft_model->header.vocab_size == vocab;
    int use_lookup = greedy && !use_draft &&
                     model->lookup_ngram > 0 && model->lookup_k > 0;

    if (use_draft) {
        inferbit_kv_clear(model->draft_model);
        rc = ib_forward(model->draft_model, input_tokens, num_input_tokens,
                        model->draft_model->buf_logits);
        if (rc != INFERBIT_OK) use_draft = 0;
    }

    if (!use_draft && !use_lookup) {
        int32_t next_token = sample_token(logits, vocab, params, input_tokens, num_input_tokens);
        out_tokens[generated++] = next_token;
        if (next_token == eos) return generated;

        while (generated < max_out_tokens) {
            rc = ib_forward(model, &next_token, 1, logits);
            if (rc != INFERBIT_OK) return rc;
            next_token = sample_token(logits, vocab, params, out_tokens, generated);
            out_tokens[generated++] = next_token;
            if (next_token == eos) break;
        }
        return generated;
    }

    /* Speculative decoding (greedy only).
     *
     * Invariant: at the top of each round, `logits` holds the main model's
     * next-token prediction given the sequence emitted so far. We produce k
     * draft candidates, verify them in a single k-length forward over the
     * main model, and emit (accepted + 1) tokens per round. KV caches are
     * rolled back to match what was actually accepted. */
    int k_cfg = use_draft ? (model->draft_tokens > 0 ? model->draft_tokens : 4)
                          : model->lookup_k;
    if (k_cfg < 1)  k_cfg = 1;
    if (k_cfg > 32) k_cfg = 32;

    int32_t candidates[32];
    int hist_cap = num_input_tokens + max_out_tokens + 8;
    int32_t* history = (int32_t*)malloc((size_t)hist_cap * sizeof(int32_t));
    if (!history) return INFERBIT_ERROR_PARAM;
    int hist_n = 0;
    for (int i = 0; i < num_input_tokens; i++) history[hist_n++] = input_tokens[i];

    float* verify_logits = (float*)malloc((size_t)k_cfg * vocab * sizeof(float));
    if (!verify_logits) { free(history); return INFERBIT_ERROR_PARAM; }

    while (generated < max_out_tokens) {
        int k = k_cfg;
        if (k > (max_out_tokens - generated)) k = max_out_tokens - generated;
        if (k < 1) break;

        /* Produce candidates. */
        int have = 0;
        if (use_lookup) {
            have = ib_lookup_candidates(history, hist_n,
                                        model->lookup_ngram, k, candidates);
        } else {
            inferbit_model* draft = model->draft_model;
            float* dlogits = draft->buf_logits;
            for (int i = 0; i < k; i++) {
                int32_t dtok = sample_argmax(dlogits, vocab);
                candidates[i] = dtok;
                rc = ib_forward(draft, &dtok, 1, dlogits);
                if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }
            }
            have = k;
        }

        int32_t mtok0 = sample_argmax(logits, vocab);

        if (have == 0 || candidates[0] != mtok0) {
            /* No speculation win — emit main's pick and continue. */
            out_tokens[generated++] = mtok0;
            history[hist_n++] = mtok0;
            if (mtok0 == eos) break;
            rc = ib_forward(model, &mtok0, 1, logits);
            if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }

            if (use_draft) {
                inferbit_model* draft = model->draft_model;
                int base = inferbit_kv_length(draft);
                inferbit_kv_truncate(draft, base - k);
                rc = ib_forward(draft, &mtok0, 1, draft->buf_logits);
                if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }
            }
            continue;
        }

        /* c[0] matches main's top pick. Verify c[1..have-1] in one forward. */
        int base_main = inferbit_kv_length(model);
        int verify_len = have;
        rc = ib_forward_positions(model, candidates, verify_len, verify_logits);
        if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }

        /* verify_logits[i] = main's prediction after consuming c[0..i]. */
        int accept = 1;  /* c[0] already matched */
        int32_t correction = -1;
        for (int i = 1; i < verify_len; i++) {
            int32_t want = sample_argmax(verify_logits + (size_t)(i - 1) * vocab, vocab);
            if (want == candidates[i]) accept++;
            else { correction = want; break; }
        }

        int32_t bonus = -1;
        if (accept == verify_len) {
            bonus = sample_argmax(verify_logits + (size_t)(verify_len - 1) * vocab, vocab);
        }

        /* Emit accepted candidates. */
        int eos_hit = 0;
        for (int i = 0; i < accept && generated < max_out_tokens; i++) {
            out_tokens[generated++] = candidates[i];
            history[hist_n++] = candidates[i];
            if (candidates[i] == eos) { eos_hit = 1; break; }
        }
        if (eos_hit) break;

        /* Roll main KV back to end of accepted prefix, then apply correction
         * or bonus token. That single-token forward re-seeds `logits`. */
        inferbit_kv_truncate(model, base_main + accept);

        int32_t seed_tok = (correction >= 0) ? correction : bonus;
        if (generated < max_out_tokens && seed_tok >= 0) {
            out_tokens[generated++] = seed_tok;
            history[hist_n++] = seed_tok;
            if (seed_tok == eos) break;
            rc = ib_forward(model, &seed_tok, 1, logits);
            if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }
        }

        if (use_draft) {
            inferbit_model* draft = model->draft_model;
            int base_draft = inferbit_kv_length(draft);
            inferbit_kv_truncate(draft, base_draft - (k - accept));
            if (seed_tok >= 0) {
                rc = ib_forward(draft, &seed_tok, 1, draft->buf_logits);
                if (rc != INFERBIT_OK) { free(history); free(verify_logits); return rc; }
            }
        }
    }

    free(history);
    free(verify_logits);
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
    int use_spec = (model->draft_model != NULL && params.temperature < 0.01f);
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

    if (!use_spec) {
        /* Standard streaming path */
        int32_t next_token = sample_token(logits, vocab, params, recent,
                                          recent_len > max_recent ? max_recent : recent_len);
        recent[recent_len++] = next_token;
        generated++;

        if (next_token == eos || callback(next_token, ctx) == 0) {
            free(recent);
            return generated;
        }

        while (generated < params.max_tokens) {
            rc = ib_forward(model, &next_token, 1, logits);
            if (rc != INFERBIT_OK) { free(recent); return rc; }

            int pen_start = recent_len > max_recent ? recent_len - max_recent : 0;
            next_token = sample_token(logits, vocab, params, recent + pen_start, recent_len - pen_start);
            recent[recent_len++] = next_token;
            generated++;

            if (next_token == eos || callback(next_token, ctx) == 0) break;
        }

        free(recent);
        return generated;
    }

    /* Speculative streaming path */
    inferbit_model* draft = model->draft_model;
    float* dlogits = draft->buf_logits;
    int draft_k = model->draft_tokens > 0 ? model->draft_tokens : 4;
    if (draft_k > 32) draft_k = 32;
    int32_t candidates[32];
    int stopped = 0;

    while (generated < params.max_tokens && !stopped) {
        int base_draft_len = inferbit_kv_length(draft);

        int k = draft_k;
        if (k > (params.max_tokens - generated)) k = params.max_tokens - generated;

        /* Draft generates k candidates */
        int32_t dtok = sample_argmax(dlogits, vocab);
        for (int i = 0; i < k; i++) {
            candidates[i] = dtok;
            rc = ib_forward(draft, &dtok, 1, dlogits);
            if (rc != INFERBIT_OK) { free(recent); return rc; }
            dtok = sample_argmax(dlogits, vocab);
        }

        /* Verify and stream accepted tokens */
        int accepted = 0;
        for (int i = 0; i < k && generated < params.max_tokens; i++) {
            int32_t mtok = sample_argmax(logits, vocab);
            if (mtok == candidates[i]) {
                generated++;
                if (mtok == eos || callback(mtok, ctx) == 0) { stopped = 1; break; }
                accepted++;
                rc = ib_forward(model, &mtok, 1, logits);
                if (rc != INFERBIT_OK) { free(recent); return rc; }
            } else {
                /* Mismatch: emit the main model's token */
                generated++;
                if (mtok == eos || callback(mtok, ctx) == 0) { stopped = 1; break; }
                rc = ib_forward(model, &mtok, 1, logits);
                if (rc != INFERBIT_OK) { free(recent); return rc; }
                /* Rollback draft KV cache */
                inferbit_kv_truncate(draft, base_draft_len + accepted);
                rc = ib_forward(draft, &mtok, 1, dlogits);
                if (rc != INFERBIT_OK) { free(recent); return rc; }
                break;
            }
        }

        /* If all k accepted, emit one more from main model */
        if (!stopped && accepted == k && generated < params.max_tokens) {
            int32_t extra = sample_argmax(logits, vocab);
            generated++;
            if (extra == eos || callback(extra, ctx) == 0) { stopped = 1; }
            else {
                rc = ib_forward(model, &extra, 1, logits);
                if (rc != INFERBIT_OK) { free(recent); return rc; }
                rc = ib_forward(draft, &extra, 1, dlogits);
                if (rc != INFERBIT_OK) { free(recent); return rc; }
            }
        }
    }

    free(recent);
    return generated;
}
