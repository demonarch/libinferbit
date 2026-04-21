/*
 * speculative.c — speculative-decoding helpers
 *
 * Thin wrappers for the draft-model API plus the prompt-lookup drafter
 * (n-gram match over running history — no external draft model required).
 *
 * The spec control flow (verify / mismatch / KV rollback) lives in
 * generate.c; this file provides only the drafting primitives.
 */

#include "inferbit_internal.h"

/* ── External-draft-model API ───────────────────────────────── */

void inferbit_set_draft_model(inferbit_model* model, inferbit_model* draft, int draft_tokens) {
    if (!model) return;
    model->draft_model  = draft;
    model->draft_tokens = draft_tokens > 0 ? draft_tokens : 4;
}

void inferbit_unset_draft_model(inferbit_model* model) {
    if (!model) return;
    model->draft_model  = NULL;
    model->draft_tokens = 0;
}

/* ── Prompt-lookup drafter (no external model) ──────────────── */

void inferbit_set_prompt_lookup(inferbit_model* model, int ngram, int k) {
    if (!model) return;
    if (ngram < 0) ngram = 0;
    if (k     < 0) k     = 0;
    if (k     > 32) k = 32;      /* Matches candidates[32] in verify loop. */
    model->lookup_ngram = ngram;
    model->lookup_k     = k;
}

/* Earliest-match search. Returns number of candidates written (0..k). */
int ib_prompt_lookup_search(const int32_t* history, int hist_len,
                            int ngram, int k, int32_t* out_candidates) {
    if (!history || !out_candidates) return 0;
    if (ngram <= 0 || k <= 0) return 0;
    if (hist_len < ngram + 1) return 0;

    const int32_t* suffix = history + (hist_len - ngram);

    /* Search positions [0 .. hist_len - ngram - 1]. The "-1" ensures there is
     * at least one token following the match that is not itself part of the
     * suffix — otherwise the "draft" would be the suffix itself, useless. */
    int search_end = hist_len - ngram - 1;
    for (int i = 0; i <= search_end; i++) {
        int match = 1;
        for (int j = 0; j < ngram; j++) {
            if (history[i + j] != suffix[j]) { match = 0; break; }
        }
        if (!match) continue;

        int start = i + ngram;
        /* Cap so candidates come from history, not from the suffix region. */
        int avail = (hist_len - ngram) - start;
        if (avail <= 0) continue;
        int take = avail < k ? avail : k;
        for (int j = 0; j < take; j++) out_candidates[j] = history[start + j];
        return take;
    }
    return 0;
}
