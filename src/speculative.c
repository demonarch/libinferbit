#include "inferbit_internal.h"

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

void inferbit_set_prompt_lookup(inferbit_model* model, int ngram, int k) {
    if (!model) return;
    if (ngram < 0) ngram = 0;
    if (k < 0) k = 0;
    if (k > 32) k = 32;
    model->lookup_ngram = ngram;
    model->lookup_k     = k;
}

/* Search the history buffer for the most recent occurrence of the suffix
 * `history[n - ngram .. n - 1]` in an earlier position, and copy up to k
 * following tokens into `out_candidates`. Returns how many candidates were
 * populated (may be fewer than k if the match is near the end of history).
 *
 * Returns 0 if no earlier match exists or if ngram is too small. */
int ib_lookup_candidates(const int32_t* history, int n, int ngram, int k,
                         int32_t* out_candidates) {
    if (ngram <= 0 || k <= 0 || n <= ngram) return 0;

    const int32_t* needle = history + (n - ngram);

    /* Search from most recent to oldest for the best locality — but skip
     * the current suffix itself (positions >= n - ngram). */
    for (int start = n - ngram - 1; start >= 0; start--) {
        /* Does history[start..start+ngram) match needle? */
        int match = 1;
        for (int j = 0; j < ngram; j++) {
            if (history[start + j] != needle[j]) { match = 0; break; }
        }
        if (!match) continue;

        int src = start + ngram;
        int avail = n - src;
        int take = k < avail ? k : avail;
        if (take <= 0) return 0;
        for (int j = 0; j < take; j++) out_candidates[j] = history[src + j];
        return take;
    }
    return 0;
}
