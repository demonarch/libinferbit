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
#include <stdlib.h>

/* ── Spec-tuning env helpers (opt-in; default path unchanged) ───
 *
 * IB_SPEC_K    : override draft length k for the self-spec / prompt-lookup /
 *                external-draft loops. Clamped to [1, IB_BATCH_MAX] because a
 *                single batched verify (ib_forward_positions) cannot exceed
 *                IB_BATCH_MAX positions. Returns `fallback` when unset/invalid.
 * IB_SPEC_TREE : when 1, prompt-lookup gathers several n-gram continuations
 *                and verifies them together in one batched forward.
 * IB_SPEC_LOG  : when 1, prints per-generation amortization telemetry.
 *
 * All three default OFF / to the caller's current value, so the default decode
 * path is byte-identical when none are set. */

int ib_spec_k_override(int fallback) {
    const char* e = getenv("IB_SPEC_K");
    if (!e || !e[0]) return fallback;
    long v = strtol(e, NULL, 10);
    if (v <= 0) return fallback;
    if (v > IB_BATCH_MAX) v = IB_BATCH_MAX;   /* one batched verify cap */
    return (int)v;
}

int ib_spec_tree_enabled(void) {
    const char* e = getenv("IB_SPEC_TREE");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

int ib_spec_log_enabled(void) {
    const char* e = getenv("IB_SPEC_LOG");
    return (e && e[0] && e[0] != '0') ? 1 : 0;
}

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

/* ── Multi-candidate (tree) prompt-lookup drafter ───────────────
 *
 * Gathers up to `max_branches` DISTINCT n-gram continuations from the running
 * history and packs them into one flat `out_tokens` buffer for a single batched
 * verify. Distinct branches are continuations whose FIRST follower token
 * differs — same-first-token matches collapse to the same accept decision at
 * position 0, so keeping only the longest per distinct first token maximises the
 * useful candidates per batched forward.
 *
 * Layout written:
 *   out_tokens   : branch0_tok0 .. branch0_tok{len0-1}, branch1_tok0 .., ...
 *   branch_off[b]: start index of branch b inside out_tokens
 *   branch_len[b]: length of branch b
 * Total positions written == sum(branch_len) and is capped at `total_cap`
 * (= IB_BATCH_MAX) so the whole tree fits one ib_forward_positions call.
 *
 * Returns the number of branches (0 on miss). Per-branch length is capped at
 * `per_branch_k`. This drafter is purely advisory — the verify against exact
 * logits is the sole arbiter, so any branch shape stays lossless. */
int ib_prompt_lookup_search_tree(const int32_t* history, int hist_len,
                                 int ngram, int per_branch_k, int max_branches,
                                 int total_cap,
                                 int32_t* out_tokens,
                                 int* branch_off, int* branch_len) {
    if (!history || !out_tokens || !branch_off || !branch_len) return 0;
    if (ngram <= 0 || per_branch_k <= 0 || max_branches <= 0) return 0;
    if (total_cap <= 0) return 0;
    if (hist_len < ngram + 1) return 0;

    const int32_t* suffix = history + (hist_len - ngram);
    int search_end = hist_len - ngram - 1;

    int n_branches = 0;
    int total = 0;

    /* Iterate matches from the most recent backward: recent continuations are
     * the most likely to recur, so they get first claim on the position budget.
     * Dedup by first-follower token to keep branches genuinely distinct. */
    for (int i = search_end; i >= 0 && n_branches < max_branches && total < total_cap; i--) {
        int match = 1;
        for (int j = 0; j < ngram; j++) {
            if (history[i + j] != suffix[j]) { match = 0; break; }
        }
        if (!match) continue;

        int start = i + ngram;
        int avail = (hist_len - ngram) - start;
        if (avail <= 0) continue;

        int32_t first = history[start];

        /* Skip if a branch with this first token was already taken. */
        int dup = 0;
        for (int b = 0; b < n_branches; b++) {
            if (out_tokens[branch_off[b]] == first) { dup = 1; break; }
        }
        if (dup) continue;

        int take = avail < per_branch_k ? avail : per_branch_k;
        if (total + take > total_cap) take = total_cap - total;
        if (take <= 0) break;

        branch_off[n_branches] = total;
        branch_len[n_branches] = take;
        for (int j = 0; j < take; j++) out_tokens[total + j] = history[start + j];
        total += take;
        n_branches++;
    }

    return n_branches;
}
