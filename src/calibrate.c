/*
 * calibrate.c — Quantization profile search with quality gates
 *
 * Tries quantization profiles in order (INT2 → INT4 → INT8),
 * converts, loads, evaluates, picks first that passes gates.
 * All in C so Python and Node get identical behavior.
 */

#include "inferbit_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Profile search ───────────��─────────────────────────────── */

/* Forward declaration from eval.c */
typedef struct {
    double perplexity;
    double tokens_per_sec;
    double ms_per_token;
    double memory_mb;
    int    passes;
    char   failed[512];
} ib_eval_result;

ib_eval_result inferbit_evaluate(
    inferbit_model* model,
    const int32_t* const* samples, const int* sample_lengths, int num_samples,
    const int32_t* prompt, int prompt_len,
    int output_tokens, int warmup_runs, int measured_runs,
    double max_perplexity, double min_tokens_per_sec, double max_memory_mb
);

int inferbit_calibrate(
    const char* input_path,
    const char* output_dir,
    const int32_t* const* samples,
    const int* sample_lengths,
    int num_samples,
    int output_tokens,
    int warmup_runs,
    int measured_runs,
    double max_perplexity,
    double min_tokens_per_sec,
    double max_memory_mb,
    int threads,
    void (*progress)(const char* stage, void* ctx),
    void* progress_ctx,
    inferbit_profile_result* results,
    int* selected_index
) {
    if (!input_path || !output_dir || !results || !selected_index) {
        ib_set_error("NULL argument to inferbit_calibrate");
        return INFERBIT_ERROR_PARAM;
    }

    struct { int bits; int sens; const char* name; } profiles[] = {
        {2, 8, "int2"},
        {4, 8, "int4"},
        {8, 8, "int8"},
    };
    int n_profiles = 3;
    *selected_index = -1;

    for (int p = 0; p < n_profiles; p++) {
        memset(&results[p], 0, sizeof(inferbit_profile_result));
        results[p].bits = profiles[p].bits;
        results[p].sensitive_bits = profiles[p].sens;
        results[p].perplexity = -1.0;

        snprintf(results[p].ibf_path, sizeof(results[p].ibf_path),
                 "%s/%s.ibf", output_dir, profiles[p].name);

        /* Convert */
        if (progress) {
            char msg[128];
            snprintf(msg, sizeof(msg), "converting %s", profiles[p].name);
            progress(msg, progress_ctx);
        }

        inferbit_convert_config ccfg = inferbit_default_convert_config();
        ccfg.default_bits = profiles[p].bits;
        ccfg.sensitive_bits = profiles[p].sens;
        ccfg.threads = threads;

        int rc = inferbit_convert(input_path, results[p].ibf_path, &ccfg);
        if (rc != INFERBIT_OK) {
            results[p].passes = 0;
            snprintf(results[p].failed, sizeof(results[p].failed), "conversion failed;");
            continue;
        }

        /* Load */
        if (progress) {
            char msg[128];
            snprintf(msg, sizeof(msg), "evaluating %s", profiles[p].name);
            progress(msg, progress_ctx);
        }

        inferbit_config* cfg = inferbit_config_create();
        if (threads > 0) inferbit_config_set_threads(cfg, threads);
        inferbit_model* model = inferbit_load(results[p].ibf_path, cfg);
        inferbit_config_free(cfg);

        if (!model) {
            results[p].passes = 0;
            snprintf(results[p].failed, sizeof(results[p].failed), "load failed;");
            continue;
        }

        /* Evaluate */
        ib_eval_result ev = inferbit_evaluate(
            model, samples, sample_lengths, num_samples,
            NULL, 0, output_tokens, warmup_runs, measured_runs,
            max_perplexity, min_tokens_per_sec, max_memory_mb
        );

        /* Copy results to public struct */
        results[p].perplexity = ev.perplexity;
        results[p].tokens_per_sec = ev.tokens_per_sec;
        results[p].ms_per_token = ev.ms_per_token;
        results[p].memory_mb = ev.memory_mb;
        results[p].passes = ev.passes;
        memcpy(results[p].failed, ev.failed, sizeof(ev.failed));

        inferbit_free(model);

        /* Check if this passes */
        if (results[p].passes && *selected_index < 0) {
            results[p].selected = 1;
            *selected_index = p;
        }
    }

    /* If none pass, select the last (INT8 — most conservative) */
    if (*selected_index < 0) {
        *selected_index = n_profiles - 1;
        results[n_profiles - 1].selected = 1;
    }

    return INFERBIT_OK;
}
