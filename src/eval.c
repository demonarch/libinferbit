/*
 * eval.c — Model evaluation: perplexity, throughput, quality gates
 *
 * These run in C so both Python and Node wrappers get the same
 * evaluation logic without reimplementing it.
 */

#include "inferbit_internal.h"
#include "platform.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Perplexity ─────────────────────────────────────────────── */

static double logsumexp(const float* logits, int n) {
    float max_val = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp((double)(logits[i] - max_val));
    }
    return (double)max_val + log(sum);
}

double inferbit_perplexity(
    inferbit_model* model,
    const int32_t* const* samples,    /* Array of token arrays */
    const int* sample_lengths,         /* Length of each sample */
    int num_samples
) {
    if (!model || !samples || !sample_lengths || num_samples <= 0) return -1.0;

    int vocab = model->header.vocab_size;
    float* logits = model->buf_logits;
    double nll = 0.0;
    int count = 0;

    for (int s = 0; s < num_samples; s++) {
        const int32_t* tokens = samples[s];
        int len = sample_lengths[s];
        if (len < 2) continue;

        inferbit_kv_clear(model);

        /* Teacher forcing: for each position, predict next token */
        int rc = ib_forward(model, &tokens[0], 1, logits);
        if (rc != INFERBIT_OK) continue;

        for (int i = 1; i < len; i++) {
            double lse = logsumexp(logits, vocab);
            nll += lse - (double)logits[tokens[i]];
            count++;

            if (i < len - 1) {
                rc = ib_forward(model, &tokens[i], 1, logits);
                if (rc != INFERBIT_OK) break;
            }
        }
    }

    if (count == 0) return -1.0;
    return exp(nll / (double)count);
}

/* ── Throughput benchmark ───────────────────────────────────── */

typedef struct {
    double tokens_per_sec;
    double ms_per_token;
} ib_throughput_result;

static double time_now_sec(void) {
    struct timespec ts;
    ib_clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

ib_throughput_result inferbit_throughput(
    inferbit_model* model,
    const int32_t* prompt,
    int prompt_len,
    int output_tokens,
    int warmup_runs,
    int measured_runs
) {
    ib_throughput_result result = {0.0, 0.0};
    if (!model || !prompt || prompt_len <= 0) return result;

    inferbit_sample_params params = inferbit_default_sample_params();
    params.temperature = 0.0f;
    params.max_tokens = output_tokens;

    int32_t* out_buf = malloc(output_tokens * sizeof(int32_t));
    if (!out_buf) return result;

    int total = warmup_runs + measured_runs;
    double sum_sec_per_tok = 0.0;
    int measured = 0;

    for (int i = 0; i < total; i++) {
        inferbit_kv_clear(model);
        double t0 = time_now_sec();
        int n = inferbit_generate(model, prompt, prompt_len, out_buf, output_tokens, params);
        double dt = time_now_sec() - t0;

        if (i >= warmup_runs && n > 0) {
            sum_sec_per_tok += dt / (double)n;
            measured++;
        }
    }

    free(out_buf);

    if (measured > 0) {
        double avg = sum_sec_per_tok / (double)measured;
        result.tokens_per_sec = 1.0 / avg;
        result.ms_per_token = avg * 1000.0;
    }
    return result;
}

/* ── Quality gates ──────────────────────────────────────────── */

typedef struct {
    double perplexity;       /* -1 if not measured */
    double tokens_per_sec;
    double ms_per_token;
    double memory_mb;
    int    passes;           /* 1 if all gates pass */
    char   failed[512];      /* Semicolon-separated failure reasons */
} ib_eval_result;

ib_eval_result inferbit_evaluate(
    inferbit_model* model,
    const int32_t* const* samples,
    const int* sample_lengths,
    int num_samples,
    const int32_t* prompt,
    int prompt_len,
    int output_tokens,
    int warmup_runs,
    int measured_runs,
    double max_perplexity,      /* 0 = no gate */
    double min_tokens_per_sec,  /* 0 = no gate */
    double max_memory_mb        /* 0 = no gate */
) {
    ib_eval_result result;
    memset(&result, 0, sizeof(result));
    result.perplexity = -1.0;
    result.passes = 1;

    if (!model) return result;

    /* Perplexity */
    if (samples && sample_lengths && num_samples > 0) {
        result.perplexity = inferbit_perplexity(model, samples, sample_lengths, num_samples);
    }

    /* Throughput */
    int32_t default_prompt[] = {1, 2, 3, 4, 5};
    if (!prompt || prompt_len <= 0) {
        prompt = default_prompt;
        prompt_len = 5;
    }
    ib_throughput_result thr = inferbit_throughput(
        model, prompt, prompt_len, output_tokens, warmup_runs, measured_runs
    );
    result.tokens_per_sec = thr.tokens_per_sec;
    result.ms_per_token = thr.ms_per_token;

    /* Memory */
    result.memory_mb = (double)inferbit_model_total_memory(model) / (1024.0 * 1024.0);

    /* Check gates */
    result.failed[0] = '\0';
    if (max_perplexity > 0 && result.perplexity > 0 && result.perplexity > max_perplexity) {
        result.passes = 0;
        snprintf(result.failed + strlen(result.failed),
                 sizeof(result.failed) - strlen(result.failed),
                 "perplexity %.3f > %.3f;", result.perplexity, max_perplexity);
    }
    if (min_tokens_per_sec > 0 && result.tokens_per_sec < min_tokens_per_sec) {
        result.passes = 0;
        snprintf(result.failed + strlen(result.failed),
                 sizeof(result.failed) - strlen(result.failed),
                 "tokens/sec %.3f < %.3f;", result.tokens_per_sec, min_tokens_per_sec);
    }
    if (max_memory_mb > 0 && result.memory_mb > max_memory_mb) {
        result.passes = 0;
        snprintf(result.failed + strlen(result.failed),
                 sizeof(result.failed) - strlen(result.failed),
                 "memory %.1fMB > %.1fMB;", result.memory_mb, max_memory_mb);
    }

    return result;
}
