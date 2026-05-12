/* Unified bench harness for the libinferbit-vs-llama.cpp comparison.
 *
 * Usage:
 *   bench_compare <model.ibf> [--backend cpu|gpu] [--prompt-tokens N]
 *                              [--gen-tokens N] [--ctx N]
 *
 * Measures, for the requested backend:
 *   - prompt-eval (prefill) time → time-to-first-token (TTFT) seen by user
 *   - decode tok/s over `gen-tokens` decoded steps
 *   - argmax sequence (printed so quality can be cross-checked offline)
 *
 * Uses a deterministic synthetic prompt (token IDs 1..N) so results are
 * reproducible and compare apples-to-apples to the llama.cpp side
 * which uses the same fixed token sequence.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "inferbit.h"
#include "metal/metal_runtime.h"
#include "inferbit_internal.h"

extern float ib_fp16_to_fp32(uint16_t h);

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int argmax_logit(const float *l, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
    return best;
}

/* Mirrors forward.c::embedding_lookup so the GPU path can hand off
 * a pre-decoded fp32 hidden state to ib_metal_forward_token. */
static void cpu_embed_lookup(const inferbit_model *m, int token, float *out) {
    int hidden = m->header.hidden_size;
    const ib_tensor_meta *e = &m->token_embedding;
    const uint8_t *base = (const uint8_t*)m->weight_data;
    const uint8_t *data_b = base + e->offset;
    const void *scales_raw = e->scale_size ? (const void*)(base + e->scale_offset) : NULL;
    if (e->pq) {
        const pqv2_t *pq = e->pq;
        uint32_t nc = pq->N / pq->G;
        uint32_t HALF = pq->half;
        uint32_t K = pq->K;
        const uint8_t *idx_base = (const uint8_t *)pq->indices;
        const int8_t *cb_q = (const int8_t *)pq->cb_q;
        const uint16_t *cb_s = (const uint16_t *)pq->cb_scale;
        float rs = pq->row_scale ? ib_fp16_to_fp32(((const uint16_t *)pq->row_scale)[token]) : 1.0f;
        for (uint32_t c = 0; c < nc; c++) {
            for (uint32_t s = 0; s < pq->n_subchunks; s++) {
                uint8_t k = idx_base[((size_t)c * pq->n_subchunks + s) * pq->M + token];
                float scl = ib_fp16_to_fp32(cb_s[s * K + k]) * rs;
                for (uint32_t h = 0; h < HALF; h++) {
                    out[c * pq->G + s * HALF + h] =
                        (float)cb_q[(s * K + k) * HALF + h] * scl;
                }
            }
        }
        (void)data_b; (void)scales_raw;
        return;
    }
    if (e->bits == 16) {
        const uint16_t *row = (const uint16_t*)data_b + (size_t)token * hidden;
        for (int i = 0; i < hidden; i++) out[i] = ib_fp16_to_fp32(row[i]);
    } else if (e->bits == 8) {
        const int8_t *row = (const int8_t*)data_b + (size_t)token * hidden;
        float scale = scales_raw
            ? ib_fp16_to_fp32(((const uint16_t*)scales_raw)[token]) : 1.0f;
        for (int i = 0; i < hidden; i++) out[i] = (float)row[i] * scale;
    } else if (e->bits == 4) {
        size_t row_bytes = (size_t)hidden / 2;
        const uint8_t *row = data_b + (size_t)token * row_bytes;
        float scale = scales_raw
            ? ib_fp16_to_fp32(((const uint16_t*)scales_raw)[token]) : 1.0f;
        for (int i = 0; i < hidden; i += 2) {
            uint8_t byte = row[i / 2];
            out[i]   = (float)((int8_t)(byte & 0x0F) - 8) * scale;
            out[i+1] = (float)((int8_t)((byte >> 4) & 0x0F) - 8) * scale;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.ibf> [--backend cpu|gpu] "
                "[--prompt-tokens N] [--gen-tokens N] [--ctx N]\n", argv[0]);
        return 1;
    }
    const char *backend = "cpu";
    int prompt_tokens = 32;
    int gen_tokens    = 64;
    int ctx_len       = 1024;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--backend") && i+1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--prompt-tokens") && i+1 < argc) prompt_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen-tokens") && i+1 < argc) gen_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i+1 < argc) ctx_len = atoi(argv[++i]);
    }
    int use_gpu = (strcmp(backend, "gpu") == 0);

    inferbit_config *cfg = inferbit_config_create();
    inferbit_config_set_context_length(cfg, ctx_len);
    inferbit_model *m = inferbit_load(argv[1], cfg);
    if (!m) { fprintf(stderr, "load failed\n"); return 2; }
    int hidden = m->header.hidden_size;
    int vocab  = m->header.vocab_size;

    int total = prompt_tokens + gen_tokens;
    if (total > ctx_len) {
        fprintf(stderr, "prompt+gen=%d > ctx=%d\n", total, ctx_len);
        return 1;
    }
    int32_t *prompt = malloc((size_t)prompt_tokens * sizeof(int32_t));
    for (int i = 0; i < prompt_tokens; i++) prompt[i] = (i + 1) % vocab;

    /* Setup GPU buffers if needed. */
    ib_metal_ctx *ctx = NULL;
    ib_metal_model_buffers *gbufs = NULL;
    float *embed_buf = malloc((size_t)hidden * sizeof(float));
    if (use_gpu) {
        ctx = ib_metal_create();
        if (!ctx) { fprintf(stderr, "Metal not available\n"); return 3; }
        gbufs = ib_metal_upload_model(ctx, m);
        if (!gbufs) { fprintf(stderr, "GPU upload failed\n"); return 4; }
        /* Opt-in via env var: ib_metal_strip_cpu_mmap drops the mmap and
         * keeps only the embedding bytes. On macOS this momentarily holds
         * mmap+Metal+embed at once, so /usr/bin/time -l peak RSS goes UP
         * by ~embedding_size, but steady-state RSS during a long inference
         * run drops by ~file_size. Off by default to keep the bench's peak
         * metric clean; flip via IB_STRIP_MMAP=1. */
        if (getenv("IB_STRIP_MMAP")) ib_metal_strip_cpu_mmap(m);
    }

    /* Opt-in batched prefill (Item 2 from the future-work list).
     * Off by default to keep existing tests deterministic; flip
     * IB_PREFILL_BATCH=1 in the env to use ib_metal_forward_prefill. */
    int use_batched_prefill = (use_gpu && getenv("IB_PREFILL_BATCH") != NULL);
    float *embed_batch = NULL;
    if (use_batched_prefill) {
        embed_batch = malloc((size_t)prompt_tokens * hidden * sizeof(float));
    }

    /* ── Warmup (post-load, mirrors llama-bench default) ─────────── */
    float *logits = malloc((size_t)vocab * sizeof(float));
    inferbit_kv_clear(m);
    if (use_gpu) ib_metal_reset_kv(gbufs);
    if (use_gpu) {
        if (use_batched_prefill) {
            for (int i = 0; i < prompt_tokens; i++)
                cpu_embed_lookup(m, prompt[i], embed_batch + (size_t)i * hidden);
            int rc = ib_metal_forward_prefill(ctx, gbufs, embed_batch,
                                                prompt_tokens, 0, logits);
            if (rc == -2) {
                fprintf(stderr, "warn: model not blk32-only; falling back to per-token\n");
                use_batched_prefill = 0;
            }
        }
        if (!use_batched_prefill) {
            for (int i = 0; i < prompt_tokens; i++) {
                cpu_embed_lookup(m, prompt[i], embed_buf);
                ib_metal_forward_token(ctx, gbufs, embed_buf, i, logits);
            }
        }
    } else {
        ib_forward(m, prompt, prompt_tokens, logits);
    }
    inferbit_kv_clear(m);
    if (use_gpu) ib_metal_reset_kv(gbufs);

    /* ── Prefill (TTFT timer) ──────────────────────────────────────── */
    double t0 = now_sec();
    if (use_gpu) {
        if (use_batched_prefill) {
            for (int i = 0; i < prompt_tokens; i++)
                cpu_embed_lookup(m, prompt[i], embed_batch + (size_t)i * hidden);
            ib_metal_forward_prefill(ctx, gbufs, embed_batch,
                                       prompt_tokens, 0, logits);
        } else {
            for (int i = 0; i < prompt_tokens; i++) {
                cpu_embed_lookup(m, prompt[i], embed_buf);
                ib_metal_forward_token(ctx, gbufs, embed_buf, i, logits);
            }
        }
    } else {
        ib_forward(m, prompt, prompt_tokens, logits);
    }
    double t_ttft = now_sec() - t0;
    int first_decode = argmax_logit(logits, vocab);

    /* ── Decode loop ──────────────────────────────────────────────── */
    int *generated = malloc((size_t)gen_tokens * sizeof(int));
    generated[0] = first_decode;
    t0 = now_sec();
    if (use_gpu) {
        /* Paginated mode: IB_DECODE_PAGE_N=N to chain N forwards per
         * command buffer with GPU argmax + embed_lookup. Off → original
         * one-CB-per-token path. */
        int page_n = 0;
        const char *page_env = getenv("IB_DECODE_PAGE_N");
        if (page_env) page_n = atoi(page_env);
        if (page_n > 0) {
            int i = 0;
            while (i < gen_tokens) {
                int n = gen_tokens - i;
                if (n > page_n) n = page_n;
                int tok = (i == 0) ? first_decode : generated[i - 1];
                cpu_embed_lookup(m, tok, embed_buf);
                int rc = ib_metal_forward_decode_n(ctx, gbufs, embed_buf,
                                                     prompt_tokens + i, n,
                                                     generated + i);
                if (rc != 0) {
                    fprintf(stderr, "forward_decode_n rc=%d, falling back\n", rc);
                    /* Fallback to per-token for remaining. */
                    for (int j = 0; j < n; j++) {
                        int t = (i + j == 0) ? first_decode : generated[i + j - 1];
                        cpu_embed_lookup(m, t, embed_buf);
                        ib_metal_forward_token(ctx, gbufs, embed_buf,
                                                prompt_tokens + i + j, logits);
                        generated[i + j] = argmax_logit(logits, vocab);
                    }
                }
                i += n;
            }
        } else {
            for (int i = 0; i < gen_tokens; i++) {
                int tok = (i == 0) ? first_decode : generated[i - 1];
                cpu_embed_lookup(m, tok, embed_buf);
                ib_metal_forward_token(ctx, gbufs, embed_buf, prompt_tokens + i, logits);
                generated[i] = argmax_logit(logits, vocab);
            }
        }
    } else {
        for (int i = 0; i < gen_tokens; i++) {
            int tok = (i == 0) ? first_decode : generated[i - 1];
            int32_t one = (int32_t)tok;
            ib_forward(m, &one, 1, logits);
            generated[i] = argmax_logit(logits, vocab);
        }
    }
    double t_decode = now_sec() - t0;

    /* ── Output (single line, machine-parseable + human readable) ── */
    printf("BACKEND=%s\n", use_gpu ? "libinferbit-gpu" : "libinferbit-cpu");
    printf("MODEL=%s\n", argv[1]);
    printf("ARCH=%s LAYERS=%d HIDDEN=%d VOCAB=%d\n",
           m->header.architecture, m->header.num_layers, hidden, vocab);
    printf("PROMPT_TOKENS=%d GEN_TOKENS=%d CTX=%d\n", prompt_tokens, gen_tokens, ctx_len);
    printf("TTFT_MS=%.3f\n", t_ttft * 1000.0);
    printf("PROMPT_EVAL_TOK_PER_SEC=%.2f\n", (double)prompt_tokens / t_ttft);
    printf("DECODE_TOK_PER_SEC=%.2f\n", (double)gen_tokens / t_decode);
    printf("DECODE_TOTAL_MS=%.3f\n", t_decode * 1000.0);
    printf("FIRST_DECODE_TOKEN=%d\n", first_decode);
    printf("GENERATED=");
    for (int i = 0; i < gen_tokens; i++) printf("%s%d", i ? "," : "", generated[i]);
    printf("\n");

    free(prompt); free(logits); free(generated); free(embed_buf);
    if (embed_batch) free(embed_batch);
    if (use_gpu) {
        ib_metal_release_model(ctx, gbufs);
        ib_metal_destroy(ctx);
    }
    inferbit_free(m);
    inferbit_config_free(cfg);
    return 0;
}
