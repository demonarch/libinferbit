/* bench_speculative — two-model speculative decoding bench.
 *
 * Loads a draft model (e.g. Llama-3.2-1B) and a target model (e.g.
 * Llama-3.1-8B), both Llama-3 family with identical tokenizer.
 * Generates a sequence two ways and compares tok/s:
 *
 *   1) Target alone: pure GPU forward_token loop on target.
 *   2) Speculative: draft generates K tokens via forward_token, target
 *      verifies them in one forward_prefill_logits_all call. Accept
 *      longest matching prefix + corrected token.
 *
 * Usage: bench_speculative <draft.ibf> <target.ibf> [K=4] [n_gen=128] [ctx=512]
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
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

/* Mirrors bench_compare's cpu_embed_lookup (INT4/INT8/fp16/PQv2 paths). */
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
                for (uint32_t h = 0; h < HALF; h++) {
                    int8_t q = cb_q[(s * K + k) * HALF + h];
                    float cb = (float)q * ib_fp16_to_fp32(cb_s[(s * K + k) * HALF + h]);
                    out[c * pq->G + s * HALF + h] = cb * rs;
                }
            }
        }
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
    } else {
        size_t row_bytes = (size_t)hidden / 2;
        const uint8_t *row = data_b + (size_t)token * row_bytes;
        float scale = scales_raw
            ? ib_fp16_to_fp32(((const uint16_t*)scales_raw)[token]) : 1.0f;
        for (int i = 0; i < hidden; i++) {
            uint8_t byte = row[i/2];
            int w = (i & 1) ? ((int)((byte >> 4) & 0x0F) - 8)
                            : ((int)(byte & 0x0F) - 8);
            out[i] = (float)w * scale;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <draft.ibf> <target.ibf> [K=4] [n_gen=128] [ctx=512]\n", argv[0]);
        return 1;
    }
    const char *draft_path  = argv[1];
    const char *target_path = argv[2];
    int K     = (argc > 3) ? atoi(argv[3]) : 4;
    int n_gen = (argc > 4) ? atoi(argv[4]) : 128;
    int ctx   = (argc > 5) ? atoi(argv[5]) : 512;

    inferbit_config *dcfg = inferbit_config_create();
    inferbit_config_set_context_length(dcfg, ctx);
    inferbit_model *draft = inferbit_load(draft_path, dcfg);
    if (!draft) { fprintf(stderr, "draft load failed\n"); return 2; }
    inferbit_config *tcfg = inferbit_config_create();
    inferbit_config_set_context_length(tcfg, ctx);
    inferbit_model *target = inferbit_load(target_path, tcfg);
    if (!target) { fprintf(stderr, "target load failed\n"); return 3; }

    int dhidden = draft->header.hidden_size;
    int thidden = target->header.hidden_size;
    int tvocab  = target->header.vocab_size;
    int dvocab  = draft->header.vocab_size;
    if (dvocab != tvocab) {
        fprintf(stderr, "vocab mismatch: draft=%d target=%d\n", dvocab, tvocab);
        return 4;
    }

    ib_metal_ctx *mctx = ib_metal_create();
    ib_metal_model_buffers *dbufs = ib_metal_upload_model(mctx, draft);
    ib_metal_model_buffers *tbufs = ib_metal_upload_model(mctx, target);
    if (!dbufs || !tbufs) { fprintf(stderr, "metal upload failed\n"); return 5; }

    /* Synthetic short prompt: token IDs 1..8 (matches bench_compare style). */
    const int prompt_len = 8;
    int prompt[prompt_len];
    for (int i = 0; i < prompt_len; i++) prompt[i] = (i + 1) % tvocab;

    float *d_embed = malloc((size_t)dhidden * sizeof(float));
    float *t_embed = malloc((size_t)thidden * sizeof(float));
    float *t_logits = malloc((size_t)tvocab * sizeof(float));
    float *d_logits = malloc((size_t)dvocab * sizeof(float));

    /* ── Warmup + prompt prefill on both models (per-token loop). ── */
    ib_metal_reset_kv(dbufs);
    ib_metal_reset_kv(tbufs);
    for (int i = 0; i < prompt_len; i++) {
        cpu_embed_lookup(draft,  prompt[i], d_embed);
        cpu_embed_lookup(target, prompt[i], t_embed);
        ib_metal_forward_token(mctx, dbufs, d_embed, i, d_logits);
        ib_metal_forward_token(mctx, tbufs, t_embed, i, t_logits);
    }
    int last_tok = argmax_logit(t_logits, tvocab);

    /* ── Reset and prefill for the TARGET-ALONE timed run. ── */
    ib_metal_reset_kv(tbufs);
    for (int i = 0; i < prompt_len; i++) {
        cpu_embed_lookup(target, prompt[i], t_embed);
        ib_metal_forward_token(mctx, tbufs, t_embed, i, t_logits);
    }
    last_tok = argmax_logit(t_logits, tvocab);

    double t0 = now_sec();
    int pos = prompt_len;
    int tokens_alone[2048];
    int n_alone = 0;
    for (int g = 0; g < n_gen && pos < ctx; g++) {
        cpu_embed_lookup(target, last_tok, t_embed);
        ib_metal_forward_token(mctx, tbufs, t_embed, pos, t_logits);
        last_tok = argmax_logit(t_logits, tvocab);
        tokens_alone[n_alone++] = last_tok;
        pos++;
    }
    double t_alone = now_sec() - t0;
    double tps_alone = (double)n_alone / t_alone;

    /* ── Speculative run. ── */
    ib_metal_reset_kv(dbufs);
    ib_metal_reset_kv(tbufs);
    for (int i = 0; i < prompt_len; i++) {
        cpu_embed_lookup(draft,  prompt[i], d_embed);
        cpu_embed_lookup(target, prompt[i], t_embed);
        ib_metal_forward_token(mctx, dbufs, d_embed, i, d_logits);
        ib_metal_forward_token(mctx, tbufs, t_embed, i, t_logits);
    }
    last_tok = argmax_logit(t_logits, tvocab);

    float *all_embeds  = malloc((size_t)K * thidden * sizeof(float));
    float *all_logits  = malloc((size_t)K * tvocab  * sizeof(float));
    int   *drafts      = malloc((size_t)K * sizeof(int));
    int   tokens_spec[2048];
    int n_spec = 0;
    int accepted_sum = 0;
    int iterations = 0;
    /* Per-iteration acceptance histogram: hist[k] = # iters with k accepted
     * drafts (k in 0..K). Reveals whether speculative is winning broadly
     * or only on a few high-streak iterations (dflash #17). */
    int *hist = calloc((size_t)(K + 1), sizeof(int));

    double t1 = now_sec();
    pos = prompt_len;
    while (n_spec < n_gen && pos + K + 1 <= ctx) {
        /* Step 1: draft generates K tokens via forward_token. */
        int cur = last_tok;
        for (int k = 0; k < K; k++) {
            cpu_embed_lookup(draft, cur, d_embed);
            ib_metal_forward_token(mctx, dbufs, d_embed, pos + k, d_logits);
            cur = argmax_logit(d_logits, dvocab);
            drafts[k] = cur;
        }

        /* Step 2: target verifies — build K embeddings: last_tok (input)
         *   actually: target's input at pos+k is the k-th token we
         *   "committed", which is last_tok for k=0 and drafts[k-1] for k≥1. */
        cpu_embed_lookup(target, last_tok, all_embeds);
        for (int k = 1; k < K; k++) {
            cpu_embed_lookup(target, drafts[k-1], all_embeds + (size_t)k * thidden);
        }
        int rc = ib_metal_forward_prefill_logits_all(
            mctx, tbufs, all_embeds, K, pos, all_logits);
        if (rc != 0) { fprintf(stderr, "verify failed rc=%d\n", rc); break; }

        /* Step 3: acceptance. At position pos+k, all_logits[k] is the
         * model's prediction for what comes AFTER the input at pos+k —
         * compare to drafts[k]. Accept while match. */
        int accepted = 0;
        int correction = -1;
        for (int k = 0; k < K; k++) {
            int targ = argmax_logit(all_logits + (size_t)k * tvocab, tvocab);
            if (targ == drafts[k]) {
                accepted++;
            } else {
                correction = targ;
                break;
            }
        }
        /* If all K accepted, the next token after drafts[K-1] is target's
         * argmax at position pos+K-1 — but we already have that as
         * all_logits[K-1] (which is what we used as the K-th comparison).
         * For accepted==K, the correction is whatever target predicts
         * after the last accepted draft — we'd need an extra position.
         * Simplification: if accepted==K, take all_logits[K-1]'s argmax
         * (already done above — that became `targ` which == drafts[K-1]
         * meaning it matched). To produce the (K+1)th token we'd need
         * one more forward pass. For simplicity, treat all-accepted as
         * "produced K tokens" — don't emit the bonus token. */
        int produced = accepted + (correction >= 0 ? 1 : 0);

        /* Step 4: emit accepted drafts + correction (if any). */
        for (int k = 0; k < accepted && n_spec < n_gen; k++) {
            tokens_spec[n_spec++] = drafts[k];
        }
        if (correction >= 0 && n_spec < n_gen) {
            tokens_spec[n_spec++] = correction;
        }

        /* Step 5: crop both KV caches back to the accepted boundary
         * (doc 36 phase 2.3 — explicit ib_kv_crop instead of the old
         * "stale KV is harmless until overwritten" trick).
         *
         * Draft wrote KV at positions [pos..pos+K-1] for drafts[0..K-1].
         * After verify only [pos..pos+accepted-1] are valid; the draft's
         * KV at pos+accepted holds the wrong token and pos+accepted+1..
         * pos+K-1 are speculative garbage. The old code relied on those
         * positions being causally-future (hence masked) until the next
         * iteration overwrites them — but that assumption breaks once a
         * rotating KV window (phase 2.2) can wrap and surface a stale
         * slot. inferbit_kv_truncate makes the eviction explicit: it
         * resets each cache's logical length so the bookkeeping matches
         * the real accepted prefix length.
         *
         * crop_to = the number of fully-committed tokens after this
         * iteration. The draft additionally needs position pos+accepted
         * refilled with the CORRECTION token (target's pick), so we crop
         * the draft to pos+accepted and re-run one forward_token. */
        int crop_to = pos + accepted;
        inferbit_kv_truncate(draft,  crop_to);
        inferbit_kv_truncate(target, crop_to);
        if (correction >= 0) {
            cpu_embed_lookup(draft, correction, d_embed);
            ib_metal_forward_token(mctx, dbufs, d_embed, crop_to, d_logits);
        }

        /* Update target's "last_tok" to the last produced token. */
        if (n_spec > 0) last_tok = tokens_spec[n_spec - 1];
        pos += produced;
        accepted_sum += accepted;
        if (accepted >= 0 && accepted <= K) hist[accepted]++;
        iterations++;
    }
    double t_spec = now_sec() - t1;
    double tps_spec = (double)n_spec / t_spec;

    /* ── Print results. ── */
    printf("BACKEND=libinferbit-gpu speculative\n");
    printf("DRAFT=%s\n", draft_path);
    printf("TARGET=%s\n", target_path);
    printf("K=%d  N_GEN=%d  CTX=%d  PROMPT_LEN=%d\n", K, n_gen, ctx, prompt_len);
    printf("\n");
    double ms_per_tok_alone = (n_alone > 0) ? (t_alone * 1000.0 / (double)n_alone) : 0.0;
    double ms_per_tok_spec  = (n_spec  > 0) ? (t_spec  * 1000.0 / (double)n_spec ) : 0.0;
    printf("TARGET_ALONE_TOKENS=%d  TARGET_ALONE_SEC=%.3f  TARGET_ALONE_TPS=%.2f  TARGET_ALONE_MS_PER_TOK=%.2f\n",
           n_alone, t_alone, tps_alone, ms_per_tok_alone);
    printf("SPECULATIVE_TOKENS=%d  SPECULATIVE_SEC=%.3f  SPECULATIVE_TPS=%.2f  SPECULATIVE_MS_PER_TOK=%.2f\n",
           n_spec, t_spec, tps_spec, ms_per_tok_spec);
    printf("SPECULATIVE_SPEEDUP=%.2fx\n", tps_spec / tps_alone);
    printf("ITERATIONS=%d  TOTAL_ACCEPTED=%d  AVG_ACCEPTED_PER_ITER=%.2f  (K=%d)\n",
           iterations, accepted_sum, (double)accepted_sum / iterations, K);
    /* Per-iteration acceptance histogram. */
    printf("ACCEPTANCE_HISTOGRAM:");
    for (int k = 0; k <= K; k++) {
        double pct = (iterations > 0) ? (100.0 * hist[k] / iterations) : 0.0;
        printf(" %d:%d(%.0f%%)", k, hist[k], pct);
    }
    printf("\n");
    printf("\nGENERATED_ALONE: ");
    for (int i = 0; i < n_alone && i < 16; i++) printf("%d,", tokens_alone[i]);
    printf("\nGENERATED_SPEC:  ");
    for (int i = 0; i < n_spec  && i < 16; i++) printf("%d,", tokens_spec[i]);
    printf("\n");

    /* Cleanup. */
    free(d_embed); free(t_embed); free(t_logits); free(d_logits);
    free(all_embeds); free(all_logits); free(drafts);
    free(hist);
    ib_metal_release_model(mctx, dbufs);
    ib_metal_release_model(mctx, tbufs);
    ib_metal_destroy(mctx);
    inferbit_free(draft);
    inferbit_free(target);
    inferbit_config_free(dcfg);
    inferbit_config_free(tcfg);
    return 0;
}
