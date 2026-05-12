/* Streaming-PPL bench for libinferbit. CPU or GPU.
 *
 * Usage:
 *   bench_ppl_run <model.ibf> <tokens.i32.bin> [--backend cpu|gpu]
 *                  [--warmup N] [--score N] [--ctx N]
 *
 * tokens.i32.bin is a flat int32 token-id file. We feed [0..warmup-1]
 * as a one-shot prefill, then stream-score the next `score` predictions.
 * PPL = exp(mean NLL).
 */
#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

/* Peak resident-set size in bytes via getrusage. ru_maxrss is BYTES on
 * macOS, KB on Linux. */
static size_t peak_rss_bytes(void) {
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r) != 0) return 0;
#if defined(__APPLE__)
    return (size_t)r.ru_maxrss;
#else
    return (size_t)r.ru_maxrss * 1024;
#endif
}

/* Current resident-set size in bytes. On macOS via mach_task_basic_info
 * (the only way to get *current* RSS — getrusage returns peak only).
 * Returns 0 on platforms where it isn't implemented. */
static size_t current_rss_bytes(void) {
#if defined(__APPLE__)
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &cnt) != KERN_SUCCESS) return 0;
    return (size_t)info.resident_size;
#else
    return 0;
#endif
}
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

/* Mirrors forward.c::embedding_lookup. */
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
        const int8_t  *cb_q  = (const int8_t  *)pq->cb_q;
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

/* Compute -log P(target) from logits using stable logsumexp. */
static double nll_from_logits(const float *logits, int target, int vocab) {
    float m = logits[0];
    for (int i = 1; i < vocab; i++) if (logits[i] > m) m = logits[i];
    double s = 0.0;
    for (int i = 0; i < vocab; i++) s += exp((double)(logits[i] - m));
    double lse = (double)m + log(s);
    return -((double)logits[target]) + lse;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.ibf> <tokens.i32.bin> [--backend cpu|gpu] "
                "[--warmup N] [--score N] [--ctx N]\n", argv[0]);
        return 1;
    }
    const char *backend = "cpu";
    int warmup = 64, score = 512, ctx_len = 1024;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--backend") && i+1 < argc) backend = argv[++i];
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--score") && i+1 < argc) score = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i+1 < argc) ctx_len = atoi(argv[++i]);
    }
    int use_gpu = (strcmp(backend, "gpu") == 0);

    /* Load tokens. */
    FILE *fp = fopen(argv[2], "rb");
    if (!fp) { fprintf(stderr, "open tokens failed\n"); return 2; }
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    int n_tokens = (int)(sz / 4);
    int32_t *tokens = malloc((size_t)n_tokens * sizeof(int32_t));
    fread(tokens, 4, n_tokens, fp); fclose(fp);
    int total_needed = warmup + score + 1;
    if (n_tokens < total_needed) {
        fprintf(stderr, "tokens file too small: have %d need %d\n", n_tokens, total_needed);
        return 3;
    }

    inferbit_config *cfg = inferbit_config_create();
    inferbit_config_set_context_length(cfg, ctx_len);
    inferbit_model *m = inferbit_load(argv[1], cfg);
    if (!m) { fprintf(stderr, "load failed\n"); return 4; }
    int vocab = m->header.vocab_size;
    int hidden = m->header.hidden_size;

    ib_metal_ctx *ctx = NULL;
    ib_metal_model_buffers *gbufs = NULL;
    float *embed_buf = malloc((size_t)hidden * sizeof(float));
    if (use_gpu) {
        ctx = ib_metal_create();
        if (!ctx) { fprintf(stderr, "metal init failed\n"); return 5; }
        gbufs = ib_metal_upload_model(ctx, m);
        if (!gbufs) { fprintf(stderr, "metal upload failed\n"); return 6; }
        if (getenv("IB_STRIP_MMAP")) ib_metal_strip_cpu_mmap(m);
    }

    float *logits = malloc((size_t)vocab * sizeof(float));
    inferbit_kv_clear(m);
    if (use_gpu) ib_metal_reset_kv(gbufs);

    /* Optional memory-pressure hog. IB_RSS_HOG_MB=N allocates N MB of
     * incompressible (random-looking) memory and mlocks it so macOS's
     * memory compressor can't sneak the pressure away. This forces the
     * kernel to actually evict file-backed mmap pages to make room. */
    void *hog = NULL;
    size_t hog_bytes = 0;
    {
        const char *hg = getenv("IB_RSS_HOG_MB");
        if (hg) {
            long mb = atol(hg);
            if (mb > 0) {
                hog_bytes = (size_t)mb * 1024 * 1024;
                hog = malloc(hog_bytes);
                if (hog) {
                    long ps = sysconf(_SC_PAGESIZE);
                    if (ps <= 0) ps = 4096;
                    /* Fill with a high-entropy pattern (linear-congruential)
                     * so the compressor can't shrink it. */
                    uint64_t seed = 0xdeadbeefcafebabeULL;
                    for (size_t off = 0; off < hog_bytes; off += (size_t)ps) {
                        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                        memcpy((char*)hog + off, &seed, sizeof(seed));
                    }
                    if (mlock(hog, hog_bytes) == 0) {
                        fprintf(stderr, "RSS hog allocated %.1f MB and mlocked\n",
                                hog_bytes / 1048576.0);
                    } else {
                        fprintf(stderr, "RSS hog allocated %.1f MB (mlock failed; may be compressed)\n",
                                hog_bytes / 1048576.0);
                    }
                }
            }
        }
    }

    /* Warmup phase: feed first `warmup` tokens. */
    if (use_gpu) {
        for (int i = 0; i < warmup; i++) {
            cpu_embed_lookup(m, tokens[i], embed_buf);
            ib_metal_forward_token(ctx, gbufs, embed_buf, i, logits);
        }
    } else {
        ib_forward(m, tokens, warmup, logits);
    }

    double nll_total = 0.0;
    int n_scored = 0;

    /* Score the prediction at position `warmup` (using last warmup-pass logits). */
    nll_total += nll_from_logits(logits, tokens[warmup], vocab);
    n_scored++;

    /* Stream-score subsequent positions. */
    double t0 = now_sec();
    for (int i = warmup; i < warmup + score; i++) {
        int tok = tokens[i];
        int target = tokens[i + 1];
        if (use_gpu) {
            cpu_embed_lookup(m, tok, embed_buf);
            ib_metal_forward_token(ctx, gbufs, embed_buf, i, logits);
        } else {
            int32_t one = (int32_t)tok;
            ib_forward(m, &one, 1, logits);
        }
        nll_total += nll_from_logits(logits, target, vocab);
        n_scored++;
    }
    double elapsed = now_sec() - t0;
    double ppl = exp(nll_total / n_scored);

    printf("BACKEND=%s\n", use_gpu ? "libinferbit-gpu" : "libinferbit-cpu");
    printf("MODEL=%s\n", argv[1]);
    printf("VOCAB=%d HIDDEN=%d LAYERS=%d\n", vocab, hidden, m->header.num_layers);
    printf("WARMUP=%d SCORE=%d CTX=%d\n", warmup, score, ctx_len);
    printf("PPL=%.6f\n", ppl);
    printf("N_SCORED=%d\n", n_scored);
    printf("SCORING_S=%.3f\n", elapsed);
    {
        size_t peak = peak_rss_bytes();
        size_t cur  = current_rss_bytes();
        printf("PEAK_RSS_BYTES=%zu\n", peak);
        printf("PEAK_RSS_MB=%.1f\n", peak / (1024.0 * 1024.0));
        printf("CUR_RSS_BYTES=%zu\n", cur);
        printf("CUR_RSS_MB=%.1f\n", cur / (1024.0 * 1024.0));
        printf("HOG_MB=%.1f\n", hog_bytes / (1024.0 * 1024.0));
        const char *rm = getenv("IB_RESIDENCY_MODE");
        printf("RESIDENCY_MODE=%s\n", rm ? rm : "ram");
    }
    if (hog) { munlock(hog, hog_bytes); free(hog); }
    printf("SCORE_TOK_PER_SEC=%.2f\n", (double)score / elapsed);

    free(tokens); free(logits); free(embed_buf);
    if (use_gpu) {
        ib_metal_release_model(ctx, gbufs);
        ib_metal_destroy(ctx);
    }
    inferbit_free(m);
    inferbit_config_free(cfg);
    return 0;
}
