/* Phase 7 end-to-end bench: load a real IBF, run N forward passes on
 * both CPU and GPU, compare logits + tok/s.
 *
 * Usage: bench_metal_real_model <model.ibf> [n_tokens] [start_token]
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

/* Mirror the CPU embedding decode (forward.c::embedding_lookup) so we
 * can hand the GPU forward an identical fp32 starting state. */
static void cpu_embed_lookup(const inferbit_model *m, int token, float *out) {
    int hidden = m->header.hidden_size;
    const ib_tensor_meta *e = &m->token_embedding;
    const uint8_t *base = (const uint8_t*)m->weight_data;
    const uint8_t *data_b = base + e->offset;
    const void *scales_raw = e->scale_size ? (const void*)(base + e->scale_offset) : NULL;

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
        fprintf(stderr, "usage: %s <model.ibf> [n_tokens=20] [start_token=1]\n", argv[0]);
        return 1;
    }
    int n_tokens = (argc > 2) ? atoi(argv[2]) : 20;
    int start_tok = (argc > 3) ? atoi(argv[3]) : 1;

    /* Load model on CPU. */
    inferbit_config *cfg = inferbit_config_create();
    inferbit_model *m = inferbit_load(argv[1], cfg);
    if (!m) { fprintf(stderr, "load failed\n"); return 2; }
    int hidden = m->header.hidden_size;
    int vocab  = m->header.vocab_size;
    printf("Loaded: %s\n", argv[1]);
    printf("  arch=%s  layers=%d  hidden=%d  intermediate=%d  vocab=%d\n",
           m->header.architecture, m->header.num_layers, hidden,
           m->header.intermediate_size, vocab);
    printf("  n_heads=%d  n_kv=%d  head_dim=%d  kv_bits=%d\n",
           m->header.num_heads, m->header.num_kv_heads, m->header.head_dim,
           m->header.kv_bits);

    /* Create Metal context + upload model. */
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 3; }
    printf("  GPU: %s\n", ib_metal_device_name(ctx));
    ib_metal_model_buffers *gbufs = ib_metal_upload_model(ctx, m);
    if (!gbufs) { fprintf(stderr, "GPU upload failed\n"); return 4; }
    printf("  GPU buffers uploaded.\n");

    /* Token sequence: linear walk through token ids 1..n_tokens, just to
     * exercise different positions. Real benchmarks would use real
     * generation; here we're measuring per-token compute, not tokenizer. */
    int *tokens = malloc((size_t)n_tokens * sizeof(int));
    for (int i = 0; i < n_tokens; i++) tokens[i] = (start_tok + i) % vocab;

    /* CPU forward via the public API (one-shot per token). */
    float *cpu_logits_last = malloc((size_t)vocab * sizeof(float));
    float *gpu_logits_last = malloc((size_t)vocab * sizeof(float));
    float *embed_buf = malloc((size_t)hidden * sizeof(float));

    /* warmup */
    for (int i = 0; i < 3; i++) {
        inferbit_kv_clear(m);
        ib_forward(m, tokens, n_tokens, cpu_logits_last);
    }

    double t0 = now_sec();
    int CPU_RUNS = 3;
    for (int r = 0; r < CPU_RUNS; r++) {
        inferbit_kv_clear(m);
        ib_forward(m, tokens, n_tokens, cpu_logits_last);
    }
    double t_cpu_total = (now_sec() - t0) / CPU_RUNS;

    /* GPU forward: per-token loop calling ib_metal_forward_token. */
    /* warmup */
    for (int w = 0; w < 2; w++) {
        ib_metal_reset_kv(gbufs);
        for (int i = 0; i < n_tokens; i++) {
            cpu_embed_lookup(m, tokens[i], embed_buf);
            ib_metal_forward_token(ctx, gbufs, embed_buf, i, gpu_logits_last);
        }
    }

    int GPU_RUNS = 5;
    t0 = now_sec();
    for (int r = 0; r < GPU_RUNS; r++) {
        ib_metal_reset_kv(gbufs);
        for (int i = 0; i < n_tokens; i++) {
            cpu_embed_lookup(m, tokens[i], embed_buf);
            ib_metal_forward_token(ctx, gbufs, embed_buf, i, gpu_logits_last);
        }
    }
    double t_gpu_total = (now_sec() - t0) / GPU_RUNS;

    /* Compare last-token logits. */
    double dot=0, na=0, nb=0, max_diff=0;
    for (int i = 0; i < vocab; i++) {
        double a = cpu_logits_last[i], bv = gpu_logits_last[i];
        dot += a*bv; na += a*a; nb += bv*bv;
        double d = fabs(a - bv); if (d > max_diff) max_diff = d;
    }
    double cos = dot / (sqrt(na) * sqrt(nb));

    /* Top-token agreement: argmax of CPU vs GPU. */
    int top_cpu = 0, top_gpu = 0;
    for (int i = 1; i < vocab; i++) {
        if (cpu_logits_last[i] > cpu_logits_last[top_cpu]) top_cpu = i;
        if (gpu_logits_last[i] > gpu_logits_last[top_gpu]) top_gpu = i;
    }
    /* Top-5 overlap. */
    /* (skip; for quick sanity we check argmax only) */

    printf("\n=== Phase 7: end-to-end forward, %d tokens ===\n", n_tokens);
    printf("  cos(CPU logits, GPU logits) = %.6f   max|diff|=%.4e\n", cos, max_diff);
    printf("  argmax — CPU=%d  GPU=%d  %s\n", top_cpu, top_gpu,
           top_cpu == top_gpu ? "MATCH" : "DIFFER");
    printf("  CPU full sweep:  %7.1f ms  →  %.1f tok/s\n",
           t_cpu_total * 1000.0, (double)n_tokens / t_cpu_total);
    printf("  GPU full sweep:  %7.1f ms  →  %.1f tok/s\n",
           t_gpu_total * 1000.0, (double)n_tokens / t_gpu_total);
    printf("  Speedup:           %.2f×%s\n", t_cpu_total / t_gpu_total,
           t_gpu_total < t_cpu_total ? "  GPU faster" : "  CPU faster");

    free(tokens); free(cpu_logits_last); free(gpu_logits_last); free(embed_buf);
    ib_metal_release_model(ctx, gbufs);
    ib_metal_destroy(ctx);
    inferbit_free(m);
    inferbit_config_free(cfg);
    return (cos < 0.99) ? 5 : 0;
}
