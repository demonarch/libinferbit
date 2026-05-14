/* Phase 6 architectural validation:
 *
 * Run a synthetic mini-pipeline N times — [rmsnorm → matmul_w4a8_fp32_in
 * → residual_add] — once per "layer". Compare two GPU strategies:
 *
 *   (a) UNBATCHED: each kernel goes in its own command buffer with its
 *       own commit/wait. Pays per-call dispatch overhead 3*N times.
 *   (b) BATCHED:   the entire N-layer pipeline records into ONE command
 *       buffer, paying dispatch overhead exactly ONCE.
 *
 * Also benchmark CPU equivalent for reference.
 *
 * If the architectural thesis is correct, batched should massively beat
 * unbatched, and may also beat CPU at TinyLlama-class shapes once enough
 * work is queued behind the single commit.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "metal/metal_runtime.h"
#include "inferbit_internal.h"

extern float ib_fp16_to_fp32(uint16_t h);

static uint16_t f32_to_fp16(float f) {
    uint32_t b; memcpy(&b, &f, 4);
    uint16_t s = (uint16_t)((b >> 16) & 0x8000);
    int32_t  e = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t m = b & 0x7FFFFF;
    if (e <= 0) return s;
    if (e >= 31) return s | 0x7C00;
    return s | (uint16_t)(e << 10) | (uint16_t)(m >> 13);
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int hidden = (argc > 1) ? atoi(argv[1]) : 2048;
    int n_layers = (argc > 2) ? atoi(argv[2]) : 22;  /* TinyLlama */
    int iters = (argc > 3) ? atoi(argv[3]) : 50;

    ib_init_kernels(ib_detect_simd());
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { fprintf(stderr, "Metal not available\n"); return 1; }
    printf("Device: %s   hidden=%d  n_layers=%d  iters=%d\n",
           ib_metal_device_name(ctx), hidden, n_layers, iters);

    /* Synthetic per-layer state.
     * State buffers (reused across all layers): x, xb, scratch_xq, scratch_xs.
     * Per-layer weights: norm_w (fp16), proj_w (uint8 [hidden, hidden/2]),
     * proj_s (fp16 [hidden]). Use the SAME weights for every layer to keep
     * the bench compact (still pays the matmul cost per layer). */
    int n_groups = (hidden + 127) / 128;

    /* Allocate weights (one set, reused across layers). */
    size_t w_bytes  = (size_t)hidden * (hidden / 2);
    size_t ws_bytes = (size_t)hidden * sizeof(uint16_t);
    size_t nw_bytes = (size_t)hidden * sizeof(uint16_t);

    uint8_t  *h_w  = malloc(w_bytes);
    uint16_t *h_ws = malloc(ws_bytes);
    uint16_t *h_nw = malloc(nw_bytes);
    float    *h_x  = malloc((size_t)hidden * sizeof(float));
    srand(0xBADF00D);
    for (size_t i = 0; i < w_bytes; i++)  h_w[i]  = (uint8_t)(rand() & 0xFF);
    for (int i = 0; i < hidden; i++)      h_ws[i] = f32_to_fp16(0.01f + ((rand() & 0xFF) / 255.0f) * 0.05f);
    for (int i = 0; i < hidden; i++)      h_nw[i] = f32_to_fp16(0.5f + ((rand() & 0xFF) / 255.0f));
    for (int i = 0; i < hidden; i++)      h_x[i]  = ((rand() & 0xFFFF) / 32767.0f - 0.5f) * 2.0f;

    /* GPU buffers */
    void *g_x   = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), h_x);
    void *g_xb  = ib_metal_alloc(ctx, (size_t)hidden * sizeof(float), NULL);
    void *g_xq  = ib_metal_alloc(ctx, (size_t)hidden, NULL);
    void *g_xs  = ib_metal_alloc(ctx, (size_t)n_groups * sizeof(float), NULL);
    void *g_w   = ib_metal_alloc(ctx, w_bytes,  h_w);
    void *g_ws  = ib_metal_alloc(ctx, ws_bytes, h_ws);
    void *g_nw  = ib_metal_alloc(ctx, nw_bytes, h_nw);

    /* Reset x to a known starting state each iteration so timings aren't
     * polluted by NaNs from runaway growth across many residual adds. */
    #define reset_x() memcpy(g_x, h_x, (size_t)hidden * sizeof(float))

    /* === UNBATCHED === */
    /* warmup */
    for (int w = 0; w < 3; w++) {
        reset_x();
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rmsnorm_fp16(ctx, g_x, g_nw, g_xb, hidden, 1e-5f);
            ib_metal_matmul_w4a8_fp32_in(ctx, g_xb, g_w, g_ws, g_xb, g_xq, g_xs, hidden, hidden);
            ib_metal_residual_add(ctx, g_x, g_xb, hidden);
        }
    }
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        reset_x();
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rmsnorm_fp16(ctx, g_x, g_nw, g_xb, hidden, 1e-5f);
            ib_metal_matmul_w4a8_fp32_in(ctx, g_xb, g_w, g_ws, g_xb, g_xq, g_xs, hidden, hidden);
            ib_metal_residual_add(ctx, g_x, g_xb, hidden);
        }
    }
    double t_unbatched = (now_sec() - t0) / iters * 1000.0;

    /* === BATCHED === */
    for (int w = 0; w < 3; w++) {
        reset_x();
        ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rec_rmsnorm_fp16(r, g_x, g_nw, g_xb, hidden, 1e-5f);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, g_w, g_ws, g_xb, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_residual_add(r, g_x, g_xb, hidden);
        }
        ib_metal_recorder_commit(r);
    }
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        reset_x();
        ib_metal_recorder *r = ib_metal_recorder_begin(ctx);
        for (int L = 0; L < n_layers; L++) {
            ib_metal_rec_rmsnorm_fp16(r, g_x, g_nw, g_xb, hidden, 1e-5f);
            ib_metal_rec_matmul_w4a8_fp32_in(r, g_xb, g_w, g_ws, g_xb, g_xq, g_xs, hidden, hidden);
            ib_metal_rec_residual_add(r, g_x, g_xb, hidden);
        }
        ib_metal_recorder_commit(r);
    }
    double t_batched = (now_sec() - t0) / iters * 1000.0;

    /* === CPU equivalent (using ib_kern.* + ib_quantize_input_int8_g128). === */
    float *cpu_x  = malloc((size_t)hidden * sizeof(float));
    float *cpu_xb = malloc((size_t)hidden * sizeof(float));
    float *cpu_nw = malloc((size_t)hidden * sizeof(float));
    float *cpu_ws = malloc((size_t)hidden * sizeof(float));
    int8_t *cpu_xq = malloc((size_t)hidden);
    float  *cpu_xs = malloc((size_t)n_groups * sizeof(float));
    for (int i = 0; i < hidden; i++) cpu_nw[i] = ib_fp16_to_fp32(h_nw[i]);
    for (int i = 0; i < hidden; i++) cpu_ws[i] = ib_fp16_to_fp32(h_ws[i]);
    for (int w = 0; w < 3; w++) {
        memcpy(cpu_x, h_x, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            ib_kern.rmsnorm(cpu_xb, cpu_x, cpu_nw, 1e-5f, hidden);
            ib_quantize_input_int8_g128(cpu_xb, cpu_xq, cpu_xs, hidden);
            ib_kern.matmul_w4a8(cpu_xb, h_w, cpu_ws, cpu_xq, cpu_xs, hidden, hidden);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
        }
    }
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memcpy(cpu_x, h_x, (size_t)hidden * sizeof(float));
        for (int L = 0; L < n_layers; L++) {
            ib_kern.rmsnorm(cpu_xb, cpu_x, cpu_nw, 1e-5f, hidden);
            ib_quantize_input_int8_g128(cpu_xb, cpu_xq, cpu_xs, hidden);
            ib_kern.matmul_w4a8(cpu_xb, h_w, cpu_ws, cpu_xq, cpu_xs, hidden, hidden);
            for (int i = 0; i < hidden; i++) cpu_x[i] += cpu_xb[i];
        }
    }
    double t_cpu = (now_sec() - t0) / iters * 1000.0;

    printf("\n=== synthetic mini-pipeline (rmsnorm → matmul_w4a8 → residual) ===\n");
    printf("  per-iter: %d layers × 3 ops = %d total kernel dispatches\n",
           n_layers, n_layers * 3);
    printf("  CPU full pipeline:    %8.3f ms/iter\n", t_cpu);
    printf("  GPU unbatched:        %8.3f ms/iter   (%d separate cb commits)\n",
           t_unbatched, n_layers * 3);
    printf("  GPU batched (1 cb):   %8.3f ms/iter   (1 commit)\n", t_batched);
    printf("  Batched / unbatched:  %.2f×%s\n", t_unbatched / t_batched,
           t_batched < t_unbatched ? "  faster (dispatch amortized)" : "");
    printf("  Batched vs CPU:       %.2f×%s\n", t_cpu / t_batched,
           t_batched < t_cpu ? "  GPU faster" : "  CPU faster");

    free(h_w); free(h_ws); free(h_nw); free(h_x);
    free(cpu_x); free(cpu_xb); free(cpu_nw); free(cpu_ws); free(cpu_xq); free(cpu_xs);
    ib_metal_free(ctx, g_x);
    ib_metal_free(ctx, g_xb);
    ib_metal_free(ctx, g_xq);
    ib_metal_free(ctx, g_xs);
    ib_metal_free(ctx, g_w);
    ib_metal_free(ctx, g_ws);
    ib_metal_free(ctx, g_nw);
    ib_metal_destroy(ctx);
    return 0;
}
