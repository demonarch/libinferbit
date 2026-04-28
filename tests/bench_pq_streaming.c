/*
 * bench_pq_streaming — cold-start inference with prefetch/evict.
 *
 * Loads a multi-tensor v5 file via mmap, then runs the SiLU-gated FFN
 * forward pass through every layer in three modes:
 *
 *   1. baseline   — no advice; OS faults pages on demand
 *   2. willneed   — issue MADV_WILLNEED for layer L+1 before computing L
 *   3. streaming  — willneed L+1 + dontneed L (the production pattern)
 *
 * Reports per-token wall + peak RSS for each. The wins show up most
 * clearly with a cold OS page cache; on macOS use `sudo purge` between
 * runs (or after a fresh boot). Linux: drop_caches.
 *
 * Usage: bench_pq_streaming <model.ibf> [<repeats>]
 */
#include "pq_decode.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static long maxrss_kib(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}
static inline float silu(float v) { return v / (1.0f + expf(-v)); }

typedef struct { const ib_pq_tensor *gate, *up, *down; } triple;

enum mode_t { MODE_BASELINE, MODE_WILLNEED, MODE_STREAMING };

static double run_pass(const triple* layers, int n_layers,
                        int hidden, int N_mid, int repeats,
                        ib_thread_pool* pool, enum mode_t mode,
                        const char* label) {
    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));
    double best = 1e18;
    long rss_pre = maxrss_kib();
    for (int r = 0; r < repeats; r++) {
        for (int j = 0; j < hidden; j++)
            x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;
        double s = now_s();
        for (int L = 0; L < n_layers; L++) {
            if (mode == MODE_WILLNEED || mode == MODE_STREAMING) {
                if (L + 1 < n_layers) {
                    const ib_pq_tensor* nxt[3] = {
                        layers[L+1].gate, layers[L+1].up, layers[L+1].down };
                    ib_pq_advise_willneed_n(nxt, 3);
                }
            }
            ib_pq_matmul_fp32_threaded(layers[L].gate, x, g, pool);
            ib_pq_matmul_fp32_threaded(layers[L].up,   x, u, pool);
            for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
            ib_pq_matmul_fp32_threaded(layers[L].down, h, y, pool);
            for (int j = 0; j < hidden; j++) x[j] = 0.9f * x[j] + 0.1f * y[j];
            if (mode == MODE_STREAMING) {
                const ib_pq_tensor* cur[3] = {
                    layers[L].gate, layers[L].up, layers[L].down };
                ib_pq_advise_dontneed_n(cur, 3);
            }
        }
        double d = now_s() - s;
        if (d < best) best = d;
    }
    long rss_post = maxrss_kib();
    fprintf(stdout, "  %-10s  %7.1f ms/tok  %.2f tok/s   RSS: pre=%ld post=%ld delta=%+ld KiB\n",
            label, best * 1e3, 1.0 / best, rss_pre, rss_post, rss_post - rss_pre);
    free(x); free(g); free(u); free(h); free(y);
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_pq_streaming <model.ibf> [<repeats>]\n");
        return 2;
    }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 3;

    ib_pq_multi m = {0};
    if (ib_pq_open_mmap(argv[1], &m) != 0) {
        fprintf(stderr, "open_mmap failed\n"); return 1;
    }
    int n_layers = 0;
    for (int i = 0; i < m.n; i++)
        if (m.names[i] && m.names[i][0] == 'L') {
            int n = atoi(m.names[i] + 1);
            if (n + 1 > n_layers) n_layers = n + 1;
        }
    triple* layers = (triple*)calloc((size_t)n_layers, sizeof(triple));
    for (int L = 0; L < n_layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "L%d_gate_proj", L); layers[L].gate = ib_pq_multi_find(&m, nm);
        snprintf(nm, sizeof(nm), "L%d_up_proj",   L); layers[L].up   = ib_pq_multi_find(&m, nm);
        snprintf(nm, sizeof(nm), "L%d_down_proj", L); layers[L].down = ib_pq_multi_find(&m, nm);
    }
    int hidden = layers[0].gate->N;
    int N_mid = layers[0].gate->M;
    fprintf(stdout, "model: hidden=%d intermediate=%d n_layers=%d\n", hidden, N_mid, n_layers);

    ib_thread_pool* pool = ib_pool_create(8);

    /* Run all three modes. The first run after open_mmap typically
     * has a cold page cache (unless the file was just used).
     * Subsequent modes run on warm cache. To benchmark cold-start
     * specifically, run each mode in its own process invocation
     * (re-launching the binary). */
    fprintf(stdout, "(first run is cold cache; subsequent are warm)\n");
    run_pass(layers, n_layers, hidden, N_mid, repeats, pool, MODE_BASELINE,  "baseline");
    run_pass(layers, n_layers, hidden, N_mid, repeats, pool, MODE_WILLNEED,  "willneed");
    run_pass(layers, n_layers, hidden, N_mid, repeats, pool, MODE_STREAMING, "streaming");

    ib_pool_destroy(pool);
    ib_pq_multi_free(&m);
    free(layers);
    return 0;
}
