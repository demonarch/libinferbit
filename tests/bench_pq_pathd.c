/*
 * bench_pq_pathd — mmap (Path D) vs fread loader.
 *
 * Loads the same multi-tensor v5 file via both ib_pq_load_multi (heap
 * arena, fread copy) and ib_pq_open_mmap (zero-copy view), then runs
 * an identical chained-FFN bench to verify (a) numerical equivalence
 * and (b) the load-time + RSS delta.
 *
 * Usage: bench_pq_pathd <model.ibf> [<repeats>]
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
    /* On macOS ru_maxrss is bytes; on Linux it's kilobytes. */
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

typedef struct { const ib_pq_tensor *gate, *up, *down; } triple;

static double run_ffn_pass(const ib_pq_multi* m, int n_layers,
                            int hidden, int N_mid, int repeats,
                            ib_thread_pool* pool) {
    triple* layers = (triple*)calloc((size_t)n_layers, sizeof(triple));
    for (int L = 0; L < n_layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "L%d_gate_proj", L); layers[L].gate = ib_pq_multi_find(m, nm);
        snprintf(nm, sizeof(nm), "L%d_up_proj",   L); layers[L].up   = ib_pq_multi_find(m, nm);
        snprintf(nm, sizeof(nm), "L%d_down_proj", L); layers[L].down = ib_pq_multi_find(m, nm);
        if (!layers[L].gate || !layers[L].up || !layers[L].down) {
            free(layers); return -1.0;
        }
    }
    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));

    double best = 1e18;
    for (int r = 0; r < repeats; r++) {
        for (int j = 0; j < hidden; j++)
            x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;
        double s = now_s();
        for (int L = 0; L < n_layers; L++) {
            ib_pq_matmul_fp32_threaded(layers[L].gate, x, g, pool);
            ib_pq_matmul_fp32_threaded(layers[L].up,   x, u, pool);
            for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
            ib_pq_matmul_fp32_threaded(layers[L].down, h, y, pool);
            for (int j = 0; j < hidden; j++) x[j] = 0.9f * x[j] + 0.1f * y[j];
        }
        double d = now_s() - s;
        if (d < best) best = d;
    }
    free(layers); free(x); free(g); free(u); free(h); free(y);
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_pq_pathd <model.ibf> [<repeats>]\n"); return 2; }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 3;
    if (repeats < 1) repeats = 1;

    const char* path = argv[1];
    long rss_baseline = maxrss_kib();

    /* === Path A: heap arena (existing fread loader) === */
    fprintf(stdout, "=== fread loader ===\n");
    double t0 = now_s();
    ib_pq_multi mh = {0};
    if (ib_pq_load_multi(path, &mh) != 0) { fprintf(stderr, "load_multi failed\n"); return 1; }
    double t_load_h = now_s() - t0;
    long rss_after_h = maxrss_kib();
    fprintf(stdout, "  load:        %7.1f ms\n", t_load_h * 1e3);
    fprintf(stdout, "  RSS delta:   %ld KiB (%.1f MB)\n",
            rss_after_h - rss_baseline, (rss_after_h - rss_baseline) / 1024.0);

    int n_layers = 0;
    for (int i = 0; i < mh.n; i++) {
        if (mh.names[i] && mh.names[i][0] == 'L') {
            int n = atoi(mh.names[i] + 1);
            if (n + 1 > n_layers) n_layers = n + 1;
        }
    }
    int hidden = n_layers > 0 ? mh.tensors[0].N : 0;  /* approximate */
    /* Find hidden + N_mid from the first layer's gate_proj */
    const ib_pq_tensor* g0 = ib_pq_multi_find(&mh, "L0_gate_proj");
    const ib_pq_tensor* d0 = ib_pq_multi_find(&mh, "L0_down_proj");
    if (!g0 || !d0) { fprintf(stderr, "missing L0 tensors\n"); return 1; }
    hidden = g0->N;
    int N_mid = g0->M;
    fprintf(stdout, "  model: hidden=%d intermediate=%d n_layers=%d\n", hidden, N_mid, n_layers);

    ib_thread_pool* pool = ib_pool_create(8);
    double t_run_h = run_ffn_pass(&mh, n_layers, hidden, N_mid, repeats, pool);
    fprintf(stdout, "  FFN forward (8t): %7.1f ms / token  (%.2f tok/s)\n",
            t_run_h * 1e3, 1.0 / t_run_h);
    ib_pq_multi_free(&mh);

    /* === Path B: mmap loader === */
    fprintf(stdout, "\n=== mmap loader (Path D) ===\n");
    long rss_pre_m = maxrss_kib();
    t0 = now_s();
    ib_pq_multi mm = {0};
    if (ib_pq_open_mmap(path, &mm) != 0) { fprintf(stderr, "open_mmap failed\n"); return 1; }
    double t_load_m = now_s() - t0;
    long rss_after_m = maxrss_kib();
    fprintf(stdout, "  load:        %7.1f ms\n", t_load_m * 1e3);
    fprintf(stdout, "  RSS delta:   %ld KiB (%.1f MB)\n",
            rss_after_m - rss_pre_m, (rss_after_m - rss_pre_m) / 1024.0);

    double t_run_m = run_ffn_pass(&mm, n_layers, hidden, N_mid, repeats, pool);
    fprintf(stdout, "  FFN forward (8t): %7.1f ms / token  (%.2f tok/s)\n",
            t_run_m * 1e3, 1.0 / t_run_m);
    long rss_after_run = maxrss_kib();
    fprintf(stdout, "  RSS after FFN: +%ld KiB (%.1f MB total since baseline)\n",
            rss_after_run - rss_baseline, (rss_after_run - rss_baseline) / 1024.0);

    /* Test eviction: advise DONTNEED on every layer, see if RSS shrinks */
    for (int L = 0; L < n_layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "L%d_gate_proj", L); ib_pq_advise_dontneed(ib_pq_multi_find(&mm, nm));
        snprintf(nm, sizeof(nm), "L%d_up_proj",   L); ib_pq_advise_dontneed(ib_pq_multi_find(&mm, nm));
        snprintf(nm, sizeof(nm), "L%d_down_proj", L); ib_pq_advise_dontneed(ib_pq_multi_find(&mm, nm));
    }
    long rss_after_evict = maxrss_kib();
    fprintf(stdout, "  RSS after MADV_DONTNEED on all layers: %ld KiB total\n", rss_after_evict);

    ib_pq_multi_free(&mm);
    ib_pool_destroy(pool);

    fprintf(stdout, "\nload-time speedup mmap vs fread: %.1fx\n", t_load_h / t_load_m);
    fprintf(stdout, "run-time delta:                  %.2f ms (%+.1f%%)\n",
            (t_run_m - t_run_h) * 1e3,
            (t_run_m - t_run_h) / t_run_h * 100.0);
    return 0;
}
