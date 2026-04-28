/*
 * bench_pq_dejavu — measure DejaVu sparsity speedup.
 *
 * Loads a v5 tensor (intended to be a `down_proj` shape, so x is the
 * post-SiLU FFN intermediate), runs the matmul with three input
 * profiles:
 *
 *   1. dense    — uniform random x, no chunks below threshold
 *   2. 60%-zero — matches Llama-8B post-SiLU sparsity at 1% threshold
 *   3. 80%-zero — aggressive case
 *
 * Reports wall + the auto-detected n_active. The kernel auto-applies
 * sparsity skipping; no API change.
 *
 * Usage: bench_pq_dejavu <down_proj.ibf> [<repeats>]
 */
#include "pq_decode.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void make_x(float* x, int N, float zero_frac, unsigned seed) {
    /* Fill all with a non-zero random uniform first, then knock out
     * exactly round(N * zero_frac) random positions to 0. */
    unsigned s = seed * 2654435761u + 0x9E3779B9u;
    for (int j = 0; j < N; j++) {
        s = s * 1664525u + 1013904223u;
        int v = (int)((s >> 16) & 0x3FF) - 512;
        if (v == 0) v = 1;  /* avoid accidental zero */
        x[j] = (float)v / 256.0f;
    }
    int n_zero = (int)((double)N * zero_frac + 0.5);
    /* Fisher-Yates partial shuffle to pick n_zero distinct indices */
    /* Simple Floyd's algorithm using the Knuth shuffle of a small set: */
    int* idx = (int*)malloc((size_t)N * sizeof(int));
    for (int j = 0; j < N; j++) idx[j] = j;
    for (int i = 0; i < n_zero; i++) {
        s = s * 1664525u + 1013904223u;
        int j = i + (int)(s % (unsigned)(N - i));
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        x[idx[i]] = 0.0f;
    }
    free(idx);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_pq_dejavu <in.ibf> [<repeats>]\n"); return 2; }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 5;

    ib_pq_tensor t;
    if (ib_pq_load_single(argv[1], &t) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    int M = t.M, N = t.N;
    fprintf(stdout, "tensor: M=%d N=%d K=%d n_levels=%d C=%d\n",
            M, N, t.K, t.n_levels, t.C);

    float* x = (float*)malloc((size_t)N * sizeof(float));
    float* y = (float*)malloc((size_t)M * sizeof(float));

    ib_thread_pool* pool = ib_pool_create(8);

    struct { const char* label; float zero_frac; } cases[] = {
        {"dense (0% zeros)", 0.0f},
        {"40%-zero (typical post-SiLU at high thresh)", 0.40f},
        {"60%-zero (Llama-8B post-SiLU @ 1% thresh)", 0.60f},
        {"80%-zero (aggressive)", 0.80f},
    };
    int ncases = sizeof(cases) / sizeof(cases[0]);
    for (int ci = 0; ci < ncases; ci++) {
        make_x(x, N, cases[ci].zero_frac, 42 + ci);
        /* Predict expected n_active chunks at this sparsity:
         * a chunk is active if any of its G inner cols has |x| > 1% maxabs.
         * Approximation for random distribution: 1 - zero_frac^G */
        int n_inner = N - t.n_outlier;
        double frac_active_pred = 1.0 - pow(cases[ci].zero_frac, t.G);
        int n_active_pred = (int)(t.C * frac_active_pred);
        (void)n_inner;

        double tmin = 1e9;
        for (int r = 0; r < repeats; r++) {
            double s = now_s();
            ib_pq_matmul_fp32_threaded(&t, x, y, pool);
            double d = now_s() - s;
            if (d < tmin) tmin = d;
        }
        fprintf(stdout, "  %-50s %7.3f ms  (~%d/%d chunks active)\n",
                cases[ci].label, tmin * 1e3, n_active_pred, t.C);
    }

    ib_pool_destroy(pool);
    free(x); free(y);
    ib_pq_free(&t);
    return 0;
}
