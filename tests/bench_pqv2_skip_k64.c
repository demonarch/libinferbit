/* Derisk options 2 (smaller K) + 3 (activation-aware skip), separately
 * and combined. Loads two PQv2-encoded variants of the SAME weight
 * tensor (one K=256, one K=64) and times:
 *   - baseline K=256 (B=4 sequential)
 *   - K=256 + skip
 *   - K=64 baseline
 *   - K=64 + skip
 * Reports time + cos vs baseline for each.
 *
 * Usage:
 *   ./bench_pqv2_skip_k64 <k256.ibf> <k64.ibf> [tensor] [iters]
 */
#include "../src/pqv2_format.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

static double max_abs_x(const float *x, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double v = fabs((double)x[i]);
        if (v > m) m = v;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *path256 = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    const char *path64  = (argc > 2) ? argv[2] : "/tmp/q_proj_k64.ibf";
    const char *tname   = (argc > 3) ? argv[3] : "L0.self_attn.q_proj";
    int iters = (argc > 4) ? atoi(argv[4]) : 200;

    ib_pqv2_file f256 = {0}, f64 = {0};
    if (ib_pqv2_file_load(path256, &f256) != 0) { fprintf(stderr, "load %s failed\n", path256); return 1; }
    if (ib_pqv2_file_load(path64,  &f64)  != 0) { fprintf(stderr, "load %s failed\n", path64);  return 1; }

    const ib_pqv2_named_tensor *nt256 = ib_pqv2_find(&f256, tname);
    const ib_pqv2_named_tensor *nt64  = ib_pqv2_find(&f64,  tname);
    if (!nt256 || !nt64 || nt256->kind != IB_PQV2_KIND_PQV2 || nt64->kind != IB_PQV2_KIND_PQV2) {
        fprintf(stderr, "tensor '%s' missing or wrong kind\n", tname);
        return 1;
    }
    const pqv2_t *t256 = &nt256->pq;
    const pqv2_t *t64  = &nt64->pq;
    if (t256->M != t64->M || t256->N != t64->N) {
        fprintf(stderr, "shape mismatch: K=256 %ux%u vs K=64 %ux%u\n",
                t256->M, t256->N, t64->M, t64->N);
        return 1;
    }
    printf("tensor %s: M=%u N=%u  (K=256 vs K=64)\n", tname, t256->M, t256->N);

    size_t N = t256->N, M = t256->M;
    float *x = aligned_alloc(64, ((size_t)4 * N * sizeof(float) + 63) & ~(size_t)63);
    float *y_base = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    float *y_skip = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    float *y_k64  = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    float *y_k64s = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);

    /* Realistic activation distribution: most channels small (typical
     * RMSNorm output), with sparse outlier channels (well-documented
     * in AWQ/GPTQ literature). About 5% of channels are outliers
     * with ~50× the magnitude of the rest. */
    unsigned int seed = 42;
    for (size_t i = 0; i < (size_t)4 * N; i++) {
        seed = seed * 1103515245 + 12345;
        int v = (int)((seed >> 16) & 0x7fff) - 16384;
        float base = (float)v * (1.0f / 16384.0f);   /* uniform [-1, 1] */
        /* Convert to gaussian-ish via Box-Muller approx: cube. */
        base = base * base * base;
        /* 5% outlier channels carry 50× larger magnitude. */
        size_t channel = i % N;
        if ((channel * 2654435761u) % 100 < 5) {
            x[i] = base * 50.0f;
        } else {
            x[i] = base * 1.0f;
        }
    }
    double xmax = max_abs_x(x, (size_t)4 * N);
    printf("input x: max|x| = %.4f, suggested skip thresholds 0.5%% / 1%% / 2%% of max\n", xmax);

    /* Baseline: K=256 sequential B=4 */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8_k256(t256, x + b*N, y_base + b*M);
    }
    double t_base = (now_sec() - t0) / iters * 1000.0;

    /* K=64 baseline (no skip) */
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8(t64, x + b*N, y_k64 + b*M);
    }
    double t_k64 = (now_sec() - t0) / iters * 1000.0;
    double cos_k64 = cosine(y_base, y_k64, (size_t)4 * M);

    /* For each skip threshold: time and quality */
    double thresholds[] = {0.005 * xmax, 0.01 * xmax, 0.02 * xmax, 0.05 * xmax};
    const char *names[] = {"0.5%", "1%", "2%", "5%"};

    printf("\n=== Baselines ===\n");
    printf("  K=256 sequential B=4:  %8.3f ms/call\n", t_base);
    printf("  K=64 sequential B=4:   %8.3f ms/call  ratio=%.3f  cos=%.5f\n",
           t_k64, t_k64 / t_base, cos_k64);

    printf("\n=== K=256 + skip ===\n");
    for (int i = 0; i < 4; i++) {
        double skip_frac = 0.0;
        /* warmup once to capture skip_frac */
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8_k256_skip(t256, x + b*N, y_skip + b*M,
                                             (float)thresholds[i], &skip_frac);
        t0 = now_sec();
        for (int it = 0; it < iters; it++) {
            for (int b = 0; b < 4; b++)
                pqv2_matvec_tbl_int8_k256_skip(t256, x + b*N, y_skip + b*M,
                                                 (float)thresholds[i], NULL);
        }
        double tt = (now_sec() - t0) / iters * 1000.0;
        double cs = cosine(y_base, y_skip, (size_t)4 * M);
        printf("  thresh=%s of max:  %8.3f ms (ratio %.3f)  skipped=%.0f%%  cos=%.5f\n",
               names[i], tt, tt / t_base, skip_frac * 100.0, cs);
    }

    printf("\n=== K=64 + skip ===\n");
    for (int i = 0; i < 4; i++) {
        double skip_frac = 0.0;
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8_skip(t64, x + b*N, y_k64s + b*M,
                                        (float)thresholds[i], &skip_frac);
        t0 = now_sec();
        for (int it = 0; it < iters; it++) {
            for (int b = 0; b < 4; b++)
                pqv2_matvec_tbl_int8_skip(t64, x + b*N, y_k64s + b*M,
                                            (float)thresholds[i], NULL);
        }
        double tt = (now_sec() - t0) / iters * 1000.0;
        double cs = cosine(y_base, y_k64s, (size_t)4 * M);
        printf("  thresh=%s of max:  %8.3f ms (ratio %.3f)  skipped=%.0f%%  cos=%.5f\n",
               names[i], tt, tt / t_base, skip_frac * 100.0, cs);
    }

    free(x); free(y_base); free(y_skip); free(y_k64); free(y_k64s);
    ib_pqv2_file_free(&f256); ib_pqv2_file_free(&f64);
    return 0;
}
