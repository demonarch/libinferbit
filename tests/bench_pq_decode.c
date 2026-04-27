/*
 * bench_pq_decode — micro-bench: PQ decode + matmul costs.
 *
 * usage: bench_pq_decode <in.ibf> [<repeats>]
 *
 * Reports for the loaded PQ tensor:
 *   - load wall time
 *   - reconstruct (materialize) wall time, GB/s on output
 *   - fused PQ-matmul wall time (mean over repeats), GFLOP/s
 *   - reference: materialize-then-FP32-matmul wall time
 *   - speedup (fused / materialize+matmul)
 *
 * Caveat: scalar reference implementations only. NEON/AVX kernels are
 * separate; this just establishes the algorithmic baseline.
 */

#include "pq_decode.h"
#include "inferbit_internal.h"  /* ib_thread_pool */

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

static void scalar_matmul_fp32(const float* W, const float* x, float* out, int M, int N) {
    for (int r = 0; r < M; r++) {
        float acc = 0.0f;
        const float* Wr = W + (size_t)r * N;
        for (int j = 0; j < N; j++) acc += Wr[j] * x[j];
        out[r] = acc;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_pq_decode <in.ibf> [<repeats>]\n");
        return 2;
    }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 3;
    if (repeats < 1) repeats = 1;

    double t0 = now_s();
    ib_pq_tensor t;
    if (ib_pq_load_single(argv[1], &t) != 0) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    double t_load = now_s() - t0;

    int M = t.M, N = t.N;
    size_t MN = (size_t)M * (size_t)N;
    fprintf(stdout, "tensor: M=%d N=%d G=%d K=%d n_levels=%d n_outlier=%d\n",
            M, N, t.G, t.K, t.n_levels, t.n_outlier);
    fprintf(stdout, "  load:                          %7.3f ms\n", t_load * 1e3);

    /* Inputs / outputs */
    float* W_fp32 = (float*)malloc(MN * sizeof(float));
    float* x      = (float*)malloc((size_t)N * sizeof(float));
    float* y_fused      = (float*)malloc((size_t)M * sizeof(float));
    float* y_reference  = (float*)malloc((size_t)M * sizeof(float));
    if (!W_fp32 || !x || !y_fused || !y_reference) {
        fprintf(stderr, "oom\n"); return 1;
    }
    /* Deterministic input vector */
    for (int j = 0; j < N; j++) x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;

    /* Materialize timing */
    double t_mat_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        if (ib_pq_reconstruct_fp32(&t, W_fp32) != 0) { fprintf(stderr, "rec failed\n"); return 1; }
        double d = now_s() - s;
        if (d < t_mat_min) t_mat_min = d;
    }
    double mat_GBps = (double)(MN * 4) / 1e9 / t_mat_min;
    fprintf(stdout, "  materialize (FP32):            %7.3f ms  %6.2f GB/s\n",
            t_mat_min * 1e3, mat_GBps);

    /* Reference: materialize + scalar FP32 matmul */
    double t_ref_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        if (ib_pq_reconstruct_fp32(&t, W_fp32) != 0) return 1;
        scalar_matmul_fp32(W_fp32, x, y_reference, M, N);
        double d = now_s() - s;
        if (d < t_ref_min) t_ref_min = d;
    }
    double ref_GFLOPs = 2.0 * (double)MN / 1e9 / t_ref_min;
    fprintf(stdout, "  materialize+FP32 matmul:       %7.3f ms  %6.2f GFLOP/s\n",
            t_ref_min * 1e3, ref_GFLOPs);

    /* Fused PQ matmul (single-threaded) */
    double t_fus_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        if (ib_pq_matmul_fp32(&t, x, y_fused) != 0) { fprintf(stderr, "mm failed\n"); return 1; }
        double d = now_s() - s;
        if (d < t_fus_min) t_fus_min = d;
    }
    double fus_GFLOPs = 2.0 * (double)MN / 1e9 / t_fus_min;
    fprintf(stdout, "  fused PQ-matmul (1 thread):    %7.3f ms  %6.2f GFLOP/s\n",
            t_fus_min * 1e3, fus_GFLOPs);

    /* Threaded fused (P=4, P=8 if available) */
    int n_thread_configs[] = {4, 8};
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        int P = n_thread_configs[p_idx];
        ib_thread_pool* pool = ib_pool_create(P);
        if (!pool) continue;
        double t_th_min = 1e9;
        for (int r = 0; r < repeats; r++) {
            double s = now_s();
            if (ib_pq_matmul_fp32_threaded(&t, x, y_fused, pool) != 0) return 1;
            double d = now_s() - s;
            if (d < t_th_min) t_th_min = d;
        }
        double th_GFLOPs = 2.0 * (double)MN / 1e9 / t_th_min;
        fprintf(stdout, "  fused PQ-matmul (%d threads):   %7.3f ms  %6.2f GFLOP/s  (%.2fx vs 1t)\n",
                P, t_th_min * 1e3, th_GFLOPs, t_fus_min / t_th_min);
        ib_pool_destroy(pool);
    }

    fprintf(stdout, "  speedup fused vs materialize+matmul: %.2fx\n",
            t_ref_min / t_fus_min);

    /* Numerical sanity: fused should match reference to fp32 noise */
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int i = 0; i < M; i++) {
        double d = (double)y_fused[i] - (double)y_reference[i];
        if (d < 0) d = -d;
        if (d > max_abs) max_abs = d;
        double a = (double)y_reference[i];
        if (a < 0) a = -a;
        double r = d / (a + 1e-6);
        if (r > max_rel) max_rel = r;
    }
    fprintf(stdout, "  fused vs reference: max|d|=%.3e  max_rel=%.3e\n", max_abs, max_rel);

    free(W_fp32); free(x); free(y_fused); free(y_reference);
    ib_pq_free(&t);
    return 0;
}
