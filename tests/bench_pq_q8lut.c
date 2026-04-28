/*
 * bench_pq_q8lut — fp32 LUT vs byte-quantised LUT.
 *
 * Verifies numerical match (within ~1e-3 frob-rel per spec) and
 * compares wall time at 1 / 4 / 8 threads on the same v5 file.
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

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_pq_q8lut <in.ibf> [<repeats>]\n"); return 2; }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 5;

    ib_pq_tensor t;
    if (ib_pq_load_single(argv[1], &t) != 0) { fprintf(stderr, "load failed\n"); return 1; }

    int M = t.M, N = t.N;
    fprintf(stdout, "tensor: M=%d N=%d K=%d n_levels=%d n_outlier=%d\n",
            M, N, t.K, t.n_levels, t.n_outlier);

    float* x = (float*)malloc((size_t)N * sizeof(float));
    float* y_fp = (float*)malloc((size_t)M * sizeof(float));
    float* y_q8 = (float*)malloc((size_t)M * sizeof(float));
    for (int j = 0; j < N; j++) x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;

    /* fp32 LUT path */
    double tf_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        ib_pq_matmul_fp32(&t, x, y_fp);
        double d = now_s() - s;
        if (d < tf_min) tf_min = d;
    }
    fprintf(stdout, "  fp32 LUT (1t):  %7.3f ms\n", tf_min * 1e3);

    /* q8 LUT path — single thread */
    double tq_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        ib_pq_matmul_fp32_q8lut(&t, x, y_q8, NULL);
        double d = now_s() - s;
        if (d < tq_min) tq_min = d;
    }
    fprintf(stdout, "  q8   LUT (1t):  %7.3f ms  (%.2fx vs fp32)\n",
            tq_min * 1e3, tf_min / tq_min);

    /* Threaded comparison */
    int n_threads_configs[] = {4, 8};
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        int P = n_threads_configs[p_idx];
        ib_thread_pool* pool = ib_pool_create(P);
        if (!pool) continue;
        double tf_t = 1e9, tq_t = 1e9;
        for (int r = 0; r < repeats; r++) {
            double s = now_s();
            ib_pq_matmul_fp32_threaded(&t, x, y_fp, pool);
            double d = now_s() - s;
            if (d < tf_t) tf_t = d;
        }
        for (int r = 0; r < repeats; r++) {
            double s = now_s();
            ib_pq_matmul_fp32_q8lut(&t, x, y_q8, pool);
            double d = now_s() - s;
            if (d < tq_t) tq_t = d;
        }
        fprintf(stdout, "  fp32 LUT (%dt): %7.3f ms\n", P, tf_t * 1e3);
        fprintf(stdout, "  q8   LUT (%dt): %7.3f ms  (%.2fx vs fp32 %dt)\n",
                P, tq_t * 1e3, tf_t / tq_t, P);
        ib_pool_destroy(pool);
    }

    /* Numerical check */
    double abs_max = 0, sq_sum = 0, ref_sq = 0;
    for (int i = 0; i < M; i++) {
        double d = (double)y_fp[i] - (double)y_q8[i];
        if (d < 0) d = -d;
        if (d > abs_max) abs_max = d;
        sq_sum += d * d;
        ref_sq += (double)y_fp[i] * (double)y_fp[i];
    }
    double frob_rel = (ref_sq > 0) ? sqrt(sq_sum) / sqrt(ref_sq) : 0;
    fprintf(stdout, "  numerical: max|d|=%.3e  frob_rel=%.3e\n", abs_max, frob_rel);

    free(x); free(y_fp); free(y_q8);
    ib_pq_free(&t);
    return 0;
}
