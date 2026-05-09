/* Derisk bench for option B: GEMM-style B=4 batched K=256 matvec.
 *
 * Loads a real PQv2 tensor (e.g., L0.q_proj from the IBF v6 file),
 * times two implementations on B=4 random inputs:
 *   1. Sequential: 4× pqv2_matvec_tbl_int8_k256
 *   2. GEMM-B4:    1× pqv2_matvec_tbl_int8_k256_gemm_b4
 * Reports cycles per matvec and ratio.
 *
 * Decision rule:
 *   ratio (gemm/seq) < 0.7  → option B viable, push spec
 *   ratio >= 0.85            → option B doesn't help, abandon spec lever
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

static double max_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    const char *tname = (argc > 2) ? argv[2] : "L0.self_attn.q_proj";
    int iters = (argc > 3) ? atoi(argv[3]) : 200;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(path, &f) != 0) {
        fprintf(stderr, "load %s failed\n", path);
        return 1;
    }
    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, tname);
    if (!nt || nt->kind != IB_PQV2_KIND_PQV2) {
        fprintf(stderr, "tensor '%s' not found or not PQv2\n", tname);
        return 1;
    }
    const pqv2_t *t = &nt->pq;
    printf("tensor %s: M=%u N=%u K=%u half=%u, n_chunks=%u\n",
           tname, t->M, t->N, t->K, t->half, t->N / t->G);

    size_t N = t->N, M = t->M;
    float *x = aligned_alloc(64, ((size_t)4 * N * sizeof(float) + 63) & ~(size_t)63);
    float *y_seq  = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    float *y_gemm = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    /* Random inputs (small magnitude). */
    unsigned int seed = 42;
    for (size_t i = 0; i < (size_t)4 * N; i++) {
        seed = seed * 1103515245 + 12345;
        int v = (int)((seed >> 16) & 0x7fff) - 16384;
        x[i] = (float)v * 0.0001f;
    }

    float *y_fp16  = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);

    /* Warmup */
    for (int b = 0; b < 4; b++)
        pqv2_matvec_tbl_int8_k256(t, x + (size_t)b * N, y_seq + (size_t)b * M);
    pqv2_matvec_tbl_int8_k256_gemm_b4(t, x, y_gemm);
    for (int b = 0; b < 4; b++)
        pqv2_matvec_tbl_int8_k256_fp16acc(t, x + (size_t)b * N, y_fp16 + (size_t)b * M);

    /* Correctness — fp16acc is approximate, gemm is exact */
    double mx_gemm = max_diff(y_seq, y_gemm, (size_t)4 * M);
    double mx_fp16 = max_diff(y_seq, y_fp16, (size_t)4 * M);
    /* Per-call cosine for fp16 quality check (use first position). */
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < M; i++) {
        dot += (double)y_seq[i] * (double)y_fp16[i];
        na  += (double)y_seq[i] * (double)y_seq[i];
        nb  += (double)y_fp16[i] * (double)y_fp16[i];
    }
    double cos_fp16 = dot / (sqrt(na) * sqrt(nb) + 1e-30);
    printf("correctness:\n");
    printf("  seq vs gemm  : max abs diff = %.6e\n", mx_gemm);
    printf("  seq vs fp16  : max abs diff = %.6e   cos = %.6f\n", mx_fp16, cos_fp16);

    /* Time sequential fp32 acc */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8_k256(t, x + (size_t)b * N, y_seq + (size_t)b * M);
    }
    double t_seq = now_sec() - t0;

    /* Time GEMM-B4 */
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        pqv2_matvec_tbl_int8_k256_gemm_b4(t, x, y_gemm);
    }
    double t_gemm = now_sec() - t0;

    /* Time fp16-acc B=4 (4 sequential calls with fp16 acc) */
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        for (int b = 0; b < 4; b++)
            pqv2_matvec_tbl_int8_k256_fp16acc(t, x + (size_t)b * N, y_fp16 + (size_t)b * M);
    }
    double t_fp16 = now_sec() - t0;

    double ms_per_seq  = 1000.0 * t_seq  / iters;
    double ms_per_gemm = 1000.0 * t_gemm / iters;
    double ms_per_fp16 = 1000.0 * t_fp16 / iters;
    printf("\n=== %d iters ===\n", iters);
    printf("  seq B=4 (4× single, fp32 acc):   %8.3f ms/call\n", ms_per_seq);
    printf("  gemm B=4 (row-tile, fp32 acc):   %8.3f ms/call  ratio=%.3f\n",
           ms_per_gemm, ms_per_gemm / ms_per_seq);
    printf("  fp16 acc B=4 (4× single, fp16):  %8.3f ms/call  ratio=%.3f\n",
           ms_per_fp16, ms_per_fp16 / ms_per_seq);
    if (ms_per_fp16 < 0.70 * ms_per_seq && cos_fp16 > 0.99) {
        printf("  VERDICT: fp16 acc viable — %.2f× faster, cos %.4f\n",
                ms_per_seq / ms_per_fp16, cos_fp16);
    } else if (ms_per_fp16 < 0.85 * ms_per_seq && cos_fp16 > 0.99) {
        printf("  VERDICT: fp16 acc marginal but quality-acceptable\n");
    } else {
        printf("  VERDICT: fp16 acc not a clear win\n");
    }

    free(x); free(y_seq); free(y_gemm); free(y_fp16);
    ib_pqv2_file_free(&f);
    return 0;
}
