/* Validate activation-aware skip on REAL TinyLlama activations.
 *
 * Loads a PQv2 weight tensor + a fp32 activation vector dumped from
 * the actual model forward pass. Runs the K=256 skip kernel at
 * various thresholds, reports skip% and cos vs full kernel.
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

static int load_fp32_bin(const char *path, float *out, size_t n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t got = fread(out, sizeof(float), n, fp);
    fclose(fp);
    return (got == n) ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *ibf_path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    const char *tname    = (argc > 2) ? argv[2] : "L0.self_attn.q_proj";
    const char *act_path = (argc > 3) ? argv[3] : "/tmp/real_acts/L0_q_proj.bin";
    int iters = (argc > 4) ? atoi(argv[4]) : 200;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(ibf_path, &f) != 0) {
        fprintf(stderr, "load %s failed\n", ibf_path);
        return 1;
    }
    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, tname);
    if (!nt || nt->kind != IB_PQV2_KIND_PQV2) {
        fprintf(stderr, "tensor %s not found / not PQv2\n", tname);
        return 1;
    }
    const pqv2_t *t = &nt->pq;
    size_t N = t->N, M = t->M;

    float *x = aligned_alloc(64, (N * sizeof(float) + 63) & ~(size_t)63);
    if (load_fp32_bin(act_path, x, N) != 0) {
        fprintf(stderr, "load %s failed (need %zu fp32 = %zu bytes)\n",
                act_path, N, N * sizeof(float));
        return 1;
    }
    /* Compute distribution stats. */
    double xmax = 0.0, xsum = 0.0, xsumsq = 0.0;
    for (size_t i = 0; i < N; i++) {
        double v = fabs((double)x[i]);
        if (v > xmax) xmax = v;
        xsum += v;
        xsumsq += v * v;
    }
    double xmean = xsum / N;
    double xrms  = sqrt(xsumsq / N);
    printf("tensor %s: M=%u N=%u  K=%u\n", tname, t->M, t->N, t->K);
    printf("activation %s: max|x|=%.4f, mean|x|=%.5f, rms|x|=%.5f\n",
           act_path, xmax, xmean, xrms);

    float *y_base = aligned_alloc(64, (M * sizeof(float) + 63) & ~(size_t)63);
    float *y_skip = aligned_alloc(64, (M * sizeof(float) + 63) & ~(size_t)63);

    /* Baseline */
    pqv2_matvec_tbl_int8_k256(t, x, y_base);
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) pqv2_matvec_tbl_int8_k256(t, x, y_base);
    double t_base = (now_sec() - t0) / iters * 1000.0;

    printf("\n=== K=256 + skip on real %s ===\n", act_path);
    printf("  baseline:                 %8.3f ms (cos=1.00000 ref)\n", t_base);
    double thresholds[] = {0.005 * xmax, 0.01 * xmax, 0.02 * xmax, 0.05 * xmax};
    const char *names[] = {"0.5%", "1%", "2%", "5%"};
    for (int i = 0; i < 4; i++) {
        double skip_frac = 0.0;
        pqv2_matvec_tbl_int8_k256_skip(t, x, y_skip,
                                          (float)thresholds[i], &skip_frac);
        t0 = now_sec();
        for (int it = 0; it < iters; it++) {
            pqv2_matvec_tbl_int8_k256_skip(t, x, y_skip,
                                              (float)thresholds[i], NULL);
        }
        double tt = (now_sec() - t0) / iters * 1000.0;
        double cs = cosine(y_base, y_skip, M);
        printf("  thresh=%s of max:        %8.3f ms (%.2f×)  skipped=%.0f%%  cos=%.6f\n",
               names[i], tt, t_base / tt, skip_frac * 100.0, cs);
    }

    free(x); free(y_base); free(y_skip);
    ib_pqv2_file_free(&f);
    return 0;
}
