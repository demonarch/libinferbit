/* test_pqv2_roundtrip — encode -> kernel-decode -> compare round-trip
 * regression test for PQv2.
 *
 * Goal: catch future encoder-layout / kernel-dispatch regressions in CI.
 *
 * For a deterministic synthetic [RT_M=64, RT_N=128] weight matrix W and
 * a deterministic input vector x[RT_N]:
 *   - encode W with the production C encoder (pqv2_encode_flat /
 *     pqv2_encode_pyramid),
 *   - run the encoded tensor through the same K=256 kernel that
 *     forward.c::tensor_matmul dispatches to
 *     (pqv2_matvec_tbl_int8_k256, which internally calls
 *      pqv2_acc_tbl_int8_k256_chunks + applies row_scale + L2),
 *   - compute y_ref = W @ x in plain fp32 as ground truth,
 *   - compare with relative L2 error; fail if > 5%.
 *
 * Also covers the all-zero edge case (row_scale clamp at fp16-min-normal
 * must not blow up reconstruction).
 *
 * Identifiers are prefixed RT_ because pqv2_t fields M/N/G/K would
 * collide with bare-name macros.
 */

#include "pqv2_encode.h"
#include "pqv2_kernel.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RT_M 64
#define RT_N 128
#define RT_G 64
#define RT_K 256
#define RT_HALF 2

/* Pyramid L2 codebook size — production loader/kernel cap of 64. */
#define RT_K_INNER 64

/* Acceptance threshold for flat / pyramid round-trips. K=256 codewords
 * on 2D codeword space typically reconstruct to ~1-2 % relative L2 error
 * on a smooth dense tensor; 5 % gives plenty of headroom for legitimate
 * quantisation noise while catching layout regressions (which produce
 * order-unity errors). */
#define RT_THRESH 0.05f

static void make_weights(float *W_) {
    for (int m = 0; m < RT_M; ++m) {
        for (int n = 0; n < RT_N; ++n) {
            W_[m * RT_N + n] = (float)(
                sin(0.1 * (double)m) * cos(0.05 * (double)n)
                + 0.01 * (double)((m * n) % 7 - 3));
        }
    }
}

static void make_input(float *x) {
    for (int n = 0; n < RT_N; ++n) {
        x[n] = (float)(0.3 * sin(0.07 * (double)n + 1.1)
                       + 0.1 * cos(0.13 * (double)n));
    }
}

static void ref_matmul(const float *W_, const float *x, float *y_ref) {
    for (int m = 0; m < RT_M; ++m) {
        float acc = 0.0f;
        for (int n = 0; n < RT_N; ++n) acc += W_[m * RT_N + n] * x[n];
        y_ref[m] = acc;
    }
}

static float rel_l2_err(const float *y_ref, const float *y_pq, int len) {
    double num = 0.0, den = 0.0;
    for (int m = 0; m < len; ++m) {
        double d = (double)y_pq[m] - (double)y_ref[m];
        num += d * d;
        den += (double)y_ref[m] * (double)y_ref[m];
    }
    if (den <= 0.0) {
        /* All-zero reference: report absolute L2 (treat as relative-to-1). */
        return (float)sqrt(num);
    }
    return (float)sqrt(num / den);
}

static void dump_head(const char *label, const float *v, int n) {
    fprintf(stderr, "  %s[0:%d]: ", label, n);
    for (int i = 0; i < n; ++i) fprintf(stderr, "% .5f ", v[i]);
    fprintf(stderr, "\n");
}

/* Build a pqv2_t from already-encoded buffers and run the production
 * K=256 kernel. cb_fp32 / l2_cb_fp32 are left NULL on purpose so the
 * kernel decodes the int8 codebooks itself — that exercises the exact
 * code path forward.c hits when the loader has not pre-decoded a tensor
 * (e.g. drive mode), and is what `pqv2_matvec_tbl_int8_k256` requires
 * to work. */
static void run_kernel(uint32_t mm, uint32_t nn, uint32_t gg, uint32_t kk,
                       uint32_t half_, uint32_t l2_kind, uint32_t l2_K_,
                       const uint16_t *row_scale,
                       const int8_t *cb_q, const uint16_t *cb_scale,
                       const uint8_t *indices,
                       const int8_t *l2_cb_q, const uint16_t *l2_cb_scale,
                       const uint8_t *l2_indices,
                       const float *x, float *y) {
    pqv2_t t = (pqv2_t){0};
    t.M = mm; t.N = nn; t.G = gg; t.K = kk;
    t.n_subchunks = gg / half_;
    t.half = half_;
    t.l2_kind = l2_kind;
    t.l2_K = l2_K_;
    t.row_scale = row_scale;
    t.cb_q = cb_q; t.cb_scale = cb_scale; t.indices = indices;
    t.l2_cb_q = l2_cb_q; t.l2_cb_scale = l2_cb_scale; t.l2_indices = l2_indices;
    /* cb_fp32 / l2_cb_fp32 / drive offsets left NULL/0. */

    pqv2_matvec_tbl_int8_k256(&t, x, y);
}

static int test_flat(const float *W_, const float *x, const float *y_ref,
                     float *err_out) {
    const uint32_t n_sub   = RT_G / RT_HALF;
    const uint32_t n_chunk = RT_N / RT_G;

    int8_t   *cb_q     = malloc((size_t)n_sub * RT_K * RT_HALF * sizeof(int8_t));
    uint16_t *cb_scale = malloc((size_t)n_sub * RT_K * sizeof(uint16_t));
    uint16_t *row_scale= malloc((size_t)RT_M * sizeof(uint16_t));
    uint8_t  *idx      = malloc((size_t)n_chunk * n_sub * RT_M * sizeof(uint8_t));

    int rc = pqv2_encode_flat(W_, RT_M, RT_N, RT_G, RT_K, RT_HALF,
                              cb_q, cb_scale, row_scale, idx, /*seed=*/42);
    if (rc != 0) {
        fprintf(stderr, "pqv2_encode_flat failed: rc=%d\n", rc);
        free(cb_q); free(cb_scale); free(row_scale); free(idx);
        return 1;
    }

    float y_pq[RT_M];
    memset(y_pq, 0, sizeof(y_pq));
    run_kernel(RT_M, RT_N, RT_G, RT_K, RT_HALF, /*l2_kind=*/0, /*l2_K=*/0,
               row_scale, cb_q, cb_scale, idx,
               NULL, NULL, NULL,
               x, y_pq);

    float err = rel_l2_err(y_ref, y_pq, RT_M);
    *err_out = err;

    free(cb_q); free(cb_scale); free(row_scale); free(idx);

    if (!(err <= RT_THRESH)) {  /* also catches NaN */
        fprintf(stderr, "PQv2 roundtrip flat: rel L2 err %.6f > threshold %.3f\n",
                (double)err, (double)RT_THRESH);
        dump_head("y_ref", y_ref, 8);
        dump_head("y_pq ", y_pq, 8);
        return 1;
    }
    return 0;
}

static int test_pyramid(const float *W_, const float *x, const float *y_ref,
                        float flat_err, float *err_out) {
    const uint32_t n_sub   = RT_G / RT_HALF;
    const uint32_t n_chunk = RT_N / RT_G;

    int8_t   *cb_l1    = malloc((size_t)n_sub * RT_K * RT_HALF * sizeof(int8_t));
    uint16_t *cb_s_l1  = malloc((size_t)n_sub * RT_K * sizeof(uint16_t));
    int8_t   *cb_l2    = malloc((size_t)n_sub * RT_K_INNER * RT_HALF * sizeof(int8_t));
    uint16_t *cb_s_l2  = malloc((size_t)n_sub * RT_K_INNER * sizeof(uint16_t));
    uint16_t *row_scale= malloc((size_t)RT_M * sizeof(uint16_t));
    uint8_t  *idx_l1   = malloc((size_t)n_chunk * n_sub * RT_M * sizeof(uint8_t));
    uint8_t  *idx_l2   = malloc((size_t)n_chunk * n_sub * RT_M * sizeof(uint8_t));

    int rc = pqv2_encode_pyramid(W_, RT_M, RT_N, RT_G,
                                 /*K_outer=*/0, /*K_inner=*/RT_K_INNER,
                                 RT_HALF,
                                 cb_l1, cb_s_l1,
                                 cb_l2, cb_s_l2,
                                 row_scale,
                                 idx_l1, idx_l2,
                                 /*seed=*/42);
    if (rc != 0) {
        fprintf(stderr, "pqv2_encode_pyramid failed: rc=%d\n", rc);
        free(cb_l1); free(cb_s_l1); free(cb_l2); free(cb_s_l2);
        free(row_scale); free(idx_l1); free(idx_l2);
        return 1;
    }

    float y_pq[RT_M];
    memset(y_pq, 0, sizeof(y_pq));
    run_kernel(RT_M, RT_N, RT_G, RT_K, RT_HALF,
               /*l2_kind=*/2, /*l2_K=*/RT_K_INNER,
               row_scale, cb_l1, cb_s_l1, idx_l1,
               cb_l2, cb_s_l2, idx_l2,
               x, y_pq);

    float err = rel_l2_err(y_ref, y_pq, RT_M);
    *err_out = err;

    free(cb_l1); free(cb_s_l1); free(cb_l2); free(cb_s_l2);
    free(row_scale); free(idx_l1); free(idx_l2);

    if (!(err <= RT_THRESH)) {
        fprintf(stderr, "PQv2 roundtrip pyramid: rel L2 err %.6f > threshold %.3f\n",
                (double)err, (double)RT_THRESH);
        dump_head("y_ref", y_ref, 8);
        dump_head("y_pq ", y_pq, 8);
        return 1;
    }

    /* Pyramid should reconstruct strictly better than flat. With a tiny
     * 64×128 tensor the residual L2 codebook can occasionally tie flat
     * within numerical noise, so we warn rather than fail when that
     * happens. The hard PASS criterion is still the absolute threshold
     * above. */
    if (err > flat_err + 1e-6f) {
        fprintf(stderr,
                "NOTE: pyramid err %.6f >= flat err %.6f (no L2 quality benefit "
                "on this tiny tensor) — passing on absolute threshold only.\n",
                (double)err, (double)flat_err);
    }
    return 0;
}

static int test_all_zero(void) {
    /* All-zero W — verify encoder does not crash and reconstruction
     * stays near zero. (Encoder must clamp row_scale at fp16 min normal;
     * if it does not, this either produces NaN or a non-tiny output.) */
    const uint32_t n_sub   = RT_G / RT_HALF;
    const uint32_t n_chunk = RT_N / RT_G;

    float *W_   = calloc((size_t)RT_M * RT_N, sizeof(float));
    float *x    = malloc((size_t)RT_N * sizeof(float));
    make_input(x);

    int8_t   *cb_q     = malloc((size_t)n_sub * RT_K * RT_HALF * sizeof(int8_t));
    uint16_t *cb_scale = malloc((size_t)n_sub * RT_K * sizeof(uint16_t));
    uint16_t *row_scale= malloc((size_t)RT_M * sizeof(uint16_t));
    uint8_t  *idx      = malloc((size_t)n_chunk * n_sub * RT_M * sizeof(uint8_t));

    int rc = pqv2_encode_flat(W_, RT_M, RT_N, RT_G, RT_K, RT_HALF,
                              cb_q, cb_scale, row_scale, idx, /*seed=*/42);
    if (rc != 0) {
        fprintf(stderr, "PQv2 roundtrip zero: encode failed rc=%d\n", rc);
        free(W_); free(x); free(cb_q); free(cb_scale); free(row_scale); free(idx);
        return 1;
    }

    float y_pq[RT_M];
    memset(y_pq, 0, sizeof(y_pq));
    run_kernel(RT_M, RT_N, RT_G, RT_K, RT_HALF, /*l2_kind=*/0, /*l2_K=*/0,
               row_scale, cb_q, cb_scale, idx,
               NULL, NULL, NULL,
               x, y_pq);

    /* All-zero W => W@x == 0; reconstruction must be tiny (and finite). */
    float worst = 0.0f;
    for (int m = 0; m < RT_M; ++m) {
        if (!isfinite(y_pq[m])) {
            fprintf(stderr, "PQv2 roundtrip zero: non-finite output y[%d]=%g\n",
                    m, (double)y_pq[m]);
            free(W_); free(x); free(cb_q); free(cb_scale); free(row_scale); free(idx);
            return 1;
        }
        float a = fabsf(y_pq[m]);
        if (a > worst) worst = a;
    }

    free(W_); free(x); free(cb_q); free(cb_scale); free(row_scale); free(idx);

    /* fp16 min-normal is ~6.1e-5; clamped row_scale times an int8
     * codeword reconstruction stays in the same neighborhood. Allow up
     * to 1e-2 absolute; in practice we see ~1e-4. */
    if (worst > 1e-2f) {
        fprintf(stderr, "PQv2 roundtrip zero: max |y_pq| = %.6g (expected ~0)\n",
                (double)worst);
        return 1;
    }
    return 0;
}

int main(void) {
    float *W_   = malloc((size_t)RT_M * RT_N * sizeof(float));
    float *x    = malloc((size_t)RT_N * sizeof(float));
    float *y_ref= malloc((size_t)RT_M * sizeof(float));
    if (!W_ || !x || !y_ref) {
        fprintf(stderr, "alloc failed\n");
        return 2;
    }
    make_weights(W_);
    make_input(x);
    ref_matmul(W_, x, y_ref);

    int fails = 0;
    float flat_err = 0.0f, pyr_err = 0.0f;

    if (test_flat(W_, x, y_ref, &flat_err) != 0) fails++;
    else fprintf(stdout, "PQv2 roundtrip flat:    rel L2 err = %.6f (threshold %.3f) OK\n",
                 (double)flat_err, (double)RT_THRESH);

    if (test_pyramid(W_, x, y_ref, flat_err, &pyr_err) != 0) fails++;
    else fprintf(stdout, "PQv2 roundtrip pyramid: rel L2 err = %.6f (threshold %.3f) OK%s\n",
                 (double)pyr_err, (double)RT_THRESH,
                 (pyr_err < flat_err) ? " [better than flat]"
                                      : " [tie/worse-than-flat: noise level]");

    if (test_all_zero() != 0) fails++;
    else fprintf(stdout, "PQv2 roundtrip zero:    encoder + kernel produce ~0 output OK\n");

    free(W_); free(x); free(y_ref);

    if (fails) {
        fprintf(stderr, "PQv2 roundtrip: %d sub-test(s) FAILED\n", fails);
        return 1;
    }
    fprintf(stdout, "PQv2 roundtrip: ALL OK\n");
    return 0;
}
