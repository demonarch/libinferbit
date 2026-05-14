/* Path 1 → INT4 (Strategy 3) derisk: re-quantize PQv2 K=256 down to flat
 * INT4 with per-row scale, then use libinferbit's matmul_w4a8 (which IS
 * already vdotq-optimized). Tests speed AND quality of double quantization.
 *
 * Memory: 550 MB (vs Path 1's 1100 MB), 30 MB less than current PQv2's 715 MB.
 * Speed: should match INT4 NEON ~30 t/s.
 * Quality: PQv2 K=256 noise + INT4 RTN noise compounded.
 */
#include "../src/inferbit_internal.h"
#include "../src/pqv2_format.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define IB_GROUP 128

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Decode PQv2 to fp32, then re-quantize to INT4 with per-row scale. */
static void decode_pqv2_to_int4(const pqv2_t *t, uint8_t *w_out, float *row_scale_out) {
    uint32_t M = t->M, N = t->N, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = N / G;

    float *cb_fp32 = NULL;
    if (t->cb_fp32) cb_fp32 = (float*)t->cb_fp32;
    else {
        cb_fp32 = malloc((size_t)ns * K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < K; k++) {
                float sc = pqv2_h2f(t->cb_scale[s * K + k]);
                const int8_t *q = &t->cb_q[(s * K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    cb_fp32[(s * K + k) * half + h] = (float)q[h] * sc;
            }
    }

    float *row = aligned_alloc(64, ((size_t)N * sizeof(float) + 63) & ~(size_t)63);

    for (uint32_t m = 0; m < M; m++) {
        float row_pq_scale = pqv2_h2f(t->row_scale[m]);
        for (uint32_t c = 0; c < n_chunks; c++) {
            for (uint32_t s = 0; s < ns; s++) {
                uint8_t idx = t->indices[((size_t)c * ns + s) * M + m];
                for (uint32_t h = 0; h < half; h++) {
                    row[c * G + s * half + h] =
                        cb_fp32[(s * K + idx) * half + h] * row_pq_scale;
                }
            }
        }
        /* Find max abs and quantize to INT4 (4 bits, range [-8, 7] +8 → [0, 15]) */
        float rmax = 0.0f;
        for (uint32_t n = 0; n < N; n++) {
            float a = fabsf(row[n]); if (a > rmax) rmax = a;
        }
        /* INT4 storage: nibbles biased by +8. range [0, 15] = signed [-8, 7]. */
        float scale = (rmax > 0) ? (rmax / 7.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        row_scale_out[m] = scale;
        /* Pack 2 nibbles per byte */
        uint8_t *row_out = w_out + (size_t)m * (N / 2);
        for (uint32_t n = 0; n < N; n += 2) {
            int q0 = (int)roundf(row[n]   * inv_scale) + 8;  /* bias to [0, 15] */
            int q1 = (int)roundf(row[n+1] * inv_scale) + 8;
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            row_out[n / 2] = (uint8_t)((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }

    free(row);
    if (!t->cb_fp32) free(cb_fp32);
}

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na += (double)a[i] * (double)a[i];
        nb += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    const char *tname = (argc > 2) ? argv[2] : "L0.self_attn.q_proj";
    int iters = (argc > 3) ? atoi(argv[3]) : 1000;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(path, &f) != 0) { fprintf(stderr, "load fail\n"); return 1; }
    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, tname);
    const pqv2_t *t = &nt->pq;
    int M = (int)t->M, N = (int)t->N;
    printf("tensor: M=%d N=%d K=%u half=%u\n", M, N, t->K, t->half);

    /* Random fp32 input */
    float *x = aligned_alloc(64, ((size_t)N * sizeof(float) + 63) & ~(size_t)63);
    unsigned s_rng = 42;
    for (int i = 0; i < N; i++) {
        s_rng = s_rng * 1103515245 + 12345;
        float u = ((float)((s_rng >> 16) & 0xffff) / 32768.0f - 1.0f);
        x[i] = u * u * u * 0.5f;
    }

    /* Reference: PQv2 TBL gather */
    float *y_ref = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    pqv2_matvec_tbl_int8_k256(t, x, y_ref);

    /* Decode PQv2 → INT4 */
    uint8_t *w_int4 = aligned_alloc(64, ((size_t)M * N / 2 + 63) & ~(size_t)63);
    float   *row_sc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    double t_decode_0 = now_sec();
    decode_pqv2_to_int4(t, w_int4, row_sc);
    double t_decode_ms = (now_sec() - t_decode_0) * 1000.0;
    printf("decode time: %.2f ms (one-time at load)\n", t_decode_ms);

    /* Use existing libinferbit matmul_w4a8 (ib_kern defined in
     * inferbit_internal.h). */
    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);

    /* w4a8 needs INT8-quantized input + per-group scale.
     * Use ib_quantize_input_int8_g128 (mirrors the production path). */
    int n_groups = (N + IB_GROUP - 1) / IB_GROUP;
    int8_t *x_q = aligned_alloc(64, ((size_t)N + 63) & ~(size_t)63);
    float  *x_sc = aligned_alloc(64, ((size_t)n_groups * sizeof(float) + 63) & ~(size_t)63);
    ib_quantize_input_int8_g128(x, x_q, x_sc, N);

    float *y_int4 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    /* Warmup */
    for (int it = 0; it < 5; it++)
        ib_kern.matmul_w4a8(y_int4, w_int4, row_sc, x_q, x_sc, M, N);

    /* Time PQv2 ref */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) pqv2_matvec_tbl_int8_k256(t, x, y_ref);
    double t_ref = (now_sec() - t0) / iters * 1000.0;

    /* Time w4a8 with quantize */
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        ib_quantize_input_int8_g128(x, x_q, x_sc, N);
        ib_kern.matmul_w4a8(y_int4, w_int4, row_sc, x_q, x_sc, M, N);
    }
    double t_int4 = (now_sec() - t0) / iters * 1000.0;

    double cs = cosine(y_ref, y_int4, (size_t)M);

    printf("\n=== PQv2 → INT4 (Strategy 3) derisk ===\n");
    printf("  reference (PQv2 TBL):        %.4f ms/call\n", t_ref);
    printf("  decoded INT4 (w4a8 vdotq):   %.4f ms/call\n", t_int4);
    printf("  speedup: %.2f×\n", t_ref / t_int4);
    printf("  cosine vs PQv2 ref:          %.6f\n", cs);
    printf("  memory ratio: %.2fx (vs current PQv2 715 MB → %.0f MB)\n",
            0.5 / 0.875, 715.0 * 0.5 / 0.875);

    free(x); free(y_ref); free(w_int4); free(row_sc);
    free(x_q); free(x_sc); free(y_int4);
    ib_pqv2_file_free(&f);
    return 0;
}
