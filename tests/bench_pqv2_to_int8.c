/* Path 1 derisk: decode a real PQv2 K=256 tensor to flat INT8 + per-row
 * scale at load time, then run vdotq_s32-based int8 matmul. Compare
 * speed and cosine against the current PQv2 TBL-gather matvec.
 *
 * Question this answers:
 *   - Does pre-decoding PQv2 to int8 actually preserve quality? (cosine)
 *   - Does the vdotq-based int8 matvec hit the speed we projected? (~5× faster)
 *
 * Per-group input quantization (g=128) to match w4a8's quality model.
 */
#include "../src/inferbit_internal.h"
#include "../src/pqv2_format.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <arm_neon.h>

#define IB_GROUP 128

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Decode a PQv2 tensor into a flat INT8 weight matrix [M, N] with one
 * fp32 scale per row. Re-quantizes by computing the full-precision
 * decoded weight per row, finding the row max, and quantizing to int8. */
static void decode_pqv2_to_int8(const pqv2_t *t, int8_t *w_out, float *row_scale_out) {
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
        /* Decode this row to fp32 first */
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
        /* Find max abs and quantize to int8 */
        float rmax = 0.0f;
        for (uint32_t n = 0; n < N; n++) {
            float a = fabsf(row[n]); if (a > rmax) rmax = a;
        }
        float scale = (rmax > 0) ? (rmax / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        row_scale_out[m] = scale;
        for (uint32_t n = 0; n < N; n++) {
            int q = (int)roundf(row[n] * inv_scale);
            if (q > 127) q = 127; if (q < -128) q = -128;
            w_out[(size_t)m * N + n] = (int8_t)q;
        }
    }

    free(row);
    if (!t->cb_fp32) free(cb_fp32);
}

/* Quantize input vector x[N] to int8 with per-group scale. Matches
 * libinferbit's w4a8 input quantization (group size = IB_GROUP). */
static void quantize_input_int8_grouped(const float *x, int N, int8_t *q, float *scales) {
    int n_groups = (N + IB_GROUP - 1) / IB_GROUP;
    for (int g = 0; g < n_groups; g++) {
        int start = g * IB_GROUP;
        int end = (start + IB_GROUP > N) ? N : start + IB_GROUP;
        float xmax = 0.0f;
        for (int i = start; i < end; i++) {
            float a = fabsf(x[i]); if (a > xmax) xmax = a;
        }
        float scale = (xmax > 0) ? (xmax / 127.0f) : 1.0f;
        scales[g] = scale;
        float inv = 1.0f / scale;
        for (int i = start; i < end; i++) {
            int v = (int)roundf(x[i] * inv);
            if (v > 127) v = 127; if (v < -128) v = -128;
            q[i] = (int8_t)v;
        }
    }
}

/* The fast int8 matmul using vdotq_s32, with per-row weight scale and
 * per-group input scale. Mirrors w4a8's structure. */
static void int8_matmul_vdotq_grouped(
    float *out, const int8_t *w, const float *w_scale,
    const int8_t *x_q, const float *x_scales,
    int M, int N)
{
    int n_groups = (N + IB_GROUP - 1) / IB_GROUP;
    for (int i = 0; i < M; i++) {
        const int8_t *row = w + (size_t)i * N;
        float row_acc = 0.0f;
        int j = 0;
        for (int g = 0; g < n_groups; g++) {
            int start = g * IB_GROUP;
            int end = (start + IB_GROUP > N) ? N : start + IB_GROUP;
            int32x4_t acc = vdupq_n_s32(0);
            for (j = start; j + 15 < end; j += 16) {
                int8x16_t w16 = vld1q_s8(row + j);
                int8x16_t x16 = vld1q_s8(x_q + j);
                acc = vdotq_s32(acc, w16, x16);
            }
            int32_t group_int = vaddvq_s32(acc);
            for (; j < end; j++) group_int += (int32_t)row[j] * (int32_t)x_q[j];
            row_acc += (float)group_int * x_scales[g];
        }
        out[i] = row_acc * w_scale[i];
    }
}

/* PQv2 reference matvec via the existing TBL-gather kernel. */
static void pqv2_matvec_reference(const pqv2_t *t, const float *x, float *y) {
    pqv2_matvec_tbl_int8_k256(t, x, y);
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
    if (ib_pqv2_file_load(path, &f) != 0) { fprintf(stderr, "load %s failed\n", path); return 1; }
    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, tname);
    if (!nt || nt->kind != IB_PQV2_KIND_PQV2) { fprintf(stderr, "tensor missing\n"); return 1; }
    const pqv2_t *t = &nt->pq;
    int M = (int)t->M, N = (int)t->N;
    printf("tensor %s: M=%d N=%d K=%u half=%u\n", tname, M, N, t->K, t->half);

    /* Random fp32 input mimicking transformer activation magnitudes */
    float *x = aligned_alloc(64, ((size_t)N * sizeof(float) + 63) & ~(size_t)63);
    unsigned s = 42;
    for (int i = 0; i < N; i++) {
        s = s * 1103515245 + 12345;
        float u = ((float)((s >> 16) & 0xffff) / 32768.0f - 1.0f);
        x[i] = u * u * u * 0.5f;   /* gaussian-ish, ~N(0,0.1) */
    }

    /* (1) Reference: PQv2 TBL-gather matvec */
    float *y_pq = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    pqv2_matvec_reference(t, x, y_pq);

    /* (2) Decode PQv2 → INT8 + per-row scale (one-time cost, off the hot path) */
    int8_t *w_int8   = aligned_alloc(64, ((size_t)M * N + 63) & ~(size_t)63);
    float  *row_sc   = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    double t_decode_0 = now_sec();
    decode_pqv2_to_int8(t, w_int8, row_sc);
    double t_decode = (now_sec() - t_decode_0) * 1000.0;
    printf("decode time: %.2f ms (one-time at load)\n", t_decode);

    /* (3) Quantize input per-group to int8 */
    int n_groups = (N + IB_GROUP - 1) / IB_GROUP;
    int8_t *x_q = aligned_alloc(64, ((size_t)N + 63) & ~(size_t)63);
    float  *x_scales = aligned_alloc(64, ((size_t)n_groups * sizeof(float) + 63) & ~(size_t)63);
    quantize_input_int8_grouped(x, N, x_q, x_scales);

    /* (4) vdotq INT8 matvec */
    float *y_int8 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    /* Warmup */
    for (int it = 0; it < 5; it++)
        int8_matmul_vdotq_grouped(y_int8, w_int8, row_sc, x_q, x_scales, M, N);

    /* Time PQv2 reference */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) pqv2_matvec_reference(t, x, y_pq);
    double t_pq_ms = (now_sec() - t0) / iters * 1000.0;

    /* Time vdotq INT8 (input quant per call) */
    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        quantize_input_int8_grouped(x, N, x_q, x_scales);
        int8_matmul_vdotq_grouped(y_int8, w_int8, row_sc, x_q, x_scales, M, N);
    }
    double t_int8_ms = (now_sec() - t0) / iters * 1000.0;

    double cos = cosine(y_pq, y_int8, (size_t)M);
    double max_err = 0;
    for (int i = 0; i < M; i++) max_err = fmax(max_err, fabs(y_pq[i] - y_int8[i]));

    printf("\n=== PQv2 → INT8 (Path 1) derisk ===\n");
    printf("  reference (PQv2 TBL-gather): %.4f ms/call\n", t_pq_ms);
    printf("  decoded INT8 (vdotq+input_q): %.4f ms/call\n", t_int8_ms);
    printf("  speedup: %.2f×\n", t_pq_ms / t_int8_ms);
    printf("  correctness:\n");
    printf("    cosine (vdotq vs PQv2 ref): %.6f\n", cos);
    printf("    max abs diff: %.4f  (relative: %.2e)\n",
            max_err, max_err / fabs(y_pq[0] + 1e-30));

    free(x); free(y_pq); free(w_int8); free(row_sc);
    free(x_q); free(x_scales); free(y_int8);
    ib_pqv2_file_free(&f);
    return 0;
}
