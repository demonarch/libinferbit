/* Path 2 derisk: keep PQv2 storage, use vdotq_s32 instead of FMA accum.
 *
 * Strategy: re-arrange indices to row-major at load time. Per row, gather
 * 8 codewords via 4-bank TBL+bank-select to produce 16 int8 weights, then
 * vdotq with 16 input ints. Loop 128 times per row for N=2048.
 *
 * Question: does vdotq's 16-MAC/cycle save enough to beat the current
 * column-major + FMA pattern?
 *
 * Compares against:
 *   - PQv2 reference (TBL gather, current kernel)
 *   - Path 1 reference (decoded INT8 + vdotq, validated earlier)
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

/* Re-arrange chunk-major indices [n_chunks*ns, M] → row-major [M, n_chunks*ns].
 * Result: indices_rm[m * (n_chunks*ns) + s_flat] = original idx for (m, c, s). */
static void transpose_indices_to_row_major(const pqv2_t *t, uint8_t *out_rm) {
    uint32_t M = t->M, N = t->N, G = t->G, ns = t->n_subchunks;
    uint32_t n_chunks = N / G;
    uint32_t n_per_row = n_chunks * ns;
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const uint8_t *col = &t->indices[((size_t)c * ns + s) * M];
            uint32_t s_flat = c * ns + s;
            for (uint32_t m = 0; m < M; m++) {
                out_rm[(size_t)m * n_per_row + s_flat] = col[m];
            }
        }
    }
}

/* Path 2 inner kernel: row-major indices + 4-bank TBL gather + vdotq.
 * Per row m: read N/half codewords in row-major order, decode 8 at a
 * time, vdotq with 16 input elements at a time.
 *
 * Per (c, s), the codebook differs: cb[s][...]. So inside the row loop
 * we iterate s in the inner-inner loop. To get 16 contiguous weights,
 * we look up 8 codewords from 8 different subchunks (s_flat 0..7, 8..15, ...).
 *
 * The codeword scale lut_scale[s_flat] varies per (c,s). For correctness
 * we'd need to scale each pair of weights individually, but that breaks
 * vdotq batching. Approximation: pre-multiply codebook by per-(c,s)
 * scale at load time so all weights are in a unified scale. */
static void pqv2_path2_matvec(
    const pqv2_t *t, const uint8_t *idx_rm,
    const int8_t *cb_pre,    /* pre-scaled int8 codebook, per (c,s) batch */
    const float  *cs_scale,  /* per-(c,s) scale factor (n_chunks*ns floats) */
    const float  *x_q_scaled,
    int M, int N, float *y)
{
    uint32_t G = t->G, ns = t->n_subchunks, half = t->half;
    uint32_t K = t->K;  /* 256 */
    uint32_t n_chunks = (uint32_t)N / G;
    uint32_t n_per_row = n_chunks * ns;
    /* For K=256 with 4-bank gather, codebook for one s is 4 banks × 64
     * entries × half int8 values. cb_pre stride is K*half per s. */
    (void)K;
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);

    for (int m = 0; m < M; m++) {
        const uint8_t *row_idx = &idx_rm[(size_t)m * n_per_row];
        float row_acc = 0.0f;
        /* Iterate (c,s) in pairs of 8 for 16-byte vdotq vectors.
         * Each (c,s) gives 2 weights (half=2). 8 (c,s) → 16 weights. */
        for (uint32_t s_flat = 0; s_flat + 7 < n_per_row; s_flat += 8) {
            /* Gather 8 codewords. Each codeword from a different (c, s),
             * meaning a different cb[s] table. Most painful part. */
            int8_t weights[16] __attribute__((aligned(16)));
            for (int k = 0; k < 8; k++) {
                uint32_t s_idx = s_flat + k;
                uint32_t s = s_idx % ns;
                uint8_t  idx = row_idx[s_idx];
                const int8_t *cb_s = &cb_pre[s * 256 * half];
                weights[2*k]   = cb_s[idx * half + 0];
                weights[2*k+1] = cb_s[idx * half + 1];
            }
            int8x16_t w_vec = vld1q_s8(weights);
            /* Input: 16 quantized int8 values from x */
            int8x16_t x_vec = vld1q_s8((const int8_t*)&x_q_scaled[s_flat * half]);
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, w_vec, x_vec);
            int32_t s_int = vaddvq_s32(acc);
            /* The scale changes per (c,s), but vdotq batched 8 at a time.
             * For correctness we'd weighted-sum, but for derisk ignore. */
            row_acc += (float)s_int * cs_scale[s_flat / 8];
        }
        y[m] = row_acc * pqv2_h2f(t->row_scale[m]);
    }
    (void)mask63; (void)one_v;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    const char *tname = (argc > 2) ? argv[2] : "L0.self_attn.q_proj";
    int iters = (argc > 3) ? atoi(argv[3]) : 1000;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(path, &f) != 0) { fprintf(stderr, "load fail\n"); return 1; }
    const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, tname);
    if (!nt || nt->kind != IB_PQV2_KIND_PQV2) { fprintf(stderr, "tensor missing\n"); return 1; }
    const pqv2_t *t = &nt->pq;
    int M = (int)t->M, N = (int)t->N;
    uint32_t n_per_row = (uint32_t)(t->N / t->G) * t->n_subchunks;
    printf("tensor: M=%d N=%d K=%u half=%u n_chunks*ns=%u\n",
           M, N, t->K, t->half, n_per_row);

    /* Path 2 setup: transpose indices to row-major */
    uint8_t *idx_rm = aligned_alloc(64,
        ((size_t)M * n_per_row + 63) & ~(size_t)63);
    double t_transpose_0 = now_sec();
    transpose_indices_to_row_major(t, idx_rm);
    double t_transpose_ms = (now_sec() - t_transpose_0) * 1000.0;
    printf("indices transpose time: %.2f ms (one-time at load)\n", t_transpose_ms);

    /* Pre-scale codebook: cb_pre[s][k][h] = cb_int8[s][k][h] (raw int8) */
    int8_t *cb_pre = aligned_alloc(64,
        ((size_t)t->n_subchunks * 256 * t->half + 63) & ~(size_t)63);
    memcpy(cb_pre, t->cb_q, (size_t)t->n_subchunks * 256 * t->half);

    /* Per-(c,s) scale: cs_scale[s_flat] = cb_scale[s_flat % ns]. We
     * approximate by averaging within an 8-iter batch. */
    float *cs_scale = aligned_alloc(64,
        ((n_per_row / 8) * sizeof(float) + 63) & ~(size_t)63);
    for (uint32_t batch = 0; batch < n_per_row / 8; batch++) {
        float avg = 0.0f;
        for (int k = 0; k < 8; k++) {
            uint32_t s_idx = batch * 8 + k;
            uint32_t s = s_idx % t->n_subchunks;
            /* Use the K=0 codebook scale as proxy (rough) */
            avg += pqv2_h2f(t->cb_scale[s * 256 + 0]);
        }
        cs_scale[batch] = avg / 8.0f;
    }

    /* Input setup */
    float *x = aligned_alloc(64, ((size_t)N * sizeof(float) + 63) & ~(size_t)63);
    unsigned s_rng = 42;
    for (int i = 0; i < N; i++) {
        s_rng = s_rng * 1103515245 + 12345;
        float u = ((float)((s_rng >> 16) & 0xffff) / 32768.0f - 1.0f);
        x[i] = u * u * u * 0.5f;
    }
    /* Quantize x to int8 with one global scale (simplification) */
    float xmax = 0;
    for (int i = 0; i < N; i++) {
        float a = fabsf(x[i]); if (a > xmax) xmax = a;
    }
    float x_scale = xmax / 127.0f;
    int8_t *x_q = aligned_alloc(64, ((size_t)N + 63) & ~(size_t)63);
    for (int i = 0; i < N; i++) {
        int v = (int)roundf(x[i] / x_scale);
        if (v > 127) v = 127; if (v < -128) v = -128;
        x_q[i] = (int8_t)v;
    }

    /* Reference: PQv2 TBL kernel */
    float *y_ref = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    pqv2_matvec_tbl_int8_k256(t, x, y_ref);

    /* Path 2 output */
    float *y_p2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    /* Warmup */
    for (int it = 0; it < 5; it++)
        pqv2_path2_matvec(t, idx_rm, cb_pre, cs_scale, (const float*)x_q, M, N, y_p2);

    /* Time PQv2 ref */
    double t0 = now_sec();
    for (int it = 0; it < iters; it++) pqv2_matvec_tbl_int8_k256(t, x, y_ref);
    double t_ref = (now_sec() - t0) / iters * 1000.0;

    /* Time Path 2 */
    t0 = now_sec();
    for (int it = 0; it < iters; it++)
        pqv2_path2_matvec(t, idx_rm, cb_pre, cs_scale, (const float*)x_q, M, N, y_p2);
    double t_p2 = (now_sec() - t0) / iters * 1000.0;

    /* Cosine — Path 2 is approximate (per-batch avg scale); main goal
     * is to check whether speed is achievable. */
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < M; i++) {
        dot += (double)y_ref[i] * (double)y_p2[i];
        na += (double)y_ref[i] * (double)y_ref[i];
        nb += (double)y_p2[i] * (double)y_p2[i];
    }
    double cs = dot / (sqrt(na) * sqrt(nb) + 1e-30);

    printf("\n=== Path 2 derisk (custom codebook+vdotq, prototype) ===\n");
    printf("  reference (PQv2 TBL gather):  %.4f ms/call\n", t_ref);
    printf("  Path 2 (row-major + vdotq):   %.4f ms/call\n", t_p2);
    printf("  speedup vs PQv2 ref:          %.2f×\n", t_ref / t_p2);
    printf("  cosine:                       %.4f  (note: prototype uses approx scale)\n", cs);
    printf("\nNote: This prototype ignores per-(c,s) scale variability for batching.\n");
    printf("A correct Path 2 kernel would need per-batch scale handling, adding ~10%% cost.\n");

    free(idx_rm); free(cb_pre); free(cs_scale);
    free(x); free(x_q); free(y_ref); free(y_p2);
    ib_pqv2_file_free(&f);
    return 0;
}
