/* bench_pqv2 — head-to-head: PQv2 matvec vs INT4 G=32 matvec.
 *
 * Inputs:
 *   argv[1] = path to .pqv2 file (Python-dumped)
 *   argv[2] = path to .int4 file (same tensor encoded as INT4 G=32)
 *
 * .int4 format (little-endian):
 *   "INT4" magic 4B, u32 M, u32 N, u32 G,
 *   q [M, N/2] uint8 packed nibbles (0..15 mapped to -8..7),
 *   scale [M, N/G] fp16
 */

#include "pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    uint32_t M, N, G;
    const uint8_t *q;
    const uint16_t *scale;
    void *_q, *_s;
} int4_t;

static int int4_load(const char *path, int4_t *out) {
    FILE *f = fopen(path, "rb"); if (!f) { perror(path); return -1; }
    char m[4]; if (fread(m, 1, 4, f) != 4 || memcmp(m, "INT4", 4)) { fclose(f); return -1; }
    uint32_t h[3]; if (fread(h, 4, 3, f) != 3) { fclose(f); return -1; }
    out->M = h[0]; out->N = h[1]; out->G = h[2];
    size_t qb = (size_t)out->M * out->N / 2;
    size_t sb = (size_t)out->M * (out->N / out->G) * 2;
    out->_q = malloc(qb); out->_s = malloc(sb);
    if (fread(out->_q, 1, qb, f) != qb || fread(out->_s, 1, sb, f) != sb) { fclose(f); return -1; }
    out->q = out->_q; out->scale = out->_s;
    fclose(f); return 0;
}

static void int4_free(int4_t *t) { free(t->_q); free(t->_s); }

#if defined(__ARM_NEON)
#include <arm_neon.h>
/* INT4 G=32 NEON matvec. Packed layout: byte i has w[2i] in lo nibble,
 * w[2i+1] in hi nibble. Group has 16 packed bytes = 32 weights. */
static void int4_matvec_neon(const int4_t *t, const float *x, float *y) {
    uint32_t M = t->M, N = t->N, G = t->G;
    uint32_t ng = N / G;
    const uint8x16_t mask_lo = vdupq_n_u8(0x0F);
    const int8x16_t bias = vdupq_n_s8(8);
    for (uint32_t m = 0; m < M; m++) {
        const uint8_t *qrow = &t->q[(size_t)m * (N / 2)];
        const uint16_t *srow = &t->scale[(size_t)m * ng];
        float total = 0.0f;
        for (uint32_t g = 0; g < ng; g++) {
            float sc = pqv2_h2f(srow[g]);
            uint8x16_t pkd = vld1q_u8(&qrow[g * 16]);
            int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(pkd, mask_lo)), bias);
            int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(pkd, 4)), bias);
            /* Re-interleave so output order is w[0], w[1], w[2], ..., w[31].
             * vzipq_s8 yields {w[0],w[1],w[2],...,w[15]} in val[0],
             *                 {w[16],...,w[31]} in val[1]. */
            int8x16x2_t z = vzipq_s8(lo, hi);
            int16x8_t a16 = vmovl_s8(vget_low_s8(z.val[0]));
            int16x8_t b16 = vmovl_s8(vget_high_s8(z.val[0]));
            int16x8_t c16 = vmovl_s8(vget_low_s8(z.val[1]));
            int16x8_t d16 = vmovl_s8(vget_high_s8(z.val[1]));
            float32x4_t a0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(a16)));
            float32x4_t a1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(a16)));
            float32x4_t a2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
            float32x4_t a3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
            float32x4_t a4 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(c16)));
            float32x4_t a5 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(c16)));
            float32x4_t a6 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(d16)));
            float32x4_t a7 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(d16)));
            const float *xg = &x[g * G];
            float32x4_t acc = vmulq_f32(a0, vld1q_f32(&xg[0]));
            acc = vfmaq_f32(acc, a1, vld1q_f32(&xg[4]));
            acc = vfmaq_f32(acc, a2, vld1q_f32(&xg[8]));
            acc = vfmaq_f32(acc, a3, vld1q_f32(&xg[12]));
            acc = vfmaq_f32(acc, a4, vld1q_f32(&xg[16]));
            acc = vfmaq_f32(acc, a5, vld1q_f32(&xg[20]));
            acc = vfmaq_f32(acc, a6, vld1q_f32(&xg[24]));
            acc = vfmaq_f32(acc, a7, vld1q_f32(&xg[28]));
            total += vaddvq_f32(acc) * sc;
        }
        y[m] = total;
    }
}
#endif

static void int4_matvec(const int4_t *t, const float *x, float *y) {
    uint32_t M = t->M, N = t->N, G = t->G;
    uint32_t ng = N / G;
    for (uint32_t m = 0; m < M; m++) {
        float acc = 0.0f;
        const uint8_t *qrow = &t->q[(size_t)m * (N / 2)];
        const uint16_t *srow = &t->scale[(size_t)m * ng];
        for (uint32_t g = 0; g < ng; g++) {
            float sc = pqv2_h2f(srow[g]);
            float gacc = 0.0f;
            for (uint32_t i = 0; i < G; i++) {
                size_t bi = (size_t)g * G + i;
                uint8_t byte = qrow[bi / 2];
                int8_t nib = (bi & 1) ? (int8_t)((byte >> 4) & 0xF) - 8
                                       : (int8_t)(byte & 0xF) - 8;
                gacc += (float)nib * x[bi];
            }
            acc += gacc * sc;
        }
        y[m] = acc;
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tensor.pqv2> <tensor.int4>\n", argv[0]);
        return 1;
    }
    pqv2_t pq; void *pq_owned;
    if (pqv2_load(argv[1], &pq, &pq_owned) != 0) return 1;
    int4_t i4; if (int4_load(argv[2], &i4) != 0) return 1;

    if (pq.M != i4.M || pq.N != i4.N) {
        fprintf(stderr, "shape mismatch\n"); return 1;
    }
    printf("tensor: M=%u N=%u G=%u K=%u n_sub=%u half=%u l2_kind=%u\n",
           pq.M, pq.N, pq.G, pq.K, pq.n_subchunks, pq.half, pq.l2_kind);

    /* Random x */
    float *x = malloc((size_t)pq.N * sizeof(float));
    float *y_pq = malloc((size_t)pq.M * sizeof(float));
    float *y_i4 = malloc((size_t)pq.M * sizeof(float));
    srand(42);
    for (uint32_t i = 0; i < pq.N; i++) x[i] = (float)((rand() % 2001) - 1000) / 1000.f;

    /* Warmup */
    pqv2_matvec_scalar(&pq, x, y_pq);
    int4_matvec(&i4, x, y_i4);

    /* Bench */
    int N_ITER = 10;
    double t0 = now_s();
    for (int it = 0; it < N_ITER; it++) pqv2_matvec_scalar(&pq, x, y_pq);
    double t_pq = (now_s() - t0) / N_ITER;

    float *y_lut = malloc((size_t)pq.M * sizeof(float));
    pqv2_matvec_lut(&pq, x, y_lut);
    t0 = now_s();
    for (int it = 0; it < N_ITER; it++) pqv2_matvec_lut(&pq, x, y_lut);
    double t_lut = (now_s() - t0) / N_ITER;

    float *y_neon = malloc((size_t)pq.M * sizeof(float));
    pqv2_matvec_lut_neon(&pq, x, y_neon);
    t0 = now_s();
    for (int it = 0; it < N_ITER; it++) pqv2_matvec_lut_neon(&pq, x, y_neon);
    double t_neon = (now_s() - t0) / N_ITER;

    float *y_tbl = malloc((size_t)pq.M * sizeof(float));
    double t_tbl = 0;
    if (pq.K <= 64) {
        pqv2_matvec_tbl_int8(&pq, x, y_tbl);
        t0 = now_s();
        for (int it = 0; it < N_ITER; it++) pqv2_matvec_tbl_int8(&pq, x, y_tbl);
        t_tbl = (now_s() - t0) / N_ITER;
    } else if (pq.K == 128) {
        pqv2_matvec_tbl_int8_k128(&pq, x, y_tbl);
        t0 = now_s();
        for (int it = 0; it < N_ITER; it++) pqv2_matvec_tbl_int8_k128(&pq, x, y_tbl);
        t_tbl = (now_s() - t0) / N_ITER;
    } else if (pq.K == 256) {
        pqv2_matvec_tbl_int8_k256(&pq, x, y_tbl);
        t0 = now_s();
        for (int it = 0; it < N_ITER; it++) pqv2_matvec_tbl_int8_k256(&pq, x, y_tbl);
        t_tbl = (now_s() - t0) / N_ITER;
        /* sanity: tbl_int8 vs LUT cosine (will lose a small bit due to int8 noise) */
        double dot = 0, n_l = 0, n_t = 0;
        for (uint32_t m = 0; m < pq.M; m++) {
            dot += y_lut[m] * y_tbl[m]; n_l += y_lut[m]*y_lut[m]; n_t += y_tbl[m]*y_tbl[m];
        }
        printf("TBL-int8 vs fp32 LUT cos: %.5f\n", dot/(sqrt(n_l)*sqrt(n_t)+1e-12));
    }

    t0 = now_s();
    for (int it = 0; it < N_ITER; it++) int4_matvec(&i4, x, y_i4);
    double t_i4 = (now_s() - t0) / N_ITER;

    float *y_i4_neon = malloc((size_t)pq.M * sizeof(float));
    double t_i4_neon = 0;
#if defined(__ARM_NEON)
    int4_matvec_neon(&i4, x, y_i4_neon);
    t0 = now_s();
    for (int it = 0; it < N_ITER; it++) int4_matvec_neon(&i4, x, y_i4_neon);
    t_i4_neon = (now_s() - t0) / N_ITER;
#endif

    /* sanity: scalar vs LUT must agree */
    double err = 0, sum = 0;
    for (uint32_t m = 0; m < pq.M; m++) {
        float d = y_pq[m] - y_lut[m]; err += d*d; sum += y_pq[m]*y_pq[m];
    }
    printf("LUT vs scalar rel-err: %.2e\n", sqrt(err)/sqrt(sum + 1e-12));

    /* Compare outputs */
    double dot = 0, n_pq = 0, n_i4 = 0;
    for (uint32_t m = 0; m < pq.M; m++) {
        dot += y_pq[m] * y_i4[m];
        n_pq += y_pq[m] * y_pq[m];
        n_i4 += y_i4[m] * y_i4[m];
    }
    double cos = dot / (sqrt(n_pq) * sqrt(n_i4) + 1e-12);

    double mb_pq = (double)((size_t)pq.M * pq.N) / 1e6;
    printf("PQv2 scalar:   %.2f ms/matvec   %.1f Gop/s\n", t_pq * 1000, mb_pq * 2 / t_pq / 1e3);
    printf("PQv2 LUT:      %.2f ms/matvec   %.1f Gop/s\n", t_lut * 1000, mb_pq * 2 / t_lut / 1e3);
    printf("PQv2 NEON:     %.2f ms/matvec   %.1f Gop/s\n", t_neon * 1000, mb_pq * 2 / t_neon / 1e3);
    if (t_tbl > 0)
        printf("PQv2 TBL-int8: %.2f ms/matvec   %.1f Gop/s\n", t_tbl * 1000, mb_pq * 2 / t_tbl / 1e3);
    printf("INT4 scalar:   %.2f ms/matvec   %.1f Gop/s\n", t_i4 * 1000, mb_pq * 2 / t_i4 / 1e3);
    if (t_i4_neon > 0)
        printf("INT4 NEON:     %.2f ms/matvec   %.1f Gop/s\n", t_i4_neon * 1000, mb_pq * 2 / t_i4_neon / 1e3);
    printf("PQv2 NEON speedup vs INT4 NEON: %.2fx\n", t_i4_neon > 0 ? t_i4_neon / t_neon : 0.0);
    printf("output cos(PQv2, INT4) = %.4f  (sanity: should be ≥0.99)\n", cos);

    free(x); free(y_pq); free(y_i4);
    pqv2_free(pq_owned);
    int4_free(&i4);
    return 0;
}
