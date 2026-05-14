#include "pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ── fp16 ─────────────────────────────────────────────────────────── */

float pqv2_h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            exp++; mant &= 0x3FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

uint16_t pqv2_f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* ── loader ───────────────────────────────────────────────────────── */
/* File format (little-endian):
 *   magic "PQV2" 4 bytes
 *   header u32: M, N, G, K, n_sub, half, l2_kind, l2_K
 *   row_scale [M] fp16
 *   cb_q [n_sub*K*half] int8
 *   cb_scale [n_sub*K] fp16
 *   indices [M*n_chunks*n_sub] u8
 *   if l2_kind==2:
 *     l2_cb_q [n_sub*l2_K*half] int8
 *     l2_cb_scale [n_sub*l2_K] fp16
 *     l2_indices [M*n_chunks*n_sub] u8
 */

typedef struct {
    void *blocks[8];
    int n;
} owned_list_t;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) { perror("malloc"); exit(1); }
    return p;
}

int pqv2_load(const char *path, pqv2_t *out, void **owned_p) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PQV2", 4) != 0) {
        fprintf(stderr, "bad magic\n"); fclose(f); return -1;
    }
    uint32_t hdr[8];
    if (fread(hdr, 4, 8, f) != 8) { fclose(f); return -1; }
    out->M = hdr[0]; out->N = hdr[1]; out->G = hdr[2]; out->K = hdr[3];
    out->n_subchunks = hdr[4]; out->half = hdr[5];
    out->l2_kind = hdr[6]; out->l2_K = hdr[7];

    size_t n_chunks = out->N / out->G;
    size_t row_bytes = (size_t)out->M * 2;
    size_t cb_q_bytes = (size_t)out->n_subchunks * out->K * out->half;
    size_t cb_s_bytes = (size_t)out->n_subchunks * out->K * 2;
    size_t idx_bytes = (size_t)out->M * n_chunks * out->n_subchunks;

    owned_list_t *L = xmalloc(sizeof(*L));
    L->n = 0;

    void *rs = xmalloc(row_bytes);  L->blocks[L->n++] = rs;
    void *cbq = xmalloc(cb_q_bytes); L->blocks[L->n++] = cbq;
    void *cbs = xmalloc(cb_s_bytes); L->blocks[L->n++] = cbs;
    void *idx = xmalloc(idx_bytes);  L->blocks[L->n++] = idx;

    if (fread(rs, 1, row_bytes, f) != row_bytes ||
        fread(cbq, 1, cb_q_bytes, f) != cb_q_bytes ||
        fread(cbs, 1, cb_s_bytes, f) != cb_s_bytes ||
        fread(idx, 1, idx_bytes, f) != idx_bytes) {
        fprintf(stderr, "short read L1\n"); fclose(f); return -1;
    }
    out->row_scale = rs; out->cb_q = cbq; out->cb_scale = cbs; out->indices = idx;
    out->l2_cb_q = NULL; out->l2_cb_scale = NULL; out->l2_indices = NULL;

    if (out->l2_kind == 2) {
        size_t l2q_bytes = (size_t)out->n_subchunks * out->l2_K * out->half;
        size_t l2s_bytes = (size_t)out->n_subchunks * out->l2_K * 2;
        void *l2q = xmalloc(l2q_bytes); L->blocks[L->n++] = l2q;
        void *l2s = xmalloc(l2s_bytes); L->blocks[L->n++] = l2s;
        void *l2i = xmalloc(idx_bytes); L->blocks[L->n++] = l2i;
        if (fread(l2q, 1, l2q_bytes, f) != l2q_bytes ||
            fread(l2s, 1, l2s_bytes, f) != l2s_bytes ||
            fread(l2i, 1, idx_bytes, f) != idx_bytes) {
            fprintf(stderr, "short read L2\n"); fclose(f); return -1;
        }
        out->l2_cb_q = l2q; out->l2_cb_scale = l2s; out->l2_indices = l2i;
    }
    fclose(f);
    *owned_p = L;
    return 0;
}

void pqv2_free(void *owned) {
    owned_list_t *L = owned;
    for (int i = 0; i < L->n; i++) free(L->blocks[i]);
    free(L);
}

/* ── matvec scalar ─────────────────────────────────────────────────
 * y[m] = sum over c, s, h of cb_recon[s, idx[m,c,s], h] * x[c*G + s*half + h]
 *        all multiplied by row_scale[m], + L2 residual same shape.
 */
void pqv2_matvec_scalar(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;

    /* Pre-decode codewords to fp32 for clarity. cb_fp32[s][k][h] */
    float *cb_fp32 = malloc((size_t)ns * K * half * sizeof(float));
    for (uint32_t s = 0; s < ns; s++) {
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(t->cb_scale[s * K + k]);
            const int8_t *q = &t->cb_q[(s * K + k) * half];
            float *o = &cb_fp32[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++) o[h] = (float)q[h] * sc;
        }
    }
    float *l2_cb = NULL;
    if (t->l2_kind == 2) {
        l2_cb = malloc((size_t)ns * t->l2_K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++) {
            for (uint32_t k = 0; k < t->l2_K; k++) {
                float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                float *o = &l2_cb[(s * t->l2_K + k) * half];
                for (uint32_t h = 0; h < half; h++) o[h] = (float)q[h] * sc;
            }
        }
    }

    for (uint32_t m = 0; m < M; m++) {
        float acc_l1 = 0.0f, acc_l2 = 0.0f;
        for (uint32_t c = 0; c < n_chunks; c++) {
            for (uint32_t s = 0; s < ns; s++) {
                uint8_t k = t->indices[((size_t)c * ns + s) * M + m];
                const float *cw = &cb_fp32[(s * K + k) * half];
                const float *xs = &x[c * G + s * half];
                float d = 0.0f;
                for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                acc_l1 += d;
                if (t->l2_indices) {
                    uint8_t k2 = t->l2_indices[((size_t)c * ns + s) * M + m];
                    const float *cw2 = &l2_cb[(s * t->l2_K + k2) * half];
                    float d2 = 0.0f;
                    for (uint32_t h = 0; h < half; h++) d2 += cw2[h] * xs[h];
                    acc_l2 += d2;
                }
            }
        }
        y[m] = acc_l1 * pqv2_h2f(t->row_scale[m]) + acc_l2;
    }
    free(cb_fp32);
    if (l2_cb) free(l2_cb);
}

/* ── matvec LUT (K=256, any half, any l2_kind ∈ {0,2}) ─────────────
 * For each (chunk c, sub-chunk s):
 *   lut[k] = dot(cb[s][k][:half], x[c*G + s*half:][:half])     for k=0..K-1
 *   for each row m: acc[m] += lut[indices[m,c,s]]
 * Then y[m] = acc[m] * row_scale[m] + l2_acc[m].
 */
void pqv2_matvec_lut(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;

    float *acc_l1 = calloc(M, sizeof(float));
    float *acc_l2 = (t->l2_kind == 2) ? calloc(M, sizeof(float)) : NULL;
    /* Pre-decoded codebooks fp32 (K * half) per sub-chunk */
    float *cb = malloc((size_t)ns * K * half * sizeof(float));
    for (uint32_t s = 0; s < ns; s++) {
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(t->cb_scale[s * K + k]);
            const int8_t *q = &t->cb_q[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++)
                cb[(s * K + k) * half + h] = (float)q[h] * sc;
        }
    }
    float *l2_cb = NULL;
    if (t->l2_kind == 2) {
        l2_cb = malloc((size_t)ns * t->l2_K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++) {
            for (uint32_t k = 0; k < t->l2_K; k++) {
                float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    l2_cb[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
            }
        }
    }

    float *lut = malloc((size_t)K * sizeof(float));
    float *l2_lut = (t->l2_kind == 2) ? malloc((size_t)t->l2_K * sizeof(float)) : NULL;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            /* Build L1 LUT */
            for (uint32_t k = 0; k < K; k++) {
                const float *cw = &cb[(s * K + k) * half];
                float d = 0.0f;
                for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                lut[k] = d;
            }
            /* Gather rows — contiguous across M.
             * Layout assumed: indices[c, s, m] (transposed from python). */
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            for (uint32_t m = 0; m < M; m++)
                acc_l1[m] += lut[idx[m]];

            if (t->l2_kind == 2) {
                for (uint32_t k = 0; k < t->l2_K; k++) {
                    const float *cw = &l2_cb[(s * t->l2_K + k) * half];
                    float d = 0.0f;
                    for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                    l2_lut[k] = d;
                }
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                for (uint32_t m = 0; m < M; m++)
                    acc_l2[m] += l2_lut[l2_idx[m]];
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc_l1[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    free(cb); if (l2_cb) free(l2_cb);
    free(lut); if (l2_lut) free(l2_lut);
    free(acc_l1); if (acc_l2) free(acc_l2);
}

/* ── NEON variant ─────────────────────────────────────────────────── */
#if defined(__ARM_NEON)
#include <arm_neon.h>

void pqv2_matvec_lut_neon(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;

    float *acc_l1 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc_l1, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    float *cb = malloc((size_t)ns * K * half * sizeof(float));
    for (uint32_t s = 0; s < ns; s++)
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(t->cb_scale[s * K + k]);
            const int8_t *q = &t->cb_q[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++)
                cb[(s * K + k) * half + h] = (float)q[h] * sc;
        }
    float *l2_cb = NULL;
    if (t->l2_kind == 2) {
        l2_cb = malloc((size_t)ns * t->l2_K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < t->l2_K; k++) {
                float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    l2_cb[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
            }
    }
    float *lut = aligned_alloc(64, ((size_t)K * sizeof(float) + 63) & ~63);
    float *l2_lut = NULL;
    if (t->l2_kind == 2)
        l2_lut = aligned_alloc(64, ((size_t)t->l2_K * sizeof(float) + 63) & ~63);

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            for (uint32_t k = 0; k < K; k++) {
                const float *cw = &cb[(s * K + k) * half];
                float d = 0.0f;
                for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                lut[k] = d;
            }
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            uint32_t m = 0;
            /* Process 16 rows at a time: gather 16 indices, look up, add. */
            for (; m + 16 <= M; m += 16) {
                if (m + 128 < M) __builtin_prefetch(&idx[m + 128]);
                /* No NEON gather instruction in baseline NEON; do scalar
                 * gather, but issue 16 in parallel to hide latency. */
                float g[16];
                #pragma clang loop unroll(full)
                for (int i = 0; i < 16; i++) g[i] = lut[idx[m + i]];
                float32x4_t a0 = vld1q_f32(&acc_l1[m + 0]);
                float32x4_t a1 = vld1q_f32(&acc_l1[m + 4]);
                float32x4_t a2 = vld1q_f32(&acc_l1[m + 8]);
                float32x4_t a3 = vld1q_f32(&acc_l1[m + 12]);
                float32x4_t v0 = vld1q_f32(&g[0]);
                float32x4_t v1 = vld1q_f32(&g[4]);
                float32x4_t v2 = vld1q_f32(&g[8]);
                float32x4_t v3 = vld1q_f32(&g[12]);
                vst1q_f32(&acc_l1[m + 0],  vaddq_f32(a0, v0));
                vst1q_f32(&acc_l1[m + 4],  vaddq_f32(a1, v1));
                vst1q_f32(&acc_l1[m + 8],  vaddq_f32(a2, v2));
                vst1q_f32(&acc_l1[m + 12], vaddq_f32(a3, v3));
            }
            for (; m < M; m++) acc_l1[m] += lut[idx[m]];

            if (t->l2_kind == 2) {
                for (uint32_t k = 0; k < t->l2_K; k++) {
                    const float *cw = &l2_cb[(s * t->l2_K + k) * half];
                    float d = 0.0f;
                    for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                    l2_lut[k] = d;
                }
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                for (uint32_t mm = 0; mm < M; mm++)
                    acc_l2[mm] += l2_lut[l2_idx[mm]];
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc_l1[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    free(cb); if (l2_cb) free(l2_cb);
    free(lut); if (l2_lut) free(l2_lut);
    free(acc_l1); if (acc_l2) free(acc_l2);
}
/* ── INT8-TBL NEON kernel ────────────────────────────────────────── */
/* Per (c, s):
 *   1. Build fp32 LUT[K] = cb[k]·x_slice  (K ≤ 64).
 *   2. lut_scale = max|LUT| / 127; lut_q[k] = round(LUT/lut_scale) int8.
 *   3. Pack lut_q as 4× int8x16 vectors = 64-byte table.
 *   4. For m in 0..M step 16: gather via vqtbl4q_s8(table, idx[m..]).
 *      Convert int8 → fp32, FMA into acc[m] with lut_scale.
 * L2 (if present) is handled the same way against acc_l2.
 */
static inline void build_lut_int8(const float *cb, const float *xs,
                                   uint32_t K, uint32_t half,
                                   int8_t *lut_q, float *lut_scale) {
    float am = 1e-12f;
    float lut_f[64];
    for (uint32_t k = 0; k < K; k++) {
        float d = 0.0f;
        const float *cw = &cb[k * half];
        for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
        lut_f[k] = d;
        float a = d < 0 ? -d : d;
        if (a > am) am = a;
    }
    float sc = am / 127.0f;
    float inv = 1.0f / sc;
    for (uint32_t k = 0; k < K; k++) {
        int v = (int)lrintf(lut_f[k] * inv);
        if (v > 127) v = 127; if (v < -128) v = -128;
        lut_q[k] = (int8_t)v;
    }
    /* Zero-pad up to 64 so vqtbl4q gather of any 0..63 index is safe */
    for (uint32_t k = K; k < 64; k++) lut_q[k] = 0;
    *lut_scale = sc;
}

void pqv2_matvec_tbl_int8(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;

    float *acc_l1 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc_l1, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    float *cb = malloc((size_t)ns * K * half * sizeof(float));
    for (uint32_t s = 0; s < ns; s++)
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(t->cb_scale[s * K + k]);
            const int8_t *q = &t->cb_q[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++)
                cb[(s * K + k) * half + h] = (float)q[h] * sc;
        }
    float *l2_cb = NULL;
    if (t->l2_kind == 2) {
        l2_cb = malloc((size_t)ns * t->l2_K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < t->l2_K; k++) {
                float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    l2_cb[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
            }
    }

    int8_t lut_q[64] __attribute__((aligned(16)));
    int8_t l2_lut_q[64] __attribute__((aligned(16)));
    float lut_scale, l2_lut_scale;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8(&cb[(size_t)s * K * half], xs, K, half, lut_q, &lut_scale);
#if defined(__ARM_NEON)
            int8x16x4_t tbl;
            tbl.val[0] = vld1q_s8(&lut_q[0]);
            tbl.val[1] = vld1q_s8(&lut_q[16]);
            tbl.val[2] = vld1q_s8(&lut_q[32]);
            tbl.val[3] = vld1q_s8(&lut_q[48]);
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            uint32_t m = 0;
            for (; m + 16 <= M; m += 16) {
                uint8x16_t i16 = vld1q_u8(&idx[m]);
                int8x16_t g = vqtbl4q_s8(tbl, i16);
                int16x8_t lo = vmovl_s8(vget_low_s8(g));
                int16x8_t hi = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
                vst1q_f32(&acc_l1[m+ 0], vfmaq_f32(vld1q_f32(&acc_l1[m+ 0]), f0, scl));
                vst1q_f32(&acc_l1[m+ 4], vfmaq_f32(vld1q_f32(&acc_l1[m+ 4]), f1, scl));
                vst1q_f32(&acc_l1[m+ 8], vfmaq_f32(vld1q_f32(&acc_l1[m+ 8]), f2, scl));
                vst1q_f32(&acc_l1[m+12], vfmaq_f32(vld1q_f32(&acc_l1[m+12]), f3, scl));
            }
            for (; m < M; m++)
                acc_l1[m] += (float)lut_q[idx[m]] * lut_scale;
#else
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            for (uint32_t m = 0; m < M; m++)
                acc_l1[m] += (float)lut_q[idx[m]] * lut_scale;
#endif
            if (t->l2_kind == 2) {
                build_lut_int8(&l2_cb[(size_t)s * t->l2_K * half], xs,
                                t->l2_K, half, l2_lut_q, &l2_lut_scale);
#if defined(__ARM_NEON)
                int8x16x4_t tbl2;
                tbl2.val[0] = vld1q_s8(&l2_lut_q[0]);
                tbl2.val[1] = vld1q_s8(&l2_lut_q[16]);
                tbl2.val[2] = vld1q_s8(&l2_lut_q[32]);
                tbl2.val[3] = vld1q_s8(&l2_lut_q[48]);
                float32x4_t scl2 = vdupq_n_f32(l2_lut_scale);
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                uint32_t m = 0;
                for (; m + 16 <= M; m += 16) {
                    uint8x16_t i16 = vld1q_u8(&l2_idx[m]);
                    int8x16_t g = vqtbl4q_s8(tbl2, i16);
                    int16x8_t lo = vmovl_s8(vget_low_s8(g));
                    int16x8_t hi = vmovl_s8(vget_high_s8(g));
                    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
                    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo)));
                    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
                    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)));
                    vst1q_f32(&acc_l2[m+ 0], vfmaq_f32(vld1q_f32(&acc_l2[m+ 0]), f0, scl2));
                    vst1q_f32(&acc_l2[m+ 4], vfmaq_f32(vld1q_f32(&acc_l2[m+ 4]), f1, scl2));
                    vst1q_f32(&acc_l2[m+ 8], vfmaq_f32(vld1q_f32(&acc_l2[m+ 8]), f2, scl2));
                    vst1q_f32(&acc_l2[m+12], vfmaq_f32(vld1q_f32(&acc_l2[m+12]), f3, scl2));
                }
                for (; m < M; m++)
                    acc_l2[m] += (float)l2_lut_q[l2_idx[m]] * l2_lut_scale;
#else
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                for (uint32_t m = 0; m < M; m++)
                    acc_l2[m] += (float)l2_lut_q[l2_idx[m]] * l2_lut_scale;
#endif
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc_l1[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    free(cb); if (l2_cb) free(l2_cb);
    free(acc_l1); if (acc_l2) free(acc_l2);
}

/* ── INT8-TBL K=128 (2-bank) ──────────────────────────────────────
 * Two banks of 64 int8 entries. High bit of index selects bank.
 * vqtbl4q_s8 with idx > 63 returns 0, so we run TBL twice (one per
 * bank) with the index masked to 6 bits, then bank-select via vbsl.
 */
static inline void build_lut_int8_k128(const float *cb, const float *xs,
                                        uint32_t half,
                                        int8_t *lut_lo, int8_t *lut_hi,
                                        float *lut_scale) {
    float lut_f[128];
    float am = 1e-12f;
    for (uint32_t k = 0; k < 128; k++) {
        float d = 0.0f;
        const float *cw = &cb[k * half];
        for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
        lut_f[k] = d;
        float a = d < 0 ? -d : d;
        if (a > am) am = a;
    }
    float sc = am / 127.0f;
    float inv = 1.0f / sc;
    for (uint32_t k = 0; k < 64; k++) {
        int v = (int)lrintf(lut_f[k] * inv);
        if (v > 127) v = 127; if (v < -128) v = -128;
        lut_lo[k] = (int8_t)v;
    }
    for (uint32_t k = 0; k < 64; k++) {
        int v = (int)lrintf(lut_f[k + 64] * inv);
        if (v > 127) v = 127; if (v < -128) v = -128;
        lut_hi[k] = (int8_t)v;
    }
    *lut_scale = sc;
}

void pqv2_matvec_tbl_int8_k128(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;
    if (K != 128) { pqv2_matvec_lut(t, x, y); return; }

    float *acc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2 && t->l2_K <= 64) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    float *cb = malloc((size_t)ns * K * half * sizeof(float));
    for (uint32_t s = 0; s < ns; s++)
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(t->cb_scale[s * K + k]);
            const int8_t *q = &t->cb_q[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++)
                cb[(s * K + k) * half + h] = (float)q[h] * sc;
        }
    /* L2 codebooks (PQ, K_L2 ≤ 64) */
    float *l2_cb = NULL;
    if (acc_l2) {
        l2_cb = malloc((size_t)ns * t->l2_K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < t->l2_K; k++) {
                float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    l2_cb[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
            }
    }
    int8_t lut_lo[64] __attribute__((aligned(16)));
    int8_t lut_hi[64] __attribute__((aligned(16)));
    int8_t l2_lut_q[64] __attribute__((aligned(16)));
    float lut_scale, l2_lut_scale;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8_k128(&cb[(size_t)s * K * half], xs, half,
                                  lut_lo, lut_hi, &lut_scale);
#if defined(__ARM_NEON)
            int8x16x4_t tbl_lo, tbl_hi;
            tbl_lo.val[0] = vld1q_s8(&lut_lo[0]);
            tbl_lo.val[1] = vld1q_s8(&lut_lo[16]);
            tbl_lo.val[2] = vld1q_s8(&lut_lo[32]);
            tbl_lo.val[3] = vld1q_s8(&lut_lo[48]);
            tbl_hi.val[0] = vld1q_s8(&lut_hi[0]);
            tbl_hi.val[1] = vld1q_s8(&lut_hi[16]);
            tbl_hi.val[2] = vld1q_s8(&lut_hi[32]);
            tbl_hi.val[3] = vld1q_s8(&lut_hi[48]);
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            uint32_t m = 0;
            const uint8x16_t mask63 = vdupq_n_u8(63);
            const uint8x16_t bank_bit = vdupq_n_u8(64);
            for (; m + 16 <= M; m += 16) {
                uint8x16_t i16 = vld1q_u8(&idx[m]);
                uint8x16_t i6 = vandq_u8(i16, mask63);
                int8x16_t glo = vqtbl4q_s8(tbl_lo, i6);
                int8x16_t ghi = vqtbl4q_s8(tbl_hi, i6);
                /* Select hi if (idx & 64) != 0 */
                uint8x16_t sel = vceqq_u8(vandq_u8(i16, bank_bit), bank_bit);
                int8x16_t g = vbslq_s8(sel, ghi, glo);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(&acc[m+ 0], vfmaq_f32(vld1q_f32(&acc[m+ 0]), f0, scl));
                vst1q_f32(&acc[m+ 4], vfmaq_f32(vld1q_f32(&acc[m+ 4]), f1, scl));
                vst1q_f32(&acc[m+ 8], vfmaq_f32(vld1q_f32(&acc[m+ 8]), f2, scl));
                vst1q_f32(&acc[m+12], vfmaq_f32(vld1q_f32(&acc[m+12]), f3, scl));
            }
            for (; m < M; m++) {
                uint8_t k = idx[m];
                int8_t v = (k & 64) ? lut_hi[k & 63] : lut_lo[k & 63];
                acc[m] += (float)v * lut_scale;
            }
            /* L2 path */
            if (acc_l2) {
                build_lut_int8(&l2_cb[(size_t)s * t->l2_K * half], xs,
                                t->l2_K, half, l2_lut_q, &l2_lut_scale);
                int8x16x4_t tbl2;
                tbl2.val[0] = vld1q_s8(&l2_lut_q[0]);
                tbl2.val[1] = vld1q_s8(&l2_lut_q[16]);
                tbl2.val[2] = vld1q_s8(&l2_lut_q[32]);
                tbl2.val[3] = vld1q_s8(&l2_lut_q[48]);
                float32x4_t scl2 = vdupq_n_f32(l2_lut_scale);
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                uint32_t mm = 0;
                for (; mm + 16 <= M; mm += 16) {
                    uint8x16_t i16 = vld1q_u8(&l2_idx[mm]);
                    int8x16_t g = vqtbl4q_s8(tbl2, i16);
                    int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                    vst1q_f32(&acc_l2[mm+ 0], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 0]), f0, scl2));
                    vst1q_f32(&acc_l2[mm+ 4], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 4]), f1, scl2));
                    vst1q_f32(&acc_l2[mm+ 8], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 8]), f2, scl2));
                    vst1q_f32(&acc_l2[mm+12], vfmaq_f32(vld1q_f32(&acc_l2[mm+12]), f3, scl2));
                }
                for (; mm < M; mm++)
                    acc_l2[mm] += (float)l2_lut_q[l2_idx[mm]] * l2_lut_scale;
            }
#endif
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    free(cb); if (l2_cb) free(l2_cb);
    free(acc); if (acc_l2) free(acc_l2);
}

/* ── INT8-TBL K=256 (4-bank) ──────────────────────────────────────
 * Four banks of 64 int8 entries. Top 2 bits of index select bank,
 * low 6 bits select within bank. Each row block fires 4 vqtbl4q
 * gathers and selects via a 2-step vbsl cascade.
 */
static inline void build_lut_int8_k256(const float *cb, const float *xs,
                                        uint32_t half,
                                        int8_t lut[4][64], float *lut_scale) {
    float lut_f[256] __attribute__((aligned(16)));
#if defined(__ARM_NEON)
    if (half == 2) {
        float32x4_t x0 = vdupq_n_f32(xs[0]);
        float32x4_t x1 = vdupq_n_f32(xs[1]);
        float32x4_t am4 = vdupq_n_f32(1e-12f);
        for (uint32_t k = 0; k < 256; k += 8) {
            /* cb is interleaved as [k0_h0, k0_h1, k1_h0, k1_h1, ...] */
            float32x4x2_t a = vld2q_f32(&cb[(k + 0) * 2]);
            float32x4x2_t b = vld2q_f32(&cb[(k + 4) * 2]);
            float32x4_t la = vmulq_f32(a.val[0], x0);
            la = vfmaq_f32(la, a.val[1], x1);
            float32x4_t lb = vmulq_f32(b.val[0], x0);
            lb = vfmaq_f32(lb, b.val[1], x1);
            vst1q_f32(&lut_f[k + 0], la);
            vst1q_f32(&lut_f[k + 4], lb);
            am4 = vmaxq_f32(am4, vabsq_f32(la));
            am4 = vmaxq_f32(am4, vabsq_f32(lb));
        }
        float am = vmaxvq_f32(am4);
        float sc = am / 127.0f;
        float inv = 1.0f / sc;
        float32x4_t inv4 = vdupq_n_f32(inv);
        for (uint32_t b = 0; b < 4; b++) {
            for (uint32_t k = 0; k < 64; k += 16) {
                float32x4_t f0 = vld1q_f32(&lut_f[b * 64 + k + 0]);
                float32x4_t f1 = vld1q_f32(&lut_f[b * 64 + k + 4]);
                float32x4_t f2 = vld1q_f32(&lut_f[b * 64 + k + 8]);
                float32x4_t f3 = vld1q_f32(&lut_f[b * 64 + k + 12]);
                int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(f0, inv4));
                int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(f1, inv4));
                int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(f2, inv4));
                int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(f3, inv4));
                int16x8_t s01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                int16x8_t s23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                int8x16_t s8 = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                vst1q_s8(&lut[b][k], s8);
            }
        }
        *lut_scale = sc;
        return;
    }
#endif
    float am = 1e-12f;
    for (uint32_t k = 0; k < 256; k++) {
        float d = 0.0f;
        const float *cw = &cb[k * half];
        for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
        lut_f[k] = d;
        float a = d < 0 ? -d : d;
        if (a > am) am = a;
    }
    float sc = am / 127.0f;
    float inv = 1.0f / sc;
    for (uint32_t b = 0; b < 4; b++) {
        for (uint32_t k = 0; k < 64; k++) {
            int v = (int)lrintf(lut_f[b * 64 + k] * inv);
            if (v > 127) v = 127; if (v < -128) v = -128;
            lut[b][k] = (int8_t)v;
        }
    }
    *lut_scale = sc;
}

/* Chunk-range accumulator (K=256). Pure accumulation into caller-owned
 * acc/acc_l2 over chunks [c_start, c_end). Caller must zero acc/acc_l2
 * before the first call. cb and l2_cb must be precomputed fp32 codebooks
 * (caller passes t->cb_fp32 / t->l2_cb_fp32 directly). */
/* Internal: chunk-range accumulator with optional activation-skip.
 * skip_thresh == 0  → no skip check, identical to original behavior.
 * skip_thresh >  0  → bypass (c,s) iters with max|x_slice| < skip_thresh. */
static void pqv2_acc_tbl_int8_k256_chunks_inner(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end,
    float skip_thresh)
{
    uint32_t M = t->M, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t G = t->G;
    if (K != 256) return;
    int8_t lut[4][64] __attribute__((aligned(16)));
    int8_t l2_lut_q[64] __attribute__((aligned(16)));
    float lut_scale, l2_lut_scale;

    for (uint32_t c = c_start; c < c_end; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            if (skip_thresh > 0.0f) {
                float xm = 0.0f;
                for (uint32_t h = 0; h < half; h++) {
                    float v = xs[h]; if (v < 0) v = -v;
                    if (v > xm) xm = v;
                }
                if (xm < skip_thresh) continue;
            }
            build_lut_int8_k256(&cb[(size_t)s * K * half], xs, half,
                                  lut, &lut_scale);
#if defined(__ARM_NEON)
            int8x16x4_t b0, b1, b2, b3;
            #define LOAD_BANK(B, ARR) \
                B.val[0] = vld1q_s8(&ARR[0]); B.val[1] = vld1q_s8(&ARR[16]); \
                B.val[2] = vld1q_s8(&ARR[32]); B.val[3] = vld1q_s8(&ARR[48]);
            LOAD_BANK(b0, lut[0]); LOAD_BANK(b1, lut[1]);
            LOAD_BANK(b2, lut[2]); LOAD_BANK(b3, lut[3]);
            #undef LOAD_BANK
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            const uint8x16_t mask63 = vdupq_n_u8(63);
            const uint8x16_t one_v = vdupq_n_u8(1);
            uint32_t m = 0;
            /* 32-row blocks with prefetch */
            for (; m + 32 <= M; m += 32) {
                if (m + 256 < M) __builtin_prefetch(&idx[m + 256], 0, 0);
                uint8x16_t iA = vld1q_u8(&idx[m]);
                uint8x16_t iB = vld1q_u8(&idx[m + 16]);
                uint8x16_t i6A = vandq_u8(iA, mask63);
                uint8x16_t i6B = vandq_u8(iB, mask63);
                int8x16_t gA0 = vqtbl4q_s8(b0, i6A); int8x16_t gB0 = vqtbl4q_s8(b0, i6B);
                int8x16_t gA1 = vqtbl4q_s8(b1, i6A); int8x16_t gB1 = vqtbl4q_s8(b1, i6B);
                int8x16_t gA2 = vqtbl4q_s8(b2, i6A); int8x16_t gB2 = vqtbl4q_s8(b2, i6B);
                int8x16_t gA3 = vqtbl4q_s8(b3, i6A); int8x16_t gB3 = vqtbl4q_s8(b3, i6B);
                uint8x16_t selA_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(iA, 6), one_v), one_v);
                uint8x16_t selA_msb = vceqq_u8(vshrq_n_u8(iA, 7), one_v);
                uint8x16_t selB_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(iB, 6), one_v), one_v);
                uint8x16_t selB_msb = vceqq_u8(vshrq_n_u8(iB, 7), one_v);
                int8x16_t gA = vbslq_s8(selA_msb, vbslq_s8(selA_lsb, gA3, gA2),
                                                    vbslq_s8(selA_lsb, gA1, gA0));
                int8x16_t gB = vbslq_s8(selB_msb, vbslq_s8(selB_lsb, gB3, gB2),
                                                    vbslq_s8(selB_lsb, gB1, gB0));
                int16x8_t lA = vmovl_s8(vget_low_s8(gA)); int16x8_t hA = vmovl_s8(vget_high_s8(gA));
                int16x8_t lB = vmovl_s8(vget_low_s8(gB)); int16x8_t hB = vmovl_s8(vget_high_s8(gB));
                float32x4_t fA0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lA)));
                float32x4_t fA1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lA)));
                float32x4_t fA2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hA)));
                float32x4_t fA3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hA)));
                float32x4_t fB0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lB)));
                float32x4_t fB1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lB)));
                float32x4_t fB2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hB)));
                float32x4_t fB3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hB)));
                vst1q_f32(&acc[m+ 0], vfmaq_f32(vld1q_f32(&acc[m+ 0]), fA0, scl));
                vst1q_f32(&acc[m+ 4], vfmaq_f32(vld1q_f32(&acc[m+ 4]), fA1, scl));
                vst1q_f32(&acc[m+ 8], vfmaq_f32(vld1q_f32(&acc[m+ 8]), fA2, scl));
                vst1q_f32(&acc[m+12], vfmaq_f32(vld1q_f32(&acc[m+12]), fA3, scl));
                vst1q_f32(&acc[m+16], vfmaq_f32(vld1q_f32(&acc[m+16]), fB0, scl));
                vst1q_f32(&acc[m+20], vfmaq_f32(vld1q_f32(&acc[m+20]), fB1, scl));
                vst1q_f32(&acc[m+24], vfmaq_f32(vld1q_f32(&acc[m+24]), fB2, scl));
                vst1q_f32(&acc[m+28], vfmaq_f32(vld1q_f32(&acc[m+28]), fB3, scl));
            }
            for (; m + 16 <= M; m += 16) {
                uint8x16_t i16 = vld1q_u8(&idx[m]);
                uint8x16_t i6 = vandq_u8(i16, mask63);
                int8x16_t g0 = vqtbl4q_s8(b0, i6);
                int8x16_t g1 = vqtbl4q_s8(b1, i6);
                int8x16_t g2 = vqtbl4q_s8(b2, i6);
                int8x16_t g3 = vqtbl4q_s8(b3, i6);
                uint8x16_t sel_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(i16, 6), one_v), one_v);
                uint8x16_t sel_msb = vceqq_u8(vshrq_n_u8(i16, 7), one_v);
                int8x16_t g = vbslq_s8(sel_msb, vbslq_s8(sel_lsb, g3, g2),
                                                    vbslq_s8(sel_lsb, g1, g0));
                int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(&acc[m+ 0], vfmaq_f32(vld1q_f32(&acc[m+ 0]), f0, scl));
                vst1q_f32(&acc[m+ 4], vfmaq_f32(vld1q_f32(&acc[m+ 4]), f1, scl));
                vst1q_f32(&acc[m+ 8], vfmaq_f32(vld1q_f32(&acc[m+ 8]), f2, scl));
                vst1q_f32(&acc[m+12], vfmaq_f32(vld1q_f32(&acc[m+12]), f3, scl));
            }
            for (; m < M; m++) {
                uint8_t k = idx[m];
                int8_t v = lut[k >> 6][k & 63];
                acc[m] += (float)v * lut_scale;
            }
            /* L2 path (K_L2 ≤ 64) */
            if (acc_l2) {
                build_lut_int8(&l2_cb[(size_t)s * t->l2_K * half], xs,
                                t->l2_K, half, l2_lut_q, &l2_lut_scale);
                int8x16x4_t tbl2;
                tbl2.val[0] = vld1q_s8(&l2_lut_q[0]);
                tbl2.val[1] = vld1q_s8(&l2_lut_q[16]);
                tbl2.val[2] = vld1q_s8(&l2_lut_q[32]);
                tbl2.val[3] = vld1q_s8(&l2_lut_q[48]);
                float32x4_t scl2 = vdupq_n_f32(l2_lut_scale);
                const uint8_t *l2_idx = &t->l2_indices[((size_t)c * ns + s) * M];
                uint32_t mm = 0;
                for (; mm + 16 <= M; mm += 16) {
                    uint8x16_t i16 = vld1q_u8(&l2_idx[mm]);
                    int8x16_t g = vqtbl4q_s8(tbl2, i16);
                    int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                    vst1q_f32(&acc_l2[mm+ 0], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 0]), f0, scl2));
                    vst1q_f32(&acc_l2[mm+ 4], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 4]), f1, scl2));
                    vst1q_f32(&acc_l2[mm+ 8], vfmaq_f32(vld1q_f32(&acc_l2[mm+ 8]), f2, scl2));
                    vst1q_f32(&acc_l2[mm+12], vfmaq_f32(vld1q_f32(&acc_l2[mm+12]), f3, scl2));
                }
                for (; mm < M; mm++)
                    acc_l2[mm] += (float)l2_lut_q[l2_idx[mm]] * l2_lut_scale;
            }
#endif
        }
    }
}

/* Public chunks accumulator (no skip, original API). */
void pqv2_acc_tbl_int8_k256_chunks(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end)
{
    pqv2_acc_tbl_int8_k256_chunks_inner(t, x, cb, l2_cb, acc, acc_l2,
                                          c_start, c_end, 0.0f);
}

/* Public skip-aware chunks accumulator. */
void pqv2_acc_tbl_int8_k256_chunks_skip(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end,
    float skip_thresh)
{
    pqv2_acc_tbl_int8_k256_chunks_inner(t, x, cb, l2_cb, acc, acc_l2,
                                          c_start, c_end, skip_thresh);
}

/* Single-thread K=256 matvec: alloc scratch, accumulate over all chunks,
 * apply row_scale + L2, write y. Threading lives outside the kernel
 * (forward.c invokes pqv2_acc_tbl_int8_k256_chunks per worker). */
void pqv2_matvec_tbl_int8_k256(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / t->G;
    if (K != 256) { pqv2_matvec_lut(t, x, y); return; }

    float *acc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2 && t->l2_K <= 64) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    /* fp32 codebook: precomputed by loader, decode locally if absent. */
    const float *cb;
    float *cb_local = NULL;
    if (t->cb_fp32) {
        cb = t->cb_fp32;
    } else {
        cb_local = malloc((size_t)ns * K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < K; k++) {
                float sc = pqv2_h2f(t->cb_scale[s * K + k]);
                const int8_t *q = &t->cb_q[(s * K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    cb_local[(s * K + k) * half + h] = (float)q[h] * sc;
            }
        cb = cb_local;
    }
    const float *l2_cb = NULL;
    float *l2_cb_local = NULL;
    if (acc_l2) {
        if (t->l2_cb_fp32) {
            l2_cb = t->l2_cb_fp32;
        } else {
            l2_cb_local = malloc((size_t)ns * t->l2_K * half * sizeof(float));
            for (uint32_t s = 0; s < ns; s++)
                for (uint32_t k = 0; k < t->l2_K; k++) {
                    float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                    const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                    for (uint32_t h = 0; h < half; h++)
                        l2_cb_local[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
                }
            l2_cb = l2_cb_local;
        }
    }

    pqv2_acc_tbl_int8_k256_chunks(t, x, cb, l2_cb, acc, acc_l2, 0, n_chunks);

    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    if (cb_local) free(cb_local);
    if (l2_cb_local) free(l2_cb_local);
    free(acc); if (acc_l2) free(acc_l2);
}

#define IB_PQV2_BATCH_MAX 8

/* Batched chunk-range accumulator (K=256). Pure accumulation into
 * caller-owned acc[B*M] over chunks [c_start, c_end). Per-position
 * summation order matches the single-position chunks variant exactly,
 * so spec verify agrees bit-for-bit with single-token decode (when
 * the threading uses the same chunk-to-slot partition).
 *
 * Just calls the single-position chunks function B times; the batched
 * kernel I tried earlier (interleaved B-lane gathers) ran into NEON
 * register pressure for B=4 and lost the win we expected. Per-position
 * sequential calls are simpler and produce identical fp32 output. */
void pqv2_acc_tbl_int8_k256_chunks_batch(
    const pqv2_t *t, const float *x_batch, int B,
    const float *cb,
    float *acc_batch,
    uint32_t c_start, uint32_t c_end)
{
    if (t->K != 256) return;
    for (int b = 0; b < B; b++) {
        const float *xb = x_batch + (size_t)b * t->N;
        float *ab = acc_batch + (size_t)b * t->M;
        pqv2_acc_tbl_int8_k256_chunks(t, xb, cb, NULL, ab, NULL,
                                        c_start, c_end);
    }
}

/* DERISK: K=256 single-position matvec with activation-aware skip.
 * Identical to pqv2_matvec_tbl_int8_k256 but skips (c,s) iterations
 * whose input slice has max|x| below skip_thresh. */
void pqv2_matvec_tbl_int8_k256_skip(
    const pqv2_t *t, const float *x, float *y,
    float skip_thresh, double *out_skip_frac)
{
    if (t->K != 256 || !t->cb_fp32) {
        pqv2_matvec_tbl_int8_k256(t, x, y);
        if (out_skip_frac) *out_skip_frac = 0.0;
        return;
    }
    uint32_t M = t->M, K = 256, ns = t->n_subchunks, half = t->half, G = t->G;
    uint32_t n_chunks = t->N / G;
    const float *cb = t->cb_fp32;
    float *acc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    memset(acc, 0, M * sizeof(float));
    long n_skipped = 0;
    long n_total = (long)n_chunks * (long)ns;
#if defined(__ARM_NEON)
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);
    int8_t lut[4][64] __attribute__((aligned(16)));
    float lut_scale;
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            /* Activation magnitude check — skip if entire input slice
             * is too small to contribute meaningfully. */
            float xm = 0.0f;
            for (uint32_t h = 0; h < half; h++) {
                float v = xs[h]; if (v < 0) v = -v;
                if (v > xm) xm = v;
            }
            if (xm < skip_thresh) { n_skipped++; continue; }
            build_lut_int8_k256(&cb[(size_t)s * K * half], xs, half,
                                  lut, &lut_scale);
            int8x16x4_t bank0, bank1, bank2, bank3;
            bank0.val[0] = vld1q_s8(&lut[0][0]);  bank0.val[1] = vld1q_s8(&lut[0][16]);
            bank0.val[2] = vld1q_s8(&lut[0][32]); bank0.val[3] = vld1q_s8(&lut[0][48]);
            bank1.val[0] = vld1q_s8(&lut[1][0]);  bank1.val[1] = vld1q_s8(&lut[1][16]);
            bank1.val[2] = vld1q_s8(&lut[1][32]); bank1.val[3] = vld1q_s8(&lut[1][48]);
            bank2.val[0] = vld1q_s8(&lut[2][0]);  bank2.val[1] = vld1q_s8(&lut[2][16]);
            bank2.val[2] = vld1q_s8(&lut[2][32]); bank2.val[3] = vld1q_s8(&lut[2][48]);
            bank3.val[0] = vld1q_s8(&lut[3][0]);  bank3.val[1] = vld1q_s8(&lut[3][16]);
            bank3.val[2] = vld1q_s8(&lut[3][32]); bank3.val[3] = vld1q_s8(&lut[3][48]);
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            for (uint32_t m = 0; m + 16 <= M; m += 16) {
                uint8x16_t i16 = vld1q_u8(&idx[m]);
                uint8x16_t i6  = vandq_u8(i16, mask63);
                uint8x16_t sel_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(i16, 6), one_v), one_v);
                uint8x16_t sel_msb = vceqq_u8(vshrq_n_u8(i16, 7), one_v);
                int8x16_t g0 = vqtbl4q_s8(bank0, i6);
                int8x16_t g1 = vqtbl4q_s8(bank1, i6);
                int8x16_t g2 = vqtbl4q_s8(bank2, i6);
                int8x16_t g3 = vqtbl4q_s8(bank3, i6);
                int8x16_t g  = vbslq_s8(sel_msb,
                                          vbslq_s8(sel_lsb, g3, g2),
                                          vbslq_s8(sel_lsb, g1, g0));
                int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(&acc[m+ 0], vfmaq_f32(vld1q_f32(&acc[m+ 0]), f0, scl));
                vst1q_f32(&acc[m+ 4], vfmaq_f32(vld1q_f32(&acc[m+ 4]), f1, scl));
                vst1q_f32(&acc[m+ 8], vfmaq_f32(vld1q_f32(&acc[m+ 8]), f2, scl));
                vst1q_f32(&acc[m+12], vfmaq_f32(vld1q_f32(&acc[m+12]), f3, scl));
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        y[m] = acc[m] * pqv2_h2f(t->row_scale[m]);
    }
#else
    pqv2_matvec_tbl_int8_k256(t, x, y);
#endif
    free(acc);
    if (out_skip_frac) *out_skip_frac = (double)n_skipped / (double)n_total;
}

/* DERISK: K≤64 single-position matvec with activation-aware skip.
 * Wraps pqv2_matvec_tbl_int8 by manually walking (c,s) and skipping. */
void pqv2_matvec_tbl_int8_skip(
    const pqv2_t *t, const float *x, float *y,
    float skip_thresh, double *out_skip_frac)
{
    if (t->K > 64) { pqv2_matvec_tbl_int8(t, x, y); if (out_skip_frac) *out_skip_frac = 0.0; return; }
    /* Lazy implementation: just walk the chunks directly with the same
     * inner kernel as pqv2_matvec_tbl_int8 but with skip. Reuses the
     * cb_fp32 if available. For K≤64 the build_lut + gather is much
     * cheaper than K=256, so the relative speedup from skipping may
     * be smaller. */
    uint32_t M = t->M, K = t->K, ns = t->n_subchunks, half = t->half, G = t->G;
    uint32_t n_chunks = t->N / G;
    const float *cb;
    float *cb_local = NULL;
    if (t->cb_fp32) cb = t->cb_fp32;
    else {
        cb_local = malloc((size_t)ns * K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++)
            for (uint32_t k = 0; k < K; k++) {
                float sc = pqv2_h2f(t->cb_scale[s * K + k]);
                const int8_t *q = &t->cb_q[(s * K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    cb_local[(s * K + k) * half + h] = (float)q[h] * sc;
            }
        cb = cb_local;
    }
    float *acc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~(size_t)63);
    memset(acc, 0, M * sizeof(float));
    long n_skipped = 0, n_total = (long)n_chunks * (long)ns;
#if defined(__ARM_NEON)
    int8_t lut_q[64] __attribute__((aligned(16)));
    float lut_scale;
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            float xm = 0.0f;
            for (uint32_t h = 0; h < half; h++) {
                float v = xs[h]; if (v < 0) v = -v; if (v > xm) xm = v;
            }
            if (xm < skip_thresh) { n_skipped++; continue; }
            build_lut_int8(&cb[(size_t)s * K * half], xs, K, half,
                            lut_q, &lut_scale);
            int8x16x4_t tbl;
            tbl.val[0] = vld1q_s8(&lut_q[0]);
            tbl.val[1] = vld1q_s8(&lut_q[16]);
            tbl.val[2] = vld1q_s8(&lut_q[32]);
            tbl.val[3] = vld1q_s8(&lut_q[48]);
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            for (uint32_t m = 0; m + 16 <= M; m += 16) {
                uint8x16_t i = vld1q_u8(&idx[m]);
                int8x16_t g = vqtbl4q_s8(tbl, i);
                int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                vst1q_f32(&acc[m+ 0], vfmaq_f32(vld1q_f32(&acc[m+ 0]), f0, scl));
                vst1q_f32(&acc[m+ 4], vfmaq_f32(vld1q_f32(&acc[m+ 4]), f1, scl));
                vst1q_f32(&acc[m+ 8], vfmaq_f32(vld1q_f32(&acc[m+ 8]), f2, scl));
                vst1q_f32(&acc[m+12], vfmaq_f32(vld1q_f32(&acc[m+12]), f3, scl));
            }
        }
    }
    for (uint32_t m = 0; m < M; m++)
        y[m] = acc[m] * pqv2_h2f(t->row_scale[m]);
#else
    pqv2_matvec_tbl_int8(t, x, y);
#endif
    free(acc); if (cb_local) free(cb_local);
    if (out_skip_frac) *out_skip_frac = (double)n_skipped / (double)n_total;
}

/* DERISK: fp16-accumulator single-position kernel.
 *
 * Same loop structure as pqv2_matvec_tbl_int8_k256 but with acc as
 * fp16 instead of fp32. The multiply by lut_scale is still fp32 (so
 * we don't lose precision per iteration); only the accumulation is
 * fp16. After the chunk loop, acc is promoted to fp32 and multiplied
 * by row_scale.
 *
 * Memory savings: M halfs (2 bytes) instead of M floats (4 bytes) for
 * acc — read+write per (c,s,m_tile) → halved traffic on the dominant
 * memory stream. */
void pqv2_matvec_tbl_int8_k256_fp16acc(
    const pqv2_t *t, const float *x, float *y)
{
    if (t->K != 256) { pqv2_matvec_tbl_int8_k256(t, x, y); return; }
    uint32_t M = t->M, K = 256, ns = t->n_subchunks, half = t->half, G = t->G;
    uint32_t n_chunks = t->N / G;
    if (M < 16 || (M % 16) != 0 || !t->cb_fp32) {
        pqv2_matvec_tbl_int8_k256(t, x, y);
        return;
    }
    const float *cb = t->cb_fp32;
    /* fp16 acc: 2 bytes per element, half the traffic of fp32 acc. */
    __fp16 *acc = aligned_alloc(64, ((size_t)M * sizeof(__fp16) + 63) & ~(size_t)63);
    memset(acc, 0, M * sizeof(__fp16));
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);
    int8_t lut[4][64] __attribute__((aligned(16)));
    float lut_scale;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8_k256(&cb[(size_t)s * K * half], xs, half,
                                  lut, &lut_scale);
            int8x16x4_t bank0, bank1, bank2, bank3;
            bank0.val[0] = vld1q_s8(&lut[0][0]);  bank0.val[1] = vld1q_s8(&lut[0][16]);
            bank0.val[2] = vld1q_s8(&lut[0][32]); bank0.val[3] = vld1q_s8(&lut[0][48]);
            bank1.val[0] = vld1q_s8(&lut[1][0]);  bank1.val[1] = vld1q_s8(&lut[1][16]);
            bank1.val[2] = vld1q_s8(&lut[1][32]); bank1.val[3] = vld1q_s8(&lut[1][48]);
            bank2.val[0] = vld1q_s8(&lut[2][0]);  bank2.val[1] = vld1q_s8(&lut[2][16]);
            bank2.val[2] = vld1q_s8(&lut[2][32]); bank2.val[3] = vld1q_s8(&lut[2][48]);
            bank3.val[0] = vld1q_s8(&lut[3][0]);  bank3.val[1] = vld1q_s8(&lut[3][16]);
            bank3.val[2] = vld1q_s8(&lut[3][32]); bank3.val[3] = vld1q_s8(&lut[3][48]);
            const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M];
            float32x4_t scl = vdupq_n_f32(lut_scale);
            for (uint32_t m = 0; m + 16 <= M; m += 16) {
                uint8x16_t i16 = vld1q_u8(&idx[m]);
                uint8x16_t i6  = vandq_u8(i16, mask63);
                uint8x16_t sel_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(i16, 6), one_v), one_v);
                uint8x16_t sel_msb = vceqq_u8(vshrq_n_u8(i16, 7), one_v);
                int8x16_t g0 = vqtbl4q_s8(bank0, i6);
                int8x16_t g1 = vqtbl4q_s8(bank1, i6);
                int8x16_t g2 = vqtbl4q_s8(bank2, i6);
                int8x16_t g3 = vqtbl4q_s8(bank3, i6);
                int8x16_t g  = vbslq_s8(sel_msb,
                                          vbslq_s8(sel_lsb, g3, g2),
                                          vbslq_s8(sel_lsb, g1, g0));
                /* int8 → int16 → fp32 (scale) → fp16 narrow → accumulate. */
                int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), scl);
                float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), scl);
                float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), scl);
                float32x4_t f3 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), scl);
                float16x4_t h0 = vcvt_f16_f32(f0);
                float16x4_t h1 = vcvt_f16_f32(f1);
                float16x4_t h2 = vcvt_f16_f32(f2);
                float16x4_t h3 = vcvt_f16_f32(f3);
                float16x8_t hlo = vcombine_f16(h0, h1);
                float16x8_t hhi = vcombine_f16(h2, h3);
                /* RMW fp16 acc — 16 lanes per iter as 2× 8-wide. */
                float16x8_t alo = vld1q_f16(&acc[m + 0]);
                float16x8_t ahi = vld1q_f16(&acc[m + 8]);
                alo = vaddq_f16(alo, hlo);
                ahi = vaddq_f16(ahi, hhi);
                vst1q_f16(&acc[m + 0], alo);
                vst1q_f16(&acc[m + 8], ahi);
            }
        }
    }
    /* Promote fp16 → fp32 and apply row_scale. */
    for (uint32_t m = 0; m < M; m++) {
        float a = (float)acc[m];
        y[m] = a * pqv2_h2f(t->row_scale[m]);
    }
#else
    pqv2_matvec_tbl_int8_k256(t, x, y);
#endif
    free(acc);
}

/* DERISK: GEMM-style B=4 K=256 matvec. Outer loop = row tiles of 16;
 * inner loop = chunks × subchunks × 4 batch positions. Per (c,s):
 * - read indices ONCE (16 bytes) — shared across all 4 positions
 * - build 4 LUTs (one per position)
 * - do 4 NEON gathers + FMAs into 4 separate acc reg sets
 * Question: does index-read amortization actually win against
 * unchanged 4× ALU work? */
void pqv2_matvec_tbl_int8_k256_gemm_b4(
    const pqv2_t *t, const float *x_batch, float *y_batch)
{
    if (t->K != 256) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, 4, y_batch);
        return;
    }
    uint32_t M = t->M, K = 256, ns = t->n_subchunks, half = t->half, G = t->G;
    uint32_t n_chunks = t->N / G;
    if (M < 16 || (M % 16) != 0 || !t->cb_fp32) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, 4, y_batch);
        return;
    }
    const float *cb = t->cb_fp32;
    float *acc = aligned_alloc(64, ((size_t)4 * M * sizeof(float) + 63) & ~(size_t)63);
    memset(acc, 0, (size_t)4 * M * sizeof(float));
#if defined(__ARM_NEON)
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);
    int8_t lut[4][4][64] __attribute__((aligned(16)));
    float lut_scale[4];

    /* Correct loop order: build LUT once per (c,s), then sweep all m
     * tiles for all 4 batch positions. acc lives in memory; the win
     * vs sequential B=4 is that we read indices ONCE per (c,s,m_tile)
     * (shared across all 4 positions) instead of 4× sequentially.
     * Same acc memory traffic as sequential. */
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            for (int b = 0; b < 4; b++) {
                const float *xs = &x_batch[(size_t)b * t->N + c * G + s * half];
                build_lut_int8_k256(&cb[(size_t)s * K * half], xs, half,
                                      lut[b], &lut_scale[b]);
            }
            const uint8_t *idx_base = &t->indices[((size_t)c * ns + s) * M];
            for (uint32_t m = 0; m + 16 <= M; m += 16) {
                /* Read indices ONCE for this 16-row tile. */
                uint8x16_t i16 = vld1q_u8(&idx_base[m]);
                uint8x16_t i6  = vandq_u8(i16, mask63);
                uint8x16_t sel_lsb = vceqq_u8(vandq_u8(vshrq_n_u8(i16, 6), one_v), one_v);
                uint8x16_t sel_msb = vceqq_u8(vshrq_n_u8(i16, 7), one_v);
                /* For each position: load LUT, gather, RMW into acc[b][m..m+15]. */
                for (int b = 0; b < 4; b++) {
                    int8x16x4_t bank0, bank1, bank2, bank3;
                    bank0.val[0] = vld1q_s8(&lut[b][0][0]);  bank0.val[1] = vld1q_s8(&lut[b][0][16]);
                    bank0.val[2] = vld1q_s8(&lut[b][0][32]); bank0.val[3] = vld1q_s8(&lut[b][0][48]);
                    bank1.val[0] = vld1q_s8(&lut[b][1][0]);  bank1.val[1] = vld1q_s8(&lut[b][1][16]);
                    bank1.val[2] = vld1q_s8(&lut[b][1][32]); bank1.val[3] = vld1q_s8(&lut[b][1][48]);
                    bank2.val[0] = vld1q_s8(&lut[b][2][0]);  bank2.val[1] = vld1q_s8(&lut[b][2][16]);
                    bank2.val[2] = vld1q_s8(&lut[b][2][32]); bank2.val[3] = vld1q_s8(&lut[b][2][48]);
                    bank3.val[0] = vld1q_s8(&lut[b][3][0]);  bank3.val[1] = vld1q_s8(&lut[b][3][16]);
                    bank3.val[2] = vld1q_s8(&lut[b][3][32]); bank3.val[3] = vld1q_s8(&lut[b][3][48]);
                    int8x16_t g0 = vqtbl4q_s8(bank0, i6);
                    int8x16_t g1 = vqtbl4q_s8(bank1, i6);
                    int8x16_t g2 = vqtbl4q_s8(bank2, i6);
                    int8x16_t g3 = vqtbl4q_s8(bank3, i6);
                    int8x16_t g  = vbslq_s8(sel_msb,
                                              vbslq_s8(sel_lsb, g3, g2),
                                              vbslq_s8(sel_lsb, g1, g0));
                    int16x8_t lo16 = vmovl_s8(vget_low_s8(g));
                    int16x8_t hi16 = vmovl_s8(vget_high_s8(g));
                    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16)));
                    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16)));
                    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16)));
                    float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16)));
                    float32x4_t scl = vdupq_n_f32(lut_scale[b]);
                    float *ap = acc + (size_t)b * M + m;
                    vst1q_f32(ap+ 0, vfmaq_f32(vld1q_f32(ap+ 0), f0, scl));
                    vst1q_f32(ap+ 4, vfmaq_f32(vld1q_f32(ap+ 4), f1, scl));
                    vst1q_f32(ap+ 8, vfmaq_f32(vld1q_f32(ap+ 8), f2, scl));
                    vst1q_f32(ap+12, vfmaq_f32(vld1q_f32(ap+12), f3, scl));
                }
            }
        }
    }
    /* Apply row_scale: y[b][m] = acc[b][m] * row_scale[m]. */
    for (int b = 0; b < 4; b++) {
        for (uint32_t m = 0; m < M; m++) {
            float rs = pqv2_h2f(t->row_scale[m]);
            y_batch[(size_t)b * M + m] = acc[(size_t)b * M + m] * rs;
        }
    }
#else
    pqv2_matvec_tbl_int8_k256_batch(t, x_batch, 4, y_batch);
#endif
    free(acc);
}

/* Batched K=256 matvec — single-thread reference. Threading lives in
 * forward.c (pqv2_threaded_matvec_k256_batch). */
void pqv2_matvec_tbl_int8_k256_batch(
    const pqv2_t *t, const float *x_batch, int B, float *y_batch)
{
    if (B <= 0) return;
    if (B == 1) { pqv2_matvec_tbl_int8_k256(t, x_batch, y_batch); return; }
    for (int b = 0; b < B; b++) {
        pqv2_matvec_tbl_int8_k256(t, x_batch + (size_t)b * t->N,
                                    y_batch + (size_t)b * t->M);
    }
}

#else
void pqv2_matvec_tbl_int8_k256_skip(
    const pqv2_t *t, const float *x, float *y, float st, double *of)
{ (void)st; pqv2_matvec_tbl_int8_k256(t, x, y); if (of) *of = 0.0; }
void pqv2_matvec_tbl_int8_skip(
    const pqv2_t *t, const float *x, float *y, float st, double *of)
{ (void)st; pqv2_matvec_tbl_int8(t, x, y); if (of) *of = 0.0; }
void pqv2_matvec_tbl_int8_k256_fp16acc(
    const pqv2_t *t, const float *x, float *y)
{
    pqv2_matvec_tbl_int8_k256(t, x, y);
}
void pqv2_matvec_tbl_int8_k256_gemm_b4(
    const pqv2_t *t, const float *x_batch, float *y_batch)
{
    for (int b = 0; b < 4; b++)
        pqv2_matvec_tbl_int8_k256(t, x_batch + (size_t)b * t->N,
                                    y_batch + (size_t)b * t->M);
}
void pqv2_matvec_tbl_int8_k256_batch(
    const pqv2_t *t, const float *x_batch, int B, float *y_batch)
{
    for (int b = 0; b < B; b++)
        pqv2_matvec_tbl_int8_k256(t, x_batch + (size_t)b * t->N,
                                    y_batch + (size_t)b * t->M);
}
void pqv2_acc_tbl_int8_k256_chunks(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end) {
    (void)t; (void)x; (void)cb; (void)l2_cb;
    (void)acc; (void)acc_l2; (void)c_start; (void)c_end;
}
void pqv2_acc_tbl_int8_k256_chunks_skip(
    const pqv2_t *t, const float *x,
    const float *cb, const float *l2_cb,
    float *acc, float *acc_l2,
    uint32_t c_start, uint32_t c_end, float skip_thresh) {
    (void)skip_thresh;
    pqv2_acc_tbl_int8_k256_chunks(t, x, cb, l2_cb, acc, acc_l2, c_start, c_end);
}
/* Non-ARM fallback for the batched chunk accumulator — the ARM variant
 * (inside the __ARM_NEON branch above) had no x86/scalar counterpart,
 * which broke the Linux-x86_64 and macOS-Intel links. Same B-loop over
 * the single-position chunks function as the ARM version. */
void pqv2_acc_tbl_int8_k256_chunks_batch(
    const pqv2_t *t, const float *x_batch, int B,
    const float *cb,
    float *acc_batch,
    uint32_t c_start, uint32_t c_end) {
    if (t->K != 256) return;
    for (int b = 0; b < B; b++) {
        pqv2_acc_tbl_int8_k256_chunks(t, x_batch + (size_t)b * t->N, cb, NULL,
                                      acc_batch + (size_t)b * t->M, NULL,
                                      c_start, c_end);
    }
}
void pqv2_matvec_lut_neon(const pqv2_t *t, const float *x, float *y) {
    pqv2_matvec_lut(t, x, y);
}
void pqv2_matvec_tbl_int8(const pqv2_t *t, const float *x, float *y) {
    pqv2_matvec_lut(t, x, y);
}
void pqv2_matvec_tbl_int8_k128(const pqv2_t *t, const float *x, float *y) {
    pqv2_matvec_lut(t, x, y);
}
void pqv2_matvec_tbl_int8_k256(const pqv2_t *t, const float *x, float *y) {
    pqv2_matvec_lut(t, x, y);
}
#endif
