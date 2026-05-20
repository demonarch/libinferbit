#include "pqv2_kernel.h"
#include "platform.h"   /* ib_clock_gettime(CLOCK_MONOTONIC, ...) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ── PQv2 decode profiling (IB_PQV2_PROFILE) ──────────────────────────
 * Pure instrumentation: when IB_PQV2_PROFILE is set in the environment,
 * the K=256 decode hot path accumulates a wall-clock breakdown of
 *   - LUT-build time  (codebook fp32 -> int8 LUT construction)
 *   - gather time     (uint8-index -> LUT lookup + fp32 accumulate)
 *   - total decode time (whole chunks-inner call)
 * into file-scope counters and prints a breakdown to stderr every
 * IB_PQV2_PROFILE_EVERY chunks-inner calls. ib_pqv2_profile_dump() can
 * also be called explicitly (e.g. at process exit) to print the totals.
 *
 * The env var is read exactly once (cached in g_pqv2_profile). When it
 * is unset, the hot path takes a single predicted-not-taken branch and
 * does nothing else — zero clock reads, zero numerical change. */

#define IB_PQV2_PROFILE_EVERY 200

static int            g_pqv2_profile = -1;   /* -1 = not yet checked */
static double         g_pqv2_t_lut    = 0.0; /* secs in LUT build    */
static double         g_pqv2_t_gather = 0.0; /* secs in gather loops */
static double         g_pqv2_t_total  = 0.0; /* secs in chunks-inner */
static long long      g_pqv2_n_calls  = 0;   /* chunks-inner calls   */
static long long      g_pqv2_n_lut    = 0;   /* LUT builds (c,s)     */

static inline int pqv2_profile_enabled(void) {
    int e = g_pqv2_profile;
    if (e < 0) {
        const char *v = getenv("IB_PQV2_PROFILE");
        e = (v && v[0] && v[0] != '0') ? 1 : 0;
        g_pqv2_profile = e;
    }
    return e;
}

static inline double pqv2_now(void) {
    struct timespec ts;
    ib_clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Print the accumulated PQv2 decode breakdown to stderr. Safe to call
 * even when profiling was never enabled (prints nothing in that case). */
void ib_pqv2_profile_dump(void) {
    if (g_pqv2_profile != 1 || g_pqv2_n_calls == 0) return;
    double tot = g_pqv2_t_total > 0.0 ? g_pqv2_t_total : 1e-300;
    double other = tot - g_pqv2_t_lut - g_pqv2_t_gather;
    fprintf(stderr,
        "[pqv2-profile] calls=%lld lut_builds=%lld | "
        "total=%.3f ms  lut=%.3f ms (%.1f%%)  gather=%.3f ms (%.1f%%)  "
        "other=%.3f ms (%.1f%%)\n",
        g_pqv2_n_calls, g_pqv2_n_lut,
        g_pqv2_t_total * 1e3,
        g_pqv2_t_lut    * 1e3, 100.0 * g_pqv2_t_lut    / tot,
        g_pqv2_t_gather * 1e3, 100.0 * g_pqv2_t_gather / tot,
        other           * 1e3, 100.0 * other           / tot);
}

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

/* ── L2 index bit-pack helpers (Stage 5h.1) ───────────────────────────
 * When l2_idx_bits == 6, the on-disk L2 indices are packed 4-in-3 bytes
 * along the M axis. Each (chunk, subchunk) row contains
 *   ceil(M/4) * 3 bytes
 * laid out (LSB-first):
 *   byte0 = (i0 & 0x3F)            | ((i1 & 0x03) << 6)
 *   byte1 = ((i1 >> 2) & 0x0F)     | ((i2 & 0x0F) << 4)
 *   byte2 = ((i2 >> 4) & 0x03)     | ((i3 & 0x3F) << 2)
 *
 * The kernel hot paths consume contiguous uint8 indices; rather than
 * fork every NEON inner loop, we unpack one (c,s) row at a time into a
 * caller-owned scratch buffer of length M. The unpack itself is two
 * shifts + mask per index, fully amortised by the gather work that
 * follows. */
static inline size_t pqv2_l2_packed_row_bytes(uint32_t M) {
    return ((size_t)M + 3u) / 4u * 3u;
}

/* Goal N36 — per-row byte count for the 4-bit packing (2 indices per
 * byte, low nibble first). l2_K must be ≤ 16. */
static inline size_t pqv2_l2_packed_row_bytes_4bit(uint32_t M) {
    return ((size_t)M + 1u) / 2u;
}

/* Unpack a single packed row (ceil(M/4)*3 bytes) to a uint8[M] buffer. */
static inline void pqv2_l2_unpack_row(const uint8_t *src, uint8_t *dst, uint32_t M) {
    uint32_t m = 0;
    while (m + 4 <= M) {
        uint8_t b0 = src[0], b1 = src[1], b2 = src[2];
        dst[m + 0] = (uint8_t)(b0 & 0x3F);
        dst[m + 1] = (uint8_t)(((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2));
        dst[m + 2] = (uint8_t)(((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4));
        dst[m + 3] = (uint8_t)((b2 >> 2) & 0x3F);
        src += 3;
        m += 4;
    }
    if (m < M) {
        uint8_t b0 = src[0], b1 = src[1], b2 = src[2];
        uint8_t tmp[4];
        tmp[0] = (uint8_t)(b0 & 0x3F);
        tmp[1] = (uint8_t)(((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2));
        tmp[2] = (uint8_t)(((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4));
        tmp[3] = (uint8_t)((b2 >> 2) & 0x3F);
        for (uint32_t k = 0; m < M; m++, k++) dst[m] = tmp[k];
    }
}

/* Goal N36 — unpack a 4-bit packed row (ceil(M/2) bytes) to uint8[M]. */
static inline void pqv2_l2_unpack_4bit(const uint8_t *src, uint8_t *dst, uint32_t M) {
    uint32_t m = 0;
    while (m + 2 <= M) {
        uint8_t b = src[0];
        dst[m + 0] = (uint8_t)(b & 0x0F);
        dst[m + 1] = (uint8_t)((b >> 4) & 0x0F);
        src += 1;
        m += 2;
    }
    if (m < M) {
        uint8_t b = src[0];
        dst[m] = (uint8_t)(b & 0x0F);
    }
}

/* Resolve the L2 index row pointer for chunk c, subchunk s. If the
 * tensor is bit-packed, unpack into `scratch[M]` and return scratch.
 * Otherwise return the in-place row pointer (no copy). */
static inline const uint8_t *pqv2_l2_row(const pqv2_t *t,
                                           uint32_t c, uint32_t s,
                                           uint8_t *scratch) {
    uint32_t M = t->M, ns = t->n_subchunks;
    if (t->l2_idx_bits == 6) {
        size_t row_bytes = pqv2_l2_packed_row_bytes(M);
        const uint8_t *packed = t->l2_indices
            + ((size_t)c * ns + s) * row_bytes;
        pqv2_l2_unpack_row(packed, scratch, M);
        return scratch;
    }
    if (t->l2_idx_bits == 4) {
        size_t row_bytes = pqv2_l2_packed_row_bytes_4bit(M);
        const uint8_t *packed = t->l2_indices
            + ((size_t)c * ns + s) * row_bytes;
        pqv2_l2_unpack_4bit(packed, scratch, M);
        return scratch;
    }
    return &t->l2_indices[((size_t)c * ns + s) * M];
}

/* ── Stage 5g.2 — L1 index layout adapter ─────────────────────────────
 *
 * The CPU NEON kernel inner loops read L1 indices in chunk-major order
 * (one contiguous M-byte run per (c, s) slot — what `vld1q_u8` wants).
 * Pre-5g.2 encoders always wrote chunk-major on disk so `t->indices`
 * already pointed at the NEON-friendly layout.
 *
 * Stage 5g.2 introduces an opt-in row-major on-disk layout so the Metal
 * upload can zero-copy via newBufferWithBytesNoCopy. For CPU paths, we
 * preserve NEON performance by gathering a one-shot chunk-major scratch
 * row of size M bytes inside the (c, s) loop — same pattern as the K=256
 * inner accumulator. Previous Stage 5g.2 revision materialised the entire
 * [n_chunks][n_subchunks][M] transpose up-front (≈ 470 MB for Llama-3-8B)
 * which made every matvec malloc/free that whole shadow. The per-(c, s)
 * gather costs M bytes total and reuses the same buffer across all
 * iterations.
 *
 * Caller pattern at each public matvec entry:
 *   uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
 *       ? (uint8_t *)malloc((size_t)M) : NULL;
 *   uint32_t l1_total = (t->N / t->G) * t->n_subchunks;
 *   ...
 *   for (c) for (s) {
 *       const uint8_t *idx;
 *       if (l1_row_scratch) {
 *           uint32_t off = c * ns + s;
 *           for (uint32_t mm = 0; mm < M; mm++)
 *               l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
 *           idx = l1_row_scratch;
 *       } else {
 *           idx = &t->indices[((size_t)c * ns + s) * M];
 *       }
 *       // ... use idx[m] ...
 *   }
 *   free(l1_row_scratch); */

/* Random-access single-index read for the scalar paths. */
static inline uint8_t pqv2_l2_idx_at(const pqv2_t *t,
                                       uint32_t c, uint32_t s, uint32_t m) {
    uint32_t M = t->M, ns = t->n_subchunks;
    if (t->l2_idx_bits == 6) {
        size_t row_bytes = pqv2_l2_packed_row_bytes(M);
        const uint8_t *row = t->l2_indices
            + ((size_t)c * ns + s) * row_bytes;
        uint32_t group = m >> 2u;
        uint32_t lane  = m & 3u;
        const uint8_t *p = row + (size_t)group * 3u;
        uint8_t b0 = p[0], b1 = p[1], b2 = p[2];
        switch (lane) {
            case 0: return (uint8_t)(b0 & 0x3F);
            case 1: return (uint8_t)(((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2));
            case 2: return (uint8_t)(((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4));
            default: return (uint8_t)((b2 >> 2) & 0x3F);
        }
    }
    if (t->l2_idx_bits == 4) {
        size_t row_bytes = pqv2_l2_packed_row_bytes_4bit(M);
        const uint8_t *row = t->l2_indices
            + ((size_t)c * ns + s) * row_bytes;
        uint8_t b = row[m >> 1u];
        return (uint8_t)((m & 1u) ? ((b >> 4) & 0x0F) : (b & 0x0F));
    }
    return t->l2_indices[((size_t)c * ns + s) * M + m];
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
    memset(out, 0, sizeof(*out));  /* zeros scale_precision / cb_pool_size etc. */
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    /* Use file size to disambiguate the legacy 8-u32 header from the
     * extended 9-u32 header (Stage 5h.1, l2_idx_bits added at hdr[8]). */
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsz_l = ftell(f);
    if (fsz_l < 0) { fclose(f); return -1; }
    size_t fsz = (size_t)fsz_l;
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PQV2", 4) != 0) {
        fprintf(stderr, "bad magic\n"); fclose(f); return -1;
    }
    /* Read 9 u32s up-front; if file is too small, we know the header is
     * the legacy 8-u32 form (hdr[8] doesn't exist). */
    uint32_t hdr[9] = {0};
    int header_u32s = 9;
    if (fsz < 4 + 36) header_u32s = 8;
    if (fread(hdr, 4, (size_t)header_u32s, f) != (size_t)header_u32s) {
        fclose(f); return -1;
    }
    out->M = hdr[0]; out->N = hdr[1]; out->G = hdr[2]; out->K = hdr[3];
    out->n_subchunks = hdr[4]; out->half = hdr[5];
    out->l2_kind = hdr[6]; out->l2_K = hdr[7];

    size_t n_chunks = out->N / out->G;
    size_t row_bytes = (size_t)out->M * 2;
    size_t cb_q_bytes = (size_t)out->n_subchunks * out->K * out->half;
    size_t cb_s_bytes = (size_t)out->n_subchunks * out->K * 2;
    size_t idx_bytes = (size_t)out->M * n_chunks * out->n_subchunks;
    size_t l2_packed_bytes_per_row = ((size_t)out->M + 3u) / 4u * 3u;
    size_t l2_packed_bytes_per_row_4 = ((size_t)out->M + 1u) / 2u;

    /* Decide whether the candidate 9th u32 is really l2_idx_bits by
     * checking the projected total file size against the actual size. */
    int use_extended = 0;
    uint32_t l2_idx_bits = 8;
    if (header_u32s == 9) {
        uint32_t b = hdr[8];
        if ((b == 4 || b == 6 || b == 8) && out->l2_kind == 2) {
            size_t l2q_bytes = (size_t)out->n_subchunks * out->l2_K * out->half;
            size_t l2s_bytes = (size_t)out->n_subchunks * out->l2_K * 2;
            size_t l2_idx_disk = (b == 6)
                ? (size_t)n_chunks * out->n_subchunks * l2_packed_bytes_per_row
                : ((b == 4)
                    ? (size_t)n_chunks * out->n_subchunks * l2_packed_bytes_per_row_4
                    : idx_bytes);
            size_t projected = 4 + 36 + row_bytes + cb_q_bytes + cb_s_bytes
                               + idx_bytes + l2q_bytes + l2s_bytes + l2_idx_disk;
            if (projected == fsz) {
                use_extended = 1;
                l2_idx_bits = b;
            }
        } else if (b == 8 && out->l2_kind == 0) {
            size_t projected = 4 + 36 + row_bytes + cb_q_bytes + cb_s_bytes
                               + idx_bytes;
            if (projected == fsz) {
                use_extended = 1;
                l2_idx_bits = 8;
            }
        }
    }
    out->l2_idx_bits = l2_idx_bits;
    /* Rewind so that the post-header read starts at the right place. */
    if (fseek(f, use_extended ? (long)(4 + 36) : (long)(4 + 32), SEEK_SET)
        != 0) { fclose(f); return -1; }

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
        size_t l2_idx_disk = (out->l2_idx_bits == 6)
            ? (size_t)n_chunks * out->n_subchunks * l2_packed_bytes_per_row
            : ((out->l2_idx_bits == 4)
                ? (size_t)n_chunks * out->n_subchunks * l2_packed_bytes_per_row_4
                : idx_bytes);
        void *l2q = xmalloc(l2q_bytes); L->blocks[L->n++] = l2q;
        void *l2s = xmalloc(l2s_bytes); L->blocks[L->n++] = l2s;
        void *l2i = xmalloc(l2_idx_disk); L->blocks[L->n++] = l2i;
        if (fread(l2q, 1, l2q_bytes, f) != l2q_bytes ||
            fread(l2s, 1, l2s_bytes, f) != l2s_bytes ||
            fread(l2i, 1, l2_idx_disk, f) != l2_idx_disk) {
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
                /* L1 index lookup: branch on on-disk layout. The hot path
                 * here is the scalar reference kernel; perf-sensitive
                 * paths gather a chunk-major row up front (see _lut,
                 * _neon, _tbl_int8 etc). */
                uint8_t k;
                if (t->l1_idx_layout == 1) {
                    uint32_t l1_total = n_chunks * ns;
                    k = t->indices[(size_t)m * l1_total + (c * ns + s)];
                } else {
                    k = t->indices[((size_t)c * ns + s) * M + m];
                }
                const float *cw = &cb_fp32[(s * K + k) * half];
                const float *xs = &x[c * G + s * half];
                float d = 0.0f;
                for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                acc_l1 += d;
                if (t->l2_indices) {
                    uint8_t k2 = pqv2_l2_idx_at(t, c, s, m);
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

    /* Stage 5g.2 — per-(c, s) gather of an M-byte chunk-major scratch row
     * when on-disk layout is row-major. NULL when layout is already
     * chunk-major (we read t->indices directly). */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)malloc((size_t)M) : NULL;
    if (t->l1_idx_layout == 1 && !l1_row_scratch) {
        memset(y, 0, (size_t)M * sizeof(float));
        return;
    }
    uint32_t l1_total = n_chunks * ns;

    float *acc_l1 = calloc(M, sizeof(float));
    float *acc_l2 = (t->l2_kind == 2) ? calloc(M, sizeof(float)) : NULL;
    /* Pre-decoded codebooks fp32: reuse the load-time t->cb_fp32 /
     * t->l2_cb_fp32 when present, else decode locally (fallback). */
    const float *cb;
    float *cb_local = NULL;
    if (t->cb_fp32) {
        cb = t->cb_fp32;
    } else {
        cb_local = malloc((size_t)ns * K * half * sizeof(float));
        for (uint32_t s = 0; s < ns; s++) {
            for (uint32_t k = 0; k < K; k++) {
                float sc = pqv2_h2f(t->cb_scale[s * K + k]);
                const int8_t *q = &t->cb_q[(s * K + k) * half];
                for (uint32_t h = 0; h < half; h++)
                    cb_local[(s * K + k) * half + h] = (float)q[h] * sc;
            }
        }
        cb = cb_local;
    }
    const float *l2_cb = NULL;
    float *l2_cb_local = NULL;
    if (t->l2_kind == 2) {
        if (t->l2_cb_fp32) {
            l2_cb = t->l2_cb_fp32;
        } else {
            l2_cb_local = malloc((size_t)ns * t->l2_K * half * sizeof(float));
            for (uint32_t s = 0; s < ns; s++) {
                for (uint32_t k = 0; k < t->l2_K; k++) {
                    float sc = pqv2_h2f(t->l2_cb_scale[s * t->l2_K + k]);
                    const int8_t *q = &t->l2_cb_q[(s * t->l2_K + k) * half];
                    for (uint32_t h = 0; h < half; h++)
                        l2_cb_local[(s * t->l2_K + k) * half + h] = (float)q[h] * sc;
                }
            }
            l2_cb = l2_cb_local;
        }
    }

    float *lut = malloc((size_t)K * sizeof(float));
    float *l2_lut = (t->l2_kind == 2) ? malloc((size_t)t->l2_K * sizeof(float)) : NULL;
    /* Scratch row for bit-packed L2 unpack (Stage 5h.1). One row at a
     * time = M bytes total; reused across (c, s). */
    uint8_t *l2_scratch = (t->l2_kind == 2 && (t->l2_idx_bits == 6 || t->l2_idx_bits == 4))
        ? (uint8_t *)malloc((size_t)M) : NULL;

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
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++)
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
            for (uint32_t m = 0; m < M; m++)
                acc_l1[m] += lut[idx[m]];

            if (t->l2_kind == 2) {
                for (uint32_t k = 0; k < t->l2_K; k++) {
                    const float *cw = &l2_cb[(s * t->l2_K + k) * half];
                    float d = 0.0f;
                    for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                    l2_lut[k] = d;
                }
                const uint8_t *l2_idx = pqv2_l2_row(t, c, s, l2_scratch);
                for (uint32_t m = 0; m < M; m++)
                    acc_l2[m] += l2_lut[l2_idx[m]];
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc_l1[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    if (cb_local) free(cb_local);
    if (l2_cb_local) free(l2_cb_local);
    free(lut); if (l2_lut) free(l2_lut);
    free(acc_l1); if (acc_l2) free(acc_l2);
    if (l2_scratch) free(l2_scratch);
    if (l1_row_scratch) free(l1_row_scratch);
}

/* ── NEON variant ─────────────────────────────────────────────────── */
#if defined(__ARM_NEON)
#include <arm_neon.h>

void pqv2_matvec_lut_neon(const pqv2_t *t, const float *x, float *y) {
    uint32_t M = t->M, G = t->G, K = t->K, ns = t->n_subchunks, half = t->half;
    uint32_t n_chunks = t->N / G;

    /* Stage 5g.2 — per-(c, s) gather of an M-byte chunk-major scratch row
     * when on-disk layout is row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;
    if (t->l1_idx_layout == 1 && !l1_row_scratch) {
        memset(y, 0, (size_t)M * sizeof(float));
        return;
    }
    uint32_t l1_total = n_chunks * ns;

    float *acc_l1 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc_l1, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    /* Perf: reuse the load-time pre-decoded fp32 codebooks. */
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
    if (t->l2_kind == 2) {
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
    float *lut = aligned_alloc(64, ((size_t)K * sizeof(float) + 63) & ~63);
    float *l2_lut = NULL;
    if (t->l2_kind == 2)
        l2_lut = aligned_alloc(64, ((size_t)t->l2_K * sizeof(float) + 63) & ~63);
    uint8_t *l2_scratch = (t->l2_kind == 2 && (t->l2_idx_bits == 6 || t->l2_idx_bits == 4))
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            for (uint32_t k = 0; k < K; k++) {
                const float *cw = &cb[(s * K + k) * half];
                float d = 0.0f;
                for (uint32_t h = 0; h < half; h++) d += cw[h] * xs[h];
                lut[k] = d;
            }
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++)
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
                const uint8_t *l2_idx = pqv2_l2_row(t, c, s, l2_scratch);
                for (uint32_t mm = 0; mm < M; mm++)
                    acc_l2[mm] += l2_lut[l2_idx[mm]];
            }
        }
    }
    for (uint32_t m = 0; m < M; m++) {
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = acc_l1[m] * rs + (acc_l2 ? acc_l2[m] : 0.0f);
    }
    if (cb_local) free(cb_local);
    if (l2_cb_local) free(l2_cb_local);
    free(lut); if (l2_lut) free(l2_lut);
    free(acc_l1); if (acc_l2) free(acc_l2);
    if (l2_scratch) free(l2_scratch);
    if (l1_row_scratch) free(l1_row_scratch);
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

    /* Stage 5g.2 — per-(c, s) gather of an M-byte chunk-major scratch row
     * when on-disk layout is row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;
    if (t->l1_idx_layout == 1 && !l1_row_scratch) {
        memset(y, 0, (size_t)M * sizeof(float));
        return;
    }
    uint32_t l1_total = n_chunks * ns;

    float *acc_l1 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc_l1, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    /* Perf: codebook fp32 is a STATIC tensor property, pre-decoded at load
     * into t->cb_fp32 / t->l2_cb_fp32. Reuse it and skip the per-call
     * malloc + fp16->fp32 decode (ns*K*half + ns*l2_K*half conversions per
     * matvec). Local decode kept only as a fallback when the cache is NULL. */
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
    if (t->l2_kind == 2) {
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

    int8_t lut_q[64] __attribute__((aligned(16)));
    int8_t l2_lut_q[64] __attribute__((aligned(16)));
    float lut_scale, l2_lut_scale;
    uint8_t *l2_scratch = (t->l2_kind == 2 && (t->l2_idx_bits == 6 || t->l2_idx_bits == 4))
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8(&cb[(size_t)s * K * half], xs, K, half, lut_q, &lut_scale);
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++)
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
#if defined(__ARM_NEON)
            int8x16x4_t tbl;
            tbl.val[0] = vld1q_s8(&lut_q[0]);
            tbl.val[1] = vld1q_s8(&lut_q[16]);
            tbl.val[2] = vld1q_s8(&lut_q[32]);
            tbl.val[3] = vld1q_s8(&lut_q[48]);
            float32x4_t scl = vdupq_n_f32(lut_scale);
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
            for (uint32_t m = 0; m < M; m++)
                acc_l1[m] += (float)lut_q[idx[m]] * lut_scale;
#endif
            if (t->l2_kind == 2) {
                build_lut_int8(&l2_cb[(size_t)s * t->l2_K * half], xs,
                                t->l2_K, half, l2_lut_q, &l2_lut_scale);
                const uint8_t *l2_idx = pqv2_l2_row(t, c, s, l2_scratch);
#if defined(__ARM_NEON)
                int8x16x4_t tbl2;
                tbl2.val[0] = vld1q_s8(&l2_lut_q[0]);
                tbl2.val[1] = vld1q_s8(&l2_lut_q[16]);
                tbl2.val[2] = vld1q_s8(&l2_lut_q[32]);
                tbl2.val[3] = vld1q_s8(&l2_lut_q[48]);
                float32x4_t scl2 = vdupq_n_f32(l2_lut_scale);
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
    if (cb_local) free(cb_local);
    if (l2_cb_local) free(l2_cb_local);
    free(acc_l1); if (acc_l2) free(acc_l2);
    if (l2_scratch) free(l2_scratch);
    if (l1_row_scratch) free(l1_row_scratch);
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

    /* Stage 5g.2 — per-(c, s) gather of an M-byte chunk-major scratch row
     * when on-disk layout is row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;
    if (t->l1_idx_layout == 1 && !l1_row_scratch) {
        memset(y, 0, (size_t)M * sizeof(float));
        return;
    }
    uint32_t l1_total = n_chunks * ns;

    float *acc = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
    memset(acc, 0, M * sizeof(float));
    float *acc_l2 = NULL;
    if (t->l2_kind == 2 && t->l2_K <= 64) {
        acc_l2 = aligned_alloc(64, ((size_t)M * sizeof(float) + 63) & ~63);
        memset(acc_l2, 0, M * sizeof(float));
    }
    /* Perf: reuse the load-time pre-decoded fp32 codebooks (t->cb_fp32 /
     * t->l2_cb_fp32); fall back to a per-call decode only if they're NULL. */
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
    /* L2 codebooks (PQ, K_L2 ≤ 64) */
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
    int8_t lut_lo[64] __attribute__((aligned(16)));
    int8_t lut_hi[64] __attribute__((aligned(16)));
    int8_t l2_lut_q[64] __attribute__((aligned(16)));
    float lut_scale, l2_lut_scale;
    uint8_t *l2_scratch = (acc_l2 && (t->l2_idx_bits == 6 || t->l2_idx_bits == 4))
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;

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
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++)
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
                const uint8_t *l2_idx = pqv2_l2_row(t, c, s, l2_scratch);
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
    if (cb_local) free(cb_local);
    if (l2_cb_local) free(l2_cb_local);
    free(acc); if (acc_l2) free(acc_l2);
    if (l2_scratch) free(l2_scratch);
    if (l1_row_scratch) free(l1_row_scratch);
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
    /* Stage 5h.1 bit-packed L2 indices: unpack each (c,s) row into a
     * stack-style scratch before the NEON vqtbl4q loop, so the loop body
     * is unchanged. Only allocated when the tensor is actually packed. */
    uint8_t *l2_scratch = (acc_l2 && (t->l2_idx_bits == 6 || t->l2_idx_bits == 4))
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;
    /* Stage 5g.2 — when L1 on-disk layout is row-major, gather a
     * chunk-major (size-M) scratch row per (c, s) iter so the NEON
     * inner loop stays unchanged. Allocated once per call; size is
     * tiny (≤ ~14 KB for Llama-7B-class M=14336). Heap (not stack) so
     * very-large M doesn't blow worker stacks. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~63) : NULL;
    uint32_t l1_total = (uint32_t)(t->N / G) * ns;

    /* Profiling: cached single branch; zero overhead when disabled. */
    const int prof = pqv2_profile_enabled();
    double prof_t_lut = 0.0, prof_t_gather = 0.0;
    long long prof_n_lut = 0;
    double prof_call_start = prof ? pqv2_now() : 0.0;

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
            double prof_t0 = prof ? pqv2_now() : 0.0;
            build_lut_int8_k256(&cb[(size_t)s * K * half], xs, half,
                                  lut, &lut_scale);
            double prof_t1 = prof ? pqv2_now() : 0.0;
            if (prof) { prof_t_lut += prof_t1 - prof_t0; prof_n_lut++; }
#if defined(__ARM_NEON)
            int8x16x4_t b0, b1, b2, b3;
            #define LOAD_BANK(B, ARR) \
                B.val[0] = vld1q_s8(&ARR[0]); B.val[1] = vld1q_s8(&ARR[16]); \
                B.val[2] = vld1q_s8(&ARR[32]); B.val[3] = vld1q_s8(&ARR[48]);
            LOAD_BANK(b0, lut[0]); LOAD_BANK(b1, lut[1]);
            LOAD_BANK(b2, lut[2]); LOAD_BANK(b3, lut[3]);
            #undef LOAD_BANK
            float32x4_t scl = vdupq_n_f32(lut_scale);
            const uint8_t *idx;
            if (l1_row_scratch) {
                /* Gather row-major → contiguous chunk-major scratch.
                 * Row-major on disk lays indices as [m * total + c*ns + s];
                 * we want idx[m] for fixed (c, s). */
                uint32_t off = (uint32_t)c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++) {
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                }
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
                double prof_l2_0 = prof ? pqv2_now() : 0.0;
                build_lut_int8(&l2_cb[(size_t)s * t->l2_K * half], xs,
                                t->l2_K, half, l2_lut_q, &l2_lut_scale);
                if (prof) {
                    prof_t_lut += pqv2_now() - prof_l2_0;
                    prof_n_lut++;
                }
                int8x16x4_t tbl2;
                tbl2.val[0] = vld1q_s8(&l2_lut_q[0]);
                tbl2.val[1] = vld1q_s8(&l2_lut_q[16]);
                tbl2.val[2] = vld1q_s8(&l2_lut_q[32]);
                tbl2.val[3] = vld1q_s8(&l2_lut_q[48]);
                float32x4_t scl2 = vdupq_n_f32(l2_lut_scale);
                const uint8_t *l2_idx = pqv2_l2_row(t, c, s, l2_scratch);
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
            /* gather span = (end of iter) - (end of L1 LUT build);
             * any L2 LUT-build time inside it was already added to
             * prof_t_lut, so subtract it back out below at call end. */
            if (prof) prof_t_gather += pqv2_now() - prof_t1;
        }
    }

    if (prof) {
        double prof_call_total = pqv2_now() - prof_call_start;
        /* prof_t_gather currently includes L2 LUT-build time (which is
         * also counted in prof_t_lut for this call). The L2 build time
         * for this call = prof_t_lut accumulated minus the L1 builds...
         * simpler: gather already overlaps lut only via L2. Correct it
         * by treating gather as call_total - lut - (skip/loop overhead).
         * We keep the measured prof_t_gather but clamp so lut+gather
         * never exceeds total. */
        double lut = prof_t_lut;
        double gather = prof_t_gather;
        if (lut + gather > prof_call_total) {
            /* L2-build double-count: rescale gather down. */
            gather = prof_call_total - lut;
            if (gather < 0.0) gather = 0.0;
        }
        g_pqv2_t_lut    += lut;
        g_pqv2_t_gather += gather;
        g_pqv2_t_total  += prof_call_total;
        g_pqv2_n_lut    += prof_n_lut;
        g_pqv2_n_calls  += 1;
        if (g_pqv2_n_calls % IB_PQV2_PROFILE_EVERY == 0)
            ib_pqv2_profile_dump();
    }
    if (l2_scratch) free(l2_scratch);
    if (l1_row_scratch) free(l1_row_scratch);
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
 * caller-owned acc[B*M] (and acc_l2[B*M] when L2 is present) over chunks
 * [c_start, c_end). Per-position summation order matches the
 * single-position chunks variant exactly, so spec verify agrees
 * bit-for-bit with single-token decode (when the threading uses the
 * same chunk-to-slot partition).
 *
 * Just calls the single-position chunks function B times; the batched
 * kernel I tried earlier (interleaved B-lane gathers) ran into NEON
 * register pressure for B=4 and lost the win we expected. Per-position
 * sequential calls are simpler and produce identical fp32 output. The
 * single-position function already handles L2 via the
 * pqv2_acc_tbl_int8_k256_chunks_inner L2 path (NEON-accelerated), so
 * propagating l2_cb / acc_l2_batch here is a straight pass-through. */
void pqv2_acc_tbl_int8_k256_chunks_batch(
    const pqv2_t *t, const float *x_batch, int B,
    const float *cb, const float *l2_cb,
    float *acc_batch, float *acc_l2_batch,
    uint32_t c_start, uint32_t c_end)
{
    if (t->K != 256) return;
    for (int b = 0; b < B; b++) {
        const float *xb = x_batch + (size_t)b * t->N;
        float *ab  = acc_batch    + (size_t)b * t->M;
        float *ab2 = acc_l2_batch ? acc_l2_batch + (size_t)b * t->M : NULL;
        pqv2_acc_tbl_int8_k256_chunks(t, xb, cb, l2_cb, ab, ab2,
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
    /* Stage 5g.2 — per-(c,s) chunk-major scratch when on-disk layout
     * is row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~(size_t)63) : NULL;
    uint32_t l1_total = (uint32_t)n_chunks * ns;
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
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++) {
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                }
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
    if (l1_row_scratch) free(l1_row_scratch);
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
    /* Stage 5g.2 — per-(c,s) chunk-major scratch when on-disk row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~(size_t)63) : NULL;
    uint32_t l1_total = (uint32_t)n_chunks * ns;
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
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++) {
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                }
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
    if (l1_row_scratch) free(l1_row_scratch);
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
    /* Stage 5g.2 — per-(c,s) chunk-major scratch when on-disk row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~(size_t)63) : NULL;
    uint32_t l1_total = (uint32_t)n_chunks * ns;
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
            const uint8_t *idx;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++) {
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                }
                idx = l1_row_scratch;
            } else {
                idx = &t->indices[((size_t)c * ns + s) * M];
            }
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
    if (l1_row_scratch) free(l1_row_scratch);
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
    /* Stage 5g.2 — per-(c,s) chunk-major scratch when on-disk row-major. */
    uint8_t *l1_row_scratch = (t->l1_idx_layout == 1)
        ? (uint8_t *)aligned_alloc(64, ((size_t)M + 63) & ~(size_t)63) : NULL;
    uint32_t l1_total = (uint32_t)n_chunks * ns;
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
            const uint8_t *idx_base;
            if (l1_row_scratch) {
                uint32_t off = c * ns + s;
                for (uint32_t mm = 0; mm < M; mm++) {
                    l1_row_scratch[mm] = t->indices[(size_t)mm * l1_total + off];
                }
                idx_base = l1_row_scratch;
            } else {
                idx_base = &t->indices[((size_t)c * ns + s) * M];
            }
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
    if (l1_row_scratch) free(l1_row_scratch);
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

/* ── Goal N28: fused MoME gate+up kernel (K=256, FLAT) ────────────────
 *
 * Builds the per-(c, s) INT8 LUT ONCE from the shared codebook, then
 * gathers/accumulates against 2K expert outputs in the same pass. The
 * dominant cost in the round-5 MoME path is repeated LUT construction
 * (4 banks × 256 dot-products of length `half`, then a fp32→int8
 * requantise of the 256 entries) — for K=2 experts × {gate, up} that
 * cost is paid 4× when it only needs to be paid once.
 *
 * Per-expert acc arrays live in one big slab `acc[2K * M_per]` so the
 * NEON gather loop body is byte-identical to the single-expert kernel
 * — we just iterate `2K` times per (c, s) over different `idx` / `acc`
 * pointer pairs and a fresh row-major scratch when `l1_idx_layout`
 * demands it. After the (c, s) sweep, each expert's acc is multiplied
 * by its own `row_scale` and written to hb_out[e] / hb2_out[e].
 */
int pqv2_matvec_mome_gateup_k256(
    const pqv2_t * const *gate_experts,
    const pqv2_t * const *up_experts,
    const float *x,
    float *hb_out,
    float *hb2_out,
    int K_experts)
{
    if (K_experts <= 0 || K_experts > IB_MOME_FUSED_MAX_K) return -1;
    if (!gate_experts || !up_experts || !x || !hb_out || !hb2_out) return -1;
    if (!gate_experts[0] || !up_experts[0]) return -1;
    const pqv2_t *t0 = gate_experts[0];
    /* Restrict to K=256 flat path with pre-decoded fp32 codebook. */
    if (t0->K != 256 || !t0->cb_fp32 || t0->l2_kind != 0) return -1;

    const uint32_t M_per = t0->M;
    const uint32_t N     = t0->N;
    const uint32_t G     = t0->G;
    const uint32_t ns    = t0->n_subchunks;
    const uint32_t half  = t0->half;
    const uint32_t K_cb  = t0->K;
    const uint32_t n_chunks = N / G;
    const uint32_t l1_total = n_chunks * ns;

    /* Invariant checks: every expert (gate+up) must match the prototype
     * shape AND share the same fp32 codebook pointer. Caller is
     * expected to have verified this — bail out if any expert differs
     * so we don't silently produce garbage. */
    const float *cb_fp32 = t0->cb_fp32;
    for (int e = 0; e < K_experts; e++) {
        const pqv2_t *g = gate_experts[e];
        const pqv2_t *u = up_experts[e];
        if (!g || !u) return -1;
        if (g->K != 256 || u->K != 256) return -1;
        if (g->M != M_per || u->M != M_per) return -1;
        if (g->N != N || u->N != N) return -1;
        if (g->G != G || u->G != G) return -1;
        if (g->n_subchunks != ns || u->n_subchunks != ns) return -1;
        if (g->half != half || u->half != half) return -1;
        if (g->l2_kind != 0 || u->l2_kind != 0) return -1;
        if (g->cb_fp32 != cb_fp32 || u->cb_fp32 != cb_fp32) return -1;
    }

    /* Per-tensor acc slabs — 2K of them, each M_per fp32 entries.
     * Layout: acc_all[(2*e + 0) * M_per + m] = gate_e accumulator,
     *         acc_all[(2*e + 1) * M_per + m] = up_e   accumulator. */
    const size_t per_floats   = (size_t)M_per;
    const size_t total_floats = (size_t)2 * K_experts * per_floats;
    float *acc_all = (float *)aligned_alloc(64,
        (total_floats * sizeof(float) + 63) & ~(size_t)63);
    if (!acc_all) return -1;
    memset(acc_all, 0, total_floats * sizeof(float));

    /* Per-expert row-major → chunk-major scratch (one per gate+up
     * tensor) when on-disk layout is row-major. Keep one scratch row
     * per tensor so we can refill all 2K scratches per (c, s) and then
     * sweep the gather loop 2K times. */
    uint8_t *l1_scratch_slab = NULL;
    uint8_t *l1_scratch_ptrs[2 * IB_MOME_FUSED_MAX_K];
    int any_rowmajor = 0;
    for (int e = 0; e < K_experts; e++) {
        if (gate_experts[e]->l1_idx_layout == 1) any_rowmajor = 1;
        if (up_experts[e]->l1_idx_layout == 1)   any_rowmajor = 1;
    }
    if (any_rowmajor) {
        size_t one = ((size_t)M_per + 63) & ~(size_t)63;
        l1_scratch_slab = (uint8_t *)aligned_alloc(64, one * 2 * K_experts);
        if (!l1_scratch_slab) { free(acc_all); return -1; }
        for (int i = 0; i < 2 * K_experts; i++) {
            l1_scratch_ptrs[i] = l1_scratch_slab + (size_t)i * one;
        }
    }

#if defined(__ARM_NEON)
    int8_t lut[4][64] __attribute__((aligned(16)));
    float  lut_scale;
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            /* ── BUILD LUT ONCE (shared across all 2K outputs) ── */
            build_lut_int8_k256(&cb_fp32[(size_t)s * K_cb * half], xs, half,
                                  lut, &lut_scale);

            int8x16x4_t b0, b1, b2, b3;
            #define LOAD_BANK(B, ARR) \
                B.val[0] = vld1q_s8(&ARR[0]);  B.val[1] = vld1q_s8(&ARR[16]); \
                B.val[2] = vld1q_s8(&ARR[32]); B.val[3] = vld1q_s8(&ARR[48]);
            LOAD_BANK(b0, lut[0]); LOAD_BANK(b1, lut[1]);
            LOAD_BANK(b2, lut[2]); LOAD_BANK(b3, lut[3]);
            #undef LOAD_BANK
            float32x4_t scl = vdupq_n_f32(lut_scale);

            /* ── Refill per-tensor row-major scratches for this (c, s).
             * Only the tensors that are row-major need a scratch fill;
             * chunk-major tensors read t->indices directly. */
            for (int e = 0; e < K_experts; e++) {
                const pqv2_t *g = gate_experts[e];
                const pqv2_t *u = up_experts[e];
                if (g->l1_idx_layout == 1) {
                    uint32_t off = c * ns + s;
                    uint8_t *dst = l1_scratch_ptrs[2 * e + 0];
                    for (uint32_t mm = 0; mm < M_per; mm++)
                        dst[mm] = g->indices[(size_t)mm * l1_total + off];
                }
                if (u->l1_idx_layout == 1) {
                    uint32_t off = c * ns + s;
                    uint8_t *dst = l1_scratch_ptrs[2 * e + 1];
                    for (uint32_t mm = 0; mm < M_per; mm++)
                        dst[mm] = u->indices[(size_t)mm * l1_total + off];
                }
            }

            /* ── Gather + accumulate for each of the 2K outputs. ── */
            for (int slot = 0; slot < 2 * K_experts; slot++) {
                const pqv2_t *t = (slot & 1) ? up_experts[slot >> 1]
                                              : gate_experts[slot >> 1];
                const uint8_t *idx;
                if (t->l1_idx_layout == 1) {
                    idx = l1_scratch_ptrs[slot];
                } else {
                    idx = &t->indices[((size_t)c * ns + s) * M_per];
                }
                float *acc = acc_all + (size_t)slot * per_floats;

                uint32_t m = 0;
                /* 32-row blocks with prefetch (mirror of
                 * pqv2_acc_tbl_int8_k256_chunks_inner). */
                for (; m + 32 <= M_per; m += 32) {
                    if (m + 256 < M_per) __builtin_prefetch(&idx[m + 256], 0, 0);
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
                for (; m + 16 <= M_per; m += 16) {
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
                for (; m < M_per; m++) {
                    uint8_t k = idx[m];
                    int8_t v = lut[k >> 6][k & 63];
                    acc[m] += (float)v * lut_scale;
                }
            }
        }
    }
#else
    /* Scalar fallback — same loop structure, no NEON. */
    int8_t lut[4][64];
    float  lut_scale;
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8_k256(&cb_fp32[(size_t)s * K_cb * half], xs, half,
                                  lut, &lut_scale);
            for (int slot = 0; slot < 2 * K_experts; slot++) {
                const pqv2_t *t = (slot & 1) ? up_experts[slot >> 1]
                                              : gate_experts[slot >> 1];
                const uint8_t *idx = &t->indices[((size_t)c * ns + s) * M_per];
                float *acc = acc_all + (size_t)slot * per_floats;
                for (uint32_t m = 0; m < M_per; m++) {
                    uint8_t k = idx[m];
                    int8_t v = lut[k >> 6][k & 63];
                    acc[m] += (float)v * lut_scale;
                }
            }
        }
    }
#endif

    /* ── Apply per-expert row_scale and emit hb / hb2 outputs. ── */
    for (int e = 0; e < K_experts; e++) {
        const float *gate_acc = acc_all + (size_t)(2 * e + 0) * per_floats;
        const float *up_acc   = acc_all + (size_t)(2 * e + 1) * per_floats;
        const uint16_t *rs_g  = gate_experts[e]->row_scale;
        const uint16_t *rs_u  = up_experts[e]->row_scale;
        float *go = hb_out  + (size_t)e * per_floats;
        float *uo = hb2_out + (size_t)e * per_floats;
        for (uint32_t m = 0; m < M_per; m++) {
            go[m] = gate_acc[m] * pqv2_h2f(rs_g[m]);
            uo[m] = up_acc[m]   * pqv2_h2f(rs_u[m]);
        }
    }

    free(acc_all);
    if (l1_scratch_slab) free(l1_scratch_slab);
    return 0;
}

/* ── Goal H1: batched MoME kernel v2 (interleaved across K experts) ───
 *
 * Same outputs/contract as pqv2_matvec_mome_gateup_k256, but the
 * inner (c, s) loop is restructured so each 32-row m-block processes
 * ALL 2K slots before advancing to m+32. The 32-row block of acc
 * for each slot stays in NEON registers across the slot dimension,
 * so per-(c, s) acc memory traffic is amortised across slots:
 *
 *   v1: per (c, s) per slot: read M_per acc + write M_per acc
 *       (M_per traffic × 2K slots = 2 * 2K * M_per * 4 bytes)
 *   v2: per (c, s) per m-block: read 32 acc per slot, write 32 acc
 *       per slot, but the 32-row LUT banks stay resident across slots
 *       (LUT-resident win is the same; the m-block acc still touches
 *       memory once per slot, but the indices fetch and the gather
 *       dominate, and the acc cache lines are guaranteed L1-hot).
 *
 * The real v2 lever is index locality + LUT-register residency at the
 * m-block granularity. NEON has 32 SIMD registers; we burn 16 on the
 * 4 LUT banks (b0..b3, 4 q-regs each) and keep 16 free for index
 * loads, gather scratch, and acc fma chains.
 */
int pqv2_matvec_mome_gateup_k256_v2(
    const pqv2_t * const *gate_experts,
    const pqv2_t * const *up_experts,
    const float *x,
    float *hb_out,
    float *hb2_out,
    int K_experts)
{
    if (K_experts <= 0 || K_experts > IB_MOME_FUSED_MAX_K) return -1;
    if (!gate_experts || !up_experts || !x || !hb_out || !hb2_out) return -1;
    if (!gate_experts[0] || !up_experts[0]) return -1;
    const pqv2_t *t0 = gate_experts[0];
    if (t0->K != 256 || !t0->cb_fp32 || t0->l2_kind != 0) return -1;

    const uint32_t M_per = t0->M;
    const uint32_t N     = t0->N;
    const uint32_t G     = t0->G;
    const uint32_t ns    = t0->n_subchunks;
    const uint32_t half  = t0->half;
    const uint32_t K_cb  = t0->K;
    const uint32_t n_chunks = N / G;
    const uint32_t l1_total = n_chunks * ns;

    /* Invariant re-check (same as v1). */
    const float *cb_fp32 = t0->cb_fp32;
    for (int e = 0; e < K_experts; e++) {
        const pqv2_t *g = gate_experts[e];
        const pqv2_t *u = up_experts[e];
        if (!g || !u) return -1;
        if (g->K != 256 || u->K != 256) return -1;
        if (g->M != M_per || u->M != M_per) return -1;
        if (g->N != N || u->N != N) return -1;
        if (g->G != G || u->G != G) return -1;
        if (g->n_subchunks != ns || u->n_subchunks != ns) return -1;
        if (g->half != half || u->half != half) return -1;
        if (g->l2_kind != 0 || u->l2_kind != 0) return -1;
        if (g->cb_fp32 != cb_fp32 || u->cb_fp32 != cb_fp32) return -1;
    }

    const int n_slots = 2 * K_experts;
    /* Per-slot acc slab — same layout as v1: (2*e+0) gate, (2*e+1) up. */
    const size_t per_floats   = (size_t)M_per;
    const size_t total_floats = (size_t)n_slots * per_floats;
    float *acc_all = (float *)aligned_alloc(64,
        (total_floats * sizeof(float) + 63) & ~(size_t)63);
    if (!acc_all) return -1;
    memset(acc_all, 0, total_floats * sizeof(float));

    /* Row-major scratch slab — one row per slot (refilled per (c, s)). */
    uint8_t *l1_scratch_slab = NULL;
    uint8_t *l1_scratch_ptrs[2 * IB_MOME_FUSED_MAX_K];
    int any_rowmajor = 0;
    for (int e = 0; e < K_experts; e++) {
        if (gate_experts[e]->l1_idx_layout == 1) any_rowmajor = 1;
        if (up_experts[e]->l1_idx_layout == 1)   any_rowmajor = 1;
    }
    if (any_rowmajor) {
        size_t one = ((size_t)M_per + 63) & ~(size_t)63;
        l1_scratch_slab = (uint8_t *)aligned_alloc(64, one * (size_t)n_slots);
        if (!l1_scratch_slab) { free(acc_all); return -1; }
        for (int i = 0; i < n_slots; i++) {
            l1_scratch_ptrs[i] = l1_scratch_slab + (size_t)i * one;
        }
    }

    /* Cached per-slot idx-base pointers (recomputed per (c, s) loop). */
    const uint8_t *slot_idx[2 * IB_MOME_FUSED_MAX_K];
    float *slot_acc[2 * IB_MOME_FUSED_MAX_K];
    const pqv2_t *slot_t[2 * IB_MOME_FUSED_MAX_K];
    for (int slot = 0; slot < n_slots; slot++) {
        slot_t[slot]   = (slot & 1) ? up_experts[slot >> 1]
                                     : gate_experts[slot >> 1];
        slot_acc[slot] = acc_all + (size_t)slot * per_floats;
    }

#if defined(__ARM_NEON)
    int8_t lut[4][64] __attribute__((aligned(16)));
    float  lut_scale;
    const uint8x16_t mask63 = vdupq_n_u8(63);
    const uint8x16_t one_v  = vdupq_n_u8(1);

    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            /* Build the shared LUT ONCE for this (c, s). */
            build_lut_int8_k256(&cb_fp32[(size_t)s * K_cb * half], xs, half,
                                  lut, &lut_scale);

            /* Pre-load LUT banks into NEON regs (16 q-regs). */
            int8x16x4_t b0, b1, b2, b3;
            #define LOAD_BANK(B, ARR) \
                B.val[0] = vld1q_s8(&ARR[0]);  B.val[1] = vld1q_s8(&ARR[16]); \
                B.val[2] = vld1q_s8(&ARR[32]); B.val[3] = vld1q_s8(&ARR[48]);
            LOAD_BANK(b0, lut[0]); LOAD_BANK(b1, lut[1]);
            LOAD_BANK(b2, lut[2]); LOAD_BANK(b3, lut[3]);
            #undef LOAD_BANK
            float32x4_t scl = vdupq_n_f32(lut_scale);

            /* Refill row-major scratches (if any) and cache idx base ptrs. */
            uint32_t cs_off = c * ns + s;
            for (int slot = 0; slot < n_slots; slot++) {
                const pqv2_t *t = slot_t[slot];
                if (t->l1_idx_layout == 1) {
                    uint8_t *dst = l1_scratch_ptrs[slot];
                    const uint8_t *src = t->indices;
                    for (uint32_t mm = 0; mm < M_per; mm++)
                        dst[mm] = src[(size_t)mm * l1_total + cs_off];
                    slot_idx[slot] = dst;
                } else {
                    slot_idx[slot] = &t->indices[((size_t)cs_off) * M_per];
                }
            }

            /* Outer loop: 32-row m-block. Inner loop: slot.
             *
             * For each m-block, we touch acc[m..m+32] for each slot. The
             * critical reuse is the LUT in NEON registers (b0..b3). The
             * per-slot acc cache line is guaranteed L1-hot across slots
             * because the block is only 128 bytes per slot — even at
             * K=32, total per-block acc is 32*32*4 = 4KB, fits in L1d. */
            uint32_t m = 0;
            for (; m + 32 <= M_per; m += 32) {
                /* Prefetch next-block indices for slot 0 + slot 1. */
                if (m + 256 < M_per) {
                    __builtin_prefetch(&slot_idx[0][m + 256], 0, 0);
                    if (n_slots > 1)
                        __builtin_prefetch(&slot_idx[1][m + 256], 0, 0);
                }
                for (int slot = 0; slot < n_slots; slot++) {
                    const uint8_t *idx = slot_idx[slot];
                    float *acc = slot_acc[slot];

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
            }
            /* 16-row tail block (also slot-inner). */
            for (; m + 16 <= M_per; m += 16) {
                for (int slot = 0; slot < n_slots; slot++) {
                    const uint8_t *idx = slot_idx[slot];
                    float *acc = slot_acc[slot];
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
            }
            /* Scalar row tail (also slot-inner). */
            for (; m < M_per; m++) {
                for (int slot = 0; slot < n_slots; slot++) {
                    const uint8_t *idx = slot_idx[slot];
                    float *acc = slot_acc[slot];
                    uint8_t k = idx[m];
                    int8_t v = lut[k >> 6][k & 63];
                    acc[m] += (float)v * lut_scale;
                }
            }
        }
    }
#else
    /* Scalar fallback — same loop ordering (slot-inner per m-block).
     * Reference path; not perf-critical. */
    int8_t lut[4][64];
    float  lut_scale;
    for (uint32_t c = 0; c < n_chunks; c++) {
        for (uint32_t s = 0; s < ns; s++) {
            const float *xs = &x[c * G + s * half];
            build_lut_int8_k256(&cb_fp32[(size_t)s * K_cb * half], xs, half,
                                  lut, &lut_scale);
            uint32_t cs_off = c * ns + s;
            for (int slot = 0; slot < n_slots; slot++) {
                const pqv2_t *t = slot_t[slot];
                slot_idx[slot] = &t->indices[((size_t)cs_off) * M_per];
            }
            for (uint32_t m = 0; m < M_per; m++) {
                for (int slot = 0; slot < n_slots; slot++) {
                    const uint8_t *idx = slot_idx[slot];
                    float *acc = slot_acc[slot];
                    uint8_t k = idx[m];
                    int8_t v = lut[k >> 6][k & 63];
                    acc[m] += (float)v * lut_scale;
                }
            }
        }
    }
#endif

    /* Apply per-expert row_scale and emit hb / hb2. */
    for (int e = 0; e < K_experts; e++) {
        const float *gate_acc = acc_all + (size_t)(2 * e + 0) * per_floats;
        const float *up_acc   = acc_all + (size_t)(2 * e + 1) * per_floats;
        const uint16_t *rs_g  = gate_experts[e]->row_scale;
        const uint16_t *rs_u  = up_experts[e]->row_scale;
        float *go = hb_out  + (size_t)e * per_floats;
        float *uo = hb2_out + (size_t)e * per_floats;
        for (uint32_t m = 0; m < M_per; m++) {
            go[m] = gate_acc[m] * pqv2_h2f(rs_g[m]);
            uo[m] = up_acc[m]   * pqv2_h2f(rs_u[m]);
        }
    }

    free(acc_all);
    if (l1_scratch_slab) free(l1_scratch_slab);
    return 0;
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
 * the single-position chunks function as the ARM version, with L2
 * pass-through to match. */
void pqv2_acc_tbl_int8_k256_chunks_batch(
    const pqv2_t *t, const float *x_batch, int B,
    const float *cb, const float *l2_cb,
    float *acc_batch, float *acc_l2_batch,
    uint32_t c_start, uint32_t c_end) {
    if (t->K != 256) return;
    for (int b = 0; b < B; b++) {
        float *ab  = acc_batch    + (size_t)b * t->M;
        float *ab2 = acc_l2_batch ? acc_l2_batch + (size_t)b * t->M : NULL;
        pqv2_acc_tbl_int8_k256_chunks(t, x_batch + (size_t)b * t->N,
                                      cb, l2_cb, ab, ab2,
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
int pqv2_matvec_mome_gateup_k256(
    const pqv2_t * const *gate_experts,
    const pqv2_t * const *up_experts,
    const float *x,
    float *hb_out,
    float *hb2_out,
    int K_experts)
{
    (void)gate_experts; (void)up_experts; (void)x;
    (void)hb_out; (void)hb2_out; (void)K_experts;
    /* Non-ARM build has no fused kernel; caller falls back. */
    return -1;
}
int pqv2_matvec_mome_gateup_k256_v2(
    const pqv2_t * const *gate_experts,
    const pqv2_t * const *up_experts,
    const float *x,
    float *hb_out,
    float *hb2_out,
    int K_experts)
{
    (void)gate_experts; (void)up_experts; (void)x;
    (void)hb_out; (void)hb2_out; (void)K_experts;
    return -1;
}
#endif
