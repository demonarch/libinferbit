/*
 * pq_decode.c — IBF v5 stacked 2D PQ tensor decoder.
 * Spec: docs/26_IBF_V5_PQ_FORMAT.md.
 */

#include "pq_decode.h"
#include "cJSON.h"
#include "inferbit_internal.h"  /* ib_thread_pool, ib_pool_run */
#include <pthread.h>
#include <stdatomic.h>

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* mmap support — POSIX. On Windows we fall back to fread-style loaders
 * (the existing ib_pq_load_single/multi paths) because Win32 file
 * mapping has a different API; that's a follow-up. */
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define IB_PQ_HAVE_MMAP 1
#endif

/* SIMD-accelerated outlier dot kernels.
 *   - ARM with +dotprod: vdotq_s32 (16x int8 dot per cycle)
 *   - x86_64 with AVX2: pmaddubsw / madd_epi16 cascade (16x int8 per call)
 *   - else: scalar fallback in apply_outliers_scalar */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define IB_PQ_HAVE_NEON_DOTPROD 1
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#define IB_PQ_HAVE_AVX2 1
#endif

/* F1.e: SIMD-accelerated fp32 dot + scaled accumulator (for attention). */
static inline float pq_dot_f32(const float* a, const float* b, int n) {
#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vfmaq_f32(acc, vld1q_f32(a + i), vld1q_f32(b + i));
    }
    float s = vaddvq_f32(acc);
    for (; i < n; i++) s += a[i] * b[i];
    return s;
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),
                              _mm256_loadu_ps(b + i), acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float total = _mm_cvtss_f32(s);
    for (; i < n; i++) total += a[i] * b[i];
    return total;
#else
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
#endif
}

static inline void pq_accum_scaled_f32(float* out, const float* v, float w, int n) {
#if defined(__ARM_NEON)
    float32x4_t wv = vdupq_n_f32(w);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t o = vld1q_f32(out + i);
        o = vfmaq_f32(o, vld1q_f32(v + i), wv);
        vst1q_f32(out + i, o);
    }
    for (; i < n; i++) out[i] += w * v[i];
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 o = _mm256_loadu_ps(out + i);
        o = _mm256_fmadd_ps(_mm256_loadu_ps(v + i), wv, o);
        _mm256_storeu_ps(out + i, o);
    }
    for (; i < n; i++) out[i] += w * v[i];
#else
    for (int i = 0; i < n; i++) out[i] += w * v[i];
#endif
}

/* Per-chunk codebook ↔ x dot table.
 * For each k in [0, K): out[k] = sum_{j in [0, half)} cb[k*half + j] * x[j].
 * Specialized for half ∈ {8, 16} on NEON / AVX2; scalar otherwise. */
static inline void pq_chunk_dot_table_f32(const float* cb, const float* x,
                                            int half, int K, float* out) {
#if defined(__ARM_NEON)
    if (half == 2) {
        /* G=4 case: 2 floats per codebook entry. Pack 4 entries (8 floats)
         * into 2 NEON regs, multiply by [x0,x1,x0,x1,x0,x1,x0,x1], use
         * vpaddq_f32 to get the 4 paired-sum dots in one vector. */
        float32x4_t xx = {x[0], x[1], x[0], x[1]};
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            float32x4_t e0 = vld1q_f32(cb + (size_t)k * 2);
            float32x4_t e1 = vld1q_f32(cb + (size_t)k * 2 + 4);
            float32x4_t p0 = vmulq_f32(e0, xx);
            float32x4_t p1 = vmulq_f32(e1, xx);
            vst1q_f32(out + k, vpaddq_f32(p0, p1));
        }
        for (; k < K; k++) {
            const float* e = cb + (size_t)k * 2;
            out[k] = e[0] * x[0] + e[1] * x[1];
        }
        return;
    }
    if (half == 16) {
        float32x4_t xv0 = vld1q_f32(x + 0);
        float32x4_t xv1 = vld1q_f32(x + 4);
        float32x4_t xv2 = vld1q_f32(x + 8);
        float32x4_t xv3 = vld1q_f32(x + 12);
        for (int k = 0; k < K; k++) {
            const float* e = cb + (size_t)k * 16;
            float32x4_t acc = vmulq_f32(vld1q_f32(e + 0),  xv0);
            acc = vfmaq_f32(acc, vld1q_f32(e + 4),  xv1);
            acc = vfmaq_f32(acc, vld1q_f32(e + 8),  xv2);
            acc = vfmaq_f32(acc, vld1q_f32(e + 12), xv3);
            out[k] = vaddvq_f32(acc);
        }
        return;
    }
    if (half == 8) {
        float32x4_t xv0 = vld1q_f32(x + 0);
        float32x4_t xv1 = vld1q_f32(x + 4);
        for (int k = 0; k < K; k++) {
            const float* e = cb + (size_t)k * 8;
            float32x4_t acc = vmulq_f32(vld1q_f32(e + 0), xv0);
            acc = vfmaq_f32(acc, vld1q_f32(e + 4), xv1);
            out[k] = vaddvq_f32(acc);
        }
        return;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    if (half == 16) {
        __m256 xv0 = _mm256_loadu_ps(x + 0);
        __m256 xv1 = _mm256_loadu_ps(x + 8);
        for (int k = 0; k < K; k++) {
            const float* e = cb + (size_t)k * 16;
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(e + 0), xv0);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(e + 8), xv1, acc);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            out[k] = _mm_cvtss_f32(s);
        }
        return;
    }
    if (half == 8) {
        __m256 xv = _mm256_loadu_ps(x);
        for (int k = 0; k < K; k++) {
            __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(cb + (size_t)k * 8), xv);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            out[k] = _mm_cvtss_f32(s);
        }
        return;
    }
#endif
    for (int k = 0; k < K; k++) {
        const float* e = cb + (size_t)k * half;
        float d = 0.0f;
        for (int j = 0; j < half; j++) d += e[j] * x[j];
        out[k] = d;
    }
}

#define IBF_MAGIC      "INFERBIT"
#define IBF_MAGIC_SIZE 8
#define IBF_PREAMBLE   32
#define IBF_VERSION_V5 5

/* ── FP16 software (IEEE 754 binary16, round-to-nearest-even) ───── */

float ib_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exp  = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            /* subnormal — normalize */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            f = (sign << 31) | ((127u - 15u - (uint32_t)e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        /* inf / nan */
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

/* Round-to-nearest-even helper. `mant` is the 23-bit fp32 mantissa with
 * its implicit leading 1 already in the appropriate position (or shifted
 * for subnormals). We keep the top 10 bits and round bit 12 (guard) plus
 * bits 0..11 (sticky). Rounds half-to-even. Returns the rounded value
 * with the trailing 13 bits cleared; caller still right-shifts by 13. */
static inline uint32_t round_half_even(uint32_t mant) {
    uint32_t guard = (mant >> 12) & 0x1u;
    uint32_t sticky = (mant & 0xFFFu) != 0;
    uint32_t keep_low = (mant >> 13) & 0x1u;
    if (guard && (sticky || keep_low)) {
        mant += 0x2000u;
    }
    return mant;
}

uint16_t ib_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    uint16_t out;
    if (exp <= 0) {
        if (exp < -10) {
            out = (uint16_t)(sign << 15);
        } else {
            /* subnormal: shift in the implicit 1 then right-shift */
            mant = (mant | 0x800000u) >> (1 - exp);
            mant = round_half_even(mant);
            out = (uint16_t)((sign << 15) | (mant >> 13));
        }
    } else if (exp >= 0x1F) {
        out = (uint16_t)((sign << 15) | (0x1Fu << 10) | (mant ? 0x200u : 0u));
    } else {
        uint32_t r = round_half_even(mant);
        if (r & 0x800000u) {
            /* mantissa overflow into exponent */
            r = 0;
            exp++;
            if (exp >= 0x1F) {
                return (uint16_t)((sign << 15) | (0x1Fu << 10));
            }
        }
        out = (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | (r >> 13));
    }
    return out;
}

/* ── JSON helpers ─────────────────────────────────────────────── */

static long json_int_field(const cJSON* obj, const char* key, long def) {
    cJSON* it = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsNumber(it) ? (long)it->valuedouble : def;
}

static int json_bool_field(const cJSON* obj, const char* key, int def) {
    cJSON* it = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsBool(it)) return cJSON_IsTrue(it);
    return def;
}

/* Read a {offset, size} block descriptor. Returns 0 on success. */
static int read_block(const cJSON* parent, const char* key, size_t* off, size_t* sz) {
    cJSON* b = cJSON_GetObjectItemCaseSensitive(parent, key);
    if (!cJSON_IsObject(b)) return -1;
    *off = (size_t)json_int_field(b, "offset", -1);
    *sz  = (size_t)json_int_field(b, "size", 0);
    return 0;
}

static ib_pq_format parse_format(const char* s) {
    if (!s) return IB_PQ_FMT_NONE;
    if (strcmp(s, "pq2d_v1_l1")      == 0) return IB_PQ_FMT_L1;
    if (strcmp(s, "pq2d_v1_l2")      == 0) return IB_PQ_FMT_L2;
    if (strcmp(s, "pq2d_v1_l1_l2")   == 0) return IB_PQ_FMT_L1_L2;
    if (strcmp(s, "pq2d_v1_pyramid") == 0) return IB_PQ_FMT_PYRAMID;
    return IB_PQ_FMT_NONE;
}

/* Forward declaration: defined later near the L2 index helpers. */
static void pq_unpack_bits_to_u16(const uint8_t* src, uint16_t* dst,
                                   size_t n_total, int n_bits);

/* ── Loader ────────────────────────────────────────────────────── */

void ib_pq_free(ib_pq_tensor* t) {
    if (!t) return;
    if (t->_arena) {
        free(t->_arena);
    }
#ifdef IB_PQ_HAVE_MMAP
    if (t->_owns_mmap && t->_mmap_base && t->_mmap_size) {
        munmap(t->_mmap_base, t->_mmap_size);
    }
#endif
    memset(t, 0, sizeof(*t));
}

/* Parse one tensor's metadata + load its blocks. The file handle is
 * positioned arbitrarily afterwards. Returns 0 on success. */
static int load_one_tensor(FILE* f, const cJSON* tm, size_t weight_data_start,
                            size_t file_sz, ib_pq_tensor* out) {
    memset(out, 0, sizeof(*out));

    const char* fmt_str = "";
    cJSON* fit = cJSON_GetObjectItemCaseSensitive(tm, "format");
    if (cJSON_IsString(fit)) fmt_str = fit->valuestring;
    out->format = parse_format(fmt_str);
    if (out->format == IB_PQ_FMT_NONE) return -1;

    cJSON* shape = cJSON_GetObjectItemCaseSensitive(tm, "shape");
    if (!cJSON_IsArray(shape) || cJSON_GetArraySize(shape) != 2) return -1;
    out->M = (int)cJSON_GetArrayItem(shape, 0)->valuedouble;
    out->N = (int)cJSON_GetArrayItem(shape, 1)->valuedouble;
    out->G = (int)json_int_field(tm, "G", 0);
    out->K = (int)json_int_field(tm, "K", 0);
    out->n_levels = (int)json_int_field(tm, "n_levels", 0);
    out->rotate   = json_bool_field(tm, "rotate", 0);

    cJSON* outlier = cJSON_GetObjectItemCaseSensitive(tm, "outlier");
    out->n_outlier = cJSON_IsObject(outlier)
                     ? (int)json_int_field(outlier, "n_cols", 0)
                     : 0;
    out->K_l2 = (int)json_int_field(tm, "K_l2", 0);   /* 0 = inherit K */

    int M = out->M, N = out->N, G = out->G, K = out->K;
    int n_inner = N - out->n_outlier;
    if (G <= 0 || K <= 0 || (G & 1) || n_inner % G != 0) return -1;
    out->C = n_inner / G;
    int C = out->C;
    int K_l2_eff = out->K_l2 > 0 ? out->K_l2 : K;
    /* Per docs/26: K_l2 must be in {16, 64, K} for L1_L2 format (with K
     * typically 256). The PYRAMID format relaxes this — K_l2 = K_outer *
     * K_inner where K_inner can be arbitrary up to 65535. */
    if (out->n_levels == 2 && out->format != IB_PQ_FMT_PYRAMID
        && K_l2_eff != 16 && K_l2_eff != 64 && K_l2_eff != K) return -1;
    int l2_packed = (K_l2_eff == 16);
    /* L2 index byte-width: PYRAMID with K_l2>256 uses uint16 (2 bytes).
     * Otherwise uint8 (1 byte). Packed 4-bit (K_l2==16) is 1 byte/2 indices. */
    out->l2_idx_bytes = (out->format == IB_PQ_FMT_PYRAMID && K_l2_eff > 256) ? 2 : 1;
    /* Phase 1.5: Bit-packed L2 indices on disk (PYRAMID only). At load we
     * unpack to uint16 in the arena. l2_packed_bits = 0 = no packing. */
    out->l2_packed_bits = (int)json_int_field(tm, "l2_packed_bits", 0);
    size_t l2_idx_per_row, l2_idx_disk_per_row;
    if (l2_packed) {
        l2_idx_per_row = (size_t)((C + 1) / 2);
        l2_idx_disk_per_row = l2_idx_per_row;
    } else {
        l2_idx_per_row = (size_t)C * (size_t)out->l2_idx_bytes;
        if (out->l2_packed_bits > 0) {
            /* Disk: ceil(C * bits / 8). RAM: still uint16 (l2_idx_per_row). */
            l2_idx_disk_per_row =
                ((size_t)C * (size_t)out->l2_packed_bits + 7) >> 3;
        } else {
            l2_idx_disk_per_row = l2_idx_per_row;
        }
    }
    (void)l2_idx_disk_per_row;  /* used in LOAD_BLOCK below */

    size_t cb_sz       = (size_t)K * (G / 2) * sizeof(uint16_t);
    size_t cb_l2_sz    = (size_t)K_l2_eff * (G / 2) * sizeof(uint16_t);
    size_t idx_sz      = (size_t)M * (size_t)C * sizeof(uint8_t);
    size_t idx_l2_sz   = (size_t)M * l2_idx_per_row;
    size_t rs_sz       = (size_t)M * sizeof(uint16_t);
    size_t oc_sz       = (size_t)out->n_outlier * sizeof(int32_t);
    size_t osc_sz      = (size_t)M * (size_t)out->n_outlier * sizeof(int8_t);
    size_t oscl_sz     = (size_t)out->n_outlier * sizeof(uint16_t);

    size_t total = 2*cb_sz + 2*idx_sz + rs_sz;
    if (out->n_levels == 2) total += 2*cb_l2_sz + 2*idx_l2_sz;
    if (out->n_outlier > 0) total += oc_sz + osc_sz + oscl_sz;

    out->_arena = malloc(total ? total : 1);
    if (!out->_arena) return -1;
    out->_arena_size = total;
    uint8_t* arena = (uint8_t*)out->_arena;
    size_t cur = 0;

    #define SLICE(ptr, type, sz) do { (ptr) = (type*)(arena + cur); cur += (sz); } while (0)
    SLICE(out->codebook_l1_l, uint16_t, cb_sz);
    SLICE(out->codebook_l1_r, uint16_t, cb_sz);
    SLICE(out->indices_l1_l,  uint8_t,  idx_sz);
    SLICE(out->indices_l1_r,  uint8_t,  idx_sz);
    SLICE(out->row_scale,     uint16_t, rs_sz);
    if (out->n_levels == 2) {
        SLICE(out->codebook_l2_l, uint16_t, cb_l2_sz);
        SLICE(out->codebook_l2_r, uint16_t, cb_l2_sz);
        SLICE(out->indices_l2_l,  uint8_t,  idx_l2_sz);
        SLICE(out->indices_l2_r,  uint8_t,  idx_l2_sz);
    }
    if (out->n_outlier > 0) {
        SLICE(out->outlier_cols,    int32_t,  oc_sz);
        SLICE(out->outlier_sidecar, int8_t,   osc_sz);
        SLICE(out->outlier_scale,   uint16_t, oscl_sz);
    }
    #undef SLICE

    #define LOAD_BLOCK(key, dst_ptr, expected_sz) do {                    \
        size_t off, sz;                                                   \
        if (read_block(tm, (key), &off, &sz) != 0) goto err;              \
        if (sz != (expected_sz)) goto err;                                \
        size_t abs = weight_data_start + off;                             \
        if (abs + sz > file_sz) goto err;                                 \
        if (fseek(f, (long)abs, SEEK_SET) != 0) goto err;                 \
        if (fread((dst_ptr), 1, sz, f) != sz) goto err;                   \
    } while (0)

    LOAD_BLOCK("codebook_l1_l", out->codebook_l1_l, cb_sz);
    LOAD_BLOCK("codebook_l1_r", out->codebook_l1_r, cb_sz);
    LOAD_BLOCK("indices_l1_l",  out->indices_l1_l,  idx_sz);
    LOAD_BLOCK("indices_l1_r",  out->indices_l1_r,  idx_sz);
    LOAD_BLOCK("row_scale",     out->row_scale,     rs_sz);
    if (out->n_levels == 2) {
        LOAD_BLOCK("codebook_l2_l", out->codebook_l2_l, cb_l2_sz);
        LOAD_BLOCK("codebook_l2_r", out->codebook_l2_r, cb_l2_sz);
        if (out->l2_packed_bits > 0 && !l2_packed) {
            /* Read packed bytes from disk, unpack into arena uint16 buffer. */
            size_t disk_per_row =
                ((size_t)C * (size_t)out->l2_packed_bits + 7) >> 3;
            size_t disk_total = (size_t)M * disk_per_row;
            uint8_t* tmp = (uint8_t*)malloc(disk_total + 4);  /* +4 pad for safe overread */
            if (!tmp) goto err;
            size_t off, sz;
            if (read_block(tm, "indices_l2_l", &off, &sz) != 0
                || sz != disk_total
                || weight_data_start + off + sz > file_sz
                || fseek(f, (long)(weight_data_start + off), SEEK_SET) != 0
                || fread(tmp, 1, sz, f) != sz) {
                free(tmp); goto err;
            }
            for (int r = 0; r < M; r++) {
                pq_unpack_bits_to_u16(tmp + (size_t)r * disk_per_row,
                                      ((uint16_t*)out->indices_l2_l) + (size_t)r * C,
                                      (size_t)C, out->l2_packed_bits);
            }
            if (read_block(tm, "indices_l2_r", &off, &sz) != 0
                || sz != disk_total
                || weight_data_start + off + sz > file_sz
                || fseek(f, (long)(weight_data_start + off), SEEK_SET) != 0
                || fread(tmp, 1, sz, f) != sz) {
                free(tmp); goto err;
            }
            for (int r = 0; r < M; r++) {
                pq_unpack_bits_to_u16(tmp + (size_t)r * disk_per_row,
                                      ((uint16_t*)out->indices_l2_r) + (size_t)r * C,
                                      (size_t)C, out->l2_packed_bits);
            }
            free(tmp);
        } else {
            LOAD_BLOCK("indices_l2_l", out->indices_l2_l, idx_l2_sz);
            LOAD_BLOCK("indices_l2_r", out->indices_l2_r, idx_l2_sz);
        }
    }
    if (out->n_outlier > 0) {
        LOAD_BLOCK("outlier_cols",    out->outlier_cols,    oc_sz);
        LOAD_BLOCK("outlier_sidecar", out->outlier_sidecar, osc_sz);
        LOAD_BLOCK("outlier_scale",   out->outlier_scale,   oscl_sz);
    }
    #undef LOAD_BLOCK
    return 0;

err:
    ib_pq_free(out);
    return -1;
}

#ifdef IB_PQ_HAVE_MMAP
/* Same shape as load_one_tensor but tensor pointers are offsets into
 * the mmap'd region. No fread, no heap arena copy. */
static int view_one_tensor(const uint8_t* mmap_base, size_t file_sz,
                            const cJSON* tm, size_t weight_data_start,
                            ib_pq_tensor* out) {
    memset(out, 0, sizeof(*out));

    const char* fmt_str = "";
    cJSON* fit = cJSON_GetObjectItemCaseSensitive(tm, "format");
    if (cJSON_IsString(fit)) fmt_str = fit->valuestring;
    out->format = parse_format(fmt_str);
    if (out->format == IB_PQ_FMT_NONE) return -1;

    cJSON* shape = cJSON_GetObjectItemCaseSensitive(tm, "shape");
    if (!cJSON_IsArray(shape) || cJSON_GetArraySize(shape) != 2) return -1;
    out->M = (int)cJSON_GetArrayItem(shape, 0)->valuedouble;
    out->N = (int)cJSON_GetArrayItem(shape, 1)->valuedouble;
    out->G = (int)json_int_field(tm, "G", 0);
    out->K = (int)json_int_field(tm, "K", 0);
    out->n_levels = (int)json_int_field(tm, "n_levels", 0);
    out->rotate   = json_bool_field(tm, "rotate", 0);

    cJSON* outlier = cJSON_GetObjectItemCaseSensitive(tm, "outlier");
    out->n_outlier = cJSON_IsObject(outlier)
                     ? (int)json_int_field(outlier, "n_cols", 0)
                     : 0;
    out->K_l2 = (int)json_int_field(tm, "K_l2", 0);

    int M = out->M, N = out->N, G = out->G, K = out->K;
    int n_inner = N - out->n_outlier;
    if (G <= 0 || K <= 0 || (G & 1) || n_inner % G != 0) return -1;
    out->C = n_inner / G;
    int K_l2_eff = out->K_l2 > 0 ? out->K_l2 : K;
    /* Stage D guard removed: matmul_impl + ib_pq_reconstruct_fp32 now
     * handle K_l2 ∈ {16, 64, 256} including 4-bit packed indices for
     * K_l2 == 16. The PYRAMID format relaxes the {16,64,K} restriction
     * (K_l2 = K_outer × K_inner, arbitrary). */
    if (out->n_levels == 2 && out->format != IB_PQ_FMT_PYRAMID
        && K_l2_eff != 16 && K_l2_eff != 64 && K_l2_eff != K) return -1;
    int l2_packed = (K_l2_eff == 16);
    out->l2_idx_bytes = (out->format == IB_PQ_FMT_PYRAMID && K_l2_eff > 256) ? 2 : 1;
    out->l2_packed_bits = (int)json_int_field(tm, "l2_packed_bits", 0);
    size_t l2_idx_per_row, l2_idx_disk_per_row;
    if (l2_packed) {
        l2_idx_per_row = (size_t)((out->C + 1) / 2);
        l2_idx_disk_per_row = l2_idx_per_row;
    } else {
        l2_idx_per_row = (size_t)out->C * (size_t)out->l2_idx_bytes;
        if (out->l2_packed_bits > 0) {
            l2_idx_disk_per_row =
                ((size_t)out->C * (size_t)out->l2_packed_bits + 7) >> 3;
        } else {
            l2_idx_disk_per_row = l2_idx_per_row;
        }
    }

    size_t cb_sz       = (size_t)K * (G / 2) * sizeof(uint16_t);
    size_t cb_l2_sz    = (size_t)K_l2_eff * (G / 2) * sizeof(uint16_t);
    size_t idx_sz      = (size_t)M * (size_t)out->C * sizeof(uint8_t);
    size_t idx_l2_sz   = (size_t)M * l2_idx_per_row;
    size_t idx_l2_disk_sz = (size_t)M * l2_idx_disk_per_row;
    size_t rs_sz       = (size_t)M * sizeof(uint16_t);
    size_t oc_sz       = (size_t)out->n_outlier * sizeof(int32_t);
    size_t osc_sz      = (size_t)M * (size_t)out->n_outlier * sizeof(int8_t);
    size_t oscl_sz     = (size_t)out->n_outlier * sizeof(uint16_t);

    #define VIEW_BLOCK(key, dst_ptr, dst_type, expected_sz) do {              \
        size_t off, sz;                                                       \
        if (read_block(tm, (key), &off, &sz) != 0) return -1;                 \
        if (sz != (expected_sz)) return -1;                                   \
        size_t abs = weight_data_start + off;                                 \
        if (abs + sz > file_sz) return -1;                                    \
        (dst_ptr) = (dst_type*)(mmap_base + abs);                             \
    } while (0)

    VIEW_BLOCK("codebook_l1_l", out->codebook_l1_l, uint16_t, cb_sz);
    VIEW_BLOCK("codebook_l1_r", out->codebook_l1_r, uint16_t, cb_sz);
    VIEW_BLOCK("indices_l1_l",  out->indices_l1_l,  uint8_t,  idx_sz);
    VIEW_BLOCK("indices_l1_r",  out->indices_l1_r,  uint8_t,  idx_sz);
    VIEW_BLOCK("row_scale",     out->row_scale,     uint16_t, rs_sz);
    if (out->n_levels == 2) {
        VIEW_BLOCK("codebook_l2_l", out->codebook_l2_l, uint16_t, cb_l2_sz);
        VIEW_BLOCK("codebook_l2_r", out->codebook_l2_r, uint16_t, cb_l2_sz);
        if (out->l2_packed_bits > 0 && !l2_packed) {
            /* Packed-on-disk indices: view the packed bytes via mmap, then
             * unpack into a heap-allocated uint16 buffer (stored in _arena
             * for cleanup). Decode path reads uint16 from heap, not mmap. */
            uint8_t* packed_l;
            uint8_t* packed_r;
            VIEW_BLOCK("indices_l2_l", packed_l, uint8_t, idx_l2_disk_sz);
            VIEW_BLOCK("indices_l2_r", packed_r, uint8_t, idx_l2_disk_sz);
            /* Allocate uint16 for L and R indices (2 × idx_l2_sz bytes). */
            size_t total = 2 * idx_l2_sz + 4;  /* +4 pad for safe over-read */
            uint8_t* buf = (uint8_t*)malloc(total);
            if (!buf) return -1;
            out->_arena = buf;
            out->_arena_size = total;
            uint16_t* unp_l = (uint16_t*)buf;
            uint16_t* unp_r = (uint16_t*)(buf + idx_l2_sz);
            for (int r = 0; r < M; r++) {
                pq_unpack_bits_to_u16(packed_l + (size_t)r * l2_idx_disk_per_row,
                                       unp_l + (size_t)r * out->C,
                                       (size_t)out->C, out->l2_packed_bits);
                pq_unpack_bits_to_u16(packed_r + (size_t)r * l2_idx_disk_per_row,
                                       unp_r + (size_t)r * out->C,
                                       (size_t)out->C, out->l2_packed_bits);
            }
            out->indices_l2_l = (uint8_t*)unp_l;
            out->indices_l2_r = (uint8_t*)unp_r;
        } else {
            VIEW_BLOCK("indices_l2_l", out->indices_l2_l, uint8_t, idx_l2_sz);
            VIEW_BLOCK("indices_l2_r", out->indices_l2_r, uint8_t, idx_l2_sz);
        }
    }
    if (out->n_outlier > 0) {
        VIEW_BLOCK("outlier_cols",    out->outlier_cols,    int32_t,  oc_sz);
        VIEW_BLOCK("outlier_sidecar", out->outlier_sidecar, int8_t,   osc_sz);
        VIEW_BLOCK("outlier_scale",   out->outlier_scale,   uint16_t, oscl_sz);
    }
    #undef VIEW_BLOCK
    return 0;
}
#endif  /* IB_PQ_HAVE_MMAP */

/* Open the file, parse preamble + JSON header. On success, returns a
 * still-open FILE* and the parsed cJSON root + sizes via outparams.
 * On failure returns NULL and zeroes outparams. */
static FILE* open_and_parse_header(const char* path, cJSON** out_root,
                                    size_t* out_weight_data_start,
                                    size_t* out_file_sz) {
    *out_root = NULL;
    *out_weight_data_start = 0;
    *out_file_sz = 0;

    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long file_sz_l = ftell(f);
    if (file_sz_l < (long)IBF_PREAMBLE) { fclose(f); return NULL; }
    *out_file_sz = (size_t)file_sz_l;
    rewind(f);

    uint8_t preamble[IBF_PREAMBLE];
    if (fread(preamble, 1, IBF_PREAMBLE, f) != IBF_PREAMBLE) { fclose(f); return NULL; }
    if (memcmp(preamble, IBF_MAGIC, IBF_MAGIC_SIZE) != 0) { fclose(f); return NULL; }

    uint32_t version, json_reserve;
    memcpy(&version, preamble + 8, 4);
    memcpy(&json_reserve, preamble + 12, 4);
    if (version != IBF_VERSION_V5) { fclose(f); return NULL; }

    char* json_buf = (char*)malloc(json_reserve + 1);
    if (!json_buf) { fclose(f); return NULL; }
    if (fread(json_buf, 1, json_reserve, f) != json_reserve) {
        free(json_buf); fclose(f); return NULL;
    }
    json_buf[json_reserve] = '\0';

    cJSON* root = cJSON_Parse(json_buf);
    free(json_buf);
    if (!root) { fclose(f); return NULL; }

    *out_root = root;
    *out_weight_data_start = (size_t)json_int_field(root, "weight_data_start", 0);
    return f;
}

int ib_pq_load_single(const char* path, ib_pq_tensor* out) {
    if (!out || !path) return -1;
    memset(out, 0, sizeof(*out));

    cJSON* root = NULL;
    size_t weight_data_start = 0, file_sz = 0;
    FILE* f = open_and_parse_header(path, &root, &weight_data_start, &file_sz);
    if (!f) return -1;

    cJSON* tensors = cJSON_GetObjectItemCaseSensitive(root, "tensors");
    if (!cJSON_IsObject(tensors)) { cJSON_Delete(root); fclose(f); return -1; }
    cJSON* tm = tensors->child;
    if (!tm) { cJSON_Delete(root); fclose(f); return -1; }

    int rc = load_one_tensor(f, tm, weight_data_start, file_sz, out);
    cJSON_Delete(root);
    fclose(f);
    return rc;
}

int ib_pq_load_multi(const char* path, ib_pq_multi* out) {
    if (!out || !path) return -1;
    memset(out, 0, sizeof(*out));

    cJSON* root = NULL;
    size_t weight_data_start = 0, file_sz = 0;
    FILE* f = open_and_parse_header(path, &root, &weight_data_start, &file_sz);
    if (!f) return -1;

    cJSON* tensors = cJSON_GetObjectItemCaseSensitive(root, "tensors");
    if (!cJSON_IsObject(tensors)) { cJSON_Delete(root); fclose(f); return -1; }

    int n = 0;
    for (cJSON* it = tensors->child; it; it = it->next) n++;

    out->n = n;
    if (n > 0) {
        out->names = (char**)calloc((size_t)n, sizeof(char*));
        out->tensors = (ib_pq_tensor*)calloc((size_t)n, sizeof(ib_pq_tensor));
        if (!out->names || !out->tensors) {
            cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
        }
    }

    int i = 0;
    for (cJSON* it = tensors->child; it; it = it->next, i++) {
        size_t name_len = it->string ? strlen(it->string) : 0;
        out->names[i] = (char*)malloc(name_len + 1);
        if (!out->names[i]) {
            cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
        }
        memcpy(out->names[i], it->string ? it->string : "", name_len);
        out->names[i][name_len] = '\0';

        if (load_one_tensor(f, it, weight_data_start, file_sz, &out->tensors[i]) != 0) {
            cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
        }
    }

    /* Phase 9: optional raw tensors + config (additive header fields). */
    cJSON* raws = cJSON_GetObjectItemCaseSensitive(root, "raw_tensors");
    if (cJSON_IsObject(raws)) {
        int rn = 0;
        for (cJSON* it = raws->child; it; it = it->next) rn++;
        if (rn > 0) {
            out->n_raw = rn;
            out->raw_tensors = (ib_pq_raw_tensor*)calloc((size_t)rn, sizeof(ib_pq_raw_tensor));
            if (!out->raw_tensors) {
                cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
            }
            int ri = 0;
            for (cJSON* it = raws->child; it; it = it->next, ri++) {
                ib_pq_raw_tensor* rt = &out->raw_tensors[ri];
                size_t nl = it->string ? strlen(it->string) : 0;
                rt->name = (char*)malloc(nl + 1);
                if (!rt->name) { cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1; }
                memcpy(rt->name, it->string ? it->string : "", nl); rt->name[nl] = '\0';

                cJSON* dtit = cJSON_GetObjectItemCaseSensitive(it, "dtype");
                const char* dt = cJSON_IsString(dtit) ? dtit->valuestring : "float32";
                if      (!strcmp(dt, "float32")) rt->dtype = IB_RAW_F32;
                else if (!strcmp(dt, "float16")) rt->dtype = IB_RAW_F16;
                else if (!strcmp(dt, "int32"))   rt->dtype = IB_RAW_I32;
                else if (!strcmp(dt, "int16"))   rt->dtype = IB_RAW_I16;
                else if (!strcmp(dt, "int8"))    rt->dtype = IB_RAW_I8;
                else if (!strcmp(dt, "uint8"))   rt->dtype = IB_RAW_U8;
                else { cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1; }

                cJSON* shape = cJSON_GetObjectItemCaseSensitive(it, "shape");
                rt->ndim = cJSON_IsArray(shape) ? cJSON_GetArraySize(shape) : 0;
                if (rt->ndim > 4) rt->ndim = 4;
                for (int d = 0; d < rt->ndim; d++) {
                    rt->shape[d] = (int)cJSON_GetArrayItem(shape, d)->valuedouble;
                }

                cJSON* oit = cJSON_GetObjectItemCaseSensitive(it, "offset");
                cJSON* sit = cJSON_GetObjectItemCaseSensitive(it, "size");
                if (!cJSON_IsNumber(oit) || !cJSON_IsNumber(sit)) {
                    cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
                }
                size_t off = (size_t)oit->valuedouble;
                size_t sz  = (size_t)sit->valuedouble;
                rt->size_bytes = sz;
                rt->data = malloc(sz);
                if (!rt->data) { cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1; }
                rt->_owns_data = 1;
                if (fseek(f, (long)(weight_data_start + off), SEEK_SET) != 0
                 || fread(rt->data, 1, sz, f) != sz) {
                    cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
                }
            }
        }
    }
    cJSON* cfg = cJSON_GetObjectItemCaseSensitive(root, "config");
    if (cJSON_IsObject(cfg)) {
        char* s = cJSON_PrintUnformatted(cfg);
        if (s) out->config_json = s;
    }

    cJSON_Delete(root);
    fclose(f);
    return 0;
}

void ib_pq_multi_free(ib_pq_multi* m) {
    if (!m) return;
    if (m->names) {
        for (int i = 0; i < m->n; i++) free(m->names[i]);
        free(m->names);
    }
    if (m->tensors) {
        /* Tensors loaded via ib_pq_open_mmap share one mmap owned by
         * the multi struct; clear their pointers before per-tensor free
         * so we don't double-munmap. */
        if (m->_mmap_base) {
            for (int i = 0; i < m->n; i++) {
                m->tensors[i]._mmap_base = NULL;
                m->tensors[i]._mmap_size = 0;
                m->tensors[i]._owns_mmap = 0;
            }
        }
        for (int i = 0; i < m->n; i++) ib_pq_free(&m->tensors[i]);
        free(m->tensors);
    }
    if (m->raw_tensors) {
        for (int i = 0; i < m->n_raw; i++) {
            free(m->raw_tensors[i].name);
            if (m->raw_tensors[i]._owns_data) free(m->raw_tensors[i].data);
        }
        free(m->raw_tensors);
    }
    free(m->config_json);
#ifdef IB_PQ_HAVE_MMAP
    if (m->_mmap_base && m->_mmap_size) {
        munmap(m->_mmap_base, m->_mmap_size);
    }
#endif
    memset(m, 0, sizeof(*m));
}

const ib_pq_tensor* ib_pq_multi_find(const ib_pq_multi* m, const char* name) {
    if (!m || !name) return NULL;
    for (int i = 0; i < m->n; i++) {
        if (m->names[i] && strcmp(m->names[i], name) == 0) return &m->tensors[i];
    }
    return NULL;
}

#ifdef IB_PQ_HAVE_MMAP
int ib_pq_open_mmap(const char* path, ib_pq_multi* out) {
    if (!path || !out) return -1;
    memset(out, 0, sizeof(*out));

    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    size_t file_sz = (size_t)st.st_size;
    if (file_sz < IBF_PREAMBLE) { close(fd); return -1; }

    void* mmap_base = mmap(NULL, file_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  /* mmap holds its own reference */
    if (mmap_base == MAP_FAILED) return -1;

    /* By default the OS reads pages on demand. Hint that we'll touch
     * the metadata + headers immediately. */
    (void)madvise(mmap_base, file_sz, MADV_RANDOM);

    const uint8_t* base = (const uint8_t*)mmap_base;
    if (memcmp(base, IBF_MAGIC, IBF_MAGIC_SIZE) != 0) {
        munmap(mmap_base, file_sz); return -1;
    }
    uint32_t version, json_reserve;
    memcpy(&version, base + 8, 4);
    memcpy(&json_reserve, base + 12, 4);
    if (version != IBF_VERSION_V5) { munmap(mmap_base, file_sz); return -1; }
    if (IBF_PREAMBLE + json_reserve > file_sz) {
        munmap(mmap_base, file_sz); return -1;
    }

    /* Parse JSON header. We have to copy out the header bytes because
     * cJSON_Parse expects a NUL-terminated string and our mmap is
     * read-only / not NUL-terminated at the right boundary. */
    char* json_buf = (char*)malloc(json_reserve + 1);
    if (!json_buf) { munmap(mmap_base, file_sz); return -1; }
    memcpy(json_buf, base + IBF_PREAMBLE, json_reserve);
    json_buf[json_reserve] = '\0';
    cJSON* root = cJSON_Parse(json_buf);
    free(json_buf);
    if (!root) { munmap(mmap_base, file_sz); return -1; }

    size_t weight_data_start = (size_t)json_int_field(root, "weight_data_start", 0);
    cJSON* tensors = cJSON_GetObjectItemCaseSensitive(root, "tensors");
    if (!cJSON_IsObject(tensors)) {
        cJSON_Delete(root); munmap(mmap_base, file_sz); return -1;
    }

    int n = 0;
    for (cJSON* it = tensors->child; it; it = it->next) n++;
    if (n <= 0) { cJSON_Delete(root); munmap(mmap_base, file_sz); return -1; }

    out->n = n;
    out->names = (char**)calloc((size_t)n, sizeof(char*));
    out->tensors = (ib_pq_tensor*)calloc((size_t)n, sizeof(ib_pq_tensor));
    out->_mmap_base = mmap_base;
    out->_mmap_size = file_sz;
    if (!out->names || !out->tensors) {
        cJSON_Delete(root); ib_pq_multi_free(out); return -1;
    }

    int i = 0;
    for (cJSON* it = tensors->child; it; it = it->next, i++) {
        size_t name_len = it->string ? strlen(it->string) : 0;
        out->names[i] = (char*)malloc(name_len + 1);
        if (!out->names[i]) {
            cJSON_Delete(root); ib_pq_multi_free(out); return -1;
        }
        memcpy(out->names[i], it->string ? it->string : "", name_len);
        out->names[i][name_len] = '\0';

        if (view_one_tensor(base, file_sz, it, weight_data_start, &out->tensors[i]) != 0) {
            cJSON_Delete(root); ib_pq_multi_free(out); return -1;
        }
        /* Tensors share the multi's mmap; only the multi struct frees it. */
        out->tensors[i]._mmap_base = mmap_base;
        out->tensors[i]._mmap_size = file_sz;
        out->tensors[i]._owns_mmap = 0;
    }

    cJSON_Delete(root);
    return 0;
}

/* Compute the byte range of a tensor's blocks within the mmap and
 * issue an madvise. We use the codebook_l1_l offset as the lower
 * bound; the blocks are not strictly contiguous, but advising over
 * the whole tensor's "footprint" is a fine approximation. */
static void pq_advise_tensor(const ib_pq_tensor* t, int advice) {
    if (!t || !t->_mmap_base) return;
    /* Find earliest and latest pointer among the tensor's blocks. */
    const uint8_t* lo = (const uint8_t*)t->codebook_l1_l;
    const uint8_t* hi = lo;

    #define ADJUST(p, sz) do {                                                \
        if ((p)) {                                                            \
            const uint8_t* a = (const uint8_t*)(p);                           \
            const uint8_t* b = a + (sz);                                      \
            if (a < lo) lo = a;                                               \
            if (b > hi) hi = b;                                               \
        }                                                                     \
    } while (0)

    int K = t->K, M = t->M, half = t->G / 2, C = t->C, no = t->n_outlier;
    int K_l2_eff = t->K_l2 > 0 ? t->K_l2 : K;
    int l2_packed = (K_l2_eff == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_idx_per_row = l2_packed ? (size_t)((C + 1) / 2)
                                      : (size_t)C * (size_t)l2_idx_bytes;
    size_t cb_sz   = (size_t)K * (size_t)half * sizeof(uint16_t);
    size_t cb_l2_sz = (size_t)K_l2_eff * (size_t)half * sizeof(uint16_t);
    size_t idx_sz  = (size_t)M * (size_t)C * sizeof(uint8_t);
    size_t idx_l2_sz = (size_t)M * l2_idx_per_row;
    size_t rs_sz   = (size_t)M * sizeof(uint16_t);

    ADJUST(t->codebook_l1_l, cb_sz);
    ADJUST(t->codebook_l1_r, cb_sz);
    ADJUST(t->indices_l1_l,  idx_sz);
    ADJUST(t->indices_l1_r,  idx_sz);
    ADJUST(t->row_scale,     rs_sz);
    if (t->codebook_l2_l) ADJUST(t->codebook_l2_l, cb_l2_sz);
    if (t->codebook_l2_r) ADJUST(t->codebook_l2_r, cb_l2_sz);
    if (t->indices_l2_l)  ADJUST(t->indices_l2_l,  idx_l2_sz);
    if (t->indices_l2_r)  ADJUST(t->indices_l2_r,  idx_l2_sz);
    if (no > 0) {
        ADJUST(t->outlier_cols,    (size_t)no * sizeof(int32_t));
        ADJUST(t->outlier_sidecar, (size_t)M * (size_t)no);
        ADJUST(t->outlier_scale,   (size_t)no * sizeof(uint16_t));
    }
    #undef ADJUST

    /* Round to page boundaries. madvise rejects unaligned start. */
    long pagesize = sysconf(_SC_PAGESIZE);
    if (pagesize <= 0) pagesize = 4096;
    uintptr_t lo_a = (uintptr_t)lo & ~(uintptr_t)(pagesize - 1);
    uintptr_t hi_a = ((uintptr_t)hi + pagesize - 1) & ~(uintptr_t)(pagesize - 1);
    (void)madvise((void*)lo_a, (size_t)(hi_a - lo_a), advice);
}

void ib_pq_advise_willneed(const ib_pq_tensor* t) { pq_advise_tensor(t, MADV_WILLNEED); }
void ib_pq_advise_dontneed(const ib_pq_tensor* t) { pq_advise_tensor(t, MADV_DONTNEED); }

void ib_pq_advise_willneed_n(const ib_pq_tensor* const* tensors, int n) {
    if (!tensors) return;
    for (int i = 0; i < n; i++) pq_advise_tensor(tensors[i], MADV_WILLNEED);
}
void ib_pq_advise_dontneed_n(const ib_pq_tensor* const* tensors, int n) {
    if (!tensors) return;
    for (int i = 0; i < n; i++) pq_advise_tensor(tensors[i], MADV_DONTNEED);
}
#else
int ib_pq_open_mmap(const char* path, ib_pq_multi* out) { (void)path; (void)out; return -1; }
void ib_pq_advise_willneed(const ib_pq_tensor* t) { (void)t; }
void ib_pq_advise_dontneed(const ib_pq_tensor* t) { (void)t; }
void ib_pq_advise_willneed_n(const ib_pq_tensor* const* tensors, int n) {
    (void)tensors; (void)n;
}
void ib_pq_advise_dontneed_n(const ib_pq_tensor* const* tensors, int n) {
    (void)tensors; (void)n;
}
#endif

/* ── L2 index unpack helper ───────────────────────────────────── */
/* L2 index byte modes:
 *   packed (K_l2==16): 4-bit indices, 2-per-byte (low nibble = even chunk).
 *   bytes==1: standard uint8 indices, K_l2 ≤ 256.
 *   bytes==2: uint16 little-endian indices, K_l2 > 256 (PYRAMID format).
 * Returns int (not uint8) to fit values up to 65535. */
static inline int pq_l2_idx_at(const uint8_t* il2, int c, int packed, int bytes) {
    if (packed) {
        uint8_t b = il2[c >> 1];
        return (c & 1) ? ((b >> 4) & 0x0F) : (b & 0x0F);
    }
    if (bytes == 2) {
        const uint16_t* il2_u16 = (const uint16_t*)il2;
        return (int)il2_u16[c];
    }
    return (int)il2[c];
}

/* Bit-unpack N-bit indices (LSB-first) into a uint16 array.
 * src: packed byte stream; dst: uint16 buffer with n_total slots; n_bits in [1, 16]. */
static void pq_unpack_bits_to_u16(const uint8_t* src, uint16_t* dst,
                                   size_t n_total, int n_bits) {
    if (n_bits <= 0 || n_bits > 16) return;
    uint32_t mask = (n_bits == 16) ? 0xFFFFu : ((1u << n_bits) - 1u);
    size_t bit_pos = 0;
    for (size_t i = 0; i < n_total; i++) {
        size_t byte_off = bit_pos >> 3;
        int bit_off = (int)(bit_pos & 7);
        /* Read up to 4 bytes to cover any 16-bit value crossing byte boundaries. */
        uint32_t word = (uint32_t)src[byte_off];
        word |= (uint32_t)src[byte_off + 1] << 8;
        if (bit_off + n_bits > 16) {
            word |= (uint32_t)src[byte_off + 2] << 16;
        }
        dst[i] = (uint16_t)((word >> bit_off) & mask);
        bit_pos += (size_t)n_bits;
    }
}

static inline int pq_K_l2_eff(const ib_pq_tensor* t) {
    return t->K_l2 > 0 ? t->K_l2 : t->K;
}

/* ── Materialize ──────────────────────────────────────────────── */

int ib_pq_reconstruct_fp32(const ib_pq_tensor* t, float* out) {
    if (!t || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C;
    int half = G / 2;

    /* Pre-decode codebooks to FP32 once. L1 has K entries; L2 has
     * K_l2 entries (defaults to K when not set). */
    int cb1_entries = t->K * half;
    int K_l2 = pq_K_l2_eff(t);
    int cb2_entries = K_l2 * half;
    int l2_packed = (K_l2 == 16);
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb2l = NULL;
    float* cb2r = NULL;
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    if (t->n_levels == 2) {
        cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!cb2l || !cb2r) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
        for (int i = 0; i < cb2_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }

    /* Build inner-column index list (cols not in outlier set) */
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }

    /* Fill output: zero first so any uninitialized columns are deterministic */
    memset(out, 0, (size_t)M * (size_t)N * sizeof(float));

    /* L2 indices array stride: bytes per row. Packed (K_l2=16): ceil(C/2).
     * uint8: C bytes. uint16 (PYRAMID K_l2>256): 2*C bytes. */
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_stride;
    if (l2_packed) {
        l2_stride = (size_t)((C + 1) / 2);
    } else {
        l2_stride = (size_t)C * (size_t)l2_idx_bytes;
    }

    for (int r = 0; r < M; r++) {
        float scale = ib_fp16_to_fp32(t->row_scale[r]);
        const uint8_t* il_l = t->indices_l1_l + (size_t)r * C;
        const uint8_t* il_r = t->indices_l1_r + (size_t)r * C;
        const uint8_t* il2l = t->indices_l2_l ? t->indices_l2_l + (size_t)r * l2_stride : NULL;
        const uint8_t* il2r = t->indices_l2_r ? t->indices_l2_r + (size_t)r * l2_stride : NULL;
        for (int c = 0; c < C; c++) {
            int base = c * G;
            const float* lvec = &cb1l[(size_t)il_l[c] * half];
            const float* rvec = &cb1r[(size_t)il_r[c] * half];
            int i2l = il2l ? pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes) : 0;
            int i2r = il2r ? pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes) : 0;
            for (int k = 0; k < half; k++) {
                float v = lvec[k];
                if (cb2l) v += cb2l[(size_t)i2l * half + k];
                int col = inner_cols[base + k];
                out[(size_t)r * N + col] = ib_fp16_to_fp32(ib_fp32_to_fp16(v * scale));
            }
            for (int k = 0; k < half; k++) {
                float v = rvec[k];
                if (cb2r) v += cb2r[(size_t)i2r * half + k];
                int col = inner_cols[base + half + k];
                out[(size_t)r * N + col] = ib_fp16_to_fp32(ib_fp32_to_fp16(v * scale));
            }
        }
    }

    /* Outliers: dequantized via fp32 then cast to fp16 (matches Python) */
    for (int j = 0; j < t->n_outlier; j++) {
        int col = t->outlier_cols[j];
        float os = ib_fp16_to_fp32(t->outlier_scale[j]);
        for (int r = 0; r < M; r++) {
            float v = (float)t->outlier_sidecar[(size_t)r * t->n_outlier + j] * os;
            out[(size_t)r * N + col] = ib_fp16_to_fp32(ib_fp32_to_fp16(v));
        }
    }

    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    return 0;
}

int ib_pq_reconstruct_fp16(const ib_pq_tensor* t, uint16_t* out) {
    if (!t || !out) return -1;
    size_t total = (size_t)t->M * (size_t)t->N;
    float* tmp = (float*)malloc(total * sizeof(float));
    if (!tmp) return -1;
    int rc = ib_pq_reconstruct_fp32(t, tmp);
    if (rc == 0) {
        for (size_t i = 0; i < total; i++) out[i] = ib_fp32_to_fp16(tmp[i]);
    }
    free(tmp);
    return rc;
}

/* ── Fused matmul (LUT path) ──────────────────────────────────── */
/*
 * out = W * x in FP32, computed without materializing W.
 *
 * Standard PQ-MATMUL-LUT optimisation:
 *   For each chunk c (column block of size G/2 each on L and R sides),
 *   precompute partial dot products against x for every codebook entry:
 *     T_L1[k, c] = sum_{i<half} cb1l[k,i] * x[inner_cols[c*G + i]]
 *     T_R1[k, c] = sum_{i<half} cb1r[k,i] * x[inner_cols[c*G + half + i]]
 *   Then per row r:
 *     out[r] = scale[r] * sum_c (T_L1[idx_l[r,c], c] + T_R1[idx_r[r,c], c])
 *           (+ same with L2 tables if n_levels == 2)
 *           + outlier contribution.
 *
 * Cost: O(K * N) for the tables, plus O(M * C) cheap lookups per row.
 * Old per-row implementation was O(M * N) with the same constants.
 *
 * Numerical contract: the materialize path round-trips each weight value
 * through fp16 once. The LUT path keeps the codebook entries in fp32
 * throughout and only does fp32 fma; this matches "ib_pq_matmul_fp32 may
 * differ from materialize+fp32_matmul by FMA reassociation" from doc 26.
 */
/* ── Outlier sidecar kernel ───────────────────────────────────── */
/*
 * For each row r:
 *   out[r] += sum_j sidecar[r, j] * outlier_scale[j] * x[outlier_cols[j]]
 *
 * Naive scalar: O(M * n_outlier) fp32 muls — disproportionately
 * expensive given n_outlier is small (~84 for s2_gate, ~288 for
 * s2_down at Llama-8B). Smart trick (item #5 in docs/27): pre-multiply
 * x×scale once, quantise to int8 with a single fp32 meta-scale, then
 * per-row `vdotq_s32` over 16 outliers at a time.
 *
 * Numerical: introduces ~1/127 relative error per outlier term. Sum of
 * 84 such terms: bounded by sqrt(84) × 0.4% × |xq|_typical ≈ 4%
 * relative on the outlier-only contribution. Outliers are ~5-10% of
 * total output magnitude → final per-element error ≤ 0.4%. Within
 * fp16 weight precision; well under doc-26 frob-rel spec.
 */

#ifdef IB_PQ_HAVE_NEON_DOTPROD
/* NEON int8 dot product. Returns 0 on success, < 0 on alloc failure. */
static int apply_outliers_neon(const ib_pq_tensor* t, const float* x,
                                float* out) {
    int no = t->n_outlier;
    if (no <= 0) return 0;
    int M = t->M;

    /* 1) xq[j] = x[col[j]] * scale[j] in fp32 + max-abs scan */
    float* xq = (float*)malloc((size_t)no * sizeof(float));
    if (!xq) return -1;
    float maxabs = 0.0f;
    for (int j = 0; j < no; j++) {
        float v = x[t->outlier_cols[j]] * ib_fp16_to_fp32(t->outlier_scale[j]);
        xq[j] = v;
        float a = v < 0 ? -v : v;
        if (a > maxabs) maxabs = a;
    }

    /* 2) Quantise xq to int8 with one fp32 meta-scale. Pad to multiple
     * of 16 with zeros so the SIMD inner loop has no tail. */
    int no_pad = (no + 15) & ~15;
    int8_t* xq_q = (int8_t*)calloc((size_t)no_pad, 1);
    if (!xq_q) { free(xq); return -1; }
    float xq_meta = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
    float xq_inv  = 1.0f / xq_meta;
    for (int j = 0; j < no; j++) {
        float scaled = xq[j] * xq_inv;
        int qi = (int)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
        if (qi >  127) qi =  127;
        if (qi < -128) qi = -128;
        xq_q[j] = (int8_t)qi;
    }
    free(xq);

    /* 3) Per-row vdotq_s32. The sidecar is M × n_outlier int8, so for
     * fixed r, sidecar[r * no .. r * no + no - 1] is contiguous —
     * perfect for vld1q_s8. We may read up to 15 bytes BEYOND the
     * row's outliers; for rows < M-1 those bytes belong to the NEXT
     * row of sidecar (still in-bounds). The xq_q tail is zero-padded
     * so those out-of-row reads multiply by zero and contribute
     * nothing. The LAST row would read past the buffer; fall back to
     * scalar for that row when (no & 15) != 0. */
    int safe_rows = (no & 15) ? (M - 1) : M;
    if (safe_rows < 0) safe_rows = 0;

    for (int r = 0; r < safe_rows; r++) {
        const int8_t* row_sc = t->outlier_sidecar + (size_t)r * no;
        int32x4_t acc = vdupq_n_s32(0);
        for (int j = 0; j < no_pad; j += 16) {
            int8x16_t a = vld1q_s8(row_sc + j);
            int8x16_t b = vld1q_s8(xq_q + j);
            acc = vdotq_s32(acc, a, b);
        }
        int32_t sum = vaddvq_s32(acc);
        out[r] += (float)sum * xq_meta;
    }
    /* Scalar tail row (at most one when no & 15 != 0) */
    for (int r = safe_rows; r < M; r++) {
        const int8_t* row_sc = t->outlier_sidecar + (size_t)r * no;
        int32_t sum = 0;
        for (int j = 0; j < no; j++) sum += (int32_t)row_sc[j] * (int32_t)xq_q[j];
        out[r] += (float)sum * xq_meta;
    }

    free(xq_q);
    return 0;
}
#endif

#ifdef IB_PQ_HAVE_AVX2
/* AVX2 int8 dot product. Mirrors apply_outliers_neon: pre-quantises
 * x×scale to int8, then per-row reads 16 sidecar bytes + 16 xq bytes,
 * sign-extends to int16, uses _mm256_madd_epi16 to compute 8-wide
 * int32 sums, accumulates, horizontal-sums at row end.
 *
 * Per chunk of 16 outliers: 1 madd_epi16 ~= 1 vdotq_s32 throughput. */
static int apply_outliers_avx2(const ib_pq_tensor* t, const float* x,
                                float* out) {
    int no = t->n_outlier;
    if (no <= 0) return 0;
    int M = t->M;

    float* xq = (float*)malloc((size_t)no * sizeof(float));
    if (!xq) return -1;
    float maxabs = 0.0f;
    for (int j = 0; j < no; j++) {
        float v = x[t->outlier_cols[j]] * ib_fp16_to_fp32(t->outlier_scale[j]);
        xq[j] = v;
        float a = v < 0 ? -v : v;
        if (a > maxabs) maxabs = a;
    }
    int no_pad = (no + 15) & ~15;
    int8_t* xq_q = (int8_t*)calloc((size_t)no_pad, 1);
    if (!xq_q) { free(xq); return -1; }
    float xq_meta = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
    float xq_inv  = 1.0f / xq_meta;
    for (int j = 0; j < no; j++) {
        float scaled = xq[j] * xq_inv;
        int qi = (int)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
        if (qi >  127) qi =  127;
        if (qi < -128) qi = -128;
        xq_q[j] = (int8_t)qi;
    }
    free(xq);

    int safe_rows = (no & 15) ? (M - 1) : M;
    if (safe_rows < 0) safe_rows = 0;

    for (int r = 0; r < safe_rows; r++) {
        const int8_t* row_sc = t->outlier_sidecar + (size_t)r * no;
        __m256i acc = _mm256_setzero_si256();
        for (int j = 0; j < no_pad; j += 16) {
            __m128i a = _mm_loadu_si128((const __m128i*)(row_sc + j));
            __m128i b = _mm_loadu_si128((const __m128i*)(xq_q + j));
            __m256i a16 = _mm256_cvtepi8_epi16(a);
            __m256i b16 = _mm256_cvtepi8_epi16(b);
            __m256i prod = _mm256_madd_epi16(a16, b16);
            acc = _mm256_add_epi32(acc, prod);
        }
        /* horizontal sum of 8 int32 lanes */
        __m128i lo128 = _mm256_castsi256_si128(acc);
        __m128i hi128 = _mm256_extracti128_si256(acc, 1);
        __m128i s = _mm_add_epi32(lo128, hi128);
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
        int32_t sum = _mm_cvtsi128_si32(s);
        out[r] += (float)sum * xq_meta;
    }
    for (int r = safe_rows; r < M; r++) {
        const int8_t* row_sc = t->outlier_sidecar + (size_t)r * no;
        int32_t sum = 0;
        for (int j = 0; j < no; j++) sum += (int32_t)row_sc[j] * (int32_t)xq_q[j];
        out[r] += (float)sum * xq_meta;
    }

    free(xq_q);
    return 0;
}
#endif

/* Scalar reference. Always available; used when NEON dotprod absent
 * or for very small n_outlier where the SIMD setup cost dominates. */
static void apply_outliers_scalar(const ib_pq_tensor* t, const float* x,
                                   float* out) {
    int no = t->n_outlier;
    if (no <= 0) return;
    int M = t->M;
    for (int j = 0; j < no; j++) {
        int col = t->outlier_cols[j];
        float os = ib_fp16_to_fp32(t->outlier_scale[j]);
        float xj = x[col];
        for (int r = 0; r < M; r++) {
            float w = (float)t->outlier_sidecar[(size_t)r * no + j] * os;
            out[r] += w * xj;
        }
    }
}

static void apply_outliers(const ib_pq_tensor* t, const float* x, float* out) {
    /* SIMD setup cost ~50µs (alloc + quantise loop); below ~32 outliers
     * the scalar path wins. */
    if (t->n_outlier >= 32) {
#ifdef IB_PQ_HAVE_NEON_DOTPROD
        if (apply_outliers_neon(t, x, out) == 0) return;
#elif defined(IB_PQ_HAVE_AVX2)
        if (apply_outliers_avx2(t, x, out) == 0) return;
#endif
    }
    apply_outliers_scalar(t, x, out);
}

/* Per-row gather context — populated by the matmul setup, consumed by
 * the gather task (single-threaded or threaded). */
typedef struct {
    const ib_pq_tensor* t;
    const float* TL1;
    const float* TR1;
    const float* TL2;   /* may be NULL */
    const float* TR2;   /* may be NULL */
    int K, K_l2, C;
    int l2_packed;       /* 1 when K_l2 == 16 (4-bit packed indices) */
    size_t l2_idx_stride; /* bytes per row for indices_l2_* */
    /* DejaVu sparsity: when active_list is non-NULL the gather walks
     * only those chunks. NULL means walk all C chunks (dense path). */
    const int32_t* active_list;
    int n_active;
    float* out;
} pq_gather_ctx;

static void pq_gather_rows(pq_gather_ctx* g, int r0, int r1) {
    const ib_pq_tensor* t = g->t;
    int C = g->C, K = g->K, K_l2 = g->K_l2;
    int l2_packed = g->l2_packed;
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_stride = g->l2_idx_stride;
    const float* TL1 = g->TL1;
    const float* TR1 = g->TR1;
    const float* TL2 = g->TL2;
    const float* TR2 = g->TR2;
    const int32_t* active = g->active_list;
    int n_active = g->n_active;
    float* out = g->out;
    for (int r = r0; r < r1; r++) {
        float scale = ib_fp16_to_fp32(t->row_scale[r]);
        const uint8_t* il_l = t->indices_l1_l + (size_t)r * C;
        const uint8_t* il_r = t->indices_l1_r + (size_t)r * C;
        const uint8_t* il2l = TL2 ? t->indices_l2_l + (size_t)r * l2_stride : NULL;
        const uint8_t* il2r = TR2 ? t->indices_l2_r + (size_t)r * l2_stride : NULL;
        float acc = 0.0f;
        if (active) {
            if (TL2) {
                for (int aii = 0; aii < n_active; aii++) {
                    int c = active[aii];
                    size_t base1 = (size_t)c * K;
                    size_t base2 = (size_t)c * K_l2;
                    acc += TL1[base1 + il_l[c]] + TR1[base1 + il_r[c]];
                    acc += TL2[base2 + pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes)]
                         + TR2[base2 + pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes)];
                }
            } else {
                for (int aii = 0; aii < n_active; aii++) {
                    int c = active[aii];
                    size_t base = (size_t)c * K;
                    acc += TL1[base + il_l[c]] + TR1[base + il_r[c]];
                }
            }
        } else {
            if (TL2) {
                for (int c = 0; c < C; c++) {
                    size_t base1 = (size_t)c * K;
                    size_t base2 = (size_t)c * K_l2;
                    acc += TL1[base1 + il_l[c]] + TR1[base1 + il_r[c]];
                    acc += TL2[base2 + pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes)]
                         + TR2[base2 + pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes)];
                }
            } else {
                for (int c = 0; c < C; c++) {
                    size_t base = (size_t)c * K;
                    acc += TL1[base + il_l[c]] + TR1[base + il_r[c]];
                }
            }
        }
        out[r] = acc * scale;
    }
}

static void pq_gather_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    pq_gather_rows((pq_gather_ctx*)arg, start, end);
}

/* Set up tables, run gather (optionally threaded), apply outliers. */
static int matmul_impl(const ib_pq_tensor* t, const float* x, float* out,
                       ib_thread_pool* pool) {
    if (!t || !x || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_idx_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    /* Codebook tables decoded to fp32 once. L1: K entries; L2: K_l2 entries. */
    int cb1_entries = K * half;
    int cb2_entries = K_l2 * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb2l = NULL, *cb2r = NULL;
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    if (t->n_levels == 2) {
        cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!cb2l || !cb2r) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
        for (int i = 0; i < cb2_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }

    /* Inner-col list */
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int kk = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[kk++] = j;
        free(mask);
    }

    /* Pre-multiplied tables. Layout: TL1/TR1 use T[c * K + k]; TL2/TR2
     * use T[c * K_l2 + k]. Per-row inner loop steps the chunk index
     * by 1 within each table contiguously. */
    size_t table1_sz = (size_t)C * K;
    size_t table2_sz = (size_t)C * K_l2;
    float* TL1 = (float*)malloc(table1_sz * sizeof(float));
    float* TR1 = (float*)malloc(table1_sz * sizeof(float));
    float* TL2 = NULL, *TR2 = NULL;
    if (!TL1 || !TR1) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(TL1); free(TR1); return -1;
    }
    if (cb2l) {
        TL2 = (float*)malloc(table2_sz * sizeof(float));
        TR2 = (float*)malloc(table2_sz * sizeof(float));
        if (!TL2 || !TR2) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            free(TL1); free(TR1); free(TL2); free(TR2); return -1;
        }
    }

    /* DejaVu sparsity auto-detect.
     *
     * When x has many near-zero entries (post-SiLU FFN intermediate is
     * the canonical case), entire chunks contribute nothing. We scan
     * x once, build a list of active chunks, skip the rest in both
     * LUT build and per-row gather. Threshold: 1% of max-abs. Below
     * that, contribution to the output is bounded by
     *     max_abs * 0.01 * codebook_max * C
     * which is well below fp16 noise on typical inputs.
     *
     * If no chunks fall below threshold, active_list stays NULL and
     * the kernel runs the dense path (no overhead).
     */
    float maxabs = 0.0f;
    int n_inner = N - t->n_outlier;
    for (int j = 0; j < n_inner; j++) {
        float v = x[inner_cols[j]];
        if (v < 0) v = -v;
        if (v > maxabs) maxabs = v;
    }
    int32_t* active_list = NULL;
    int n_active = 0;
    if (maxabs > 0.0f) {
        float thresh = maxabs * 0.01f;
        active_list = (int32_t*)malloc((size_t)C * sizeof(int32_t));
        if (!active_list) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            free(TL1); free(TR1); free(TL2); free(TR2); return -1;
        }
        for (int c = 0; c < C; c++) {
            int base = c * G;
            float chunk_maxabs = 0.0f;
            for (int i = 0; i < G; i++) {
                float v = x[inner_cols[base + i]];
                if (v < 0) v = -v;
                if (v > chunk_maxabs) chunk_maxabs = v;
            }
            if (chunk_maxabs >= thresh) {
                active_list[n_active++] = c;
            }
        }
        /* If everything is active, dense path is just as fast and the
         * gather inner loop is one fewer indirection. */
        if (n_active == C) {
            free(active_list);
            active_list = NULL;
        }
    }

    /* Build the partial-dot tables. L1 uses K entries per chunk; L2
     * uses K_l2 entries per chunk (which may differ from K, e.g.
     * K_l2=16 for the 6-bpw S1 variant). */
    #define BUILD_LUT_FOR_CHUNK(c) do {                                      \
        int base = (c) * G;                                                  \
        float xL[16], xR[16];                                                \
        for (int i = 0; i < half; i++) xL[i] = x[inner_cols[base + i]];      \
        for (int i = 0; i < half; i++) xR[i] = x[inner_cols[base + half + i]]; \
        /* L1: K entries */                                                  \
        for (int k = 0; k < K; k++) {                                        \
            float sl = 0.0f, sr = 0.0f;                                      \
            const float* lv = &cb1l[(size_t)k * half];                       \
            const float* rv = &cb1r[(size_t)k * half];                       \
            for (int i = 0; i < half; i++) sl += lv[i] * xL[i];              \
            for (int i = 0; i < half; i++) sr += rv[i] * xR[i];              \
            TL1[(size_t)(c) * K + k] = sl;                                   \
            TR1[(size_t)(c) * K + k] = sr;                                   \
        }                                                                    \
        /* L2: K_l2 entries (may be != K) */                                 \
        if (cb2l) {                                                          \
            for (int k = 0; k < K_l2; k++) {                                 \
                float sl2 = 0.0f, sr2 = 0.0f;                                \
                const float* lv2 = &cb2l[(size_t)k * half];                  \
                const float* rv2 = &cb2r[(size_t)k * half];                  \
                for (int i = 0; i < half; i++) sl2 += lv2[i] * xL[i];        \
                for (int i = 0; i < half; i++) sr2 += rv2[i] * xR[i];        \
                TL2[(size_t)(c) * K_l2 + k] = sl2;                           \
                TR2[(size_t)(c) * K_l2 + k] = sr2;                           \
            }                                                                \
        }                                                                    \
    } while (0)
    if (active_list) {
        for (int aii = 0; aii < n_active; aii++) {
            int c = active_list[aii];
            BUILD_LUT_FOR_CHUNK(c);
        }
    } else {
        for (int c = 0; c < C; c++) {
            BUILD_LUT_FOR_CHUNK(c);
        }
    }
    #undef BUILD_LUT_FOR_CHUNK

    /* Per-row gather. Sparse path walks active_list when present. */
    pq_gather_ctx ctx = {
        .t = t, .TL1 = TL1, .TR1 = TR1, .TL2 = TL2, .TR2 = TR2,
        .K = K, .K_l2 = K_l2, .C = C,
        .l2_packed = l2_packed, .l2_idx_stride = l2_idx_stride,
        .active_list = active_list, .n_active = n_active,
        .out = out,
    };
    if (pool) {
        ib_pool_run(pool, pq_gather_task, &ctx, M, 0);
    } else {
        pq_gather_rows(&ctx, 0, M);
    }

    /* Outliers — see apply_outliers (item #5: NEON int8 dot for n>=32). */
    apply_outliers(t, x, out);

    free(cb1l); free(cb1r); free(cb2l); free(cb2r);
    free(inner_cols); free(TL1); free(TR1); free(TL2); free(TR2);
    free(active_list);
    return 0;
}

int ib_pq_matmul_fp32(const ib_pq_tensor* t, const float* x, float* out) {
    return matmul_impl(t, x, out, NULL);
}

/* Phase 2: streaming-precompute matmul. Per-chunk codebook dots stay
 * in cache for the row accumulation pass over that chunk. Scalar
 * implementation; SIMD variants in 2.x. */
int ib_pq_matmul_fp32_streaming(const ib_pq_tensor* t, const float* x, float* out) {
    if (!t || !x || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    int has_l2 = (t->n_levels == 2);

    /* Decode codebooks to fp32 once. */
    int cb1_entries = K * half;
    int cb2_entries = K_l2 * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    float* cb2l = NULL;
    float* cb2r = NULL;
    if (has_l2) {
        cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!cb2l || !cb2r) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
        }
        for (int i = 0; i < cb2_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }

    /* Build inner-column index list (skip outlier columns). */
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
    }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            return -1;
        }
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }

    /* Per-chunk lookup tables (stay in cache). */
    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    float* C2R_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    if (!C1L_dot_x || !C1R_dot_x || (has_l2 && (!C2L_dot_x || !C2R_dot_x))) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));

    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    /* Stream chunks. For each chunk position c, build the per-chunk lookup
     * tables, then accumulate across all M output rows. Tables stay in L1. */
    for (int c = 0; c < C; c++) {
        int base = c * G;
        /* Extract the activation half-chunks for this chunk's columns. */
        float xL[16];  /* G/2 ≤ 16 in any sane config; G is small */
        float xR[16];
        for (int j = 0; j < half; j++) {
            xL[j] = x[inner_cols[base + j]];
            xR[j] = x[inner_cols[base + half + j]];
        }
        pq_chunk_dot_table_f32(cb1l, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cb1r, xR, half, K, C1R_dot_x);
        if (has_l2) {
            pq_chunk_dot_table_f32(cb2l, xL, half, K_l2, C2L_dot_x);
            pq_chunk_dot_table_f32(cb2r, xR, half, K_l2, C2R_dot_x);
        }

        /* Walk all output rows; accumulate from cache-resident tables. */
        const uint8_t* il_l_col = t->indices_l1_l;
        const uint8_t* il_r_col = t->indices_l1_r;
        for (int r = 0; r < M; r++) {
            int i1l = il_l_col[(size_t)r * C + c];
            int i1r = il_r_col[(size_t)r * C + c];
            float v = C1L_dot_x[i1l] + C1R_dot_x[i1r];
            if (has_l2) {
                const uint8_t* il2l = t->indices_l2_l + (size_t)r * l2_stride;
                const uint8_t* il2r = t->indices_l2_r + (size_t)r * l2_stride;
                int i2l = pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes);
                int i2r = pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes);
                v += C2L_dot_x[i2l] + C2R_dot_x[i2r];
            }
            out[r] += v;
        }
    }

    /* Apply per-row scale. */
    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }

    /* Add outlier sidecar contribution (existing helper). */
    apply_outliers(t, x, out);

    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return 0;
}

int ib_pq_matmul_fp32_threaded(const ib_pq_tensor* t, const float* x,
                                float* out, void* pool) {
    return matmul_impl(t, x, out, (ib_thread_pool*)pool);
}

/* Phase 4: activation-sparse streaming matmul. Skip chunks whose
 * input x values are all below threshold. */
int ib_pq_matmul_fp32_streaming_sparse(const ib_pq_tensor* t, const float* x,
                                        float* out, float act_threshold) {
    if (!t || !x || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    int has_l2 = (t->n_levels == 2);

    int cb1_entries = K * half;
    int cb2_entries = K_l2 * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    float* cb2l = NULL;
    float* cb2r = NULL;
    if (has_l2) {
        cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!cb2l || !cb2r) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
        }
        for (int i = 0; i < cb2_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
    }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            return -1;
        }
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }
    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    float* C2R_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    if (!C1L_dot_x || !C1R_dot_x || (has_l2 && (!C2L_dot_x || !C2R_dot_x))) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));
    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    int n_skipped = 0, n_total_chunks = 0;
    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        for (int j = 0; j < half; j++) {
            xL[j] = x[inner_cols[base + j]];
            xR[j] = x[inner_cols[base + half + j]];
        }
        n_total_chunks++;
        /* Activation-sparsity check: skip if all x values in this chunk
         * are below threshold (their L1+L2 contribution is bounded by
         * ||codebook||*||x_chunk|| which is tiny). */
        if (act_threshold > 0.0f) {
            int all_small = 1;
            for (int j = 0; j < half; j++) {
                if (fabsf(xL[j]) > act_threshold || fabsf(xR[j]) > act_threshold) {
                    all_small = 0; break;
                }
            }
            if (all_small) { n_skipped++; continue; }
        }

        pq_chunk_dot_table_f32(cb1l, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cb1r, xR, half, K, C1R_dot_x);
        if (has_l2) {
            pq_chunk_dot_table_f32(cb2l, xL, half, K_l2, C2L_dot_x);
            pq_chunk_dot_table_f32(cb2r, xR, half, K_l2, C2R_dot_x);
        }

        for (int r = 0; r < M; r++) {
            int i1l = t->indices_l1_l[(size_t)r * C + c];
            int i1r = t->indices_l1_r[(size_t)r * C + c];
            float v = C1L_dot_x[i1l] + C1R_dot_x[i1r];
            if (has_l2) {
                const uint8_t* il2l = t->indices_l2_l + (size_t)r * l2_stride;
                const uint8_t* il2r = t->indices_l2_r + (size_t)r * l2_stride;
                int i2l = pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes);
                int i2r = pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes);
                v += C2L_dot_x[i2l] + C2R_dot_x[i2r];
            }
            out[r] += v;
        }
    }

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);

    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return 0;
}

/* Phase 8.F: variance-bounded L2 skip variant of streaming matmul.
 * Adds per-cluster ||C2||_max precompute and per-(row, chunk) skip
 * decision. */
int ib_pq_matmul_fp32_streaming_l2skip(const ib_pq_tensor* t, const float* x,
                                        float* out, float skip_threshold) {
    if (!t || !x || !out) return -1;
    if (t->n_levels != 2) return ib_pq_matmul_fp32_streaming(t, x, out);
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;

    int cb1_entries = K * half;
    int cb2_entries = K_l2 * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
    float* cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
    if (!cb1l || !cb1r || !cb2l || !cb2r) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
    }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    for (int i = 0; i < cb2_entries; i++) {
        cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
        cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
    }
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
    }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            return -1;
        }
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }

    /* Phase 8.F+8.I: precompute per-outer-cluster ||C2[k]||_max for L and R.
     * For PYRAMID format the conditional L2 is logically [K1][K_inner][G/2];
     * the flat storage is [K_l2 = K1*K_inner][G/2] with index = k1*K_inner + k_inner.
     * Per-cluster bound (tight): max over k_inner of ||C2[k1*K_inner + k_inner]||.
     * For L1_L2 (non-pyramid): K_inner = K_l2 (global bound, fall through). */
    int is_pyramid = (t->format == IB_PQ_FMT_PYRAMID);
    int K_inner = (is_pyramid && K > 0) ? (K_l2 / K) : K_l2;
    int n_outer = (is_pyramid && K > 0) ? K : 1;
    float* c2l_max_per_cluster = (float*)calloc((size_t)n_outer, sizeof(float));
    float* c2r_max_per_cluster = (float*)calloc((size_t)n_outer, sizeof(float));
    if (!c2l_max_per_cluster || !c2r_max_per_cluster) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(c2l_max_per_cluster); free(c2r_max_per_cluster);
        return -1;
    }
    for (int k1 = 0; k1 < n_outer; k1++) {
        float ml = 0.0f, mr = 0.0f;
        int k_lo = is_pyramid ? k1 * K_inner : 0;
        int k_hi = is_pyramid ? (k1 + 1) * K_inner : K_l2;
        for (int k = k_lo; k < k_hi; k++) {
            float nl = 0.0f, nr = 0.0f;
            for (int j = 0; j < half; j++) {
                nl += cb2l[k * half + j] * cb2l[k * half + j];
                nr += cb2r[k * half + j] * cb2r[k * half + j];
            }
            nl = sqrtf(nl); nr = sqrtf(nr);
            if (nl > ml) ml = nl;
            if (nr > mr) mr = nr;
        }
        c2l_max_per_cluster[k1] = ml;
        c2r_max_per_cluster[k1] = mr;
    }

    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = (float*)malloc((size_t)K_l2 * sizeof(float));
    float* C2R_dot_x = (float*)malloc((size_t)K_l2 * sizeof(float));
    if (!C1L_dot_x || !C1R_dot_x || !C2L_dot_x || !C2R_dot_x) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(c2l_max_per_cluster); free(c2r_max_per_cluster);
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));
    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    int n_skipped = 0, n_evaluated = 0;
    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        float xL_norm2 = 0.0f, xR_norm2 = 0.0f;
        for (int j = 0; j < half; j++) {
            xL[j] = x[inner_cols[base + j]];
            xR[j] = x[inner_cols[base + half + j]];
            xL_norm2 += xL[j] * xL[j];
            xR_norm2 += xR[j] * xR[j];
        }
        float xL_norm = sqrtf(xL_norm2);
        float xR_norm = sqrtf(xR_norm2);

        pq_chunk_dot_table_f32(cb1l, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cb1r, xR, half, K, C1R_dot_x);
        pq_chunk_dot_table_f32(cb2l, xL, half, K_l2, C2L_dot_x);
        pq_chunk_dot_table_f32(cb2r, xR, half, K_l2, C2R_dot_x);

        for (int r = 0; r < M; r++) {
            int i1l = t->indices_l1_l[(size_t)r * C + c];
            int i1r = t->indices_l1_r[(size_t)r * C + c];
            float l1_contrib = C1L_dot_x[i1l] + C1R_dot_x[i1r];
            float v = l1_contrib;
            /* Per-cluster L2 bound (Phase 8.I): tighter than global bound;
             * uses ||C2[k1, *]||_max for the cluster k1 the row picked. */
            int oc_l = is_pyramid ? i1l : 0;
            int oc_r = is_pyramid ? i1r : 0;
            float l2_bound = skip_threshold * (c2l_max_per_cluster[oc_l] * xL_norm
                                             + c2r_max_per_cluster[oc_r] * xR_norm);
            if (skip_threshold > 0.0f && fabsf(l1_contrib) > l2_bound) {
                n_skipped++;
            } else {
                const uint8_t* il2l = t->indices_l2_l + (size_t)r * l2_stride;
                const uint8_t* il2r = t->indices_l2_r + (size_t)r * l2_stride;
                int i2l = pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes);
                int i2r = pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes);
                v += C2L_dot_x[i2l] + C2R_dot_x[i2r];
            }
            n_evaluated++;
            out[r] += v;
        }
    }

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);

    (void)n_skipped; (void)n_evaluated;  /* could expose stats via a debug API */

    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    free(c2l_max_per_cluster); free(c2r_max_per_cluster);
    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return 0;
}

/* Phase 5: L1-only matmul. Same as streaming matmul but pretends
 * n_levels=1 — skip the L2 codebook contribution. Cheap pass for
 * top-K candidate filtering. */
int ib_pq_matmul_fp32_l1_only(const ib_pq_tensor* t, const float* x, float* out) {
    if (!t || !x || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;

    /* Decode L1 codebooks to fp32 once. */
    int cb1_entries = K * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) { free(cb1l); free(cb1r); return -1; }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) { free(cb1l); free(cb1r); free(inner_cols); return -1; }
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }
    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    if (!C1L_dot_x || !C1R_dot_x) {
        free(cb1l); free(cb1r); free(inner_cols); free(C1L_dot_x); free(C1R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));
    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        for (int j = 0; j < half; j++) {
            xL[j] = x[inner_cols[base + j]];
            xR[j] = x[inner_cols[base + half + j]];
        }
        pq_chunk_dot_table_f32(cb1l, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cb1r, xR, half, K, C1R_dot_x);
        for (int r = 0; r < M; r++) {
            int i1l = t->indices_l1_l[(size_t)r * C + c];
            int i1r = t->indices_l1_r[(size_t)r * C + c];
            out[r] += C1L_dot_x[i1l] + C1R_dot_x[i1r];
        }
    }
    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);
    free(cb1l); free(cb1r); free(inner_cols);
    free(C1L_dot_x); free(C1R_dot_x);
    return 0;
}

/* Phase 5: subset matmul. Full pyramid logits for the selected row
 * indices only. n_rows-element output. Linear in n_rows × C, vs
 * full matmul which is M × C. For n_rows ≪ M (top-K refinement),
 * a fraction of the full matmul cost. */
int ib_pq_matmul_fp32_subset(const ib_pq_tensor* t, const float* x,
                              const int32_t* row_indices, int n_rows,
                              float* out) {
    if (!t || !x || !out || (!row_indices && n_rows > 0)) return -1;
    if (n_rows <= 0) return 0;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    int has_l2 = (t->n_levels == 2);

    /* Bounds check row indices. */
    for (int i = 0; i < n_rows; i++) {
        if (row_indices[i] < 0 || row_indices[i] >= M) return -1;
    }

    int cb1_entries = K * half;
    int cb2_entries = K_l2 * half;
    float* cb1l = (float*)malloc((size_t)cb1_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb1_entries * sizeof(float));
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb1_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    float* cb2l = NULL;
    float* cb2r = NULL;
    if (has_l2) {
        cb2l = (float*)malloc((size_t)cb2_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!cb2l || !cb2r) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
        }
        for (int i = 0; i < cb2_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1;
    }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) {
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            return -1;
        }
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[k++] = j;
        free(mask);
    }

    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    /* For each selected row, walk all chunks and accumulate. Per-chunk
     * lookup tables are per-row here (we don't share across rows, since
     * rows are sparse). For small n_rows ≪ M this is still cheaper than
     * the full matmul which precomputes K dot products M times. */
    for (int ri = 0; ri < n_rows; ri++) {
        int r = row_indices[ri];
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            int base = c * G;
            int i1l = t->indices_l1_l[(size_t)r * C + c];
            int i1r = t->indices_l1_r[(size_t)r * C + c];
            const float* eL = &cb1l[(size_t)i1l * half];
            const float* eR = &cb1r[(size_t)i1r * half];
            for (int j = 0; j < half; j++) {
                acc += eL[j] * x[inner_cols[base + j]];
                acc += eR[j] * x[inner_cols[base + half + j]];
            }
            if (has_l2) {
                const uint8_t* il2l = t->indices_l2_l + (size_t)r * l2_stride;
                const uint8_t* il2r = t->indices_l2_r + (size_t)r * l2_stride;
                int i2l = pq_l2_idx_at(il2l, c, l2_packed, l2_idx_bytes);
                int i2r = pq_l2_idx_at(il2r, c, l2_packed, l2_idx_bytes);
                const float* eL2 = &cb2l[(size_t)i2l * half];
                const float* eR2 = &cb2r[(size_t)i2r * half];
                for (int j = 0; j < half; j++) {
                    acc += eL2[j] * x[inner_cols[base + j]];
                    acc += eR2[j] * x[inner_cols[base + half + j]];
                }
            }
        }
        acc *= ib_fp16_to_fp32(t->row_scale[r]);
        /* Outlier sidecar contribution for this row. */
        if (t->n_outlier > 0) {
            for (int j = 0; j < t->n_outlier; j++) {
                int col = t->outlier_cols[j];
                float os = ib_fp16_to_fp32(t->outlier_scale[j]);
                float w = (float)t->outlier_sidecar[(size_t)r * t->n_outlier + j] * os;
                acc += w * x[col];
            }
        }
        out[ri] = acc;
    }

    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    return 0;
}

/* Phase 5: top-K orchestrator. Two-stage:
 *   1. L1-only matmul for full-vocab cheap logits.
 *   2. Partial sort to extract top-K candidate row indices.
 *   3. Subset matmul for full-pyramid logits on those K rows.
 *   4. Sort candidates by refined logit, fill outputs descending.
 *
 * Memory: O(M) scratch for stage 1, O(K) for stage 3.
 */
int ib_pq_lm_head_topk(const ib_pq_tensor* t, const float* x, int K_top,
                        float* out_logits, int32_t* out_token_ids) {
    if (!t || !x || !out_logits || !out_token_ids) return -1;
    if (K_top <= 0 || K_top > t->M) return -1;
    int M = t->M;

    /* Stage 1: cheap L1-only logits over full vocab. */
    float* coarse = (float*)malloc((size_t)M * sizeof(float));
    if (!coarse) return -1;
    int rc = ib_pq_matmul_fp32_l1_only(t, x, coarse);
    if (rc != 0) { free(coarse); return rc; }

    /* Stage 2: extract top-K row indices by coarse logit (partial sort).
     * Use a simple max-heap-like O(M log K) selection. */
    int32_t* top_ids = (int32_t*)malloc((size_t)K_top * sizeof(int32_t));
    float* top_vals = (float*)malloc((size_t)K_top * sizeof(float));
    if (!top_ids || !top_vals) {
        free(coarse); free(top_ids); free(top_vals); return -1;
    }
    /* Initialize heap with the first K elements (min-heap so we pop the
     * smallest when a larger candidate arrives). */
    for (int i = 0; i < K_top; i++) { top_ids[i] = i; top_vals[i] = coarse[i]; }
    /* Build min-heap (sift-down from middle). */
    for (int start = K_top / 2 - 1; start >= 0; start--) {
        int i = start;
        while (1) {
            int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
            if (l < K_top && top_vals[l] < top_vals[smallest]) smallest = l;
            if (r < K_top && top_vals[r] < top_vals[smallest]) smallest = r;
            if (smallest == i) break;
            float tv = top_vals[i]; top_vals[i] = top_vals[smallest]; top_vals[smallest] = tv;
            int32_t ti = top_ids[i]; top_ids[i] = top_ids[smallest]; top_ids[smallest] = ti;
            i = smallest;
        }
    }
    /* Stream the rest: if val > heap-min, replace and sift down. */
    for (int j = K_top; j < M; j++) {
        if (coarse[j] > top_vals[0]) {
            top_vals[0] = coarse[j];
            top_ids[0] = (int32_t)j;
            /* Sift down. */
            int i = 0;
            while (1) {
                int l = 2 * i + 1, r = 2 * i + 2, smallest = i;
                if (l < K_top && top_vals[l] < top_vals[smallest]) smallest = l;
                if (r < K_top && top_vals[r] < top_vals[smallest]) smallest = r;
                if (smallest == i) break;
                float tv = top_vals[i]; top_vals[i] = top_vals[smallest]; top_vals[smallest] = tv;
                int32_t ti = top_ids[i]; top_ids[i] = top_ids[smallest]; top_ids[smallest] = ti;
                i = smallest;
            }
        }
    }
    free(coarse);

    /* Stage 3: full-pyramid refinement on top-K. */
    float* refined = (float*)malloc((size_t)K_top * sizeof(float));
    if (!refined) { free(top_ids); free(top_vals); return -1; }
    rc = ib_pq_matmul_fp32_subset(t, x, top_ids, K_top, refined);
    if (rc != 0) { free(top_ids); free(top_vals); free(refined); return rc; }

    /* Stage 4: sort by refined logit descending. Insertion sort for K small. */
    /* Build pairs (refined, id) and sort. */
    for (int i = 1; i < K_top; i++) {
        float v = refined[i];
        int32_t id = top_ids[i];
        int j = i - 1;
        while (j >= 0 && refined[j] < v) {
            refined[j + 1] = refined[j];
            top_ids[j + 1] = top_ids[j];
            j--;
        }
        refined[j + 1] = v;
        top_ids[j + 1] = id;
    }

    for (int i = 0; i < K_top; i++) {
        out_logits[i] = refined[i];
        out_token_ids[i] = top_ids[i];
    }
    free(top_ids); free(top_vals); free(refined);
    return 0;
}

/* ── Byte-quantised LUT path ───────────────────────────────────── */
/*
 * Same math as matmul_impl but the per-chunk partial-products tables
 * are stored as int8 + per-(chunk,side) fp32 scale. Per-row inner
 * loop reads 2 bytes per chunk (vs 8 bytes fp32), then a single
 * fp32 fma per chunk to apply the chunk scales.
 *
 * Key trade-off:
 *   - 4x less per-chunk table size (2 bytes/k vs 8 bytes/k)
 *   - 32x less per-row L2 traffic on cache-line basis
 *   - +1 fp32 multiply per chunk in the inner loop
 *   - quantisation error: ~1 fp16-ULP per chunk-side (bounded)
 */

typedef struct {
    const int8_t* T1q;       /* M x C x K x 2 int8 (L,R interleaved per index) */
    const float*  T1s;       /* C x 2 fp32 (L scale, R scale) per chunk */
    const int8_t* T2q;       /* may be NULL */
    const float*  T2s;
    const ib_pq_tensor* t;
    int K, C;
    float* out;
} pq_q8_ctx;

static void pq_q8_gather_rows(pq_q8_ctx* g, int r0, int r1) {
    const ib_pq_tensor* t = g->t;
    int C = g->C, K = g->K;
    const int8_t* T1q = g->T1q;
    const float*  T1s = g->T1s;
    const int8_t* T2q = g->T2q;
    const float*  T2s = g->T2s;
    float* out = g->out;
    for (int r = r0; r < r1; r++) {
        float scale = ib_fp16_to_fp32(t->row_scale[r]);
        const uint8_t* il_l = t->indices_l1_l + (size_t)r * C;
        const uint8_t* il_r = t->indices_l1_r + (size_t)r * C;
        const uint8_t* il2l = T2q ? t->indices_l2_l + (size_t)r * C : NULL;
        const uint8_t* il2r = T2q ? t->indices_l2_r + (size_t)r * C : NULL;
        float acc = 0.0f;
        if (T2q) {
            for (int c = 0; c < C; c++) {
                size_t base = (size_t)c * (size_t)K * 2;
                const float* sc1 = T1s + (size_t)c * 2;
                const float* sc2 = T2s + (size_t)c * 2;
                acc += T1q[base + (size_t)il_l[c] * 2 + 0] * sc1[0]
                     + T1q[base + (size_t)il_r[c] * 2 + 1] * sc1[1];
                acc += T2q[base + (size_t)il2l[c] * 2 + 0] * sc2[0]
                     + T2q[base + (size_t)il2r[c] * 2 + 1] * sc2[1];
            }
        } else {
            for (int c = 0; c < C; c++) {
                size_t base = (size_t)c * (size_t)K * 2;
                const float* sc = T1s + (size_t)c * 2;
                acc += T1q[base + (size_t)il_l[c] * 2 + 0] * sc[0]
                     + T1q[base + (size_t)il_r[c] * 2 + 1] * sc[1];
            }
        }
        out[r] = acc * scale;
    }
}

static void pq_q8_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    pq_q8_gather_rows((pq_q8_ctx*)arg, start, end);
}

/* Build the int8 LUT and per-chunk scales from x and the codebook for
 * one side. side_off picks which half of each chunk's columns to dot
 * against (0 = first half, `half` = second half). */
static int build_q8_lut(int C, int K, int half, int G,
                         const float* cb,
                         const int* inner_cols, const float* x,
                         int side_off,
                         int8_t** out_q, float** out_s) {
    size_t fp_sz = (size_t)C * K;
    float* tmp = (float*)malloc(fp_sz * sizeof(float));
    if (!tmp) return -1;
    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xs[16];
        for (int i = 0; i < half; i++) xs[i] = x[inner_cols[base + side_off + i]];
        for (int k = 0; k < K; k++) {
            float s = 0.0f;
            const float* cv = &cb[(size_t)k * half];
            for (int i = 0; i < half; i++) s += cv[i] * xs[i];
            tmp[(size_t)c * K + k] = s;
        }
    }

    int8_t* q = (int8_t*)malloc(fp_sz);
    float*  s = (float*)malloc((size_t)C * sizeof(float));
    if (!q || !s) { free(tmp); free(q); free(s); return -1; }

    for (int c = 0; c < C; c++) {
        const float* row = tmp + (size_t)c * K;
        float maxabs = 0.0f;
        for (int k = 0; k < K; k++) {
            float v = row[k];
            float av = v < 0 ? -v : v;
            if (av > maxabs) maxabs = av;
        }
        float sc = (maxabs > 0.0f) ? (maxabs / 127.0f) : 1.0f;
        float inv = 1.0f / sc;
        s[c] = sc;
        for (int k = 0; k < K; k++) {
            float scaled = row[k] * inv;
            int qi = (int)(scaled >= 0 ? scaled + 0.5f : scaled - 0.5f);
            if (qi >  127) qi =  127;
            if (qi < -128) qi = -128;
            q[(size_t)c * K + k] = (int8_t)qi;
        }
    }
    free(tmp);
    *out_q = q;
    *out_s = s;
    return 0;
}

int ib_pq_matmul_fp32_q8lut(const ib_pq_tensor* t, const float* x,
                             float* out, void* pool_v) {
    if (!t || !x || !out) return -1;
    ib_thread_pool* pool = (ib_thread_pool*)pool_v;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;
    (void)M;

    /* Decode codebooks to fp32 once. */
    int cb_entries = K * half;
    float* cb1l = (float*)malloc((size_t)cb_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb_entries * sizeof(float));
    float* cb2l = NULL, *cb2r = NULL;
    if (!cb1l || !cb1r) { free(cb1l); free(cb1r); return -1; }
    for (int i = 0; i < cb_entries; i++) {
        cb1l[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        cb1r[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    if (t->n_levels == 2) {
        cb2l = (float*)malloc((size_t)cb_entries * sizeof(float));
        cb2r = (float*)malloc((size_t)cb_entries * sizeof(float));
        if (!cb2l || !cb2r) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
        for (int i = 0; i < cb_entries; i++) {
            cb2l[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            cb2r[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }
    }

    /* Inner-cols list */
    int* inner_cols = (int*)malloc((size_t)(N - t->n_outlier) * sizeof(int));
    if (!inner_cols) { free(cb1l); free(cb1r); free(cb2l); free(cb2r); return -1; }
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int kk = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) inner_cols[kk++] = j;
        free(mask);
    }

    /* Build per-side LUTs in q8 form, then interleave into the
     * inner-loop layout T1q[c*K*2 + k*2 + side]. */
    int8_t* qL = NULL; float* sL = NULL;
    int8_t* qR = NULL; float* sR = NULL;
    int8_t* qL2 = NULL; float* sL2 = NULL;
    int8_t* qR2 = NULL; float* sR2 = NULL;
    if (build_q8_lut(C, K, half, G, cb1l, inner_cols, x, 0,    &qL, &sL) != 0
        || build_q8_lut(C, K, half, G, cb1r, inner_cols, x, half, &qR, &sR) != 0) {
        goto fail;
    }
    if (cb2l) {
        if (build_q8_lut(C, K, half, G, cb2l, inner_cols, x, 0,    &qL2, &sL2) != 0
            || build_q8_lut(C, K, half, G, cb2r, inner_cols, x, half, &qR2, &sR2) != 0) {
            goto fail;
        }
    }

    size_t T_sz = (size_t)C * K * 2;
    int8_t* T1q = (int8_t*)malloc(T_sz);
    float*  T1s = (float*)malloc((size_t)C * 2 * sizeof(float));
    int8_t* T2q = NULL; float* T2s = NULL;
    if (!T1q || !T1s) { free(T1q); free(T1s); goto fail; }
    if (cb2l) {
        T2q = (int8_t*)malloc(T_sz);
        T2s = (float*)malloc((size_t)C * 2 * sizeof(float));
        if (!T2q || !T2s) { free(T1q); free(T1s); free(T2q); free(T2s); goto fail; }
    }

    for (int c = 0; c < C; c++) {
        T1s[c*2 + 0] = sL[c];
        T1s[c*2 + 1] = sR[c];
        const int8_t* lq = qL + (size_t)c * K;
        const int8_t* rq = qR + (size_t)c * K;
        int8_t* dst = T1q + (size_t)c * K * 2;
        for (int k = 0; k < K; k++) {
            dst[k*2 + 0] = lq[k];
            dst[k*2 + 1] = rq[k];
        }
        if (cb2l) {
            T2s[c*2 + 0] = sL2[c];
            T2s[c*2 + 1] = sR2[c];
            const int8_t* l2q = qL2 + (size_t)c * K;
            const int8_t* r2q = qR2 + (size_t)c * K;
            int8_t* d2 = T2q + (size_t)c * K * 2;
            for (int k = 0; k < K; k++) {
                d2[k*2 + 0] = l2q[k];
                d2[k*2 + 1] = r2q[k];
            }
        }
    }
    free(qL); free(sL); free(qR); free(sR);
    free(qL2); free(sL2); free(qR2); free(sR2);
    qL = qR = qL2 = qR2 = NULL;
    sL = sR = sL2 = sR2 = NULL;

    /* Zero output: outlier path adds in-place. */
    pq_q8_ctx ctx = {
        .T1q = T1q, .T1s = T1s, .T2q = T2q, .T2s = T2s,
        .t = t, .K = K, .C = C, .out = out,
    };
    if (pool) ib_pool_run(pool, pq_q8_task, &ctx, t->M, 0);
    else      pq_q8_gather_rows(&ctx, 0, t->M);

    /* Outliers (same NEON int8 dot path as fp32 LUT). */
    apply_outliers(t, x, out);

    free(cb1l); free(cb1r); free(cb2l); free(cb2r);
    free(inner_cols); free(T1q); free(T1s); free(T2q); free(T2s);
    return 0;

fail:
    free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
    free(qL); free(sL); free(qR); free(sR);
    free(qL2); free(sL2); free(qR2); free(sR2);
    return -1;
}

/* ── Phase 6: persistent codebook + inner_cols + per-cluster bound cache ── */

struct ib_pq_lut_cache {
    float* cb1l_fp32;
    float* cb1r_fp32;
    float* cb2l_fp32;
    float* cb2r_fp32;
    int*   inner_cols;
    float* c2l_max_per_cluster;
    float* c2r_max_per_cluster;
    int M, N, G, K, K_l2, half, n_levels;
    int n_inner;
    int is_pyramid;
    int n_outer;
    int K_inner;
    /* Phase 3: optional int8 quantized codebooks (NULL until quantize_int8). */
    int8_t* cb1l_q8;
    int8_t* cb1r_q8;
    int8_t* cb2l_q8;
    int8_t* cb2r_q8;
    float* cb1l_scale;  /* [K] */
    float* cb1r_scale;
    float* cb2l_scale;  /* [K_l2] */
    float* cb2r_scale;
    /* Phase 8.X: transposed indices [C][M] for sequential reads in row loop.
     * For L2: stored as flat uint16 to avoid runtime unpacking inside the
     * hot loop (handles K_l2 in [1, 65535]). */
    uint8_t*  i1l_T;  /* [C][M] uint8 */
    uint8_t*  i1r_T;
    uint16_t* i2l_T;  /* [C][M] uint16, valid when n_levels == 2 */
    uint16_t* i2r_T;
    int has_idx_T;
    /* F1.h: packed [C][M] uint64 layout for n_levels=2:
     *   bits  [0..8)  i1l   (uint8)
     *   bits  [8..16) i1r   (uint8)
     *   bits  [16..32) i2l  (uint16)
     *   bits  [32..48) i2r  (uint16)
     * One sequential 8-byte load per row in the streaming kernel
     * instead of 4 strided loads from 4 separate streams. */
    uint64_t* idx_T_packed;
};

int ib_pq_lut_cache_create(const ib_pq_tensor* t, ib_pq_lut_cache** out_p) {
    if (!t || !out_p) return -1;
    *out_p = NULL;
    int M = t->M, N = t->N, G = t->G, K = t->K;
    int half = G / 2;
    int K_l2 = pq_K_l2_eff(t);
    int has_l2 = (t->n_levels == 2);

    ib_pq_lut_cache* c = (ib_pq_lut_cache*)calloc(1, sizeof(*c));
    if (!c) return -1;
    c->M = M; c->N = N; c->G = G; c->K = K; c->K_l2 = K_l2;
    c->half = half; c->n_levels = t->n_levels;
    c->is_pyramid = (t->format == IB_PQ_FMT_PYRAMID);
    c->n_outer = (c->is_pyramid && K > 0) ? K : 1;
    c->K_inner = (c->is_pyramid && K > 0) ? (K_l2 / K) : K_l2;
    c->n_inner = N - t->n_outlier;

    int cb1_entries = K * half;
    c->cb1l_fp32 = (float*)malloc((size_t)cb1_entries * sizeof(float));
    c->cb1r_fp32 = (float*)malloc((size_t)cb1_entries * sizeof(float));
    if (!c->cb1l_fp32 || !c->cb1r_fp32) goto fail;
    for (int i = 0; i < cb1_entries; i++) {
        c->cb1l_fp32[i] = ib_fp16_to_fp32(t->codebook_l1_l[i]);
        c->cb1r_fp32[i] = ib_fp16_to_fp32(t->codebook_l1_r[i]);
    }
    if (has_l2) {
        int cb2_entries = K_l2 * half;
        c->cb2l_fp32 = (float*)malloc((size_t)cb2_entries * sizeof(float));
        c->cb2r_fp32 = (float*)malloc((size_t)cb2_entries * sizeof(float));
        if (!c->cb2l_fp32 || !c->cb2r_fp32) goto fail;
        for (int i = 0; i < cb2_entries; i++) {
            c->cb2l_fp32[i] = ib_fp16_to_fp32(t->codebook_l2_l[i]);
            c->cb2r_fp32[i] = ib_fp16_to_fp32(t->codebook_l2_r[i]);
        }

        c->c2l_max_per_cluster = (float*)calloc((size_t)c->n_outer, sizeof(float));
        c->c2r_max_per_cluster = (float*)calloc((size_t)c->n_outer, sizeof(float));
        if (!c->c2l_max_per_cluster || !c->c2r_max_per_cluster) goto fail;
        for (int k1 = 0; k1 < c->n_outer; k1++) {
            float ml = 0.0f, mr = 0.0f;
            int k_lo = c->is_pyramid ? k1 * c->K_inner : 0;
            int k_hi = c->is_pyramid ? (k1 + 1) * c->K_inner : K_l2;
            for (int k = k_lo; k < k_hi; k++) {
                float nl = 0.0f, nr = 0.0f;
                for (int j = 0; j < half; j++) {
                    nl += c->cb2l_fp32[k * half + j] * c->cb2l_fp32[k * half + j];
                    nr += c->cb2r_fp32[k * half + j] * c->cb2r_fp32[k * half + j];
                }
                nl = sqrtf(nl); nr = sqrtf(nr);
                if (nl > ml) ml = nl;
                if (nr > mr) mr = nr;
            }
            c->c2l_max_per_cluster[k1] = ml;
            c->c2r_max_per_cluster[k1] = mr;
        }
    }

    c->inner_cols = (int*)malloc((size_t)c->n_inner * sizeof(int));
    if (!c->inner_cols) goto fail;
    {
        uint8_t* mask = (uint8_t*)calloc((size_t)N, 1);
        if (!mask) goto fail;
        for (int i = 0; i < t->n_outlier; i++) mask[t->outlier_cols[i]] = 1;
        int k = 0;
        for (int j = 0; j < N; j++) if (!mask[j]) c->inner_cols[k++] = j;
        free(mask);
    }

    /* Phase 8.X: transpose indices to [C][M] for sequential row-loop reads. */
    int C = t->C;
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;
    c->i1l_T = (uint8_t*)malloc((size_t)C * (size_t)M);
    c->i1r_T = (uint8_t*)malloc((size_t)C * (size_t)M);
    if (!c->i1l_T || !c->i1r_T) goto fail;
    for (int r = 0; r < M; r++) {
        for (int cc = 0; cc < C; cc++) {
            c->i1l_T[(size_t)cc * M + r] = t->indices_l1_l[(size_t)r * C + cc];
            c->i1r_T[(size_t)cc * M + r] = t->indices_l1_r[(size_t)r * C + cc];
        }
    }
    if (has_l2) {
        c->i2l_T = (uint16_t*)malloc((size_t)C * (size_t)M * sizeof(uint16_t));
        c->i2r_T = (uint16_t*)malloc((size_t)C * (size_t)M * sizeof(uint16_t));
        if (!c->i2l_T || !c->i2r_T) goto fail;
        for (int r = 0; r < M; r++) {
            const uint8_t* il2l = t->indices_l2_l + (size_t)r * l2_stride;
            const uint8_t* il2r = t->indices_l2_r + (size_t)r * l2_stride;
            for (int cc = 0; cc < C; cc++) {
                c->i2l_T[(size_t)cc * M + r] = (uint16_t)pq_l2_idx_at(il2l, cc, l2_packed, l2_idx_bytes);
                c->i2r_T[(size_t)cc * M + r] = (uint16_t)pq_l2_idx_at(il2r, cc, l2_packed, l2_idx_bytes);
            }
        }
    }
    c->has_idx_T = 1;

    /* F1.h: pack i1l, i1r, i2l, i2r into one [C][M] uint64 stream.
     * Only when n_levels==2; otherwise the i1l/i1r uint8 path is small
     * enough on its own. */
    if (has_l2) {
        c->idx_T_packed = (uint64_t*)malloc((size_t)C * (size_t)M * sizeof(uint64_t));
        if (!c->idx_T_packed) goto fail;
        for (int cc = 0; cc < C; cc++) {
            const uint8_t*  i1l = c->i1l_T + (size_t)cc * M;
            const uint8_t*  i1r = c->i1r_T + (size_t)cc * M;
            const uint16_t* i2l = c->i2l_T + (size_t)cc * M;
            const uint16_t* i2r = c->i2r_T + (size_t)cc * M;
            uint64_t* dst = c->idx_T_packed + (size_t)cc * M;
            for (int r = 0; r < M; r++) {
                dst[r] = ((uint64_t)i1l[r])
                       | ((uint64_t)i1r[r]    << 8)
                       | ((uint64_t)i2l[r]    << 16)
                       | ((uint64_t)i2r[r]    << 32);
            }
        }
    }

    *out_p = c;
    return 0;

fail:
    ib_pq_lut_cache_free(c);
    return -1;
}

void ib_pq_lut_cache_free(ib_pq_lut_cache* c) {
    if (!c) return;
    free(c->cb1l_fp32); free(c->cb1r_fp32);
    free(c->cb2l_fp32); free(c->cb2r_fp32);
    free(c->inner_cols);
    free(c->c2l_max_per_cluster); free(c->c2r_max_per_cluster);
    free(c->cb1l_q8); free(c->cb1r_q8);
    free(c->cb2l_q8); free(c->cb2r_q8);
    free(c->cb1l_scale); free(c->cb1r_scale);
    free(c->cb2l_scale); free(c->cb2r_scale);
    free(c->i1l_T); free(c->i1r_T);
    free(c->i2l_T); free(c->i2r_T);
    free(c->idx_T_packed);
    free(c);
}

/* Quantize a [K, half] fp32 codebook to int8 with per-row scale.
 * scale[k] = max|cb[k]|/127; q[k][j] = round(cb[k][j] / scale[k]) clamped to [-127, 127].
 */
static int pq_quantize_codebook_int8(const float* cb_fp32, int K, int half,
                                       int8_t** out_q, float** out_scale) {
    int8_t* q = (int8_t*)malloc((size_t)K * half);
    float* s = (float*)malloc((size_t)K * sizeof(float));
    if (!q || !s) { free(q); free(s); return -1; }
    for (int k = 0; k < K; k++) {
        const float* row = cb_fp32 + (size_t)k * half;
        float m = 0.0f;
        for (int j = 0; j < half; j++) {
            float a = fabsf(row[j]);
            if (a > m) m = a;
        }
        float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
        float inv = 1.0f / scale;
        for (int j = 0; j < half; j++) {
            float v = row[j] * inv;
            int qi = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
            if (qi >  127) qi =  127;
            if (qi < -127) qi = -127;
            q[(size_t)k * half + j] = (int8_t)qi;
        }
        s[k] = scale;
    }
    *out_q = q; *out_scale = s;
    return 0;
}

int ib_pq_lut_cache_quantize_int8(ib_pq_lut_cache* c) {
    if (!c) return -1;
    if (c->cb1l_q8 != NULL) return 0;  /* idempotent */
    int K = c->K, K_l2 = c->K_l2, half = c->half;
    int has_l2 = (c->n_levels == 2);

    if (pq_quantize_codebook_int8(c->cb1l_fp32, K, half,
                                    &c->cb1l_q8, &c->cb1l_scale) != 0) return -1;
    if (pq_quantize_codebook_int8(c->cb1r_fp32, K, half,
                                    &c->cb1r_q8, &c->cb1r_scale) != 0) return -1;
    if (has_l2) {
        if (pq_quantize_codebook_int8(c->cb2l_fp32, K_l2, half,
                                        &c->cb2l_q8, &c->cb2l_scale) != 0) return -1;
        if (pq_quantize_codebook_int8(c->cb2r_fp32, K_l2, half,
                                        &c->cb2r_q8, &c->cb2r_scale) != 0) return -1;
    }
    return 0;
}

/* INT8 chunk dot table: int_dot[k] = sum_j q_cb[k][j] * q_x[j];
 * out[k] = int_dot[k] * cb_scale[k] * x_scale.
 * Specialized for half ∈ {8, 16}; NEON dotprod path when available. */
static inline void pq_chunk_dot_table_int8(const int8_t* cb_q, const float* cb_scale,
                                             const int8_t* xq, float x_scale,
                                             int half, int K, float* out) {
#if defined(IB_PQ_HAVE_NEON_DOTPROD)
    if (half == 16) {
        int8x16_t xv = vld1q_s8(xq);
        for (int k = 0; k < K; k++) {
            int8x16_t ev = vld1q_s8(cb_q + (size_t)k * 16);
            int32x4_t acc = vdupq_n_s32(0);
            acc = vdotq_s32(acc, ev, xv);
            int32_t s = vaddvq_s32(acc);
            out[k] = (float)s * cb_scale[k] * x_scale;
        }
        return;
    }
#endif
    for (int k = 0; k < K; k++) {
        const int8_t* e = cb_q + (size_t)k * half;
        int32_t s = 0;
        for (int j = 0; j < half; j++) s += (int32_t)e[j] * (int32_t)xq[j];
        out[k] = (float)s * cb_scale[k] * x_scale;
    }
}

/* Quantize a small fp32 chunk (length half) to int8 with single scale. */
static inline float pq_quantize_chunk_int8(const float* x, int half, int8_t* xq) {
    float m = 0.0f;
    for (int j = 0; j < half; j++) {
        float a = fabsf(x[j]);
        if (a > m) m = a;
    }
    float scale = (m > 0.0f) ? (m / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    for (int j = 0; j < half; j++) {
        float v = x[j] * inv;
        int qi = (int)(v >= 0.0f ? (v + 0.5f) : (v - 0.5f));
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        xq[j] = (int8_t)qi;
    }
    return scale;
}

int ib_pq_matmul_fp32_streaming_int8_cached(const ib_pq_tensor* t,
                                              const ib_pq_lut_cache* cache,
                                              const float* x, float* out) {
    if (!t || !cache || !x || !out) return -1;
    if (!cache->cb1l_q8) return -1;  /* must call quantize_int8 first */
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int has_l2 = (cache->n_levels == 2);

    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    float* C2R_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    if (!C1L_dot_x || !C1R_dot_x || (has_l2 && (!C2L_dot_x || !C2R_dot_x))) {
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));

    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        int8_t xLq[16], xRq[16];
        for (int j = 0; j < half; j++) {
            xL[j] = x[cache->inner_cols[base + j]];
            xR[j] = x[cache->inner_cols[base + half + j]];
        }
        float xL_scale = pq_quantize_chunk_int8(xL, half, xLq);
        float xR_scale = pq_quantize_chunk_int8(xR, half, xRq);

        pq_chunk_dot_table_int8(cache->cb1l_q8, cache->cb1l_scale,
                                xLq, xL_scale, half, K, C1L_dot_x);
        pq_chunk_dot_table_int8(cache->cb1r_q8, cache->cb1r_scale,
                                xRq, xR_scale, half, K, C1R_dot_x);
        if (has_l2) {
            pq_chunk_dot_table_int8(cache->cb2l_q8, cache->cb2l_scale,
                                    xLq, xL_scale, half, K_l2, C2L_dot_x);
            pq_chunk_dot_table_int8(cache->cb2r_q8, cache->cb2r_scale,
                                    xRq, xR_scale, half, K_l2, C2R_dot_x);
        }

        const uint8_t* i1l_row = cache->i1l_T + (size_t)c * M;
        const uint8_t* i1r_row = cache->i1r_T + (size_t)c * M;
        if (has_l2) {
            const uint16_t* i2l_row = cache->i2l_T + (size_t)c * M;
            const uint16_t* i2r_row = cache->i2r_T + (size_t)c * M;
            for (int r = 0; r < M; r++) {
                out[r] += C1L_dot_x[i1l_row[r]] + C1R_dot_x[i1r_row[r]]
                        + C2L_dot_x[i2l_row[r]] + C2R_dot_x[i2r_row[r]];
            }
        } else {
            for (int r = 0; r < M; r++) {
                out[r] += C1L_dot_x[i1l_row[r]] + C1R_dot_x[i1r_row[r]];
            }
        }
    }

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);

    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return 0;
}

/* F1.a: shared kernel with caller-supplied scratch. */
static int streaming_cached_kernel(const ib_pq_tensor* t,
                                    const ib_pq_lut_cache* cache,
                                    const float* x, float* out,
                                    float* C1L_dot_x, float* C1R_dot_x,
                                    float* C2L_dot_x, float* C2R_dot_x) {
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int has_l2 = (cache->n_levels == 2);

    memset(out, 0, (size_t)M * sizeof(float));

    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        for (int j = 0; j < half; j++) {
            xL[j] = x[cache->inner_cols[base + j]];
            xR[j] = x[cache->inner_cols[base + half + j]];
        }
        pq_chunk_dot_table_f32(cache->cb1l_fp32, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cache->cb1r_fp32, xR, half, K, C1R_dot_x);
        if (has_l2) {
            pq_chunk_dot_table_f32(cache->cb2l_fp32, xL, half, K_l2, C2L_dot_x);
            pq_chunk_dot_table_f32(cache->cb2r_fp32, xR, half, K_l2, C2R_dot_x);
        }

        if (has_l2 && cache->idx_T_packed) {
            /* F1.h: one sequential u64 load per row, all 4 indices unpacked
             * with bit ops. ~4× fewer cache lines touched in the hot loop. */
            const uint64_t* p_row = cache->idx_T_packed + (size_t)c * M;
            if (c + 1 < C) __builtin_prefetch(p_row + M, 0, 1);
            for (int r = 0; r < M; r++) {
                uint64_t p = p_row[r];
                int i1l = (int)( p        & 0xff);
                int i1r = (int)((p >>  8) & 0xff);
                int i2l = (int)((p >> 16) & 0xffff);
                int i2r = (int)((p >> 32) & 0xffff);
                out[r] += C1L_dot_x[i1l] + C1R_dot_x[i1r]
                        + C2L_dot_x[i2l] + C2R_dot_x[i2r];
            }
        } else {
            const uint8_t* i1l_row = cache->i1l_T + (size_t)c * M;
            const uint8_t* i1r_row = cache->i1r_T + (size_t)c * M;
            if (c + 1 < C) {
                __builtin_prefetch(cache->i1l_T + (size_t)(c + 1) * M, 0, 1);
                __builtin_prefetch(cache->i1r_T + (size_t)(c + 1) * M, 0, 1);
            }
            for (int r = 0; r < M; r++) {
                out[r] += C1L_dot_x[i1l_row[r]] + C1R_dot_x[i1r_row[r]];
            }
        }
    }

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);
    return 0;
}

int ib_pq_matmul_fp32_streaming_cached(const ib_pq_tensor* t,
                                        const ib_pq_lut_cache* cache,
                                        const float* x, float* out) {
    if (!t || !cache || !x || !out) return -1;
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int has_l2 = (cache->n_levels == 2);

    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    float* C2R_dot_x = has_l2 ? (float*)malloc((size_t)K_l2 * sizeof(float)) : NULL;
    if (!C1L_dot_x || !C1R_dot_x || (has_l2 && (!C2L_dot_x || !C2R_dot_x))) {
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }
    int rc = streaming_cached_kernel(t, cache, x, out,
                                       C1L_dot_x, C1R_dot_x, C2L_dot_x, C2R_dot_x);
    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return rc;
}

/* Threaded streaming_cached: each thread processes a row slice across
 * all chunks. LUT scratch is allocated per-call (thread-local on stack
 * or one shared malloc partitioned by tid). */
typedef struct {
    const ib_pq_tensor* t;
    const ib_pq_lut_cache* cache;
    const float* x;
    float* out;
    int M, G, C, K, K_l2, half;
    int has_l2;
    int n_threads;
    /* Per-thread scratch: 4 buffers each [K + K_l2] floats. Allocated
     * by the caller as one contiguous block. */
    float* scratch_base;
    size_t scratch_per_thread;
} pq_streaming_cached_threaded_args;

static void pq_streaming_cached_row_task(void* arg, int tid, int r0, int r1) {
    pq_streaming_cached_threaded_args* a = (pq_streaming_cached_threaded_args*)arg;
    const ib_pq_lut_cache* cache = a->cache;
    int M = a->M, G = a->G, C = a->C, K = a->K, K_l2 = a->K_l2, half = a->half;
    int has_l2 = a->has_l2;

    /* Per-thread LUT scratch slice. */
    float* base = a->scratch_base + (size_t)tid * a->scratch_per_thread;
    float* C1L = base;
    float* C1R = base + K;
    float* C2L = has_l2 ? (base + 2 * K) : NULL;
    float* C2R = has_l2 ? (base + 2 * K + K_l2) : NULL;

    for (int c = 0; c < C; c++) {
        int bs = c * G;
        float xL[16], xR[16];
        for (int j = 0; j < half; j++) {
            xL[j] = a->x[cache->inner_cols[bs + j]];
            xR[j] = a->x[cache->inner_cols[bs + half + j]];
        }
        pq_chunk_dot_table_f32(cache->cb1l_fp32, xL, half, K, C1L);
        pq_chunk_dot_table_f32(cache->cb1r_fp32, xR, half, K, C1R);
        if (has_l2) {
            pq_chunk_dot_table_f32(cache->cb2l_fp32, xL, half, K_l2, C2L);
            pq_chunk_dot_table_f32(cache->cb2r_fp32, xR, half, K_l2, C2R);
        }

        if (has_l2 && cache->idx_T_packed) {
            /* F1.h packed indices: one u64 load per row, no strided streams. */
            const uint64_t* p_row = cache->idx_T_packed + (size_t)c * M;
            for (int r = r0; r < r1; r++) {
                uint64_t p = p_row[r];
                a->out[r] += C1L[(int)( p        & 0xff)]
                            + C1R[(int)((p >>  8) & 0xff)]
                            + C2L[(int)((p >> 16) & 0xffff)]
                            + C2R[(int)((p >> 32) & 0xffff)];
            }
        } else {
            const uint8_t* i1l_row = cache->i1l_T + (size_t)c * M;
            const uint8_t* i1r_row = cache->i1r_T + (size_t)c * M;
            for (int r = r0; r < r1; r++) {
                a->out[r] += C1L[i1l_row[r]] + C1R[i1r_row[r]];
            }
        }
    }
}

/* Forward declarations for spin-pool (defined below). */
typedef struct pq_spin_pool pq_spin_pool;
typedef void (*pq_spin_fn)(void* arg, int tid, int s, int e);
static void pq_spin_pool_run(pq_spin_pool* p, pq_spin_fn fn, void* arg,
                              int total, int chunk_size);

static int ib_pq_matmul_fp32_streaming_cached_spin(const ib_pq_tensor* t,
                                                     const ib_pq_lut_cache* cache,
                                                     const float* x, float* out,
                                                     pq_spin_pool* sp, int n_threads,
                                                     float* scratch, size_t per_thread) {
    if (!sp || n_threads <= 1) return -1;
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int has_l2 = (cache->n_levels == 2);

    memset(out, 0, (size_t)M * sizeof(float));

    pq_streaming_cached_threaded_args args = {
        .t = t, .cache = cache, .x = x, .out = out,
        .M = M, .G = G, .C = C, .K = K, .K_l2 = K_l2, .half = half,
        .has_l2 = has_l2,
        .n_threads = n_threads,
        .scratch_base = scratch,
        .scratch_per_thread = per_thread,
    };

    int chunk = (M + n_threads - 1) / n_threads;
    if (chunk < 1) chunk = 1;
    pq_spin_pool_run(sp, pq_streaming_cached_row_task, &args, M, chunk);

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);
    return 0;
}

static int ib_pq_matmul_fp32_streaming_cached_threaded(const ib_pq_tensor* t,
                                                         const ib_pq_lut_cache* cache,
                                                         const float* x, float* out,
                                                         ib_thread_pool* pool, int n_threads) {
    if (!pool || n_threads <= 1) {
        return ib_pq_matmul_fp32_streaming_cached(t, cache, x, out);
    }
    if (!t || !cache || !x || !out) return -1;
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int has_l2 = (cache->n_levels == 2);

    /* Per-thread scratch: 4 LUT buffers (C1L, C1R, C2L, C2R). */
    size_t per_thread = (size_t)(2 * K + (has_l2 ? 2 * K_l2 : 0));
    size_t total = per_thread * (size_t)n_threads;
    float* scratch = (float*)malloc(total * sizeof(float));
    if (!scratch) {
        return ib_pq_matmul_fp32_streaming_cached(t, cache, x, out);
    }

    memset(out, 0, (size_t)M * sizeof(float));

    pq_streaming_cached_threaded_args args = {
        .t = t, .cache = cache, .x = x, .out = out,
        .M = M, .G = G, .C = C, .K = K, .K_l2 = K_l2, .half = half,
        .has_l2 = has_l2,
        .n_threads = n_threads,
        .scratch_base = scratch,
        .scratch_per_thread = per_thread,
    };

    /* One row slice per thread. */
    int chunk = (M + n_threads - 1) / n_threads;
    if (chunk < 1) chunk = 1;
    ib_pool_run(pool, pq_streaming_cached_row_task, &args, M, chunk);

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);

    free(scratch);
    return 0;
}

int ib_pq_matmul_fp32_streaming_l2skip_cached(const ib_pq_tensor* t,
                                                const ib_pq_lut_cache* cache,
                                                const float* x, float* out,
                                                float skip_threshold) {
    if (!t || !cache || !x || !out) return -1;
    if (cache->n_levels != 2) return ib_pq_matmul_fp32_streaming_cached(t, cache, x, out);
    int M = t->M, G = t->G, C = t->C, K = t->K;
    int half = cache->half;
    int K_l2 = cache->K_l2;
    int is_pyramid = cache->is_pyramid;

    float* C1L_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C1R_dot_x = (float*)malloc((size_t)K * sizeof(float));
    float* C2L_dot_x = (float*)malloc((size_t)K_l2 * sizeof(float));
    float* C2R_dot_x = (float*)malloc((size_t)K_l2 * sizeof(float));
    if (!C1L_dot_x || !C1R_dot_x || !C2L_dot_x || !C2R_dot_x) {
        free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
        return -1;
    }

    memset(out, 0, (size_t)M * sizeof(float));

    for (int c = 0; c < C; c++) {
        int base = c * G;
        float xL[16], xR[16];
        float xL_n2 = 0.0f, xR_n2 = 0.0f;
        for (int j = 0; j < half; j++) {
            xL[j] = x[cache->inner_cols[base + j]];
            xR[j] = x[cache->inner_cols[base + half + j]];
            xL_n2 += xL[j] * xL[j];
            xR_n2 += xR[j] * xR[j];
        }
        float xL_norm = sqrtf(xL_n2);
        float xR_norm = sqrtf(xR_n2);

        pq_chunk_dot_table_f32(cache->cb1l_fp32, xL, half, K, C1L_dot_x);
        pq_chunk_dot_table_f32(cache->cb1r_fp32, xR, half, K, C1R_dot_x);
        pq_chunk_dot_table_f32(cache->cb2l_fp32, xL, half, K_l2, C2L_dot_x);
        pq_chunk_dot_table_f32(cache->cb2r_fp32, xR, half, K_l2, C2R_dot_x);

        const uint8_t* i1l_row = cache->i1l_T + (size_t)c * M;
        const uint8_t* i1r_row = cache->i1r_T + (size_t)c * M;
        const uint16_t* i2l_row = cache->i2l_T + (size_t)c * M;
        const uint16_t* i2r_row = cache->i2r_T + (size_t)c * M;
        for (int r = 0; r < M; r++) {
            int i1l = i1l_row[r];
            int i1r = i1r_row[r];
            float l1_contrib = C1L_dot_x[i1l] + C1R_dot_x[i1r];
            float v = l1_contrib;
            int oc_l = is_pyramid ? i1l : 0;
            int oc_r = is_pyramid ? i1r : 0;
            float l2_bound = skip_threshold
                * (cache->c2l_max_per_cluster[oc_l] * xL_norm
                 + cache->c2r_max_per_cluster[oc_r] * xR_norm);
            if (skip_threshold > 0.0f && fabsf(l1_contrib) > l2_bound) {
                /* skip L2 */
            } else {
                v += C2L_dot_x[i2l_row[r]] + C2R_dot_x[i2r_row[r]];
            }
            out[r] += v;
        }
    }

    for (int r = 0; r < M; r++) {
        out[r] *= ib_fp16_to_fp32(t->row_scale[r]);
    }
    apply_outliers(t, x, out);

    free(C1L_dot_x); free(C1R_dot_x); free(C2L_dot_x); free(C2R_dot_x);
    return 0;
}

/* ── Phase 9: multi-tensor cache fleet ── */

struct ib_pq_multi_caches {
    int n;
    ib_pq_lut_cache** caches;
    char** names;            /* shallow refs to multi->names */
};

int ib_pq_multi_caches_create(const ib_pq_multi* multi, ib_pq_multi_caches** out_p) {
    if (!multi || !out_p) return -1;
    *out_p = NULL;
    ib_pq_multi_caches* mc = (ib_pq_multi_caches*)calloc(1, sizeof(*mc));
    if (!mc) return -1;
    mc->n = multi->n;
    if (multi->n > 0) {
        mc->caches = (ib_pq_lut_cache**)calloc((size_t)multi->n, sizeof(*mc->caches));
        mc->names  = (char**)calloc((size_t)multi->n, sizeof(*mc->names));
        if (!mc->caches || !mc->names) { ib_pq_multi_caches_free(mc); return -1; }
    }
    for (int i = 0; i < multi->n; i++) {
        mc->names[i] = multi->names[i];
        if (ib_pq_lut_cache_create(&multi->tensors[i], &mc->caches[i]) != 0) {
            ib_pq_multi_caches_free(mc);
            return -1;
        }
    }
    *out_p = mc;
    return 0;
}

void ib_pq_multi_caches_free(ib_pq_multi_caches* mc) {
    if (!mc) return;
    if (mc->caches) {
        for (int i = 0; i < mc->n; i++) ib_pq_lut_cache_free(mc->caches[i]);
        free(mc->caches);
    }
    free(mc->names);
    free(mc);
}

const ib_pq_lut_cache* ib_pq_multi_caches_get(const ib_pq_multi_caches* mc, const char* name) {
    if (!mc || !name) return NULL;
    for (int i = 0; i < mc->n; i++) {
        if (mc->names[i] && strcmp(mc->names[i], name) == 0) return mc->caches[i];
    }
    return NULL;
}

int ib_pq_multi_caches_quantize_all_int8(ib_pq_multi_caches* mc) {
    if (!mc) return -1;
    for (int i = 0; i < mc->n; i++) {
        if (ib_pq_lut_cache_quantize_int8(mc->caches[i]) != 0) return -1;
    }
    return 0;
}

/* ── Session: IBF + cache fleet + per-tensor policy ── */

/* F1.b: minimal spin pool — workers busy-wait on a generation counter
 * instead of cond_wait. Sync latency drops from O(50–200 µs) to O(<1 µs)
 * which is what makes per-call threading viable in a 154-call/token loop. */

struct pq_spin_pool {
    int n_threads;
    pthread_t* threads;
    _Atomic int generation;
    _Atomic int next_chunk;
    _Atomic int n_done;
    _Atomic int shutdown;
    pq_spin_fn fn;
    void* arg;
    int total;
    int chunk_size;
    int* tids;
    /* F1.b: hybrid spin-then-sleep — workers spin SPIN_LIMIT pq_cpu_pauses
     * before falling back to cond_wait; main always cond_broadcasts after
     * publishing the new generation, so a sleeping worker still wakes. */
    pthread_mutex_t mu;
    pthread_cond_t  cv;
};

#define PQ_SPIN_LIMIT 50000  /* ~50 µs on M-series */

static inline void pq_cpu_pause(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#endif
}

static void* pq_spin_worker(void* raw) {
    int tid_box = *(int*)raw;
    pq_spin_pool* p = *(pq_spin_pool**)((char*)raw + sizeof(int));
    int my_gen = 0;
    while (1) {
        /* Spin briefly first (hot back-to-back call case). */
        int spins = 0;
        int gen;
        while (1) {
            gen = atomic_load_explicit(&p->generation, memory_order_acquire);
            if (gen != my_gen) break;
            if (atomic_load_explicit(&p->shutdown, memory_order_acquire)) return NULL;
            if (++spins >= PQ_SPIN_LIMIT) break;
            pq_cpu_pause();
        }
        if (gen == my_gen) {
            /* Cold path: sleep on condvar. */
            pthread_mutex_lock(&p->mu);
            while ((gen = atomic_load_explicit(&p->generation, memory_order_acquire)) == my_gen
                && !atomic_load_explicit(&p->shutdown, memory_order_acquire)) {
                pthread_cond_wait(&p->cv, &p->mu);
            }
            pthread_mutex_unlock(&p->mu);
        }
        if (atomic_load_explicit(&p->shutdown, memory_order_acquire)) return NULL;
        my_gen = gen;
        pq_spin_fn fn = p->fn;
        void* arg = p->arg;
        int total = p->total;
        int chunk = p->chunk_size;
        while (1) {
            int s = atomic_fetch_add_explicit(&p->next_chunk, chunk, memory_order_relaxed);
            if (s >= total) break;
            int e = s + chunk; if (e > total) e = total;
            fn(arg, tid_box, s, e);
        }
        atomic_fetch_add_explicit(&p->n_done, 1, memory_order_release);
    }
}

static pq_spin_pool* pq_spin_pool_create(int n_threads) {
    if (n_threads <= 1) return NULL;
    pq_spin_pool* p = (pq_spin_pool*)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;
    p->threads = (pthread_t*)calloc((size_t)n_threads, sizeof(pthread_t));
    p->tids = (int*)calloc((size_t)n_threads,
                              sizeof(int) + sizeof(pq_spin_pool*));
    if (!p->threads || !p->tids) { free(p->threads); free(p->tids); free(p); return NULL; }
    atomic_store(&p->generation, 0);
    atomic_store(&p->shutdown, 0);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    /* macOS: give workers user-interactive QoS so the scheduler keeps
     * them on P-cores instead of dumping them on E-cores. Big M-series
     * impact on hot-loop throughput. */
    pthread_attr_t attr;
    pthread_attr_init(&attr);
#if defined(__APPLE__)
    pthread_attr_set_qos_class_np(&attr, QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (int i = 0; i < n_threads; i++) {
        char* slot = (char*)p->tids + (size_t)i * (sizeof(int) + sizeof(pq_spin_pool*));
        *(int*)slot = i;
        *(pq_spin_pool**)(slot + sizeof(int)) = p;
        pthread_create(&p->threads[i], &attr, pq_spin_worker, slot);
    }
    pthread_attr_destroy(&attr);
    return p;
}

static void pq_spin_pool_destroy(pq_spin_pool* p) {
    if (!p) return;
    atomic_store_explicit(&p->shutdown, 1, memory_order_release);
    /* Bump generation + wake any sleepers. */
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->n_threads; i++) pthread_join(p->threads[i], NULL);
    pthread_cond_destroy(&p->cv);
    pthread_mutex_destroy(&p->mu);
    free(p->threads); free(p->tids); free(p);
}

static void pq_spin_pool_run(pq_spin_pool* p, pq_spin_fn fn, void* arg,
                              int total, int chunk_size) {
    if (!p || total <= 0) {
        if (fn && total > 0) fn(arg, 0, 0, total);
        return;
    }
    if (chunk_size <= 0) chunk_size = (total + p->n_threads - 1) / p->n_threads;
    if (chunk_size < 1) chunk_size = 1;
    p->fn = fn; p->arg = arg; p->total = total; p->chunk_size = chunk_size;
    atomic_store_explicit(&p->next_chunk, 0, memory_order_relaxed);
    atomic_store_explicit(&p->n_done, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&p->generation, 1, memory_order_release);
    /* Wake any sleeping workers. Spinning ones already see the new gen. */
    pthread_mutex_lock(&p->mu);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    /* Main thread also takes work — turns N+1 active cores into useful
     * cores instead of having main idle. tid n_threads is the main slot. */
    while (1) {
        int s = atomic_fetch_add_explicit(&p->next_chunk, chunk_size, memory_order_relaxed);
        if (s >= total) break;
        int e = s + chunk_size; if (e > total) e = total;
        fn(arg, p->n_threads, s, e);
    }
    while (atomic_load_explicit(&p->n_done, memory_order_acquire) < p->n_threads) {
        pq_cpu_pause();
    }
}

struct ib_pq_session {
    ib_pq_multi multi;            /* owned */
    ib_pq_multi_caches* mc;       /* owned */
    ib_pq_policy default_policy;
    ib_pq_policy* policies;       /* parallel to multi.tensors[i]; NULL => default */
    int int8_quantized;
    ib_thread_pool* pool;         /* legacy cond_wait pool (kept for compat) */
    pq_spin_pool* spin;           /* F1.b: hot-loop spin pool used by matmul */
    int n_threads;
    /* F1.a: preallocated LUT scratch sized to max(K), max(K_l2) across the
     * fleet. One block, four buffers laid out contiguously. */
    float* scratch_C1L;
    float* scratch_C1R;
    float* scratch_C2L;  /* may be NULL if no L2 in fleet */
    float* scratch_C2R;
    int scratch_K;
    int scratch_K_l2;
    /* F1.b: per-thread scratch for spin-pool matmul (4 LUTs per thread). */
    float* thread_scratch;
    size_t thread_scratch_per;
    /* Phase 8.E: per-tensor inv_act_scale pointer (NULL if absent),
     * indexed parallel to multi.tensors[i]. The session scratch
     * x_scratch is used to materialize x * inv_act_scale once per matmul
     * call; sized to max(N) across the fleet. */
    const float** act_scale_inv_per_tensor;
    float* x_scratch;
    int x_scratch_n;
    /* INT4 weight path: raw_tensor pair `<base>__i4_q` (uint8 [M, N/2])
     * + `<base>__i4_s` (fp16 [M, N/G]). Detected at session_open;
     * routes ahead of PQ in session_matmul. */
    int n_int4;
    struct ib_int4_tensor {
        char* name;            /* base name (no suffix) */
        const uint8_t* q;      /* [M, N/2] packed nibbles */
        const uint16_t* s;     /* [M, N/G] fp16 group scales */
        const float* inv_act;  /* optional [N] fp32, NULL if absent (AWQ) */
        int M, N, G;
    } *int4_tensors;
};

static const struct ib_int4_tensor* session_int4_find(const ib_pq_session* s,
                                                        const char* name) {
    if (!s || !name) return NULL;
    for (int i = 0; i < s->n_int4; i++) {
        if (strcmp(s->int4_tensors[i].name, name) == 0) return &s->int4_tensors[i];
    }
    return NULL;
}

static int int4_matmul_fp32_raw(const struct ib_int4_tensor* t,
                                   const float* x, float* out) {
    const int M = t->M, N = t->N, G = t->G;
    const int n_groups = N / G;
    const uint8_t* q = t->q;
    const uint16_t* sh = t->s;
#if defined(__ARM_NEON)
    const int neon_ok = (G % 16 == 0);
#else
    const int neon_ok = 0;
#endif
    for (int m = 0; m < M; m++) {
        const uint8_t* qrow = q + (size_t)m * (N / 2);
        const uint16_t* srow = sh + (size_t)m * n_groups;
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            float scale = ib_fp16_to_fp32(srow[g]);
            const float* xg = x + g * G;
            const uint8_t* qg = qrow + (g * G) / 2;
            float gacc = 0.0f;
#if defined(__ARM_NEON)
            if (neon_ok) {
                /* NEON int4 dot: 16 columns per iter, 8 packed bytes,
                 * decode to fp32, FMA with x. */
                float32x4_t v0 = vdupq_n_f32(0.0f);
                float32x4_t v1 = vdupq_n_f32(0.0f);
                const int16x8_t bias = vdupq_n_s16(8);
                for (int j = 0; j < G; j += 16) {
                    uint8x8_t b = vld1_u8(qg + j / 2);
                    uint8x8_t lo_u = vand_u8(b, vdup_n_u8(0x0F));
                    uint8x8_t hi_u = vshr_n_u8(b, 4);
                    int16x8_t lo_s = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_u8(lo_u)), bias);
                    int16x8_t hi_s = vsubq_s16(
                        vreinterpretq_s16_u16(vmovl_u8(hi_u)), bias);
                    float32x4_t lo_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_s)));
                    float32x4_t lo_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_s)));
                    float32x4_t hi_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_s)));
                    float32x4_t hi_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_s)));
                    /* Interleave: x is laid out [col0 col1 col2 ...]
                     * lo holds even cols, hi holds odd cols. zip to
                     * get [lo0 hi0 lo1 hi1] etc. */
                    float32x4x2_t za = vzipq_f32(lo_lo, hi_lo);
                    float32x4x2_t zb = vzipq_f32(lo_hi, hi_hi);
                    float32x4_t x0 = vld1q_f32(xg + j);
                    float32x4_t x1 = vld1q_f32(xg + j + 4);
                    float32x4_t x2 = vld1q_f32(xg + j + 8);
                    float32x4_t x3 = vld1q_f32(xg + j + 12);
                    v0 = vfmaq_f32(v0, za.val[0], x0);
                    v1 = vfmaq_f32(v1, za.val[1], x1);
                    v0 = vfmaq_f32(v0, zb.val[0], x2);
                    v1 = vfmaq_f32(v1, zb.val[1], x3);
                }
                gacc = vaddvq_f32(vaddq_f32(v0, v1));
            } else
#endif
            {
                for (int j = 0; j < G; j += 2) {
                    uint8_t b = qg[j / 2];
                    int lo = (int)(b & 0x0F) - 8;
                    int hi = (int)(b >> 4) - 8;
                    gacc += (float)lo * xg[j] + (float)hi * xg[j + 1];
                }
            }
            acc += scale * gacc;
        }
        out[m] = acc;
    }
    return 0;
}

static int session_find_index(const ib_pq_session* s, const char* name) {
    if (!s || !name) return -1;
    for (int i = 0; i < s->multi.n; i++) {
        if (s->multi.names[i] && strcmp(s->multi.names[i], name) == 0) return i;
    }
    return -1;
}

int ib_pq_session_open(const char* ibf_path, ib_pq_session** out_p) {
    if (!ibf_path || !out_p) return -1;
    *out_p = NULL;
    ib_pq_session* s = (ib_pq_session*)calloc(1, sizeof(*s));
    if (!s) return -1;
    s->default_policy.variant = IB_PQ_VARIANT_STREAMING;
    s->default_policy.skip_threshold = 0.0f;
    s->default_policy.act_threshold = 0.0f;

    if (ib_pq_load_multi(ibf_path, &s->multi) != 0) {
        free(s); return -1;
    }
    if (ib_pq_multi_caches_create(&s->multi, &s->mc) != 0) {
        ib_pq_multi_free(&s->multi); free(s); return -1;
    }
    if (s->multi.n > 0) {
        s->policies = (ib_pq_policy*)calloc((size_t)s->multi.n, sizeof(*s->policies));
        if (!s->policies) {
            ib_pq_multi_caches_free(s->mc);
            ib_pq_multi_free(&s->multi); free(s); return -1;
        }
    }
    /* Sentinel: variant=-1 means "use default". */
    for (int i = 0; i < s->multi.n; i++) s->policies[i].variant = -1;

    /* Phase 8.E: precompute per-tensor inv_act_scale pointer (NULL if no
     * matching raw block) + size x_scratch to max(N). */
    s->act_scale_inv_per_tensor = (s->multi.n > 0)
        ? (const float**)calloc((size_t)s->multi.n,
                                  sizeof(*s->act_scale_inv_per_tensor))
        : NULL;
    int max_N = 0;
    char buf[256];
    for (int i = 0; i < s->multi.n; i++) {
        const ib_pq_tensor* tt = &s->multi.tensors[i];
        if (tt->N > max_N) max_N = tt->N;
        const char* name = s->multi.names[i];
        if (!name || !s->act_scale_inv_per_tensor) continue;
        snprintf(buf, sizeof(buf), "%s__act_scale", name);
        const void* data = NULL; int dtype = 0, ndim = 0; int shape[4] = {0};
        if (ib_pq_session_raw_get(s, buf, &data, &dtype, shape, &ndim) == 0
         && dtype == IB_RAW_F32 && ndim == 1 && shape[0] == tt->N) {
            s->act_scale_inv_per_tensor[i] = (const float*)data;
        }
    }
    if (max_N > 0) {
        s->x_scratch = (float*)malloc((size_t)max_N * sizeof(float));
        s->x_scratch_n = max_N;
    }

    /* INT4 weight path: scan raw_tensors for `__i4_q` + `__i4_s` pairs. */
    {
        const char* I4Q = "__i4_q";
        const char* I4S = "__i4_s";
        const size_t qlen = strlen(I4Q), slen = strlen(I4S);
        int cap = 0;
        for (int i = 0; i < s->multi.n_raw; i++) {
            const char* nm = s->multi.raw_tensors[i].name;
            size_t L = nm ? strlen(nm) : 0;
            if (L > qlen && strcmp(nm + L - qlen, I4Q) == 0) cap++;
        }
        if (cap > 0) {
            s->int4_tensors = (struct ib_int4_tensor*)calloc((size_t)cap,
                                  sizeof(struct ib_int4_tensor));
            for (int i = 0; i < s->multi.n_raw; i++) {
                ib_pq_raw_tensor* rq = &s->multi.raw_tensors[i];
                if (!rq->name) continue;
                size_t L = strlen(rq->name);
                if (L <= qlen || strcmp(rq->name + L - qlen, I4Q) != 0) continue;
                if (rq->dtype != IB_RAW_U8 || rq->ndim != 2) continue;
                size_t base_len = L - qlen;
                char base[256];
                if (base_len >= sizeof(base)) continue;
                memcpy(base, rq->name, base_len); base[base_len] = '\0';
                char sname[300];
                snprintf(sname, sizeof(sname), "%s%s", base, I4S);
                ib_pq_raw_tensor* rs = NULL;
                for (int j = 0; j < s->multi.n_raw; j++) {
                    if (s->multi.raw_tensors[j].name
                     && strcmp(s->multi.raw_tensors[j].name, sname) == 0) {
                        rs = &s->multi.raw_tensors[j]; break;
                    }
                }
                if (!rs || rs->dtype != IB_RAW_F16 || rs->ndim != 2) continue;
                int M = rq->shape[0];
                int N = rq->shape[1] * 2;
                int n_groups = rs->shape[1];
                if (n_groups <= 0 || N % n_groups != 0) continue;
                int G = N / n_groups;
                struct ib_int4_tensor* t = &s->int4_tensors[s->n_int4++];
                t->name = (char*)malloc(base_len + 1);
                memcpy(t->name, base, base_len + 1);
                t->q = (const uint8_t*)rq->data;
                t->s = (const uint16_t*)rs->data;
                t->M = M; t->N = N; t->G = G;
                if (N > max_N) max_N = N;
                /* AWQ: optional <base>__act_scale fp32[N] */
                char asname[300];
                snprintf(asname, sizeof(asname), "%s__act_scale", base);
                for (int j = 0; j < s->multi.n_raw; j++) {
                    ib_pq_raw_tensor* ra = &s->multi.raw_tensors[j];
                    if (ra->name && strcmp(ra->name, asname) == 0
                     && ra->dtype == IB_RAW_F32 && ra->ndim == 1
                     && ra->shape[0] == N) {
                        t->inv_act = (const float*)ra->data; break;
                    }
                }
                (void)slen;
            }
        }
    }
    if (max_N > 0 && !s->x_scratch) {
        s->x_scratch = (float*)malloc((size_t)max_N * sizeof(float));
        s->x_scratch_n = max_N;
    }

    /* F1.a: preallocate scratch sized to fleet max. */
    int max_K = 0, max_Kl2 = 0;
    for (int i = 0; i < s->multi.n; i++) {
        const ib_pq_lut_cache* c = s->mc->caches[i];
        if (c->K > max_K) max_K = c->K;
        if (c->n_levels == 2 && c->K_l2 > max_Kl2) max_Kl2 = c->K_l2;
    }
    s->scratch_K = max_K;
    s->scratch_K_l2 = max_Kl2;
    if (max_K > 0) {
        s->scratch_C1L = (float*)malloc((size_t)max_K * sizeof(float));
        s->scratch_C1R = (float*)malloc((size_t)max_K * sizeof(float));
    }
    if (max_Kl2 > 0) {
        s->scratch_C2L = (float*)malloc((size_t)max_Kl2 * sizeof(float));
        s->scratch_C2R = (float*)malloc((size_t)max_Kl2 * sizeof(float));
    }

    /* Thread pool. Off by default — pool sync overhead exceeds the
     * row-work win for typical Llama-1B matmuls under the cached path.
     * Set IB_PQ_THREADS > 1 to enable; only matmuls with M*C above the
     * threshold below dispatch to the pool. */
    int n_threads = 1;
    const char* env = getenv("IB_PQ_THREADS");
    if (env && *env) { int v = atoi(env); if (v > 0) n_threads = v; }
    s->n_threads = n_threads;
    if (n_threads > 1) {
        s->pool = ib_pool_create(n_threads);
        s->spin = pq_spin_pool_create(n_threads);
        size_t per_thread = (size_t)(2 * max_K + 2 * max_Kl2);
        if (per_thread > 0) {
            /* +1 slot: main thread uses tid = n_threads in pq_spin_pool_run. */
            s->thread_scratch = (float*)malloc(per_thread * (size_t)(n_threads + 1) * sizeof(float));
            s->thread_scratch_per = per_thread;
        }
    }

    *out_p = s;
    return 0;
}

void ib_pq_session_close(ib_pq_session* s) {
    if (!s) return;
    if (s->spin) pq_spin_pool_destroy(s->spin);
    if (s->pool) ib_pool_destroy(s->pool);
    free(s->scratch_C1L); free(s->scratch_C1R);
    free(s->scratch_C2L); free(s->scratch_C2R);
    free(s->thread_scratch);
    free(s->act_scale_inv_per_tensor);
    free(s->x_scratch);
    if (s->int4_tensors) {
        for (int i = 0; i < s->n_int4; i++) free(s->int4_tensors[i].name);
        free(s->int4_tensors);
    }
    ib_pq_multi_caches_free(s->mc);
    ib_pq_multi_free(&s->multi);
    free(s->policies);
    free(s);
}

int ib_pq_session_set_default_policy(ib_pq_session* s, ib_pq_policy p) {
    if (!s) return -1;
    s->default_policy = p;
    if (p.variant == IB_PQ_VARIANT_INT8 && !s->int8_quantized) {
        if (ib_pq_multi_caches_quantize_all_int8(s->mc) != 0) return -1;
        s->int8_quantized = 1;
    }
    return 0;
}

int ib_pq_session_set_policy(ib_pq_session* s, const char* name, ib_pq_policy p) {
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    s->policies[i] = p;
    if (p.variant == IB_PQ_VARIANT_INT8 && !s->int8_quantized) {
        if (ib_pq_multi_caches_quantize_all_int8(s->mc) != 0) return -1;
        s->int8_quantized = 1;
    }
    return 0;
}

/* F1.c: reentrant matmul that takes its scratch + x_scratch as args.
 * No use of session->scratch_C* / session->x_scratch / session->spin.
 * Used by forward_step_batch where multiple threads run independent
 * forward steps in parallel and each thread brings its own scratch. */
static int session_matmul_with_scratch(ib_pq_session* s, const char* name,
                                          const float* x, float* out,
                                          float* C1L, float* C1R,
                                          float* C2L, float* C2R,
                                          float* x_scratch, int x_scratch_n) {
    const struct ib_int4_tensor* i4 = session_int4_find(s, name);
    if (i4) {
        const float* xk = x;
        if (i4->inv_act && x_scratch && i4->N <= x_scratch_n) {
            for (int j = 0; j < i4->N; j++) x_scratch[j] = x[j] * i4->inv_act[j];
            xk = x_scratch;
        }
        return int4_matmul_fp32_raw(i4, xk, out);
    }
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    const ib_pq_tensor* t = &s->multi.tensors[i];
    const ib_pq_lut_cache* c = s->mc->caches[i];
    ib_pq_policy p = (s->policies[i].variant >= 0) ? s->policies[i] : s->default_policy;

    /* Phase 8.E: pre-scale x by inv_act_scale if calibration was used. */
    const float* x_for_kernel = x;
    const float* inv_s = (s->act_scale_inv_per_tensor
                            ? s->act_scale_inv_per_tensor[i] : NULL);
    if (inv_s && x_scratch && t->N <= x_scratch_n) {
        for (int j = 0; j < t->N; j++) x_scratch[j] = x[j] * inv_s[j];
        x_for_kernel = x_scratch;
    }

    switch (p.variant) {
    case IB_PQ_VARIANT_L1_ONLY:
        return ib_pq_matmul_fp32_l1_only(t, x_for_kernel, out);
    case IB_PQ_VARIANT_L2SKIP:
        return ib_pq_matmul_fp32_streaming_l2skip_cached(t, c, x_for_kernel, out, p.skip_threshold);
    case IB_PQ_VARIANT_SPARSE:
        return ib_pq_matmul_fp32_streaming_sparse(t, x_for_kernel, out, p.act_threshold);
    case IB_PQ_VARIANT_INT8:
        return ib_pq_matmul_fp32_streaming_int8_cached(t, c, x_for_kernel, out);
    case IB_PQ_VARIANT_STREAMING:
    default:
        return streaming_cached_kernel(t, c, x_for_kernel, out, C1L, C1R, C2L, C2R);
    }
}

int ib_pq_session_matmul(ib_pq_session* s, const char* name,
                          const float* x, float* out) {
    const struct ib_int4_tensor* i4 = session_int4_find(s, name);
    if (i4) {
        const float* xk = x;
        if (i4->inv_act && s->x_scratch && i4->N <= s->x_scratch_n) {
            for (int j = 0; j < i4->N; j++) s->x_scratch[j] = x[j] * i4->inv_act[j];
            xk = s->x_scratch;
        }
        return int4_matmul_fp32_raw(i4, xk, out);
    }
    /* Public entry: uses session-level shared scratch. */
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    const ib_pq_tensor* t = &s->multi.tensors[i];
    const ib_pq_lut_cache* c = s->mc->caches[i];
    ib_pq_policy p = (s->policies[i].variant >= 0) ? s->policies[i] : s->default_policy;

    const float* x_for_kernel = x;
    const float* inv_s = (s->act_scale_inv_per_tensor
                            ? s->act_scale_inv_per_tensor[i] : NULL);
    if (inv_s && s->x_scratch && t->N <= s->x_scratch_n) {
        for (int j = 0; j < t->N; j++) s->x_scratch[j] = x[j] * inv_s[j];
        x_for_kernel = s->x_scratch;
    }

    switch (p.variant) {
    case IB_PQ_VARIANT_L1_ONLY:
        return ib_pq_matmul_fp32_l1_only(t, x_for_kernel, out);
    case IB_PQ_VARIANT_L2SKIP:
        return ib_pq_matmul_fp32_streaming_l2skip_cached(t, c, x_for_kernel, out, p.skip_threshold);
    case IB_PQ_VARIANT_SPARSE:
        return ib_pq_matmul_fp32_streaming_sparse(t, x_for_kernel, out, p.act_threshold);
    case IB_PQ_VARIANT_INT8:
        return ib_pq_matmul_fp32_streaming_int8_cached(t, c, x_for_kernel, out);
    case IB_PQ_VARIANT_STREAMING:
    default:
        if (s->spin && s->n_threads > 1 && s->thread_scratch
         && (size_t)t->M * (size_t)t->C >= 1500000) {
            return ib_pq_matmul_fp32_streaming_cached_spin(t, c, x_for_kernel, out,
                                                              s->spin, s->n_threads,
                                                              s->thread_scratch,
                                                              s->thread_scratch_per);
        }
        return streaming_cached_kernel(t, c, x_for_kernel, out,
                                          s->scratch_C1L, s->scratch_C1R,
                                          s->scratch_C2L, s->scratch_C2R);
    }
}

int ib_pq_session_lm_head_topk(ib_pq_session* s, const char* name,
                                const float* x, int K_top,
                                float* out_logits, int32_t* out_ids) {
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    return ib_pq_lm_head_topk(&s->multi.tensors[i], x, K_top, out_logits, out_ids);
}

int ib_pq_session_tensor_shape(const ib_pq_session* s, const char* name,
                                int* out_M, int* out_N) {
    const struct ib_int4_tensor* i4 = session_int4_find(s, name);
    if (i4) {
        if (out_M) *out_M = i4->M;
        if (out_N) *out_N = i4->N;
        return 0;
    }
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    if (out_M) *out_M = s->multi.tensors[i].M;
    if (out_N) *out_N = s->multi.tensors[i].N;
    return 0;
}

int ib_pq_session_tensor_count(const ib_pq_session* s) {
    return s ? s->multi.n : 0;
}

const char* ib_pq_session_tensor_name(const ib_pq_session* s, int i) {
    if (!s || i < 0 || i >= s->multi.n) return NULL;
    return s->multi.names[i];
}

int ib_pq_session_raw_count(const ib_pq_session* s) {
    return s ? s->multi.n_raw : 0;
}

const char* ib_pq_session_raw_name(const ib_pq_session* s, int i) {
    if (!s || i < 0 || i >= s->multi.n_raw) return NULL;
    return s->multi.raw_tensors[i].name;
}

int ib_pq_session_raw_get(const ib_pq_session* s, const char* name,
                           const void** out_data, int* out_dtype,
                           int* out_shape, int* out_ndim) {
    if (!s || !name) return -1;
    for (int i = 0; i < s->multi.n_raw; i++) {
        ib_pq_raw_tensor* rt = &s->multi.raw_tensors[i];
        if (rt->name && strcmp(rt->name, name) == 0) {
            if (out_data)  *out_data  = rt->data;
            if (out_dtype) *out_dtype = rt->dtype;
            if (out_ndim)  *out_ndim  = rt->ndim;
            if (out_shape) for (int d = 0; d < rt->ndim; d++) out_shape[d] = rt->shape[d];
            return 0;
        }
    }
    return -1;
}

const char* ib_pq_session_config_json(const ib_pq_session* s) {
    return s ? s->multi.config_json : NULL;
}

/* ── Phase 9: forward-pass primitives ── */

void ib_rmsnorm_f32(float* out, const float* x, const float* w, int H, float eps) {
    double s = 0.0;
    for (int i = 0; i < H; i++) s += (double)x[i] * (double)x[i];
    float inv = 1.0f / sqrtf((float)(s / (double)H) + eps);
    for (int i = 0; i < H; i++) out[i] = x[i] * inv * w[i];
}

void ib_silu_gate_f32(float* out, const float* gate, const float* up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

void ib_residual_add_f32(float* x, const float* delta, int n) {
    for (int i = 0; i < n; i++) x[i] += delta[i];
}

void ib_rope_f32(float* x, int n_heads, int head_dim, int pos, float theta) {
    /* HF/Llama NEOX-style RoPE: split the head into two halves and rotate
     * (x[i], x[i + half]) by angle pos * theta^(-2i/head_dim) for i in [0, half).
     *   y[i]        = x[i]        * cos - x[i + half] * sin
     *   y[i + half] = x[i + half] * cos + x[i]        * sin
     */
    int half = head_dim / 2;
    for (int h = 0; h < n_heads; h++) {
        float* xh = x + (size_t)h * head_dim;
        for (int i = 0; i < half; i++) {
            float freq = powf(theta, -2.0f * (float)i / (float)head_dim);
            float angle = (float)pos * freq;
            float c = cosf(angle), s = sinf(angle);
            float a = xh[i];
            float b = xh[i + half];
            xh[i]        = a * c - b * s;
            xh[i + half] = b * c + a * s;
        }
    }
}

void ib_softmax_f32(float* x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); sum += x[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

int ib_fwht_norm_f32(float* x, int n) {
    int npow = 1;
    while (npow < n) npow *= 2;
    if (npow != n) {
        for (int i = n; i < npow; i++) x[i] = 0.0f;
    }
    for (int h = 1; h < npow; h *= 2) {
        for (int i = 0; i < npow; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float inv = 1.0f / sqrtf((float)npow);
    for (int i = 0; i < npow; i++) x[i] *= inv;
    return npow;
}

/* ── Phase 9: KV cache + single-token forward ── */

struct ib_pq_kv_cache {
    int n_layers;
    int max_seq;
    int kv_dim;        /* num_kv_heads * head_dim */
    int length;
    /* Storage layout: [layer][pos][kv_dim].
     * F1.d / Phase 7: storage selects fp32 (default), fp16, int8, or
     * pyramid (Phase 7 v2):
     *   IB_PQ_KV_FP16=1     → fp16 backing
     *   IB_PQ_KV_INT8=1     → int8 backing + per-token scale (4× vs fp16)
     *   IB_PQ_KV_PYRAMID=1  → pyramid PQ + per-token scale (2× vs int8,
     *                          requires K_codebook_L<N>_l/_r and
     *                          V_codebook_L<N>_l/_r raw tensors)
     */
    int storage_fp16;
    int storage_int8;
    int storage_pyramid;
    float*    k_f32;
    float*    v_f32;
    uint16_t* k_f16;
    uint16_t* v_f16;
    int8_t*   k_q8;
    int8_t*   v_q8;
    float*    k_q8_scale;
    float*    v_q8_scale;
    /* Pyramid storage: per (layer, pos), 2*C uint8 indices for K and V
     * (i_L per chunk + i_R per chunk). Plus per-(layer, pos) row scale
     * for K and V. */
    uint8_t*  k_pq_idx;     /* [n_layers][max_seq][2*C] */
    uint8_t*  v_pq_idx;
    float*    k_pq_scale;   /* [n_layers][max_seq] */
    float*    v_pq_scale;
    int       pq_C;         /* kv_dim / G */
    int       pq_G;
    int       pq_K1;
    /* Codebook pointers (borrowed from session raw tensors). */
    const uint16_t** k_cb_l;  /* [n_layers] */
    const uint16_t** k_cb_r;
    const uint16_t** v_cb_l;
    const uint16_t** v_cb_r;
};

static int session_config_int(const char* json, const char* key, int def) {
    /* Tiny JSON int extractor. Looks for "key": <number>. Avoids cJSON
     * dependency churn here; the strings come from our own writer so the
     * format is predictable. */
    if (!json) return def;
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* p = strstr(json, pat);
    if (!p) return def;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    return (int)atol(p);
}

static float session_config_float(const char* json, const char* key, float def) {
    if (!json) return def;
    char pat[64]; snprintf(pat, sizeof(pat), "\"%s\":", key);
    const char* p = strstr(json, pat);
    if (!p) return def;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    return (float)atof(p);
}

int ib_pq_kv_cache_create(const ib_pq_session* s, int max_seq_len,
                           ib_pq_kv_cache** out_p) {
    if (!s || !out_p || max_seq_len <= 0) return -1;
    const char* cfg = ib_pq_session_config_json(s);
    int n_layers = session_config_int(cfg, "num_layers", 0);
    int n_kv     = session_config_int(cfg, "num_kv_heads", 0);
    int head_dim = session_config_int(cfg, "head_dim", 0);
    if (n_layers <= 0 || n_kv <= 0 || head_dim <= 0) return -1;
    int kv_dim = n_kv * head_dim;

    ib_pq_kv_cache* kv = (ib_pq_kv_cache*)calloc(1, sizeof(*kv));
    if (!kv) return -1;
    kv->n_layers = n_layers; kv->max_seq = max_seq_len; kv->kv_dim = kv_dim;
    const char* fp16_env = getenv("IB_PQ_KV_FP16");
    const char* int8_env = getenv("IB_PQ_KV_INT8");
    const char* pq_env   = getenv("IB_PQ_KV_PYRAMID");
    kv->storage_pyramid = (pq_env && atoi(pq_env) > 0) ? 1 : 0;
    kv->storage_int8 = (!kv->storage_pyramid && int8_env && atoi(int8_env) > 0) ? 1 : 0;
    kv->storage_fp16 = (!kv->storage_int8 && !kv->storage_pyramid && fp16_env && atoi(fp16_env) > 0) ? 1 : 0;
    /* Pyramid mode requires codebooks present in the session — verify
     * they exist for L0; assume the rest follow. */
    if (kv->storage_pyramid) {
        const void* d=NULL; int dt=0,nd=0; int sh[4]={0};
        if (ib_pq_session_raw_get(s, "K_codebook_L0_l", &d, &dt, sh, &nd) != 0
         || dt != IB_RAW_F16 || nd != 2) {
            /* Fall back to fp32 silently. */
            kv->storage_pyramid = 0;
        } else {
            kv->pq_K1 = sh[0];
            int half = sh[1];
            kv->pq_G = half * 2;
            kv->pq_C = kv_dim / kv->pq_G;
            if (kv->pq_C * kv->pq_G != kv_dim) kv->storage_pyramid = 0;
        }
    }
    size_t n_elem = (size_t)n_layers * max_seq_len * kv_dim;
    size_t n_slots = (size_t)n_layers * max_seq_len;
    if (kv->storage_pyramid) {
        size_t idx_bytes = (size_t)n_layers * max_seq_len * (size_t)(2 * kv->pq_C);
        kv->k_pq_idx = (uint8_t*)malloc(idx_bytes);
        kv->v_pq_idx = (uint8_t*)malloc(idx_bytes);
        kv->k_pq_scale = (float*)malloc(n_slots * sizeof(float));
        kv->v_pq_scale = (float*)malloc(n_slots * sizeof(float));
        kv->k_cb_l = (const uint16_t**)calloc((size_t)n_layers, sizeof(*kv->k_cb_l));
        kv->k_cb_r = (const uint16_t**)calloc((size_t)n_layers, sizeof(*kv->k_cb_r));
        kv->v_cb_l = (const uint16_t**)calloc((size_t)n_layers, sizeof(*kv->v_cb_l));
        kv->v_cb_r = (const uint16_t**)calloc((size_t)n_layers, sizeof(*kv->v_cb_r));
        if (!kv->k_pq_idx || !kv->v_pq_idx || !kv->k_pq_scale || !kv->v_pq_scale
         || !kv->k_cb_l || !kv->k_cb_r || !kv->v_cb_l || !kv->v_cb_r) {
            free(kv->k_pq_idx); free(kv->v_pq_idx);
            free(kv->k_pq_scale); free(kv->v_pq_scale);
            free(kv->k_cb_l); free(kv->k_cb_r); free(kv->v_cb_l); free(kv->v_cb_r);
            free(kv); return -1;
        }
        char buf[64];
        for (int L = 0; L < n_layers; L++) {
            const void* d; int dt, nd; int sh[4];
            snprintf(buf, sizeof(buf), "K_codebook_L%d_l", L);
            if (ib_pq_session_raw_get(s, buf, &d, &dt, sh, &nd) != 0) { free(kv); return -1; }
            kv->k_cb_l[L] = (const uint16_t*)d;
            snprintf(buf, sizeof(buf), "K_codebook_L%d_r", L);
            if (ib_pq_session_raw_get(s, buf, &d, &dt, sh, &nd) != 0) { free(kv); return -1; }
            kv->k_cb_r[L] = (const uint16_t*)d;
            snprintf(buf, sizeof(buf), "V_codebook_L%d_l", L);
            if (ib_pq_session_raw_get(s, buf, &d, &dt, sh, &nd) != 0) { free(kv); return -1; }
            kv->v_cb_l[L] = (const uint16_t*)d;
            snprintf(buf, sizeof(buf), "V_codebook_L%d_r", L);
            if (ib_pq_session_raw_get(s, buf, &d, &dt, sh, &nd) != 0) { free(kv); return -1; }
            kv->v_cb_r[L] = (const uint16_t*)d;
        }
    } else if (kv->storage_int8) {
        kv->k_q8 = (int8_t*)malloc(n_elem);
        kv->v_q8 = (int8_t*)malloc(n_elem);
        kv->k_q8_scale = (float*)malloc(n_slots * sizeof(float));
        kv->v_q8_scale = (float*)malloc(n_slots * sizeof(float));
        if (!kv->k_q8 || !kv->v_q8 || !kv->k_q8_scale || !kv->v_q8_scale) {
            free(kv->k_q8); free(kv->v_q8);
            free(kv->k_q8_scale); free(kv->v_q8_scale);
            free(kv); return -1;
        }
    } else if (kv->storage_fp16) {
        kv->k_f16 = (uint16_t*)malloc(n_elem * sizeof(uint16_t));
        kv->v_f16 = (uint16_t*)malloc(n_elem * sizeof(uint16_t));
        if (!kv->k_f16 || !kv->v_f16) {
            free(kv->k_f16); free(kv->v_f16); free(kv); return -1;
        }
    } else {
        kv->k_f32 = (float*)malloc(n_elem * sizeof(float));
        kv->v_f32 = (float*)malloc(n_elem * sizeof(float));
        if (!kv->k_f32 || !kv->v_f32) {
            free(kv->k_f32); free(kv->v_f32); free(kv); return -1;
        }
    }
    *out_p = kv;
    return 0;
}

void ib_pq_kv_cache_free(ib_pq_kv_cache* kv) {
    if (!kv) return;
    free(kv->k_f32); free(kv->v_f32);
    free(kv->k_f16); free(kv->v_f16);
    free(kv->k_q8); free(kv->v_q8);
    free(kv->k_q8_scale); free(kv->v_q8_scale);
    free(kv->k_pq_idx); free(kv->v_pq_idx);
    free(kv->k_pq_scale); free(kv->v_pq_scale);
    free(kv->k_cb_l); free(kv->k_cb_r);
    free(kv->v_cb_l); free(kv->v_cb_r);
    free(kv);
}

void ib_pq_kv_cache_clear(ib_pq_kv_cache* kv) { if (kv) kv->length = 0; }
int  ib_pq_kv_cache_length(const ib_pq_kv_cache* kv) { return kv ? kv->length : 0; }

/* Embedding lookup: out[i] = embed[token_id][i]. Embed is stored fp16. */
static int embed_lookup(ib_pq_session* s, int token_id, float* out, int hidden) {
    /* Embeddings-PQ: prefer 'tok_embed' as a PQ tensor (smaller bundle).
     * Fall back to raw fp16 / fp32 storage. */
    int idx = session_find_index(s, "tok_embed");
    if (idx >= 0) {
        const ib_pq_tensor* t = &s->multi.tensors[idx];
        if (t->N != hidden) return -1;
        if (token_id < 0 || token_id >= t->M) return -1;
        return ib_pq_session_reconstruct_row(s, "tok_embed", token_id, out);
    }
    const void* data = NULL;
    int dtype = 0, ndim = 0;
    int shape[4] = {0};
    if (ib_pq_session_raw_get(s, "tok_embed", &data, &dtype, shape, &ndim) != 0) return -1;
    if (ndim != 2 || shape[1] != hidden) return -1;
    if (token_id < 0 || token_id >= shape[0]) return -1;
    if (dtype == IB_RAW_F16) {
        const uint16_t* row = (const uint16_t*)data + (size_t)token_id * hidden;
        for (int i = 0; i < hidden; i++) out[i] = ib_fp16_to_fp32(row[i]);
    } else if (dtype == IB_RAW_F32) {
        const float* row = (const float*)data + (size_t)token_id * hidden;
        memcpy(out, row, (size_t)hidden * sizeof(float));
    } else return -1;
    return 0;
}

static int load_norm_weight(const ib_pq_session* s, const char* name,
                              const float** out_w, int hidden) {
    const void* data = NULL;
    int dtype = 0, ndim = 0;
    int shape[4] = {0};
    if (ib_pq_session_raw_get(s, name, &data, &dtype, shape, &ndim) != 0) return -1;
    if (ndim != 1 || shape[0] != hidden || dtype != IB_RAW_F32) return -1;
    *out_w = (const float*)data;
    return 0;
}

/* F1.c: per-call scratch for thread-safe forward steps. NULL = use
 * session's shared scratch (single-threaded path). */
typedef struct {
    float* C1L;
    float* C1R;
    float* C2L;
    float* C2R;
    float* x;       /* x_scratch buffer for AWQ pre-scaling */
    int    x_n;
} forward_scratch;

static int session_matmul_via(ib_pq_session* s, const char* name,
                                const float* x, float* out,
                                const forward_scratch* sc) {
    if (sc) {
        return session_matmul_with_scratch(s, name, x, out,
                                              sc->C1L, sc->C1R, sc->C2L, sc->C2R,
                                              sc->x, sc->x_n);
    }
    return ib_pq_session_matmul(s, name, x, out);
}

/* Internal worker. logits != NULL: write vocab logits via lm_head.
 * hidden_out != NULL: write post-final-norm hidden state. Both NULL:
 * skip the tail (prefill no-logits). sc != NULL: use the provided
 * per-thread scratch instead of session-shared scratch (for batched
 * prefill threading). */
static int forward_step_internal_sc(ib_pq_session* s, ib_pq_kv_cache* kv,
                                       int token_id, int pos,
                                       float* logits, float* hidden_out,
                                       const forward_scratch* sc) {
    if (!s) return -1;
    const char* cfg = ib_pq_session_config_json(s);
    int n_layers = session_config_int(cfg, "num_layers", 0);
    int hidden   = session_config_int(cfg, "hidden_size", 0);
    int n_heads  = session_config_int(cfg, "num_heads", 0);
    int n_kv     = session_config_int(cfg, "num_kv_heads", n_heads);
    int head_dim = session_config_int(cfg, "head_dim", hidden / n_heads);
    int inter    = session_config_int(cfg, "intermediate_size", 0);
    int vocab    = session_config_int(cfg, "vocab_size", 0);
    float eps    = session_config_float(cfg, "rms_norm_eps", 1e-5f);
    float theta  = session_config_float(cfg, "rope_theta", 10000.0f);
    int kv_dim   = n_kv * head_dim;
    if (n_layers <= 0 || hidden <= 0 || n_heads <= 0 || vocab <= 0) return -1;
    if (kv && (pos < 0 || pos >= kv->max_seq)) return -1;

    float* x        = (float*)calloc((size_t)hidden, sizeof(float));
    float* xb       = (float*)calloc((size_t)hidden, sizeof(float));   /* normed */
    float* xb2      = (float*)calloc((size_t)hidden, sizeof(float));   /* attn out */
    float* q        = (float*)calloc((size_t)hidden, sizeof(float));   /* full Q */
    float* k_now    = (float*)calloc((size_t)kv_dim, sizeof(float));
    float* v_now    = (float*)calloc((size_t)kv_dim, sizeof(float));
    float* gate     = (float*)calloc((size_t)inter,  sizeof(float));
    float* up       = (float*)calloc((size_t)inter,  sizeof(float));
    float* mlp_out  = (float*)calloc((size_t)hidden, sizeof(float));
    float* scores   = (float*)calloc((size_t)(pos + 1), sizeof(float));
    if (!x || !xb || !xb2 || !q || !k_now || !v_now || !gate || !up || !mlp_out || !scores) {
        free(x); free(xb); free(xb2); free(q); free(k_now); free(v_now);
        free(gate); free(up); free(mlp_out); free(scores); return -1;
    }

    if (embed_lookup(s, token_id, x, hidden) != 0) {
        free(x); free(xb); free(xb2); free(q); free(k_now); free(v_now);
        free(gate); free(up); free(mlp_out); free(scores); return -1;
    }

    char buf[64];
    int rc = 0;
    /* F1.i: detect fused qkv / gateup at start; fused = present in IBF. */
    int fused_qkv = (ib_pq_session_tensor_shape(s, "L0_qkv_proj", NULL, NULL) == 0);
    int fused_gu  = (ib_pq_session_tensor_shape(s, "L0_gateup_proj", NULL, NULL) == 0);
    /* qkv buffer (hidden + 2*kv_dim); gateup buffer (2*inter). Allocate
     * lazily (only if a fused tensor exists). */
    float* qkv_buf = fused_qkv ? (float*)malloc((size_t)(hidden + 2 * kv_dim) * sizeof(float)) : NULL;
    float* gu_buf  = fused_gu  ? (float*)malloc((size_t)(2 * inter) * sizeof(float)) : NULL;
    for (int L = 0; L < n_layers && rc == 0; L++) {
        const float* w_in = NULL; const float* w_pn = NULL;
        snprintf(buf, sizeof(buf), "L%d_input_norm", L);
        if (load_norm_weight(s, buf, &w_in, hidden) != 0) { rc = -1; break; }
        snprintf(buf, sizeof(buf), "L%d_post_attn_norm", L);
        if (load_norm_weight(s, buf, &w_pn, hidden) != 0) { rc = -1; break; }

        /* ── Attention block ── */
        ib_rmsnorm_f32(xb, x, w_in, hidden, eps);

        if (fused_qkv) {
            snprintf(buf, sizeof(buf), "L%d_qkv_proj", L);
            if (session_matmul_via(s, buf, xb, qkv_buf, sc) != 0) { rc = -1; break; }
            memcpy(q,     qkv_buf,                          (size_t)hidden * sizeof(float));
            memcpy(k_now, qkv_buf + hidden,                 (size_t)kv_dim * sizeof(float));
            memcpy(v_now, qkv_buf + hidden + kv_dim,        (size_t)kv_dim * sizeof(float));
        } else {
            snprintf(buf, sizeof(buf), "L%d_q_proj", L);
            if (session_matmul_via(s, buf, xb, q, sc) != 0) { rc = -1; break; }
            snprintf(buf, sizeof(buf), "L%d_k_proj", L);
            if (session_matmul_via(s, buf, xb, k_now, sc) != 0) { rc = -1; break; }
            snprintf(buf, sizeof(buf), "L%d_v_proj", L);
            if (session_matmul_via(s, buf, xb, v_now, sc) != 0) { rc = -1; break; }
        }

        ib_rope_f32(q,     n_heads, head_dim, pos, theta);
        ib_rope_f32(k_now, n_kv,    head_dim, pos, theta);

        /* Write into kv cache. fp32 path: memcpy. fp16 path: convert.
         * int8 path: per-vector scale + round. pyramid: per-vector scale
         * + nearest codeword per chunk half. */
        if (kv) {
            size_t kv_offset = ((size_t)L * kv->max_seq + pos) * kv_dim;
            size_t slot = (size_t)L * kv->max_seq + pos;
            if (kv->storage_pyramid) {
                int G = kv->pq_G, C = kv->pq_C, half = G / 2, K1 = kv->pq_K1;
                /* per-vector row scale (max_abs / 7), normalize, then
                 * find nearest codeword per chunk-half. */
                float k_max = 1e-12f, v_max = 1e-12f;
                for (int d = 0; d < kv_dim; d++) {
                    float ak = fabsf(k_now[d]); if (ak > k_max) k_max = ak;
                    float av = fabsf(v_now[d]); if (av > v_max) v_max = av;
                }
                float k_s = k_max / 7.0f, v_s = v_max / 7.0f;
                if (k_s < 1e-12f) k_s = 1e-12f;
                if (v_s < 1e-12f) v_s = 1e-12f;
                float k_inv = 1.0f / k_s, v_inv = 1.0f / v_s;
                kv->k_pq_scale[slot] = k_s;
                kv->v_pq_scale[slot] = v_s;
                uint8_t* k_idx = kv->k_pq_idx + slot * (size_t)(2 * C);
                uint8_t* v_idx = kv->v_pq_idx + slot * (size_t)(2 * C);
                const uint16_t* kcb_l = kv->k_cb_l[L];
                const uint16_t* kcb_r = kv->k_cb_r[L];
                const uint16_t* vcb_l = kv->v_cb_l[L];
                const uint16_t* vcb_r = kv->v_cb_r[L];
                for (int c = 0; c < C; c++) {
                    /* For each chunk c, normalize the K and V sub-vectors,
                     * then find the nearest codeword in their L/R half. */
                    float kL[16], kR[16], vL[16], vR[16];
                    for (int j = 0; j < half; j++) {
                        kL[j] = k_now[c * G + j] * k_inv;
                        kR[j] = k_now[c * G + half + j] * k_inv;
                        vL[j] = v_now[c * G + j] * v_inv;
                        vR[j] = v_now[c * G + half + j] * v_inv;
                    }
                    int best_kl = 0, best_kr = 0, best_vl = 0, best_vr = 0;
                    float best_kl_d = 1e30f, best_kr_d = 1e30f;
                    float best_vl_d = 1e30f, best_vr_d = 1e30f;
                    for (int k = 0; k < K1; k++) {
                        float dkl = 0, dkr = 0, dvl = 0, dvr = 0;
                        for (int j = 0; j < half; j++) {
                            float ekl = ib_fp16_to_fp32(kcb_l[(size_t)k*half+j]);
                            float ekr = ib_fp16_to_fp32(kcb_r[(size_t)k*half+j]);
                            float evl = ib_fp16_to_fp32(vcb_l[(size_t)k*half+j]);
                            float evr = ib_fp16_to_fp32(vcb_r[(size_t)k*half+j]);
                            float ed;
                            ed = kL[j] - ekl; dkl += ed*ed;
                            ed = kR[j] - ekr; dkr += ed*ed;
                            ed = vL[j] - evl; dvl += ed*ed;
                            ed = vR[j] - evr; dvr += ed*ed;
                        }
                        if (dkl < best_kl_d) { best_kl_d = dkl; best_kl = k; }
                        if (dkr < best_kr_d) { best_kr_d = dkr; best_kr = k; }
                        if (dvl < best_vl_d) { best_vl_d = dvl; best_vl = k; }
                        if (dvr < best_vr_d) { best_vr_d = dvr; best_vr = k; }
                    }
                    k_idx[c]         = (uint8_t)best_kl;
                    k_idx[C + c]     = (uint8_t)best_kr;
                    v_idx[c]         = (uint8_t)best_vl;
                    v_idx[C + c]     = (uint8_t)best_vr;
                }
            } else if (kv->storage_int8) {
                int8_t* k_slot = kv->k_q8 + kv_offset;
                int8_t* v_slot = kv->v_q8 + kv_offset;
                float k_max = 1e-12f, v_max = 1e-12f;
                for (int d = 0; d < kv_dim; d++) {
                    float ak = fabsf(k_now[d]); if (ak > k_max) k_max = ak;
                    float av = fabsf(v_now[d]); if (av > v_max) v_max = av;
                }
                float k_s = k_max / 127.0f, v_s = v_max / 127.0f;
                float k_inv = 1.0f / k_s, v_inv = 1.0f / v_s;
                kv->k_q8_scale[slot] = k_s;
                kv->v_q8_scale[slot] = v_s;
                for (int d = 0; d < kv_dim; d++) {
                    float kq = k_now[d] * k_inv;
                    float vq = v_now[d] * v_inv;
                    int kqi = (int)(kq >= 0 ? kq + 0.5f : kq - 0.5f);
                    int vqi = (int)(vq >= 0 ? vq + 0.5f : vq - 0.5f);
                    if (kqi >  127) kqi =  127;
                    if (kqi < -127) kqi = -127;
                    if (vqi >  127) vqi =  127;
                    if (vqi < -127) vqi = -127;
                    k_slot[d] = (int8_t)kqi;
                    v_slot[d] = (int8_t)vqi;
                }
            } else if (kv->storage_fp16) {
                uint16_t* k_slot = kv->k_f16 + kv_offset;
                uint16_t* v_slot = kv->v_f16 + kv_offset;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
                int d = 0;
                for (; d + 4 <= kv_dim; d += 4) {
                    float16x4_t kh = vcvt_f16_f32(vld1q_f32(k_now + d));
                    float16x4_t vh = vcvt_f16_f32(vld1q_f32(v_now + d));
                    vst1_u16(k_slot + d, vreinterpret_u16_f16(kh));
                    vst1_u16(v_slot + d, vreinterpret_u16_f16(vh));
                }
                for (; d < kv_dim; d++) {
                    k_slot[d] = ib_fp32_to_fp16(k_now[d]);
                    v_slot[d] = ib_fp32_to_fp16(v_now[d]);
                }
#else
                for (int d = 0; d < kv_dim; d++) {
                    k_slot[d] = ib_fp32_to_fp16(k_now[d]);
                    v_slot[d] = ib_fp32_to_fp16(v_now[d]);
                }
#endif
            } else {
                float* k_slot = kv->k_f32 + kv_offset;
                float* v_slot = kv->v_f32 + kv_offset;
                memcpy(k_slot, k_now, (size_t)kv_dim * sizeof(float));
                memcpy(v_slot, v_now, (size_t)kv_dim * sizeof(float));
            }
        }

        /* Multi-head attention with grouped-query (each query head h
         * attends to the kv head h / (n_heads / n_kv)). */
        int q_per_kv = n_heads / n_kv;
        float inv_sqrt = 1.0f / sqrtf((float)head_dim);
        /* For fp16-backed KV, dequant the slice for this layer + head into
         * stack scratch once per (t, h) so the dot loop reads contiguous fp32
         * (and SIMD on it). head_dim ≤ 256 in practice. */
        float k_scratch[256];
        float v_scratch[256];
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / q_per_kv;
            const float* qh = q + (size_t)h * head_dim;
            for (int t = 0; t <= pos; t++) {
                const float* k_t;
                if (kv && kv->storage_pyramid) {
                    int G = kv->pq_G, C = kv->pq_C, half = G / 2;
                    size_t slot = (size_t)L * kv->max_seq + t;
                    const uint8_t* k_idx = kv->k_pq_idx + slot * (size_t)(2 * C);
                    const uint16_t* cb_l = kv->k_cb_l[L];
                    const uint16_t* cb_r = kv->k_cb_r[L];
                    float scale = kv->k_pq_scale[slot];
                    /* head_dim/G chunks of this kv head: starts at chunk
                     * (kv_h * head_dim) / G; head_dim must be multiple of G. */
                    int chunk0 = (kv_h * head_dim) / G;
                    int n_ch   = head_dim / G;
                    for (int cc = 0; cc < n_ch; cc++) {
                        int ci = chunk0 + cc;
                        const uint16_t* eL = cb_l + (size_t)k_idx[ci] * half;
                        const uint16_t* eR = cb_r + (size_t)k_idx[C + ci] * half;
                        for (int j = 0; j < half; j++) {
                            k_scratch[cc * G + j]        = ib_fp16_to_fp32(eL[j]) * scale;
                            k_scratch[cc * G + half + j] = ib_fp16_to_fp32(eR[j]) * scale;
                        }
                    }
                    k_t = k_scratch;
                } else if (kv && kv->storage_int8) {
                    size_t base = ((size_t)L * kv->max_seq + t) * kv_dim
                                + (size_t)kv_h * head_dim;
                    const int8_t* k_src = kv->k_q8 + base;
                    float k_s = kv->k_q8_scale[(size_t)L * kv->max_seq + t];
                    for (int d = 0; d < head_dim; d++) k_scratch[d] = (float)k_src[d] * k_s;
                    k_t = k_scratch;
                } else if (kv && kv->storage_fp16) {
                    const uint16_t* k_src = kv->k_f16
                        + ((size_t)L * kv->max_seq + t) * kv_dim
                        + (size_t)kv_h * head_dim;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
                    int d = 0;
                    for (; d + 4 <= head_dim; d += 4) {
                        float16x4_t hv = vreinterpret_f16_u16(vld1_u16(k_src + d));
                        vst1q_f32(k_scratch + d, vcvt_f32_f16(hv));
                    }
                    for (; d < head_dim; d++) k_scratch[d] = ib_fp16_to_fp32(k_src[d]);
#else
                    for (int d = 0; d < head_dim; d++) k_scratch[d] = ib_fp16_to_fp32(k_src[d]);
#endif
                    k_t = k_scratch;
                } else if (kv) {
                    k_t = kv->k_f32 + ((size_t)L * kv->max_seq + t) * kv_dim
                                    + (size_t)kv_h * head_dim;
                } else {
                    k_t = k_now + (size_t)kv_h * head_dim;
                }
                scores[t] = pq_dot_f32(qh, k_t, head_dim) * inv_sqrt;
            }
            ib_softmax_f32(scores, pos + 1);
            float* outh = xb2 + (size_t)h * head_dim;
            memset(outh, 0, (size_t)head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float* v_t;
                if (kv && kv->storage_pyramid) {
                    int G = kv->pq_G, C = kv->pq_C, half = G / 2;
                    size_t slot = (size_t)L * kv->max_seq + t;
                    const uint8_t* v_idx = kv->v_pq_idx + slot * (size_t)(2 * C);
                    const uint16_t* cb_l = kv->v_cb_l[L];
                    const uint16_t* cb_r = kv->v_cb_r[L];
                    float scale = kv->v_pq_scale[slot];
                    int chunk0 = (kv_h * head_dim) / G;
                    int n_ch   = head_dim / G;
                    for (int cc = 0; cc < n_ch; cc++) {
                        int ci = chunk0 + cc;
                        const uint16_t* eL = cb_l + (size_t)v_idx[ci] * half;
                        const uint16_t* eR = cb_r + (size_t)v_idx[C + ci] * half;
                        for (int j = 0; j < half; j++) {
                            v_scratch[cc * G + j]        = ib_fp16_to_fp32(eL[j]) * scale;
                            v_scratch[cc * G + half + j] = ib_fp16_to_fp32(eR[j]) * scale;
                        }
                    }
                    v_t = v_scratch;
                } else if (kv && kv->storage_int8) {
                    size_t base = ((size_t)L * kv->max_seq + t) * kv_dim
                                + (size_t)kv_h * head_dim;
                    const int8_t* v_src = kv->v_q8 + base;
                    float v_s = kv->v_q8_scale[(size_t)L * kv->max_seq + t];
                    for (int d = 0; d < head_dim; d++) v_scratch[d] = (float)v_src[d] * v_s;
                    v_t = v_scratch;
                } else if (kv && kv->storage_fp16) {
                    const uint16_t* v_src = kv->v_f16
                        + ((size_t)L * kv->max_seq + t) * kv_dim
                        + (size_t)kv_h * head_dim;
#if defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
                    int d = 0;
                    for (; d + 4 <= head_dim; d += 4) {
                        float16x4_t hv = vreinterpret_f16_u16(vld1_u16(v_src + d));
                        vst1q_f32(v_scratch + d, vcvt_f32_f16(hv));
                    }
                    for (; d < head_dim; d++) v_scratch[d] = ib_fp16_to_fp32(v_src[d]);
#else
                    for (int d = 0; d < head_dim; d++) v_scratch[d] = ib_fp16_to_fp32(v_src[d]);
#endif
                    v_t = v_scratch;
                } else if (kv) {
                    v_t = kv->v_f32 + ((size_t)L * kv->max_seq + t) * kv_dim
                                    + (size_t)kv_h * head_dim;
                } else {
                    v_t = v_now + (size_t)kv_h * head_dim;
                }
                pq_accum_scaled_f32(outh, v_t, scores[t], head_dim);
            }
        }

        /* o_proj + residual */
        snprintf(buf, sizeof(buf), "L%d_o_proj", L);
        if (session_matmul_via(s, buf, xb2, xb, sc) != 0) { rc = -1; break; }
        ib_residual_add_f32(x, xb, hidden);

        /* ── MLP block ── */
        ib_rmsnorm_f32(xb, x, w_pn, hidden, eps);
        if (fused_gu) {
            snprintf(buf, sizeof(buf), "L%d_gateup_proj", L);
            if (session_matmul_via(s, buf, xb, gu_buf, sc) != 0) { rc = -1; break; }
            memcpy(gate, gu_buf,         (size_t)inter * sizeof(float));
            memcpy(up,   gu_buf + inter, (size_t)inter * sizeof(float));
        } else {
            snprintf(buf, sizeof(buf), "L%d_gate_proj", L);
            if (session_matmul_via(s, buf, xb, gate, sc) != 0) { rc = -1; break; }
            snprintf(buf, sizeof(buf), "L%d_up_proj", L);
            if (session_matmul_via(s, buf, xb, up, sc) != 0) { rc = -1; break; }
        }
        ib_silu_gate_f32(gate, gate, up, inter);
        snprintf(buf, sizeof(buf), "L%d_down_proj", L);
        if (session_matmul_via(s, buf, gate, mlp_out, sc) != 0) { rc = -1; break; }
        ib_residual_add_f32(x, mlp_out, hidden);
    }
    free(qkv_buf); free(gu_buf);

    if (rc == 0 && (logits || hidden_out)) {
        const float* w_final = NULL;
        if (load_norm_weight(s, "final_norm", &w_final, hidden) != 0) rc = -1;
        else {
            ib_rmsnorm_f32(xb, x, w_final, hidden, eps);
            if (hidden_out) memcpy(hidden_out, xb, (size_t)hidden * sizeof(float));
            if (logits) {
                if (session_matmul_via(s, "lm_head", xb, logits, sc) != 0) rc = -1;
            }
            (void)vocab;
        }
    }

    if (kv && rc == 0) kv->length = pos + 1;

    free(x); free(xb); free(xb2); free(q); free(k_now); free(v_now);
    free(gate); free(up); free(mlp_out); free(scores);
    return rc;
}

int ib_pq_forward_step(ib_pq_session* s, ib_pq_kv_cache* kv,
                        int token_id, int pos, float* logits) {
    return forward_step_internal_sc(s, kv, token_id, pos, logits, NULL, NULL);
}

int ib_pq_forward_step_to_hidden(ib_pq_session* s, ib_pq_kv_cache* kv,
                                   int token_id, int pos, float* hidden_out) {
    if (!hidden_out) return -1;
    return forward_step_internal_sc(s, kv, token_id, pos, NULL, hidden_out, NULL);
}

int ib_pq_generate_greedy(ib_pq_session* s, ib_pq_kv_cache* kv,
                            const int* prompt_ids, int n_prompt,
                            int max_new, int eos_token_id,
                            int* out_ids, int* n_out,
                            ib_pq_token_cb cb, void* cb_ctx) {
    if (!s || !kv || !prompt_ids || n_prompt <= 0 || max_new < 0
     || !out_ids || !n_out) return -1;
    const char* cfg = ib_pq_session_config_json(s);
    int hidden = session_config_int(cfg, "hidden_size", 0);
    if (hidden <= 0) return -1;
    float* hbuf = (float*)malloc((size_t)hidden * sizeof(float));
    if (!hbuf) return -1;

    int pos = 0;
    /* F1.c: batched prefill — runs the prompt through one barrier-coupled
     * parallel pass per layer (B/T inputs per worker), bit-exact vs
     * sequential. Final norm is applied to the last slot. */
    if (ib_pq_forward_step_batch(s, kv, prompt_ids, n_prompt, 0, hbuf) != 0) {
        free(hbuf); return -1;
    }
    pos = n_prompt;

    int written = 0;
    /* K_TOP_GREEDY: 8 candidates is enough to virtually always include
     * the true argmax (test 50: K=32 captured full argmax 100% of runs;
     * 8 is the sweet spot for cost/safety). Output is sorted desc, so
     * element 0 is the refined best. */
    enum { K_TOP_GREEDY = 8 };
    float top_logits[K_TOP_GREEDY];
    int32_t top_ids[K_TOP_GREEDY];
    for (int g = 0; g < max_new; g++) {
        if (ib_pq_session_lm_head_topk(s, "lm_head", hbuf, K_TOP_GREEDY,
                                          top_logits, top_ids) != 0) {
            free(hbuf); *n_out = written; return -1;
        }
        int best = (int)top_ids[0];
        out_ids[written++] = best;
        if (cb && cb(best, cb_ctx) != 0) break;
        if (eos_token_id >= 0 && best == eos_token_id) break;
        if (pos >= kv->max_seq) break;

        if (forward_step_internal_sc(s, kv, best, pos, NULL, hbuf, NULL) != 0) {
            free(hbuf); *n_out = written; return -1;
        }
        pos++;
    }

    *n_out = written;
    free(hbuf);
    return 0;
}

/* ── Sampling helpers ── */

/* Tiny xorshift64 RNG. Deterministic from seed. */
typedef struct { uint64_t s; } pq_rng;
static inline uint32_t pq_rng_u32(pq_rng* r) {
    uint64_t x = r->s; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    r->s = x;
    return (uint32_t)x;
}
static inline float pq_rng_unit(pq_rng* r) {
    return (float)(pq_rng_u32(r) >> 8) * (1.0f / (float)(1 << 24));
}

/* Sample a token from logits given (temperature, top_k, top_p). */
static int sample_from_logits(const float* logits, int vocab,
                                ib_pq_sample_params params, pq_rng* rng) {
    /* Greedy fallback. */
    if (params.temperature <= 0.0f) {
        int best = 0; float bv = logits[0];
        for (int v = 1; v < vocab; v++) if (logits[v] > bv) { bv = logits[v]; best = v; }
        return best;
    }
    /* Build probs with temperature. */
    float invT = 1.0f / params.temperature;
    /* Optional top-K: gather top_k indices first. */
    int k = (params.top_k > 0 && params.top_k < vocab) ? params.top_k : vocab;
    int* idx = (int*)malloc((size_t)vocab * sizeof(int));
    float* sc = (float*)malloc((size_t)vocab * sizeof(float));
    if (!idx || !sc) { free(idx); free(sc); return 0; }
    for (int v = 0; v < vocab; v++) { idx[v] = v; sc[v] = logits[v]; }

    if (k < vocab) {
        /* Partial sort: pull top-K to front (descending). Quickselect-ish
         * via simple selection — fine at K << vocab. */
        for (int i = 0; i < k; i++) {
            int max_j = i;
            for (int j = i + 1; j < vocab; j++) if (sc[j] > sc[max_j]) max_j = j;
            float tf = sc[i]; sc[i] = sc[max_j]; sc[max_j] = tf;
            int   ti = idx[i]; idx[i] = idx[max_j]; idx[max_j] = ti;
        }
    } else {
        /* Full vocab — sort descending for top-P. */
        for (int i = 0; i < vocab; i++) {
            int max_j = i;
            for (int j = i + 1; j < vocab; j++) if (sc[j] > sc[max_j]) max_j = j;
            float tf = sc[i]; sc[i] = sc[max_j]; sc[max_j] = tf;
            int   ti = idx[i]; idx[i] = idx[max_j]; idx[max_j] = ti;
        }
    }

    /* Softmax over the kept top-K (or full sorted vocab) with temperature. */
    int n_keep = k;
    float m = sc[0] * invT;
    float sum = 0.0f;
    for (int i = 0; i < n_keep; i++) {
        sc[i] = expf(sc[i] * invT - m);
        sum += sc[i];
    }
    float invs = 1.0f / sum;
    for (int i = 0; i < n_keep; i++) sc[i] *= invs;

    /* Top-P: keep smallest prefix whose cumprob >= p. */
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum = 0.0f;
        int trunc = n_keep;
        for (int i = 0; i < n_keep; i++) {
            cum += sc[i];
            if (cum >= params.top_p) { trunc = i + 1; break; }
        }
        n_keep = trunc;
        /* renormalize */
        float s2 = 0.0f;
        for (int i = 0; i < n_keep; i++) s2 += sc[i];
        float inv2 = 1.0f / s2;
        for (int i = 0; i < n_keep; i++) sc[i] *= inv2;
    }

    /* Sample by inverse CDF. */
    float u = pq_rng_unit(rng);
    float c = 0.0f;
    int picked = idx[n_keep - 1];
    for (int i = 0; i < n_keep; i++) {
        c += sc[i];
        if (u <= c) { picked = idx[i]; break; }
    }
    free(idx); free(sc);
    return picked;
}

/* F1.j-sample: when top_k > 0, sample from the lm_head_topk subset
 * (sorted desc) with temperature + top_p truncation. */
static int sample_from_topk(const float* top_logits, const int32_t* top_ids,
                              int K, ib_pq_sample_params params, pq_rng* rng) {
    if (K <= 0) return 0;
    if (params.temperature <= 0.0f) return (int)top_ids[0];

    float invT = 1.0f / params.temperature;
    float pr[256];
    if (K > 256) K = 256;
    float m = top_logits[0] * invT;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        pr[i] = expf(top_logits[i] * invT - m);
        sum += pr[i];
    }
    float invs = 1.0f / sum;
    for (int i = 0; i < K; i++) pr[i] *= invs;

    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        float cum = 0.0f; int trunc = K;
        for (int i = 0; i < K; i++) {
            cum += pr[i];
            if (cum >= params.top_p) { trunc = i + 1; break; }
        }
        K = trunc;
        float s2 = 0.0f;
        for (int i = 0; i < K; i++) s2 += pr[i];
        float inv2 = 1.0f / s2;
        for (int i = 0; i < K; i++) pr[i] *= inv2;
    }

    float u = pq_rng_unit(rng);
    float c = 0.0f;
    int picked = (int)top_ids[K - 1];
    for (int i = 0; i < K; i++) {
        c += pr[i];
        if (u <= c) { picked = (int)top_ids[i]; break; }
    }
    return picked;
}

int ib_pq_generate_sample(ib_pq_session* s, ib_pq_kv_cache* kv,
                            const int* prompt_ids, int n_prompt,
                            int max_new, int eos_token_id,
                            ib_pq_sample_params params,
                            int* out_ids, int* n_out,
                            ib_pq_token_cb cb, void* cb_ctx) {
    if (!s || !kv || !prompt_ids || n_prompt <= 0 || max_new < 0
     || !out_ids || !n_out) return -1;
    const char* cfg = ib_pq_session_config_json(s);
    int vocab = session_config_int(cfg, "vocab_size", 0);
    int hidden = session_config_int(cfg, "hidden_size", 0);
    if (vocab <= 0 || hidden <= 0) return -1;

    pq_rng rng;
    rng.s = params.seed ? (uint64_t)params.seed : 0x9E3779B97F4A7C15ULL;
    rng.s ^= rng.s << 21; rng.s ^= rng.s >> 35; rng.s ^= rng.s << 4;

    /* If top_k is in a reasonable bound use the lm_head_topk fast path —
     * only K candidates need the refined matmul; the full M=vocab logits
     * pass is replaced by the L1-only filter + K-row refine. */
    int use_topk = (params.top_k > 0 && params.top_k <= 256);
    int K_use = use_topk ? params.top_k : 0;

    float* logits = NULL;
    float* hbuf = NULL;
    float top_logits[256];
    int32_t top_ids[256];
    if (use_topk) {
        hbuf = (float*)malloc((size_t)hidden * sizeof(float));
        if (!hbuf) return -1;
    } else {
        logits = (float*)malloc((size_t)vocab * sizeof(float));
        if (!logits) return -1;
    }

    int pos = 0;
    if (use_topk) {
        /* F1.c: batched prefill into the last slot's hidden. */
        if (ib_pq_forward_step_batch(s, kv, prompt_ids, n_prompt, 0, hbuf) != 0) {
            free(logits); free(hbuf); return -1;
        }
        pos = n_prompt;
    } else {
        /* Full-vocab sample path keeps the sequential prefill (still
         * skips lm_head on non-final positions). */
        for (int i = 0; i < n_prompt; i++) {
            int last = (i == n_prompt - 1);
            int rc = forward_step_internal_sc(s, kv, prompt_ids[i], pos,
                                                 last ? logits : NULL, NULL, NULL);
            if (rc != 0) { free(logits); free(hbuf); return -1; }
            pos++;
        }
    }
    int written = 0;
    for (int g = 0; g < max_new; g++) {
        int tok;
        if (use_topk) {
            if (ib_pq_session_lm_head_topk(s, "lm_head", hbuf, K_use,
                                              top_logits, top_ids) != 0) {
                free(hbuf); *n_out = written; return -1;
            }
            tok = sample_from_topk(top_logits, top_ids, K_use, params, &rng);
        } else {
            tok = sample_from_logits(logits, vocab, params, &rng);
        }
        out_ids[written++] = tok;
        if (cb && cb(tok, cb_ctx) != 0) break;
        if (eos_token_id >= 0 && tok == eos_token_id) break;
        if (pos >= kv->max_seq) break;
        int rc;
        if (use_topk) {
            rc = forward_step_internal_sc(s, kv, tok, pos, NULL, hbuf, NULL);
        } else {
            rc = forward_step_internal_sc(s, kv, tok, pos, logits, NULL, NULL);
        }
        if (rc != 0) { free(logits); free(hbuf); *n_out = written; return -1; }
        pos++;
    }
    *n_out = written;
    free(logits); free(hbuf);
    return 0;
}

int ib_pq_forward_step_no_logits(ib_pq_session* s, ib_pq_kv_cache* kv,
                                   int token_id, int pos) {
    return ib_pq_forward_step(s, kv, token_id, pos, NULL);
}

/* ── F1.c full batched prefill ── */

typedef struct {
    /* Per-batch-slot state. All buffers are slot-local; threads never share. */
    int   token_id;
    int   pos;
    float* x;        /* [hidden] residual stream */
    float* xb;       /* [hidden] norm output */
    float* xb2;      /* [hidden] attn output */
    float* q;        /* [n_heads * head_dim] */
    float* k_now;    /* [kv_dim] */
    float* v_now;    /* [kv_dim] */
    float* gate;     /* [intermediate] */
    float* up;       /* [intermediate] */
    float* mlp_out;  /* [hidden] */
    float* scores;   /* [max_seq] */
    /* F1.i: optional fused-output buffers (NULL if not fused). */
    float* qkv_buf;  /* [hidden + 2*kv_dim] */
    float* gu_buf;   /* [2*intermediate] */
    forward_scratch sc;
} pq_batch_slot;

typedef struct {
    ib_pq_session* s;
    ib_pq_kv_cache* kv;
    pq_batch_slot* slots;
    int B;
    int L;             /* current layer */
    /* Cached config & layer weights for the current layer. */
    int hidden, n_heads, n_kv, head_dim, kv_dim, inter;
    float eps, theta;
    int q_per_kv;
    const float* w_in;
    const float* w_pn;
    char  q_name[32], k_name[32], v_name[32], o_name[32];
    char  gate_name[32], up_name[32], down_name[32];
    char  qkv_name[32], gu_name[32];
    int   fused_qkv;
    int   fused_gu;
} pq_batch_ctx;

/* PHASE 1 (pre-attn): norm → q/k/v_proj → RoPE → KV write. */
static void batch_pre_attn_task(void* arg, int tid, int b0, int b1) {
    (void)tid;
    pq_batch_ctx* ctx = (pq_batch_ctx*)arg;
    int hidden = ctx->hidden, kv_dim = ctx->kv_dim;
    for (int b = b0; b < b1; b++) {
        pq_batch_slot* s = &ctx->slots[b];
        ib_rmsnorm_f32(s->xb, s->x, ctx->w_in, hidden, ctx->eps);
        if (ctx->fused_qkv) {
            session_matmul_via(ctx->s, ctx->qkv_name, s->xb, s->qkv_buf, &s->sc);
            memcpy(s->q,     s->qkv_buf,                   (size_t)hidden * sizeof(float));
            memcpy(s->k_now, s->qkv_buf + hidden,          (size_t)kv_dim * sizeof(float));
            memcpy(s->v_now, s->qkv_buf + hidden + kv_dim, (size_t)kv_dim * sizeof(float));
        } else {
            session_matmul_via(ctx->s, ctx->q_name, s->xb, s->q, &s->sc);
            session_matmul_via(ctx->s, ctx->k_name, s->xb, s->k_now, &s->sc);
            session_matmul_via(ctx->s, ctx->v_name, s->xb, s->v_now, &s->sc);
        }
        ib_rope_f32(s->q,     ctx->n_heads, ctx->head_dim, s->pos, ctx->theta);
        ib_rope_f32(s->k_now, ctx->n_kv,    ctx->head_dim, s->pos, ctx->theta);
        /* KV write at slot's absolute position. fp32 path; int8/fp16
         * paths are not currently used by the prefill helper to keep
         * this commit small. */
        if (ctx->kv && !ctx->kv->storage_int8 && !ctx->kv->storage_fp16) {
            size_t off = ((size_t)ctx->L * ctx->kv->max_seq + s->pos) * kv_dim;
            memcpy(ctx->kv->k_f32 + off, s->k_now, (size_t)kv_dim * sizeof(float));
            memcpy(ctx->kv->v_f32 + off, s->v_now, (size_t)kv_dim * sizeof(float));
        } else if (ctx->kv && ctx->kv->storage_fp16) {
            uint16_t* kp = ctx->kv->k_f16 + ((size_t)ctx->L * ctx->kv->max_seq + s->pos) * kv_dim;
            uint16_t* vp = ctx->kv->v_f16 + ((size_t)ctx->L * ctx->kv->max_seq + s->pos) * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                kp[d] = ib_fp32_to_fp16(s->k_now[d]);
                vp[d] = ib_fp32_to_fp16(s->v_now[d]);
            }
        }
        /* int8 path falls back to sequential — not used here. */
    }
}

/* PHASE 2 (post-attn): attention reads → o_proj → residual → norm →
 * gate/up → silu_gate → down → residual. */
static void batch_post_attn_task(void* arg, int tid, int b0, int b1) {
    (void)tid;
    pq_batch_ctx* ctx = (pq_batch_ctx*)arg;
    int hidden = ctx->hidden, n_heads = ctx->n_heads, n_kv = ctx->n_kv;
    int head_dim = ctx->head_dim, kv_dim = ctx->kv_dim, inter = ctx->inter;
    int q_per_kv = ctx->q_per_kv;
    float inv_sqrt = 1.0f / sqrtf((float)head_dim);
    float k_scratch[256], v_scratch[256];
    for (int b = b0; b < b1; b++) {
        pq_batch_slot* sl = &ctx->slots[b];
        /* Multi-head causal attention over kv positions [0, sl->pos]. */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / q_per_kv;
            const float* qh = sl->q + (size_t)h * head_dim;
            for (int t = 0; t <= sl->pos; t++) {
                const float* k_t;
                if (ctx->kv->storage_fp16) {
                    const uint16_t* k_src = ctx->kv->k_f16
                        + ((size_t)ctx->L * ctx->kv->max_seq + t) * kv_dim
                        + (size_t)kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++) k_scratch[d] = ib_fp16_to_fp32(k_src[d]);
                    k_t = k_scratch;
                } else {
                    k_t = ctx->kv->k_f32 + ((size_t)ctx->L * ctx->kv->max_seq + t) * kv_dim
                                         + (size_t)kv_h * head_dim;
                }
                sl->scores[t] = pq_dot_f32(qh, k_t, head_dim) * inv_sqrt;
            }
            ib_softmax_f32(sl->scores, sl->pos + 1);
            float* outh = sl->xb2 + (size_t)h * head_dim;
            memset(outh, 0, (size_t)head_dim * sizeof(float));
            for (int t = 0; t <= sl->pos; t++) {
                const float* v_t;
                if (ctx->kv->storage_fp16) {
                    const uint16_t* v_src = ctx->kv->v_f16
                        + ((size_t)ctx->L * ctx->kv->max_seq + t) * kv_dim
                        + (size_t)kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++) v_scratch[d] = ib_fp16_to_fp32(v_src[d]);
                    v_t = v_scratch;
                } else {
                    v_t = ctx->kv->v_f32 + ((size_t)ctx->L * ctx->kv->max_seq + t) * kv_dim
                                          + (size_t)kv_h * head_dim;
                }
                pq_accum_scaled_f32(outh, v_t, sl->scores[t], head_dim);
            }
        }
        session_matmul_via(ctx->s, ctx->o_name, sl->xb2, sl->xb, &sl->sc);
        ib_residual_add_f32(sl->x, sl->xb, hidden);

        ib_rmsnorm_f32(sl->xb, sl->x, ctx->w_pn, hidden, ctx->eps);
        if (ctx->fused_gu) {
            session_matmul_via(ctx->s, ctx->gu_name, sl->xb, sl->gu_buf, &sl->sc);
            memcpy(sl->gate, sl->gu_buf,         (size_t)inter * sizeof(float));
            memcpy(sl->up,   sl->gu_buf + inter, (size_t)inter * sizeof(float));
        } else {
            session_matmul_via(ctx->s, ctx->gate_name, sl->xb, sl->gate, &sl->sc);
            session_matmul_via(ctx->s, ctx->up_name, sl->xb, sl->up, &sl->sc);
        }
        ib_silu_gate_f32(sl->gate, sl->gate, sl->up, inter);
        session_matmul_via(ctx->s, ctx->down_name, sl->gate, sl->mlp_out, &sl->sc);
        ib_residual_add_f32(sl->x, sl->mlp_out, hidden);
    }
}

int ib_pq_forward_step_batch(ib_pq_session* s, ib_pq_kv_cache* kv,
                               const int* tokens, int B, int pos_start,
                               float* hidden_last_out) {
    if (!s || !kv || !tokens || B <= 0) return -1;
    if (kv->storage_int8 || kv->storage_pyramid) {
        /* int8 / pyramid KV paths use per-vector quantization (or NN
         * encoding) that's awkward to write from multiple threads
         * cleanly. Fall back to sequential. */
        int rc = 0;
        for (int b = 0; b < B && rc == 0; b++) {
            float* hbuf = (b == B - 1) ? hidden_last_out : NULL;
            rc = forward_step_internal_sc(s, kv, tokens[b], pos_start + b,
                                            NULL, hbuf, NULL);
        }
        return rc;
    }
    if (B == 1) {
        return forward_step_internal_sc(s, kv, tokens[0], pos_start,
                                          NULL, hidden_last_out, NULL);
    }

    const char* cfg = ib_pq_session_config_json(s);
    int n_layers = session_config_int(cfg, "num_layers", 0);
    int hidden   = session_config_int(cfg, "hidden_size", 0);
    int n_heads  = session_config_int(cfg, "num_heads", 0);
    int n_kv     = session_config_int(cfg, "num_kv_heads", n_heads);
    int head_dim = session_config_int(cfg, "head_dim", hidden / n_heads);
    int inter    = session_config_int(cfg, "intermediate_size", 0);
    float eps    = session_config_float(cfg, "rms_norm_eps", 1e-5f);
    float theta  = session_config_float(cfg, "rope_theta", 10000.0f);
    int kv_dim   = n_kv * head_dim;
    if (n_layers <= 0 || hidden <= 0) return -1;

    /* Per-slot allocation. All slots own their state independently. */
    int max_seq = kv->max_seq;
    pq_batch_slot* slots = (pq_batch_slot*)calloc((size_t)B, sizeof(*slots));
    if (!slots) return -1;
    int max_K = 0, max_Kl2 = 0;
    for (int i = 0; i < s->multi.n; i++) {
        const ib_pq_lut_cache* c = s->mc->caches[i];
        if (c->K > max_K) max_K = c->K;
        if (c->n_levels == 2 && c->K_l2 > max_Kl2) max_Kl2 = c->K_l2;
    }
    int fused_qkv = (ib_pq_session_tensor_shape(s, "L0_qkv_proj", NULL, NULL) == 0);
    int fused_gu  = (ib_pq_session_tensor_shape(s, "L0_gateup_proj", NULL, NULL) == 0);
    int rc = 0;
    for (int b = 0; b < B; b++) {
        pq_batch_slot* sl = &slots[b];
        sl->token_id = tokens[b];
        sl->pos = pos_start + b;
        sl->x       = (float*)calloc((size_t)hidden, sizeof(float));
        sl->xb      = (float*)calloc((size_t)hidden, sizeof(float));
        sl->xb2     = (float*)calloc((size_t)hidden, sizeof(float));
        sl->q       = (float*)calloc((size_t)hidden, sizeof(float));
        sl->k_now   = (float*)calloc((size_t)kv_dim, sizeof(float));
        sl->v_now   = (float*)calloc((size_t)kv_dim, sizeof(float));
        sl->gate    = (float*)calloc((size_t)inter,  sizeof(float));
        sl->up      = (float*)calloc((size_t)inter,  sizeof(float));
        sl->mlp_out = (float*)calloc((size_t)hidden, sizeof(float));
        sl->scores  = (float*)calloc((size_t)(max_seq + 1), sizeof(float));
        sl->sc.C1L  = (float*)malloc((size_t)max_K   * sizeof(float));
        sl->sc.C1R  = (float*)malloc((size_t)max_K   * sizeof(float));
        sl->sc.C2L  = (float*)malloc((size_t)max_Kl2 * sizeof(float));
        sl->sc.C2R  = (float*)malloc((size_t)max_Kl2 * sizeof(float));
        sl->sc.x    = (float*)malloc((size_t)hidden  * sizeof(float));
        sl->sc.x_n  = hidden;
        sl->qkv_buf = fused_qkv ? (float*)malloc((size_t)(hidden + 2*kv_dim) * sizeof(float)) : NULL;
        sl->gu_buf  = fused_gu  ? (float*)malloc((size_t)(2 * inter) * sizeof(float)) : NULL;
        if (!sl->x || !sl->xb || !sl->xb2 || !sl->q || !sl->k_now || !sl->v_now
         || !sl->gate || !sl->up || !sl->mlp_out || !sl->scores
         || !sl->sc.C1L || !sl->sc.C1R || !sl->sc.C2L || !sl->sc.C2R || !sl->sc.x
         || (fused_qkv && !sl->qkv_buf) || (fused_gu && !sl->gu_buf)) {
            rc = -1;
        }
        /* Embedding lookup. */
        if (rc == 0 && embed_lookup(s, sl->token_id, sl->x, hidden) != 0) rc = -1;
    }

    if (rc == 0) {
        pq_batch_ctx ctx = {0};
        ctx.s = s; ctx.kv = kv; ctx.slots = slots; ctx.B = B;
        ctx.hidden = hidden; ctx.n_heads = n_heads; ctx.n_kv = n_kv;
        ctx.head_dim = head_dim; ctx.kv_dim = kv_dim; ctx.inter = inter;
        ctx.eps = eps; ctx.theta = theta;
        ctx.q_per_kv = n_heads / n_kv;
        ctx.fused_qkv = fused_qkv;
        ctx.fused_gu  = fused_gu;

        for (int L = 0; L < n_layers; L++) {
            ctx.L = L;
            const float* w_in = NULL; const float* w_pn = NULL;
            char buf[64];
            snprintf(buf, sizeof(buf), "L%d_input_norm", L);
            if (load_norm_weight(s, buf, &w_in, hidden) != 0) { rc = -1; break; }
            snprintf(buf, sizeof(buf), "L%d_post_attn_norm", L);
            if (load_norm_weight(s, buf, &w_pn, hidden) != 0) { rc = -1; break; }
            ctx.w_in = w_in; ctx.w_pn = w_pn;
            snprintf(ctx.q_name, sizeof(ctx.q_name), "L%d_q_proj", L);
            snprintf(ctx.k_name, sizeof(ctx.k_name), "L%d_k_proj", L);
            snprintf(ctx.v_name, sizeof(ctx.v_name), "L%d_v_proj", L);
            snprintf(ctx.o_name, sizeof(ctx.o_name), "L%d_o_proj", L);
            snprintf(ctx.gate_name, sizeof(ctx.gate_name), "L%d_gate_proj", L);
            snprintf(ctx.up_name, sizeof(ctx.up_name), "L%d_up_proj", L);
            snprintf(ctx.down_name, sizeof(ctx.down_name), "L%d_down_proj", L);
            snprintf(ctx.qkv_name, sizeof(ctx.qkv_name), "L%d_qkv_proj", L);
            snprintf(ctx.gu_name, sizeof(ctx.gu_name), "L%d_gateup_proj", L);

            /* PHASE 1: pre-attn (norm + qkv + RoPE + KV write). */
            if (s->spin && s->n_threads > 1) {
                int chunk = (B + s->n_threads - 1) / s->n_threads;
                if (chunk < 1) chunk = 1;
                pq_spin_pool_run(s->spin, batch_pre_attn_task, &ctx, B, chunk);
            } else {
                batch_pre_attn_task(&ctx, 0, 0, B);
            }

            /* PHASE 2: attn read + remainder. KV writes from PHASE 1 are
             * fully synchronized at this point (pool returned). */
            if (s->spin && s->n_threads > 1) {
                int chunk = (B + s->n_threads - 1) / s->n_threads;
                if (chunk < 1) chunk = 1;
                pq_spin_pool_run(s->spin, batch_post_attn_task, &ctx, B, chunk);
            } else {
                batch_post_attn_task(&ctx, 0, 0, B);
            }
        }
        if (rc == 0) kv->length = pos_start + B;

        if (rc == 0 && hidden_last_out) {
            const float* w_final = NULL;
            if (load_norm_weight(s, "final_norm", &w_final, hidden) != 0) rc = -1;
            else {
                ib_rmsnorm_f32(hidden_last_out, slots[B - 1].x, w_final, hidden, eps);
            }
        }
    }

    for (int b = 0; b < B; b++) {
        pq_batch_slot* sl = &slots[b];
        free(sl->x); free(sl->xb); free(sl->xb2); free(sl->q);
        free(sl->k_now); free(sl->v_now);
        free(sl->gate); free(sl->up); free(sl->mlp_out); free(sl->scores);
        free(sl->sc.C1L); free(sl->sc.C1R); free(sl->sc.C2L); free(sl->sc.C2R);
        free(sl->sc.x);
        free(sl->qkv_buf); free(sl->gu_buf);
    }
    free(slots);
    return rc;
}

int ib_pq_session_reconstruct_row(ib_pq_session* s, const char* name,
                                    int row, float* out_row) {
    if (!s || !name || !out_row) return -1;
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    const ib_pq_tensor* t = &s->multi.tensors[i];
    const ib_pq_lut_cache* c = s->mc->caches[i];
    if (row < 0 || row >= t->M) return -1;
    int N = t->N, G = t->G, C = t->C;
    int half = c->half;
    int K = t->K, K_l2 = c->K_l2;
    int has_l2 = (c->n_levels == 2);
    int l2_packed = (K_l2 == 16);
    int l2_idx_bytes = t->l2_idx_bytes > 0 ? t->l2_idx_bytes : 1;
    size_t l2_stride = l2_packed
        ? (size_t)((C + 1) / 2)
        : (size_t)C * (size_t)l2_idx_bytes;

    /* Initialize output to zero so outliers + chunks accumulate cleanly. */
    memset(out_row, 0, (size_t)N * sizeof(float));

    /* Walk chunks and place reconstructed half-values into the
     * corresponding columns. inner_cols maps chunk position → original
     * column index. Per row scale + outlier sidecar applied at end. */
    float row_scale = ib_fp16_to_fp32(t->row_scale[row]);
    for (int cc = 0; cc < C; cc++) {
        int base = cc * G;
        int i1l = t->indices_l1_l[(size_t)row * C + cc];
        int i1r = t->indices_l1_r[(size_t)row * C + cc];
        const float* eL = c->cb1l_fp32 + (size_t)i1l * half;
        const float* eR = c->cb1r_fp32 + (size_t)i1r * half;
        for (int j = 0; j < half; j++) {
            int col_l = c->inner_cols[base + j];
            int col_r = c->inner_cols[base + half + j];
            out_row[col_l] = eL[j];
            out_row[col_r] = eR[j];
        }
        if (has_l2) {
            const uint8_t* il2l = t->indices_l2_l + (size_t)row * l2_stride;
            const uint8_t* il2r = t->indices_l2_r + (size_t)row * l2_stride;
            int i2l = pq_l2_idx_at(il2l, cc, l2_packed, l2_idx_bytes);
            int i2r = pq_l2_idx_at(il2r, cc, l2_packed, l2_idx_bytes);
            const float* eL2 = c->cb2l_fp32 + (size_t)i2l * half;
            const float* eR2 = c->cb2r_fp32 + (size_t)i2r * half;
            for (int j = 0; j < half; j++) {
                int col_l = c->inner_cols[base + j];
                int col_r = c->inner_cols[base + half + j];
                out_row[col_l] += eL2[j];
                out_row[col_r] += eR2[j];
            }
        }
    }
    /* Apply per-row scale to the reconstructed values. */
    for (int n = 0; n < N; n++) out_row[n] *= row_scale;

    /* Outlier columns: dequant int8 sidecar with per-column scale. */
    if (t->n_outlier > 0) {
        for (int j = 0; j < t->n_outlier; j++) {
            int col = t->outlier_cols[j];
            float os = ib_fp16_to_fp32(t->outlier_scale[j]);
            float w = (float)t->outlier_sidecar[(size_t)row * t->n_outlier + j] * os;
            out_row[col] = w;
        }
    }
    return 0;
}
