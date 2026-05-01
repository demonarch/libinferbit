/*
 * pq_decode.c — IBF v5 stacked 2D PQ tensor decoder.
 * Spec: docs/26_IBF_V5_PQ_FORMAT.md.
 */

#include "pq_decode.h"
#include "cJSON.h"
#include "inferbit_internal.h"  /* ib_thread_pool, ib_pool_run */

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

/* Per-chunk codebook ↔ x dot table.
 * For each k in [0, K): out[k] = sum_{j in [0, half)} cb[k*half + j] * x[j].
 * Specialized for half ∈ {8, 16} on NEON / AVX2; scalar otherwise. */
static inline void pq_chunk_dot_table_f32(const float* cb, const float* x,
                                            int half, int K, float* out) {
#if defined(__ARM_NEON)
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
    if (n <= 0) { cJSON_Delete(root); fclose(f); return -1; }

    out->n = n;
    out->names = (char**)calloc((size_t)n, sizeof(char*));
    out->tensors = (ib_pq_tensor*)calloc((size_t)n, sizeof(ib_pq_tensor));
    if (!out->names || !out->tensors) {
        cJSON_Delete(root); fclose(f); ib_pq_multi_free(out); return -1;
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
    mc->caches = (ib_pq_lut_cache**)calloc((size_t)multi->n, sizeof(*mc->caches));
    mc->names  = (char**)calloc((size_t)multi->n, sizeof(*mc->names));
    if (!mc->caches || !mc->names) { ib_pq_multi_caches_free(mc); return -1; }
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

struct ib_pq_session {
    ib_pq_multi multi;            /* owned */
    ib_pq_multi_caches* mc;       /* owned */
    ib_pq_policy default_policy;
    ib_pq_policy* policies;       /* parallel to multi.tensors[i]; NULL => default */
    int int8_quantized;
};

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
    s->policies = (ib_pq_policy*)calloc((size_t)s->multi.n, sizeof(*s->policies));
    if (!s->policies) {
        ib_pq_multi_caches_free(s->mc);
        ib_pq_multi_free(&s->multi); free(s); return -1;
    }
    /* Sentinel: variant=-1 means "use default". */
    for (int i = 0; i < s->multi.n; i++) s->policies[i].variant = -1;

    *out_p = s;
    return 0;
}

void ib_pq_session_close(ib_pq_session* s) {
    if (!s) return;
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

int ib_pq_session_matmul(ib_pq_session* s, const char* name,
                          const float* x, float* out) {
    int i = session_find_index(s, name);
    if (i < 0) return -1;
    const ib_pq_tensor* t = &s->multi.tensors[i];
    const ib_pq_lut_cache* c = s->mc->caches[i];
    ib_pq_policy p = (s->policies[i].variant >= 0) ? s->policies[i] : s->default_policy;

    switch (p.variant) {
    case IB_PQ_VARIANT_L1_ONLY:
        return ib_pq_matmul_fp32_l1_only(t, x, out);
    case IB_PQ_VARIANT_L2SKIP:
        return ib_pq_matmul_fp32_streaming_l2skip_cached(t, c, x, out, p.skip_threshold);
    case IB_PQ_VARIANT_SPARSE:
        return ib_pq_matmul_fp32_streaming_sparse(t, x, out, p.act_threshold);
    case IB_PQ_VARIANT_INT8:
        return ib_pq_matmul_fp32_streaming_int8_cached(t, c, x, out);
    case IB_PQ_VARIANT_STREAMING:
    default:
        return ib_pq_matmul_fp32_streaming_cached(t, c, x, out);
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
