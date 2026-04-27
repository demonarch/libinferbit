/*
 * pq_decode.c — IBF v5 stacked 2D PQ tensor decoder.
 * Spec: docs/26_IBF_V5_PQ_FORMAT.md.
 */

#include "pq_decode.h"
#include "cJSON.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (strcmp(s, "pq2d_v1_l1")    == 0) return IB_PQ_FMT_L1;
    if (strcmp(s, "pq2d_v1_l2")    == 0) return IB_PQ_FMT_L2;
    if (strcmp(s, "pq2d_v1_l1_l2") == 0) return IB_PQ_FMT_L1_L2;
    return IB_PQ_FMT_NONE;
}

/* ── Loader ────────────────────────────────────────────────────── */

void ib_pq_free(ib_pq_tensor* t) {
    if (!t) return;
    free(t->_arena);
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

    int M = out->M, N = out->N, G = out->G, K = out->K;
    int n_inner = N - out->n_outlier;
    if (G <= 0 || K <= 0 || (G & 1) || n_inner % G != 0) return -1;
    out->C = n_inner / G;
    int C = out->C;

    size_t cb_sz   = (size_t)K * (G / 2) * sizeof(uint16_t);
    size_t idx_sz  = (size_t)M * (size_t)C * sizeof(uint8_t);
    size_t rs_sz   = (size_t)M * sizeof(uint16_t);
    size_t oc_sz   = (size_t)out->n_outlier * sizeof(int32_t);
    size_t osc_sz  = (size_t)M * (size_t)out->n_outlier * sizeof(int8_t);
    size_t oscl_sz = (size_t)out->n_outlier * sizeof(uint16_t);

    size_t total = 2*cb_sz + 2*idx_sz + rs_sz;
    if (out->n_levels == 2) total += 2*cb_sz + 2*idx_sz;
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
        SLICE(out->codebook_l2_l, uint16_t, cb_sz);
        SLICE(out->codebook_l2_r, uint16_t, cb_sz);
        SLICE(out->indices_l2_l,  uint8_t,  idx_sz);
        SLICE(out->indices_l2_r,  uint8_t,  idx_sz);
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
        LOAD_BLOCK("codebook_l2_l", out->codebook_l2_l, cb_sz);
        LOAD_BLOCK("codebook_l2_r", out->codebook_l2_r, cb_sz);
        LOAD_BLOCK("indices_l2_l",  out->indices_l2_l,  idx_sz);
        LOAD_BLOCK("indices_l2_r",  out->indices_l2_r,  idx_sz);
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
        for (int i = 0; i < m->n; i++) ib_pq_free(&m->tensors[i]);
        free(m->tensors);
    }
    memset(m, 0, sizeof(*m));
}

const ib_pq_tensor* ib_pq_multi_find(const ib_pq_multi* m, const char* name) {
    if (!m || !name) return NULL;
    for (int i = 0; i < m->n; i++) {
        if (m->names[i] && strcmp(m->names[i], name) == 0) return &m->tensors[i];
    }
    return NULL;
}

/* ── Materialize ──────────────────────────────────────────────── */

int ib_pq_reconstruct_fp32(const ib_pq_tensor* t, float* out) {
    if (!t || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C;
    int half = G / 2;

    /* Pre-decode codebooks to FP32 once */
    int cb_entries = t->K * half;
    float* cb1l = (float*)malloc((size_t)cb_entries * sizeof(float));
    float* cb1r = (float*)malloc((size_t)cb_entries * sizeof(float));
    float* cb2l = NULL;
    float* cb2r = NULL;
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

    for (int r = 0; r < M; r++) {
        float scale = ib_fp16_to_fp32(t->row_scale[r]);
        const uint8_t* il_l = t->indices_l1_l + (size_t)r * C;
        const uint8_t* il_r = t->indices_l1_r + (size_t)r * C;
        const uint8_t* il2l = t->indices_l2_l ? t->indices_l2_l + (size_t)r * C : NULL;
        const uint8_t* il2r = t->indices_l2_r ? t->indices_l2_r + (size_t)r * C : NULL;
        for (int c = 0; c < C; c++) {
            int base = c * G;
            const float* lvec = &cb1l[(size_t)il_l[c] * half];
            const float* rvec = &cb1r[(size_t)il_r[c] * half];
            for (int k = 0; k < half; k++) {
                float v = lvec[k];
                if (cb2l) v += cb2l[(size_t)il2l[c] * half + k];
                int col = inner_cols[base + k];
                /* Match Python: chunk *= row_scale, then cast to fp16,
                 * so we go fp32 -> fp16 -> fp32 to be bit-identical. */
                out[(size_t)r * N + col] = ib_fp16_to_fp32(ib_fp32_to_fp16(v * scale));
            }
            for (int k = 0; k < half; k++) {
                float v = rvec[k];
                if (cb2r) v += cb2r[(size_t)il2r[c] * half + k];
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
int ib_pq_matmul_fp32(const ib_pq_tensor* t, const float* x, float* out) {
    if (!t || !x || !out) return -1;
    int M = t->M, N = t->N, G = t->G, C = t->C, K = t->K;
    int half = G / 2;

    /* Codebook tables decoded to fp32 once. */
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

    /* Pre-multiplied tables. Layout: T_*[c * K + k] (chunk-major so the
     * per-row inner loop steps the chunk index by 1 contiguously per c). */
    size_t table_sz = (size_t)C * K;
    float* TL1 = (float*)malloc(table_sz * sizeof(float));
    float* TR1 = (float*)malloc(table_sz * sizeof(float));
    float* TL2 = NULL, *TR2 = NULL;
    if (!TL1 || !TR1) {
        free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
        free(TL1); free(TR1); return -1;
    }
    if (cb2l) {
        TL2 = (float*)malloc(table_sz * sizeof(float));
        TR2 = (float*)malloc(table_sz * sizeof(float));
        if (!TL2 || !TR2) { /* leak-and-fail; matmul must abort */
            free(cb1l); free(cb1r); free(cb2l); free(cb2r); free(inner_cols);
            free(TL1); free(TR1); free(TL2); free(TR2); return -1;
        }
    }

    /* Build the partial-dot tables. For each chunk c and codebook
     * entry k: T_L*[c*K + k] = sum_{i<half} cb*l[k,i] * x[inner_cols[c*G+i]]. */
    for (int c = 0; c < C; c++) {
        int base = c * G;
        /* Cache the two x slices for this chunk. */
        float xL[16], xR[16];   /* half ≤ 16 in any sane config */
        for (int i = 0; i < half; i++) xL[i] = x[inner_cols[base + i]];
        for (int i = 0; i < half; i++) xR[i] = x[inner_cols[base + half + i]];
        for (int k = 0; k < K; k++) {
            float sl = 0.0f, sr = 0.0f;
            const float* lv = &cb1l[(size_t)k * half];
            const float* rv = &cb1r[(size_t)k * half];
            for (int i = 0; i < half; i++) sl += lv[i] * xL[i];
            for (int i = 0; i < half; i++) sr += rv[i] * xR[i];
            TL1[(size_t)c * K + k] = sl;
            TR1[(size_t)c * K + k] = sr;
            if (cb2l) {
                float sl2 = 0.0f, sr2 = 0.0f;
                const float* lv2 = &cb2l[(size_t)k * half];
                const float* rv2 = &cb2r[(size_t)k * half];
                for (int i = 0; i < half; i++) sl2 += lv2[i] * xL[i];
                for (int i = 0; i < half; i++) sr2 += rv2[i] * xR[i];
                TL2[(size_t)c * K + k] = sl2;
                TR2[(size_t)c * K + k] = sr2;
            }
        }
    }

    /* Per-row gather. */
    for (int r = 0; r < M; r++) {
        float scale = ib_fp16_to_fp32(t->row_scale[r]);
        const uint8_t* il_l = t->indices_l1_l + (size_t)r * C;
        const uint8_t* il_r = t->indices_l1_r + (size_t)r * C;
        const uint8_t* il2l = TL2 ? t->indices_l2_l + (size_t)r * C : NULL;
        const uint8_t* il2r = TR2 ? t->indices_l2_r + (size_t)r * C : NULL;
        float acc = 0.0f;
        if (TL2) {
            for (int c = 0; c < C; c++) {
                size_t base = (size_t)c * K;
                acc += TL1[base + il_l[c]] + TR1[base + il_r[c]];
                acc += TL2[base + il2l[c]] + TR2[base + il2r[c]];
            }
        } else {
            for (int c = 0; c < C; c++) {
                size_t base = (size_t)c * K;
                acc += TL1[base + il_l[c]] + TR1[base + il_r[c]];
            }
        }
        out[r] = acc * scale;
    }

    /* Outliers — small (n_outlier ≪ N), kept as scalar fp32 fma. */
    for (int j = 0; j < t->n_outlier; j++) {
        int col = t->outlier_cols[j];
        float os = ib_fp16_to_fp32(t->outlier_scale[j]);
        float xj = x[col];
        for (int r = 0; r < M; r++) {
            float w = (float)t->outlier_sidecar[(size_t)r * t->n_outlier + j] * os;
            out[r] += w * xj;
        }
    }

    free(cb1l); free(cb1r); free(cb2l); free(cb2r);
    free(inner_cols); free(TL1); free(TR1); free(TL2); free(TR2);
    return 0;
}
