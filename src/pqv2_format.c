#include "pqv2_format.h"
#include "pqv2_kernel.h"
#include "platform.h"   /* ib_open/ib_mmap/ib_close + cross-platform I/O */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>

#define IB_PQV2_MAGIC "IBFV6PQ2"
#define IB_PQV2_VERSION 1u

static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

/* Per-row packed-index byte count.
 *   bits == 6 (Stage 5h.1): ceil(M/4)*3 — 4 indices in 3 bytes.
 *   bits == 4 (Goal N36)  : ceil(M/2)   — 2 indices per byte (l2_K ≤ 16).
 * Matches the encoder's M-axis packing in pqv2_encode.c. */
static inline size_t pqv2_l2_packed_bytes_per_row_b(uint32_t M, uint32_t bits) {
    if (bits == 4u) return ((size_t)M + 1u) / 2u;
    return ((size_t)M + 3u) / 4u * 3u;
}
static inline size_t pqv2_l2_packed_bytes_per_row(uint32_t M) {
    return pqv2_l2_packed_bytes_per_row_b(M, 6u);
}

/* ── Stage 5k: fp8 E4M3 decode for row_scale + cb_scale ────────────────
 * Inverse of pqv2_encode.c's enc_f32_to_e4m3 / enc_pack_row_scale_e4m3.
 * Called once per tensor at load time, so cost is amortised over every
 * matmul that uses the tensor — the hot kernel sees only the resulting
 * fp16 arrays and stays byte-identical to the legacy layout.
 *
 * H2 sp2 redesign (Agent 3 round 1 fix): row_scale used to disk-pack as
 * int8[M] + fp16 row_max (linear quantization with a per-tensor anchor).
 * That codec floored the small-magnitude rows of LLM weight matrices to
 * zero — Llama-3 / TinyLlama row_scales span 6-10 decades and a linear
 * 127-step anchor can only resolve ~3. The codec is now fp8 E4M3[M]
 * (logarithmic, ~10 decades dynamic range, 6-12% per-row relative error)
 * — same codec as cb_scale. Old sp=2 IBF files written before this
 * change are unreadable; rerun the encoder. */

static inline float pqv2_e4m3_to_f32(uint8_t b) {
    uint32_t sign = (uint32_t)(b >> 7) & 0x1u;
    uint32_t exp  = (uint32_t)(b >> 3) & 0xFu;
    uint32_t mant = (uint32_t)b & 0x7u;
    if (exp == 0xFu && mant == 0x7u) {
        uint32_t nan_bits = (sign << 31) | 0x7FC00000u;
        float f; memcpy(&f, &nan_bits, 4); return f;
    }
    float val;
    if (exp == 0u) {
        val = (float)mant * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        int e = (int)exp - 7;
        float mantissa = 1.0f + (float)mant * (1.0f / 8.0f);
        val = ldexpf(mantissa, e);
    }
    return sign ? -val : val;
}

/* fp32 → fp16 (round to nearest even, no NaN/Inf re-tagging). Used to
 * normalize Stage 5k decoded scales into the fp16 buffer the kernel
 * already consumes. */
static inline uint16_t pqv2_f32_to_fp16_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int      exp  = (int)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)  return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

/* Owned-pointer bag filled by parse_pqv2_blob when the on-disk layout
 * forces the loader to materialise fp16/int8 scratch (Stages 5k + 5j).
 * Caller (= ib_pqv2_file_load) copies these into the named-tensor entry
 * so ib_pqv2_file_free can release them. Any field left NULL means the
 * corresponding pqv2_t pointer is into the mmap'd file (zero-copy). */
typedef struct {
    void *row_scale;     /* fp16[M], allocated when scale_precision >= 1 */
    void *cb_scale;      /* fp16[rows*K], allocated when scale_precision >= 2 */
    void *l2_cb_scale;
    void *cb_q;          /* int8[ns*K*half], allocated when pool expanded */
    void *l2_cb_q;
} pqv2_blob_owned;

/* Decode an fp8 E4M3 row_scale array (stored as M bytes on disk) into
 * a newly-allocated fp16[M] buffer. Mirrors pqv2_decode_cb_scale_e4m3. */
static uint16_t *pqv2_decode_row_scale_e4m3(const uint8_t *src, uint32_t M) {
    uint16_t *out = (uint16_t *)malloc((size_t)M * sizeof(uint16_t));
    if (!out) return NULL;
    for (uint32_t m = 0; m < M; m++) {
        out[m] = pqv2_f32_to_fp16_bits(pqv2_e4m3_to_f32(src[m]));
    }
    return out;
}

/* Decode an fp8 E4M3 cb_scale block into fp16. */
static uint16_t *pqv2_decode_cb_scale_e4m3(const uint8_t *src, size_t n) {
    uint16_t *out = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!out) return NULL;
    for (size_t i = 0; i < n; i++) {
        out[i] = pqv2_f32_to_fp16_bits(pqv2_e4m3_to_f32(src[i]));
    }
    return out;
}

/* Stage 5j — expand a pooled codebook back into a per-slot [n_sub*K**]
 * buffer. v1 ships pool_size == n_subchunks with identity pool_id, so
 * the result is byte-identical to a non-pooled layout; the path
 * exercises the loader/kernel pool plumbing so future agents can wire
 * real clustering without re-touching the read side. Used for both
 * cb_q (int8) and cb_scale (fp16). */
static void *pqv2_expand_pool_cbq(const int8_t *pool_q, const uint8_t *pool_id,
                                    uint32_t ns, uint32_t K, uint32_t half) {
    size_t row_bytes = (size_t)K * half;
    int8_t *out = (int8_t *)malloc(row_bytes * ns);
    if (!out) return NULL;
    for (uint32_t s = 0; s < ns; s++) {
        uint8_t p = pool_id[s];
        memcpy(out + (size_t)s * row_bytes,
               pool_q + (size_t)p * row_bytes,
               row_bytes);
    }
    return out;
}
static void *pqv2_expand_pool_cbs_fp16(const uint16_t *pool_s, const uint8_t *pool_id,
                                          uint32_t ns, uint32_t K) {
    size_t row_count = (size_t)K;
    uint16_t *out = (uint16_t *)malloc(row_count * ns * sizeof(uint16_t));
    if (!out) return NULL;
    for (uint32_t s = 0; s < ns; s++) {
        uint8_t p = pool_id[s];
        memcpy(out + (size_t)s * row_count,
               pool_s + (size_t)p * row_count,
               row_count * sizeof(uint16_t));
    }
    return out;
}

/* Parse a PQV2 single-tensor blob. Pointers into the result either index
 * into 'buf' directly (zero-copy, legacy) or point at heap allocations
 * recorded in `*out_owned` so the file freer can release them.
 *
 * Header history (append-only):
 *    8 u32 — original (M,N,G,K,n_sub,half,l2_kind,l2_K).
 *    9 u32 — Stage 5h.1: + l2_idx_bits (6=packed, 8=legacy).
 *   10 u32 — Stage 5c:   + residency_hint (0=AUTO, 1=RAM, 2=DRIVE).
 *   11 u32 — Stage 5k:   + scale_precision (0=fp16/fp16, 2=fp8 E4M3 / fp8 E4M3
 *                          — H2 sp2 redesign; was int8+row_max / fp8 E4M3).
 *   13 u32 — Stage 5j:   + cb_pool_size + l2_cb_pool_size.
 *   14 u32 — Stage 5g.2: + l1_idx_layout (0=chunk-major, 1=row-major).
 *
 * Disambiguates without a version field by reconciling header size +
 * data layout against the blob's total length: pick the largest header
 * layout whose projected total equals `size`. */
static int parse_pqv2_blob(const uint8_t *buf, size_t size, pqv2_t *out,
                            int *out_residency_hint,
                            pqv2_blob_owned *out_owned) {
    memset(out_owned, 0, sizeof(*out_owned));
    if (size < 4 + 32) return -1;
    if (memcmp(buf, "PQV2", 4) != 0) return -1;
    const uint32_t *hdr = (const uint32_t *)(buf + 4);
    out->M = hdr[0]; out->N = hdr[1]; out->G = hdr[2]; out->K = hdr[3];
    out->n_subchunks = hdr[4]; out->half = hdr[5];
    out->l2_kind = hdr[6]; out->l2_K = hdr[7];
    if (out_residency_hint) *out_residency_hint = 0;

    size_t n_chunks = out->N / out->G;
    size_t idx_bytes = (size_t)out->M * n_chunks * out->n_subchunks;

    /* Helper: project total blob bytes given the candidate header layout
     * (in u32s), l2_idx_bits, scale_precision, and pool sizes. */
    #define PQV2_PROJ_SIZE_FULL(hdr_u32, b, sp, p1, p2)                        \
        (                                                                       \
          /* header */                                                          \
          (size_t)(4 + (hdr_u32) * 4)                                           \
          /* row_scale: fp16[M] (legacy) OR fp8 E4M3[M] (Stage 5k H2 sp2) */   \
          + ((sp) >= 1 ? (size_t)out->M : (size_t)out->M * 2u)                 \
          /* cb_q + cb_scale (rows = p1 if > 0 else n_sub) — cb_scale  */     \
          /* is ALWAYS fp16 (Goal I2 rollback). See note at decode site. */     \
          + ((size_t)((p1) > 0 ? (p1) : out->n_subchunks)                       \
                * out->K * out->half                                            \
             + (size_t)((p1) > 0 ? (p1) : out->n_subchunks)                     \
                * out->K * 2u)                                                  \
          /* cb_pool_id[n_sub] only when p1 > 0 */                              \
          + ((p1) > 0 ? (size_t)out->n_subchunks : 0u)                          \
          /* L1 indices */                                                      \
          + idx_bytes                                                           \
          /* L2 region */                                                       \
          + ((out->l2_kind == 2)                                                \
              ? ((size_t)((p2) > 0 ? (p2) : out->n_subchunks)                   \
                    * out->l2_K * out->half                                     \
                 + (size_t)((p2) > 0 ? (p2) : out->n_subchunks)                 \
                    * out->l2_K * 2u                                            \
                 + ((p2) > 0 ? (size_t)out->n_subchunks : 0u)                   \
                 + (((b) == 6 || (b) == 4)                                     \
                     ? (size_t)n_chunks * out->n_subchunks                      \
                           * pqv2_l2_packed_bytes_per_row_b(out->M, (b))        \
                     : idx_bytes))                                              \
              : 0u)                                                             \
        )

    /* Disambiguation: try the layouts from newest (most fields) to oldest;
     * pick the first whose projected total matches `size` exactly. */
    uint32_t maybe_bits = 8;
    size_t header_bytes = 4 + 32;
    int hint = 0;
    uint32_t scale_precision = 0;
    uint32_t cb_pool_size = 0;
    uint32_t l2_cb_pool_size = 0;
    uint32_t l1_idx_layout = 0;
    int resolved = 0;

    /* 14-u32 layout (Stage 5g.2 — adds l1_idx_layout). */
    if (!resolved && size >= 4 + 56) {
        uint32_t b = hdr[8], r = hdr[9], sp = hdr[10],
                 p1 = hdr[11], p2 = hdr[12], lay = hdr[13];
        int bits_ok = (b == 4 || b == 6 || b == 8);
        int hint_ok = (r <= 2);
        int sp_ok   = (sp == 0 || sp == 2);
        int p1_ok   = (p1 == 0 || p1 == out->n_subchunks);
        int p2_ok   = (p2 == 0 || p2 == out->n_subchunks);
        int lay_ok  = (lay == 0 || lay == 1);
        if (bits_ok && hint_ok && sp_ok && p1_ok && p2_ok && lay_ok) {
            size_t projected = PQV2_PROJ_SIZE_FULL(14, b, sp, p1, p2);
            if (projected == size) {
                maybe_bits = b;
                header_bytes = 4 + 56;
                hint = (int)r;
                scale_precision = sp;
                cb_pool_size = p1;
                l2_cb_pool_size = p2;
                l1_idx_layout = lay;
                resolved = 1;
            }
        }
    }
    /* 13-u32 layout (Stage 5j). */
    if (!resolved && size >= 4 + 52) {
        uint32_t b = hdr[8], r = hdr[9], sp = hdr[10],
                 p1 = hdr[11], p2 = hdr[12];
        int bits_ok = (b == 4 || b == 6 || b == 8);
        int hint_ok = (r <= 2);
        int sp_ok   = (sp == 0 || sp == 2);
        int p1_ok   = (p1 == 0 || p1 == out->n_subchunks);
        int p2_ok   = (p2 == 0 || p2 == out->n_subchunks);
        if (bits_ok && hint_ok && sp_ok && p1_ok && p2_ok) {
            size_t projected = PQV2_PROJ_SIZE_FULL(13, b, sp, p1, p2);
            if (projected == size) {
                maybe_bits = b;
                header_bytes = 4 + 52;
                hint = (int)r;
                scale_precision = sp;
                cb_pool_size = p1;
                l2_cb_pool_size = p2;
                resolved = 1;
            }
        }
    }
    /* 11-u32 layout (Stage 5k, pre-5j). */
    if (!resolved && size >= 4 + 44) {
        uint32_t b = hdr[8], r = hdr[9], sp = hdr[10];
        int bits_ok = (b == 4 || b == 6 || b == 8);
        int hint_ok = (r <= 2);
        int sp_ok   = (sp == 0 || sp == 2);
        if (bits_ok && hint_ok && sp_ok) {
            size_t projected = PQV2_PROJ_SIZE_FULL(11, b, sp, 0u, 0u);
            if (projected == size) {
                maybe_bits = b;
                header_bytes = 4 + 44;
                hint = (int)r;
                scale_precision = sp;
                resolved = 1;
            }
        }
    }
    /* 10-u32 layout (Stage 5c). */
    if (!resolved && size >= 4 + 40) {
        uint32_t b = hdr[8], r = hdr[9];
        int bits_ok = (b == 4 || b == 6 || b == 8);
        int hint_ok = (r <= 2);
        if (bits_ok && hint_ok) {
            size_t projected = PQV2_PROJ_SIZE_FULL(10, b, 0u, 0u, 0u);
            if (projected == size) {
                maybe_bits = b;
                header_bytes = 4 + 40;
                hint = (int)r;
                resolved = 1;
            }
        }
    }
    /* 9-u32 layout (Stage 5h.1). */
    if (!resolved && size >= 4 + 36) {
        uint32_t b = hdr[8];
        if ((b == 4 || b == 6 || b == 8) &&
            (out->l2_kind == 2 || (b == 8 && out->l2_kind == 0))) {
            size_t projected = PQV2_PROJ_SIZE_FULL(9, b, 0u, 0u, 0u);
            if (projected == size) {
                maybe_bits = b;
                header_bytes = 4 + 36;
                resolved = 1;
            }
        }
    }
    /* Legacy 8-u32 layout falls through; the cursor checks below will
     * reject if the byte budget doesn't add up. */
    #undef PQV2_PROJ_SIZE_FULL

    out->l2_idx_bits = maybe_bits;
    out->scale_precision = scale_precision;
    out->cb_pool_size = cb_pool_size;
    out->l2_cb_pool_size = l2_cb_pool_size;
    out->l1_idx_layout = l1_idx_layout;
    if (out_residency_hint) *out_residency_hint = hint;
    size_t cursor = header_bytes;

    /* On any error past this point we must release the partial owned
     * scratch — caller only ever sees `out_owned` populated when the
     * parse succeeds. */
    #define PQV2_PARSE_FAIL() do {                                            \
        if (out_owned->row_scale)     free(out_owned->row_scale);             \
        if (out_owned->cb_scale)      free(out_owned->cb_scale);              \
        if (out_owned->l2_cb_scale)   free(out_owned->l2_cb_scale);           \
        if (out_owned->cb_q)          free(out_owned->cb_q);                  \
        if (out_owned->l2_cb_q)       free(out_owned->l2_cb_q);               \
        memset(out_owned, 0, sizeof(*out_owned));                             \
        return -1;                                                            \
    } while (0)

    /* row_scale: fp16[M] (legacy) OR fp8 E4M3[M] (Stage 5k H2 sp2). */
    size_t row_disk_bytes = (scale_precision >= 1)
                              ? (size_t)out->M
                              : (size_t)out->M * 2u;
    if (cursor + row_disk_bytes > size) PQV2_PARSE_FAIL();
    if (scale_precision >= 1) {
        const uint8_t *rs_e4m3 = (const uint8_t *)(buf + cursor);
        uint16_t *rs = pqv2_decode_row_scale_e4m3(rs_e4m3, out->M);
        if (!rs) PQV2_PARSE_FAIL();
        out->row_scale = rs;
        out_owned->row_scale = rs;
    } else {
        out->row_scale = (const uint16_t *)(buf + cursor);
    }
    cursor += row_disk_bytes;

    /* cb_q + cb_scale: rows = cb_pool_size if > 0 else n_subchunks.
     *
     * Goal I2: cb_scale is ALWAYS fp16 on disk, regardless of
     * scale_precision. The Stage 5k sp=2 redesign briefly packed
     * cb_scale as fp8 E4M3 (saving ns*K bytes/tensor), but probe_sp2
     * showed cb_scale's distribution clusters around 1e-3..5e-3 with
     * a ~300× range, putting ~30% of codewords in E4M3's subnormal
     * band where the small ones flush to zero (RMSE_rel 38.8% vs 2.6%
     * for row_scale). cb_scale was the actual driver of the sp=2 PPL
     * regression (50.4 → 48.9 after row-scale fp8 fix). Old sp=2
     * files written by the both-fp8 encoder are no longer readable —
     * rerun the encoder. */
    uint32_t cb_rows = cb_pool_size > 0 ? cb_pool_size : out->n_subchunks;
    size_t cb_q_disk_bytes  = (size_t)cb_rows * out->K * out->half;
    size_t cb_s_disk_bytes  = (size_t)cb_rows * out->K * 2u;
    if (cursor + cb_q_disk_bytes + cb_s_disk_bytes > size) PQV2_PARSE_FAIL();
    const int8_t  *cb_q_disk  = (const int8_t  *)(buf + cursor);
    cursor += cb_q_disk_bytes;
    const uint8_t *cb_s_disk  = (const uint8_t *)(buf + cursor);
    cursor += cb_s_disk_bytes;

    /* cb_scale always fp16; point directly at the mmap'd file. */
    const uint16_t *cb_scale_decoded = (const uint16_t *)cb_s_disk;

    /* Pool expansion (Stage 5j). When pool_size > 0, pool_id[n_subchunks]
     * follows the cb_scale block; expand pool → per-slot. v1 identity
     * mapping makes this a memcpy round-trip (no kernel change). */
    if (cb_pool_size > 0) {
        if (cursor + (size_t)out->n_subchunks > size) PQV2_PARSE_FAIL();
        const uint8_t *pool_id = buf + cursor;
        cursor += out->n_subchunks;
        for (uint32_t s = 0; s < out->n_subchunks; s++) {
            if (pool_id[s] >= cb_pool_size) PQV2_PARSE_FAIL();
        }
        void *expanded_q = pqv2_expand_pool_cbq(cb_q_disk, pool_id,
                                                  out->n_subchunks,
                                                  out->K, out->half);
        if (!expanded_q) PQV2_PARSE_FAIL();
        void *expanded_s = pqv2_expand_pool_cbs_fp16(cb_scale_decoded, pool_id,
                                                       out->n_subchunks,
                                                       out->K);
        if (!expanded_s) { free(expanded_q); PQV2_PARSE_FAIL(); }
        out->cb_q = (const int8_t *)expanded_q;
        out->cb_scale = (const uint16_t *)expanded_s;
        out_owned->cb_q = expanded_q;
        /* If we owned the decoded fp16 cb_scale already, free it — the
         * expanded buffer supersedes it. */
        if (out_owned->cb_scale) free(out_owned->cb_scale);
        out_owned->cb_scale = expanded_s;
    } else {
        out->cb_q = cb_q_disk;
        out->cb_scale = cb_scale_decoded;
    }

    if (cursor + idx_bytes > size) PQV2_PARSE_FAIL();
    out->indices = (const uint8_t *)(buf + cursor);
    cursor += idx_bytes;
    out->l2_cb_q = NULL; out->l2_cb_scale = NULL; out->l2_indices = NULL;

    if (out->l2_kind == 2) {
        uint32_t l2_rows = l2_cb_pool_size > 0 ? l2_cb_pool_size : out->n_subchunks;
        size_t l2q_disk_bytes = (size_t)l2_rows * out->l2_K * out->half;
        /* Goal I2: l2_cb_scale is always fp16, matching L1 cb_scale. */
        size_t l2s_disk_bytes = (size_t)l2_rows * out->l2_K * 2u;
        size_t l2_idx_disk =
            (out->l2_idx_bits == 6 || out->l2_idx_bits == 4)
                ? (size_t)n_chunks * out->n_subchunks
                      * pqv2_l2_packed_bytes_per_row_b(out->M,
                                                         out->l2_idx_bits)
                : idx_bytes;
        if (cursor + l2q_disk_bytes + l2s_disk_bytes > size) PQV2_PARSE_FAIL();
        const int8_t  *l2_q_disk = (const int8_t  *)(buf + cursor);
        cursor += l2q_disk_bytes;
        const uint8_t *l2_s_disk = (const uint8_t *)(buf + cursor);
        cursor += l2s_disk_bytes;

        const uint16_t *l2_scale_decoded = (const uint16_t *)l2_s_disk;

        if (l2_cb_pool_size > 0) {
            if (cursor + (size_t)out->n_subchunks > size) PQV2_PARSE_FAIL();
            const uint8_t *pool_id = buf + cursor;
            cursor += out->n_subchunks;
            for (uint32_t s = 0; s < out->n_subchunks; s++) {
                if (pool_id[s] >= l2_cb_pool_size) PQV2_PARSE_FAIL();
            }
            void *expanded_q = pqv2_expand_pool_cbq(l2_q_disk, pool_id,
                                                      out->n_subchunks,
                                                      out->l2_K, out->half);
            if (!expanded_q) PQV2_PARSE_FAIL();
            void *expanded_s = pqv2_expand_pool_cbs_fp16(l2_scale_decoded,
                                                           pool_id,
                                                           out->n_subchunks,
                                                           out->l2_K);
            if (!expanded_s) { free(expanded_q); PQV2_PARSE_FAIL(); }
            out->l2_cb_q = (const int8_t *)expanded_q;
            out->l2_cb_scale = (const uint16_t *)expanded_s;
            out_owned->l2_cb_q = expanded_q;
            if (out_owned->l2_cb_scale) free(out_owned->l2_cb_scale);
            out_owned->l2_cb_scale = expanded_s;
        } else {
            out->l2_cb_q = l2_q_disk;
            out->l2_cb_scale = l2_scale_decoded;
        }

        if (cursor + l2_idx_disk > size) PQV2_PARSE_FAIL();
        out->l2_indices = (const uint8_t *)(buf + cursor);
        cursor += l2_idx_disk;
    }
    #undef PQV2_PARSE_FAIL
    out->cb_fp32 = NULL;
    out->l2_cb_fp32 = NULL;
    return 0;
}

/* Precompute fp32 codebooks once per tensor. Kernel uses them on every
 * matvec call instead of re-decoding the int8 codebook each time. */
static float* decode_codebook_fp32(const int8_t *cb_q, const uint16_t *cb_scale,
                                     uint32_t ns, uint32_t K, uint32_t half) {
    float *cb = aligned_alloc(64,
        ((size_t)ns * K * half * sizeof(float) + 63) & ~(size_t)63);
    if (!cb) return NULL;
    for (uint32_t s = 0; s < ns; s++) {
        for (uint32_t k = 0; k < K; k++) {
            float sc = pqv2_h2f(cb_scale[s * K + k]);
            const int8_t *q = &cb_q[(s * K + k) * half];
            for (uint32_t h = 0; h < half; h++)
                cb[(s * K + k) * half + h] = (float)q[h] * sc;
        }
    }
    return cb;
}

int ib_pqv2_file_load(const char *path, ib_pqv2_file *out) {
    memset(out, 0, sizeof(*out));
    int fd = ib_open(path, O_RDONLY);
    if (fd < 0) return -1;
    ib_struct_stat st;
    if (ib_fstat(fd, &st) < 0) { ib_close(fd); return -1; }
    size_t fsz = (size_t)st.st_size;

    /* Drive mode (IB_RESIDENCY_MODE=drive): bypass the OS page cache
     * so resident set stays small under pressure.
     *  - Darwin: F_NOCACHE on the fd makes pread skip UBC, MAP_NOCACHE
     *    keeps the mmap pages from accumulating.
     *  - Linux:  POSIX_MADV_DONTNEED on the mapping marks it
     *    low-retention; there is no per-fd no-buffering knob.
     *  - Windows: OfferVirtualMemory marks pages reclaimable. The
     *    file-open cache-bypass hint (FILE_FLAG_SEQUENTIAL_SCAN) is
     *    *not* retrofitted — the fd was opened by ib_open without
     *    flag passthrough. ib_set_drive_hint_fd is a no-op there;
     *    ib_advise_dontneed below still gives a real working-set
     *    reduction under pressure. TODO: thread a dedicated open-
     *    drive-mode shim through ib_open if peak RSS still tracks
     *    RAM mode on Windows. */
    int drive_mode = 0;
    {
        const char *rm = getenv("IB_RESIDENCY_MODE");
        drive_mode = (rm && (!strcmp(rm, "drive") || !strcmp(rm, "1"))) ? 1 : 0;
    }
    if (drive_mode) (void)ib_set_drive_hint_fd(fd);

    /* MAP_NOCACHE is Darwin-only and has no Windows/Linux mmap analog;
     * keep it gated. Cross-platform reclaim lives in ib_advise_dontneed. */
    int map_flags = MAP_PRIVATE;
#if defined(__APPLE__) && defined(MAP_NOCACHE)
    if (drive_mode) map_flags |= MAP_NOCACHE;
#endif
    void *buf = ib_mmap(NULL, fsz, PROT_READ, map_flags, fd, 0);
    if (buf == MAP_FAILED) { ib_close(fd); return -1; }
    if (drive_mode) (void)ib_advise_dontneed(buf, fsz);
    out->_buffer = buf; out->_buffer_size = fsz; out->_is_mmap = 1; out->_fd = fd;

    const uint8_t *p = (const uint8_t *)buf;
    if (fsz < 24) goto fail;
    if (memcmp(p, IB_PQV2_MAGIC, 8) != 0) goto fail;
    uint32_t version = *(const uint32_t *)(p + 8);
    if (version != IB_PQV2_VERSION) goto fail;
    int n = (int)*(const uint32_t *)(p + 12);
    uint32_t manifest_size = *(const uint32_t *)(p + 16);
    if (24 + manifest_size > fsz) goto fail;

    out->n_tensors = n;
    out->tensors = calloc((size_t)n, sizeof(*out->tensors));
    if (!out->tensors) goto fail;

    const uint8_t *m = p + 24;
    const uint8_t *m_end = m + manifest_size;
    for (int i = 0; i < n; i++) {
        if (m + 2 > m_end) goto fail;
        uint16_t nl = *(const uint16_t *)m; m += 2;
        if (m + nl > m_end) goto fail;
        ib_pqv2_named_tensor *t = &out->tensors[i];
        t->name = malloc(nl + 1);
        memcpy(t->name, m, nl); t->name[nl] = '\0';
        m += nl;

        if (m + 1 + 1 + 2 + 16 + 8 + 8 > m_end) goto fail;
        t->kind = m[0]; t->ndim = m[1]; m += 4;
        memcpy(t->shape, m, 16); m += 16;
        uint64_t blob_off = *(const uint64_t *)m; m += 8;
        uint64_t blob_size = *(const uint64_t *)m; m += 8;
        if (blob_off + blob_size > fsz) goto fail;

        if (t->kind == IB_PQV2_KIND_PQV2) {
            int hint = 0;
            pqv2_blob_owned owned;
            if (parse_pqv2_blob(p + blob_off, (size_t)blob_size,
                                 &t->pq, &hint, &owned) != 0) goto fail;
            t->residency_hint = hint;
            /* Stage 5k / 5j — record loader-allocated buffers for cleanup.
             * NULL fields mean the corresponding pq pointer is into the
             * mmap'd file (zero-copy / legacy). */
            t->owned_row_scale     = owned.row_scale;
            t->owned_cb_scale      = owned.cb_scale;
            t->owned_l2_cb_scale   = owned.l2_cb_scale;
            t->owned_cb_q          = owned.cb_q;
            t->owned_l2_cb_q       = owned.l2_cb_q;
            /* Precompute fp32 codebooks so the hot kernel skips the
             * int8→fp32 decode loop on every matvec call. */
            t->pq.cb_fp32 = decode_codebook_fp32(
                t->pq.cb_q, t->pq.cb_scale,
                t->pq.n_subchunks, t->pq.K, t->pq.half);
            if (t->pq.l2_kind == 2 && t->pq.l2_cb_q) {
                t->pq.l2_cb_fp32 = decode_codebook_fp32(
                    t->pq.l2_cb_q, t->pq.l2_cb_scale,
                    t->pq.n_subchunks, t->pq.l2_K, t->pq.half);
            }
        } else {
            t->raw_data = p + blob_off;
            t->raw_size = (size_t)blob_size;
            t->residency_hint = 0;  /* AUTO for raw tensors (norms, router) */
        }
    }
    return 0;

fail:
    ib_pqv2_file_free(out);
    return -1;
}

void ib_pqv2_file_free(ib_pqv2_file *f) {
    if (f->tensors) {
        for (int i = 0; i < f->n_tensors; i++) {
            ib_pqv2_named_tensor *t = &f->tensors[i];
            free(t->name);
            if (t->kind == IB_PQV2_KIND_PQV2) {
                if (t->pq.cb_fp32)    free((void*)t->pq.cb_fp32);
                if (t->pq.l2_cb_fp32) free((void*)t->pq.l2_cb_fp32);
                /* Stage 5k / 5j scratch (NULL when zero-copy from mmap). */
                if (t->owned_row_scale)   free(t->owned_row_scale);
                if (t->owned_cb_scale)    free(t->owned_cb_scale);
                if (t->owned_l2_cb_scale) free(t->owned_l2_cb_scale);
                if (t->owned_cb_q)        free(t->owned_cb_q);
                if (t->owned_l2_cb_q)     free(t->owned_l2_cb_q);
            }
        }
        free(f->tensors);
    }
    if (f->_buffer) {
        if (f->_is_mmap) ib_munmap(f->_buffer, f->_buffer_size);
        else free(f->_buffer);
    }
    if (f->_fd >= 0) ib_close(f->_fd);
    memset(f, 0, sizeof(*f));
}

const ib_pqv2_named_tensor *ib_pqv2_find(const ib_pqv2_file *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}
