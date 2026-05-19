#ifndef PQV2_FORMAT_H
#define PQV2_FORMAT_H

#include "pqv2_kernel.h"
#include <stdint.h>
#include <stddef.h>

/* IBF v6 / PQv2 file format — multi-tensor container.
 *
 * File layout (little-endian):
 *   [0..7]    magic     = "IBFV6PQ2"
 *   [8..11]   version   = 1
 *   [12..15]  n_tensors
 *   [16..19]  manifest_size_bytes
 *   [20..23]  reserved (0)
 *
 *   manifest[n_tensors]:
 *     uint16  name_len
 *     bytes   name (utf-8, name_len)
 *     uint8   kind  (0 = pqv2, 1 = raw_fp16, 2 = raw_fp32, 3 = raw_int32)
 *     uint8   ndim
 *     uint8   reserved[2]
 *     int32   shape[4]   (unused dims = 1)
 *     uint64  blob_offset (from start of file)
 *     uint64  blob_size
 *
 *   <align to 64>
 *
 *   For each tensor at blob_offset:
 *     If kind == 0 (PQV2): the "PQV2" single-tensor blob from pqv2_kernel.c:
 *       magic "PQV2", header u32×N (M,N,G,K,n_sub,half,l2_kind,l2_K,
 *                                    [l2_idx_bits], [residency_hint],
 *                                    [scale_precision],
 *                                    [cb_pool_size], [l2_cb_pool_size]),
 *       row_scale fp16[M] (legacy) OR fp8 E4M3[M] (Stage 5k H2 sp2,
 *         scale_precision >= 1; ~10-decade dynamic range — replaces the
 *         original int8[M]+fp16 row_max codec that lost small rows),
 *       cb_q[rows*K*half] int8 — rows = cb_pool_size if > 0 else ns (Stage 5j),
 *       cb_scale[rows*K] fp16 (legacy) OR fp8 E4M3 (Stage 5k, scale_precision >= 2),
 *       cb_pool_id[ns] u8 — ONLY present when cb_pool_size > 0 (Stage 5j),
 *       indices[M*nc*ns] u8 (transposed [nc, ns, M]),
 *       L2 (if l2_kind==2): l2_cb_q[l2_rows*l2_K*half],
 *         l2_cb_scale[l2_rows*l2_K] fp16/fp8,
 *         l2_cb_pool_id[ns] u8 (only when l2_cb_pool_size > 0),
 *         l2_indices.
 *       Header sizes evolve in append-only fashion:
 *          8 u32 — legacy v0.4.0.
 *          9 u32 — Stage 5h.1: + l2_idx_bits.
 *         10 u32 — Stage 5c   : + residency_hint (0=AUTO, 1=RAM, 2=DRIVE).
 *         11 u32 — Stage 5k   : + scale_precision (0=fp16/fp16,
 *                                                  2=fp8 E4M3 / fp8 E4M3
 *                                                  — H2 sp2 redesign,
 *                                                  see pqv2_encode.c).
 *         13 u32 — Stage 5j   : + cb_pool_size + l2_cb_pool_size.
 *       Parser disambiguates by reconciling header size against the
 *       blob's total length (see parse_pqv2_blob in pqv2_format.c). Old
 *       files default new fields to 0 — bit-identical legacy behavior.
 *     If kind == 1/2/3: raw bytes of the tensor in the given dtype.
 *
 * Caller must keep the entire file mapped/loaded for the lifetime of
 * the parsed tensors (pointers index into the file buffer).
 *
 * ─── MoME naming convention (Stage 3a, docs/v2/00_CORRECTION.md) ────
 *
 * The v6 manifest does NOT carry an explicit `mome_experts` field.
 * Instead the encoder + loader agree on a naming convention:
 *
 *   Lk.mlp.gate_proj           — present iff mome_experts == 1
 *   Lk.mlp.up_proj             — same.
 *   Lk.mlp.down_proj           — same.
 *
 *   Lk.mlp.gate_proj.expert0   — present iff mome_experts > 1.
 *   Lk.mlp.gate_proj.expert1   …  up to expertN-1.
 *   Lk.mlp.up_proj.expert{e}
 *   Lk.mlp.down_proj.expert{e}
 *   Lk.mlp.router              — [K, hidden] raw fp16 router weight
 *                                (zero-init in v1, calibrated later).
 *                                Rows = experts (output), cols =
 *                                hidden (input). Same orientation as
 *                                a regular [M, N] weight so the runtime
 *                                fp16 matmul can consume it without
 *                                transposing.
 *
 * Each expert sub-tensor is a normal PQv2 tensor (kind == 0). The
 * shapes are:
 *   gate/up expert e : [M / K, hidden]   — trivial row-split.
 *   down expert e    : [hidden, M / K]   — matching column slice.
 *
 * Reading the count: the loader scans the manifest, finds the
 * highest `.expert{N}` index per layer, and sets
 * `ib_layer_meta.mome_experts = N + 1`. When N == 0 (no expert
 * tensors found), `mome_experts` stays 1 and the legacy
 * gate_proj/up_proj/down_proj slots are populated as before.
 *
 * Forward compatibility: older loaders without the MoME parser will
 * silently ignore tensors whose name doesn't match the legacy
 * pattern. They will fail to find gate_proj/up_proj/down_proj and
 * the model won't run — but they will not crash. New loaders see
 * both kinds and pick the right slot.
 */

typedef enum {
    IB_PQV2_KIND_PQV2 = 0,
    IB_PQV2_KIND_RAW_FP16 = 1,
    IB_PQV2_KIND_RAW_FP32 = 2,
    IB_PQV2_KIND_RAW_INT32 = 3,
} ib_pqv2_kind;

typedef struct {
    char *name;          /* heap-allocated */
    int kind;            /* ib_pqv2_kind */
    int ndim;
    int shape[4];
    /* For PQV2 kind: parsed pqv2_t (pointers index into file buffer) */
    pqv2_t pq;
    /* For RAW kind: raw data pointer (into file buffer) */
    const void *raw_data;
    size_t raw_size;
    /* Stage 5c — per-tensor residency hint parsed from the on-disk
     * blob header (PQv2 kind) or defaulted to AUTO (raw kinds). Values
     * match the public inferbit_residency enum: 0=AUTO, 1=RAM, 2=DRIVE.
     * Stored as int so this header stays free of the public-API
     * inclusion ordering. Default 0 (AUTO) for older files whose blob
     * predates this field — disambiguated by the same blob-size
     * heuristic that handles the Stage 5h.1 l2_idx_bits field. */
    int residency_hint;
    /* Stage 5k / 5j — when the on-disk blob uses lower-precision scales
     * (scale_precision != 0) or a codebook pool (cb_pool_size > 0), the
     * loader allocates fp16/int8 buffers that the kernel can consume
     * unchanged (decoded from int8/fp8 / expanded from the pool). These
     * pointers track ownership so ib_pqv2_file_free can release them;
     * NULL means the corresponding `pq.row_scale` / `pq.cb_scale` /
     * `pq.cb_q` etc. point directly at the mmap'd file (legacy). */
    void *owned_row_scale;     /* fp16[M] when scale_precision >= 1 */
    void *owned_cb_scale;      /* fp16[rows*K] when scale_precision >= 2 */
    void *owned_l2_cb_scale;
    void *owned_cb_q;          /* int8[ns*K*half] when pool expanded */
    void *owned_l2_cb_q;
} ib_pqv2_named_tensor;

typedef struct {
    int n_tensors;
    ib_pqv2_named_tensor *tensors;
    /* File backing: either heap-loaded buffer or mmap'd region. */
    void *_buffer;
    size_t _buffer_size;
    int _is_mmap;        /* 1 if mmap, 0 if malloc */
    int _fd;
} ib_pqv2_file;

/* Load an IBF v6 PQv2 file. Returns 0 on success.
 * Uses mmap when possible (read-only). Caller frees via ib_pqv2_file_free. */
int ib_pqv2_file_load(const char *path, ib_pqv2_file *out);

/* Free file + all tensor metadata. Tensor data pointers become invalid. */
void ib_pqv2_file_free(ib_pqv2_file *f);

/* Look up a tensor by name. Returns NULL if not found. */
const ib_pqv2_named_tensor *ib_pqv2_find(const ib_pqv2_file *f, const char *name);

#endif
