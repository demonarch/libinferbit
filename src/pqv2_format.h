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
 *                                    [reserved=0],
 *                                    [cb_pool_size], [l2_cb_pool_size]),
 *       row_scale fp16[M],
 *       cb_q[rows*K*half] int8 — rows = cb_pool_size if > 0 else ns (Stage 5j),
 *       cb_scale[rows*K] fp16,
 *       cb_pool_id[ns] u8 — ONLY present when cb_pool_size > 0 (Stage 5j),
 *       indices[M*nc*ns] u8 (chunk-major [nc, ns, M]),
 *       L2 (if l2_kind==2): l2_cb_q[l2_rows*l2_K*half],
 *         l2_cb_scale[l2_rows*l2_K] fp16,
 *         l2_cb_pool_id[ns] u8 (only when l2_cb_pool_size > 0),
 *         l2_indices.
 *       Header sizes evolve in append-only fashion (the sp2 11-u32 and
 *       rowmajor 14-u32 variants were retired — no longer creatable):
 *          8 u32 — legacy v0.4.0.
 *          9 u32 — Stage 5h.1: + l2_idx_bits.
 *         10 u32 — Stage 5c   : + residency_hint (0=AUTO, 1=RAM, 2=DRIVE).
 *         13 u32 — Stage 5j   : + reserved(=0) + cb_pool_size
 *                                  + l2_cb_pool_size.
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
 *
 * ─── Training-free sparse-FFN cluster record (2026-05-23) ───────────
 *
 * An OPTIONAL per-layer record that lets the runtime cheaply predict
 * which contiguous slices ("clusters") of a layer's FFN intermediate
 * dimension are likely to fire for a given input, so it can skip the
 * rest. It is DATA-FREE: the clustering uses only the gate_proj weight
 * rows (cosine similarity), no calibration set.
 *
 * This record is ORTHOGONAL to MoME (the .expert{e} split above): it is
 * emitted only on the NON-MoME FFN path (single gate_proj/up_proj/
 * down_proj tensors per layer). It is gated behind ffn_clusters > 1 at
 * convert time (env IB_FFN_CLUSTERS, see pqv2_encode.c). When disabled
 * (ffn_clusters <= 1) NO record is emitted and the file is byte-
 * identical to today's output.
 *
 * On-disk presence: a single manifest tensor per layer, named
 *
 *   Lk.mlp.ffn_clusters    — kind == IB_PQV2_KIND_RAW_INT32 (3).
 *
 * The blob is a raw little-endian byte stream with this EXACT layout
 * (field order is load-bearing — the loader mirrors it verbatim). It is
 * SELF-DESCRIBING: a 28-byte fixed header (== sizeof(ib_ffn_cluster_hdr)
 * in src/sparse_gate.h, the authoritative struct) preceded by magic +
 * version, then the variable arrays:
 *
 *   char   magic[8] = "IBFFNCL1" // IB_FFN_CLUSTER_MAGIC
 *   uint32 version  = 1          // IB_FFN_CLUSTER_VERSION
 *   uint32 flags                 // bit0 = inter_perm present (always set
 *                             //    here — the encoder always writes perm).
 *   uint32 ffn_n_clusters     // N. 0 or 1 = disabled (record absent in
 *                             //    practice; never written when N<=1).
 *   uint32 ffn_inter          // = intermediate_size (sanity check).
 *   uint32 ffn_hidden         // = hidden_size       (sanity check).
 *   uint32 cluster_offsets[ffn_n_clusters + 1]
 *                             // start row (in the PERMUTED inter space)
 *                             // of each cluster. Clusters are CONTIGUOUS.
 *                             // cluster_offsets[0] == 0,
 *                             // cluster_offsets[ffn_n_clusters] == ffn_inter.
 *   fp16   centroids[ffn_n_clusters * ffn_hidden]
 *                             // per-cluster centroid in INPUT (hidden)
 *                             // space = the mean of that cluster's
 *                             // gate_proj rows (each gate row is a
 *                             // hidden-dim vector). Stored as IEEE
 *                             // half-precision (uint16 bit pattern),
 *                             // row-major [cluster][hidden].
 *   uint32 inter_perm[ffn_inter]
 *                             // present iff flags bit0. inter_perm[new_row]
 *                             //   = old_row, i.e. the permutation already
 *                             //   baked into the stored gate/up rows and
 *                             //   down columns. NOT required at runtime
 *                             //   (the perm is baked into the weights);
 *                             //   present so the perm can be inspected.
 *
 * The blob's total size is therefore:
 *   28                                  (magic+version+flags+3×uint32 hdr)
 *   + 4 * (ffn_n_clusters + 1)          (cluster_offsets)
 *   + 2 * ffn_n_clusters * ffn_hidden   (centroids, fp16)
 *   + 4 * ffn_inter                     (inter_perm, present when flag set)
 * The manifest entry's blob_size is authoritative; a loader can also
 * recompute it from the header fields to validate.
 *
 * Weight consistency invariant (REQUIRED for exactness): when this
 * record is present, the encoder has applied the SAME inter-permutation
 * to (a) the ROWS of gate_proj, (b) the ROWS of up_proj, and (c) the
 * COLUMNS of down_proj, BEFORE PQ-encoding them. Because the FFN sums
 * down_proj @ (silu(gate) * up) over the inter axis, this permutation
 * cancels out and the FFN output is mathematically unchanged when ALL
 * clusters run. The stored PQ indices are thus already in permuted
 * order; the loader needs no un-permutation. ffn_inter / ffn_hidden are
 * stored so the loader can assert the record matches the layer it loads.
 *
 * Magic/version: the file magic stays "IBFV6PQ2" and version stays 1.
 * This record adds NO new file-level header field and NO new tensor
 * kind — it is a normal RAW_INT32 manifest tensor whose NAME suffix
 * (".mlp.ffn_clusters") and documented blob layout are the contract.
 * Older loaders that do not know the suffix silently ignore the tensor
 * (same forward-compat property as the MoME expert tensors above), so
 * adding the record does not break old readers. No magic/version bump
 * is needed.
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
    /* Stage 5j — when the on-disk blob uses a codebook pool
     * (cb_pool_size > 0), the loader allocates int8/fp16 buffers that the
     * kernel can consume unchanged (expanded from the pool). These
     * pointers track ownership so ib_pqv2_file_free can release them;
     * NULL means the corresponding `pq.cb_scale` / `pq.cb_q` etc. point
     * directly at the mmap'd file (zero-copy). row_scale + cb_scale are
     * always plain fp16 on disk now (the fp8 sp2 variant was retired). */
    void *owned_cb_scale;      /* fp16[rows*K] when pool expanded */
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
