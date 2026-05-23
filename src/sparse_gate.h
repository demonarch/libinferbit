#ifndef SPARSE_GATE_H
#define SPARSE_GATE_H

#include <stdint.h>
#include <stddef.h>

/* ════════════════════════════════════════════════════════════════════
 * sparse_gate — training-free sparse-FFN cluster gate (loader + runtime)
 * ════════════════════════════════════════════════════════════════════
 *
 * A model may ship, per transformer layer, an optional "FFN cluster"
 * record describing how the FFN intermediate dimension was partitioned
 * into `n_clusters` contiguous row-ranges (in a permuted inter space),
 * together with one cheap centroid signature per cluster in the
 * hidden(input) space. At runtime the gate scores the incoming hidden
 * vector `x` against every centroid and selects the clusters worth
 * computing — letting the FFN path skip the rest.
 *
 * OFF BY DEFAULT: when a layer has no record (n_clusters <= 1) the
 * loader leaves the fields zero/disabled and the FFN path runs every
 * row exactly as before — byte-identical to today.
 *
 * ── ON-DISK RECORD CONTRACT (CANONICAL — both ends mirror this) ──────
 *
 * The encoder (pqv2_encode.c::push_ffn_cluster_record) emits one record
 * PER LAYER as a NAMED RAW_INT32 manifest tensor in the IBF v6 file:
 *
 *       L<li>.mlp.ffn_clusters         (e.g. layer 3 → "L3.mlp.ffn_clusters")
 *
 * The loader (ibf_loader.c::ib_resolve_ffn_clusters) LOCATES the record by
 * that exact tensor name via ib_pqv2_find() on the model's pqv2 file
 * backing, then parses the self-describing blob below. (A legacy v5 JSON
 * path can instead stage an `ffn_cluster` {offset,size} pointer into the
 * weight blob; the blob layout is identical either way.) The byte layout
 * is the shared contract both sides mirror verbatim; the encoder builds it
 * through the ib_ffn_cluster_hdr struct so the field set can't drift.
 * Little-endian throughout.
 *
 *   [0..7]   char     magic = "IBFFNCL1"  (IBF FFN CLuster, v1)
 *   [8..11]  uint32   version            (= IB_FFN_CLUSTER_VERSION)
 *   [12..15] uint32   flags              (bit0 = inter_perm present;
 *                       the encoder ALWAYS sets it — perm is written)
 *   [16..19] uint32   n_clusters         (0 or 1 => disabled)
 *   [20..23] uint32   inter              (FFN intermediate size)
 *   [24..27] uint32   hidden             (FFN input/hidden size)
 *   [28..]   uint32   cluster_offsets[n_clusters + 1]
 *                       contiguous cluster start rows in permuted inter
 *                       space; offsets[0] = 0, offsets[n_clusters] = inter
 *            fp16     centroids[n_clusters * hidden]
 *                       per-cluster centroid in hidden(input) space
 *            uint32   inter_perm[inter]   (ONLY if flags bit0 set;
 *                       read+stored but unused at runtime)
 *
 * The fixed header is exactly 28 bytes == sizeof(ib_ffn_cluster_hdr).
 * All multi-byte fields are tightly packed (no interior padding). The
 * loader references the cluster_offsets / centroids arrays directly in the
 * mmap (zero-copy) — the mapping outlives the model.
 */

#define IB_FFN_CLUSTER_MAGIC      "IBFFNCL1"
#define IB_FFN_CLUSTER_MAGIC_SIZE 8u
#define IB_FFN_CLUSTER_VERSION    1u
#define IB_FFN_CLUSTER_FLAG_PERM  0x1u   /* inter_perm trailing array present */

/* Fixed-size record header that precedes the variable arrays. The
 * cluster_offsets / centroids / inter_perm arrays follow immediately
 * after this struct in the blob. Packed so sizeof() == on-disk bytes. */
#if defined(_MSC_VER)
#pragma pack(push, 1)
typedef struct {
#else
typedef struct __attribute__((packed)) {
#endif
    char     magic[8];     /* "IBFFNCL1" */
    uint32_t version;      /* IB_FFN_CLUSTER_VERSION */
    uint32_t flags;        /* IB_FFN_CLUSTER_FLAG_* */
    uint32_t n_clusters;   /* 0 or 1 => disabled */
    uint32_t inter;        /* FFN intermediate size */
    uint32_t hidden;       /* FFN input/hidden size */
} ib_ffn_cluster_hdr;
#if defined(_MSC_VER)
#pragma pack(pop)
#endif

/* ── Runtime gate ───────────────────────────────────────────────────
 *
 * Pure functions: no model / forward dependencies beyond the args. Both
 * are deterministic and allocation-free (the caller supplies output
 * buffers). They run per FFN per token, doing hidden×n_clusters work.
 */

/* Score every cluster against `x` and select the active ones.
 *
 *   centroids_fp16 : n_clusters * hidden  fp16 bit patterns (row-major,
 *                    cluster c at centroids_fp16 + c*hidden)
 *   n_clusters     : number of clusters (>= 1; <= 1 means caller should
 *                    not gate — this still returns a well-defined result)
 *   hidden         : centroid / x dimensionality
 *   x              : hidden-dim fp32 input vector
 *   thresh         : magnitude threshold on the silu score
 *   top_min        : minimum number of clusters to keep
 *   active_out     : caller-provided int[n_clusters]; filled with the
 *                    selected cluster indices [0..k-1]
 *
 * Selection: keep every cluster c with |score[c]| >= thresh. If fewer
 * than `top_min` qualify, pad the selection with the highest-|score|
 * clusters not already kept until `min(top_min, n_clusters)` are chosen.
 * score[c] = silu(dot(x, centroid_c)), silu(v) = v / (1 + exp(-v)).
 *
 * Tie-breaking is deterministic: the threshold pass keeps clusters in
 * ascending index order; the top-up pass selects strictly by descending
 * |score|, and on equal |score| prefers the lower cluster index. The
 * returned active_out preserves ascending cluster-index order.
 *
 * Returns k, the number of selected clusters (0 only if n_clusters<=0).
 */
int sparse_gate_select(const uint16_t *centroids_fp16, int n_clusters,
                       int hidden, const float *x,
                       float thresh, int top_min, int *active_out);

/* Telemetry/debug: write silu(dot(x, centroid_c)) for every cluster
 * into out_scores[0..n_clusters-1]. Allocation-free; out_scores must be
 * float[n_clusters]. */
void sparse_gate_scores(const uint16_t *centroids_fp16, int n_clusters,
                        int hidden, const float *x, float *out_scores);

#endif /* SPARSE_GATE_H */
