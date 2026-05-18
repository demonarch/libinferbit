/*
 * mome.h — Mixture-of-Mini-Experts runtime helpers (Stage 3a v1).
 *
 * docs/v2/00_CORRECTION.md, section 3a. Vision context in
 * docs/v1/24_MOME_VISION_AND_EXECUTION.md.
 *
 * Scope (v1 / scaffolding):
 *   - All helpers are pure C, CPU-only. The Metal MoME path is a
 *     follow-up; the recorder falls back to CPU at the layer level
 *     whenever `mome_experts > 1` (forward.c::forward_single_ex).
 *   - The matmul kernels themselves are unchanged: each expert sub-
 *     tensor is a normal PQv2 tensor consumed by the existing
 *     pqv2_matvec_* code. mome.c only adds the *routing* layer above.
 *   - "Trivial row-split" — expert e covers the contiguous row range
 *     [e * (M/K), (e+1) * (M/K)) of the original gate/up matrix and
 *     the matching column range of the down_proj. Real co-activation
 *     clustering is a Python calibration follow-up (Stage 3a.1-3a.2).
 *
 * Correctness invariant for v1 zero-init router:
 *   When every router weight is 0, softmax(router_logits) is uniform
 *   1/K for all experts. With the trivial row-split, summing
 *   (1/K) * (per-expert down_proj output) across all K experts equals
 *   running the un-split down_proj on the full M-row gate*up stack
 *   times the constant 1/K — which is then undone at calibration
 *   time. To keep "MoME-on" bit-identical to "MoME-off" pending
 *   calibration, the runtime detects the zero-router case and runs
 *   the FULL FFN (= all experts with weight 1.0, no normalisation) so
 *   the output exactly reproduces the non-MoME forward pass.
 *
 *   Once a real router is fitted, the runtime switches to the
 *   softmax-weighted top-N dispatch and accepts whatever the
 *   calibration teaches.
 */
#ifndef IB_MOME_H
#define IB_MOME_H

#include <stdint.h>
#include "inferbit_internal.h"   /* inferbit_model, ib_layer_meta, ib_tensor_meta */

#ifdef __cplusplus
extern "C" {
#endif

/* Hard cap on K. Anything larger than this would need a heap
 * allocation in the inner-loop scratch; the design target is
 * K ∈ {2, 4, 8, 16}. */
#define IB_MOME_MAX_EXPERTS    32
/* Hard cap on N (top-N selection width). With K=8 the practical
 * working range is N=2 or N=3. */
#define IB_MOME_MAX_TOP_N      8

/* Resolve the runtime top-N selection width.
 *
 *   K       : number of experts on this layer.
 *   returns : top-N value to use, clamped to [1, K] and
 *             [1, IB_MOME_MAX_TOP_N]. The default is min(2, K). The
 *             environment variable IB_MOME_TOP_N overrides; an out-of-
 *             range value is clamped.
 *
 * Read once per layer; tiny so re-reading the env each call is fine,
 * but callers may cache. */
int mome_get_top_n(int K);

/* Detect a zero-init router.
 *
 *   router : either NULL or an ib_tensor_meta whose `bits == 16` and
 *            `offset/size` point at a [hidden, K] raw fp16 buffer.
 *   returns: 1 if the router weight has ANY non-zero element,
 *            0 if it is the zero placeholder (or router == NULL,
 *            or the data pointer is unresolvable).
 *
 * Cheap O(hidden * K) scan; called once per layer per forward pass.
 * A future calibration step will write real weights and this returns
 * 1 from then on. */
int mome_router_is_nonzero(const inferbit_model *m,
                           const ib_tensor_meta *router);

/* Top-N selection over an unsorted logits array.
 *
 *   logits      : [K] fp32 router logits.
 *   K           : number of experts.
 *   top_n       : how many to select (1 ≤ top_n ≤ K).
 *   out_indices : caller-allocated [top_n] int, receives the indices
 *                 of the top-N experts in descending logit order.
 *
 * Uses a tiny in-place partial-selection (insertion sort over a
 * length-top_n buffer) — O(K * top_n), no allocations, fine for
 * K ≤ 32 and top_n ≤ 8. Stable on ties: earlier (lower) index wins. */
void mome_top_n(const float *logits, int K, int top_n, int *out_indices);

/* MoME-aware FFN dispatch.
 *
 * Walks the layer's gate_proj_experts / up_proj_experts /
 * down_proj_experts (which are valid only when
 * `layer->mome_experts > 1`). For each active expert:
 *
 *     gate_e = gate_proj_experts[e] @ x        (sub-rows of M = inter/K)
 *     up_e   = up_proj_experts[e]   @ x
 *     ffn_e  = SiLU(gate_e) * up_e             (length M/K)
 *     out   += w_e * down_proj_experts[e] @ ffn_e
 *
 * where w_e is softmax(router_logits[active]) when the router is
 * non-zero, or 1.0 when the router is zero (v1 correctness path —
 * see header doc).
 *
 *   m              : owning model.
 *   layer          : current layer (must have mome_experts > 1).
 *   x              : [hidden] post-norm input to the FFN.
 *   hb, hb2        : [inter] caller-allocated scratch buffers
 *                    (the regular `buf_mlp` / `buf_mlp2`).
 *   xb_out         : [hidden] caller-allocated FFN output (overwritten).
 *   router_logits  : [layer->mome_experts] from the router matmul, or
 *                    NULL when the router is zero (uniform weights).
 *   active         : [top_n] indices selected from logits (or NULL
 *                    when running ALL experts in the zero-router
 *                    fallback).
 *   n_active       : top_n, or layer->mome_experts in the zero-router
 *                    fallback.
 *   scale_buf      : passed through to the underlying tensor_matmul.
 *
 * The implementation reuses the existing tensor_matmul kernel for
 * each sub-tensor — no new matmul kernels needed in v1. */
void mome_dispatch_ffn(inferbit_model *m,
                       const ib_layer_meta *layer,
                       const float *x,
                       float *hb, float *hb2, float *xb_out,
                       const float *router_logits,
                       const int *active, int n_active,
                       float *scale_buf);

#ifdef __cplusplus
}
#endif
#endif /* IB_MOME_H */
