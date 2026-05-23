/*
 * inferbit_internal.h — Internal types and helpers
 *
 * Not part of the public API. Shared across source files within libinferbit.
 */

#ifndef INFERBIT_INTERNAL_H
#define INFERBIT_INTERNAL_H

#include "inferbit.h"
#include <stdbool.h>

/* ── Thread-local error ─────────────────────────────────────── */

void ib_set_error(const char* fmt, ...);

/* ── IBF header (parsed from JSON) ──────────────────────────── */

typedef struct {
    /* Model identity */
    char architecture[64];
    char name[128];

    /* Architecture params */
    int num_layers;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int intermediate_size;
    int vocab_size;
    int max_context_length;

    /* RoPE */
    float rope_theta;

    /* Normalization */
    float norm_epsilon;
    char  norm_type[16];      /* "rmsnorm" or "layernorm" */

    /* Activation */
    char activation[16];      /* "silu", "gelu", "gelu_tanh" */

    /* Flags */
    bool tie_word_embeddings;
    bool attention_bias;
    bool mlp_bias;

    /* Special tokens */
    int bos_token_id;
    int eos_token_id;

    /* Quantization */
    int   default_bits;
    int   sensitive_bits;
    float sparsity;
    int   block_size;

    /* KV cache */
    int kv_bits;
    /* KV-cache storage format (Stage 3b of docs/v2/00_CORRECTION.md).
     * 0 = FP16 (default), 1 = INT8, 2 = PQ8. When non-zero, overrides
     * `kv_bits` for storage-layout decisions. Populated from the
     * inferbit_config at model load (see ibf_loader.c / pqv2_model.c). */
    int kv_format;

    /* Data section */
    size_t weight_data_offset;
    size_t weight_data_size;
    int    alignment;
} ib_ibf_header;

/* ── Per-tensor metadata ────────────────────────────────────── */

#include "pqv2_kernel.h"   /* pqv2_t */

typedef struct {
    size_t offset;        /* Offset within weight data section */
    size_t size;          /* Size in bytes */
    int    shape[4];      /* Dimensions (unused dims = 0) */
    int    ndim;          /* Number of dimensions */
    int    bits;          /* Quantization bits (2, 4, 8, 16) */
    size_t scale_offset;  /* Offset of scale factors (0 = none) */
    size_t scale_size;    /* Size of scale data */
    bool   has_bias;
    /* PQv2 dispatch: when non-NULL, matmul uses pqv2_matvec_*.
     * Pointer is owned by the IBF v6 file backing (mmap or heap).
     * NULL for legacy v5 / INT4 / INT8 / FP16 tensors. */
    const pqv2_t* pq;
    /* Stage 5d (docs/v2/00_CORRECTION.md): hybrid backend hint. Values
     * match the public inferbit_backend enum exactly (0=AUTO, 1=CPU,
     * 2=METAL). Stored as int to keep this header free of the public
     * header inclusion ordering. Set at load time (or lazily at first
     * forward) from IB_HYBRID_* env knobs. The CPU forward checks this
     * before each FFN matmul and, when METAL, dispatches that single
     * matmul through ib_metal_run_single_matmul. */
    int preferred_backend;
    /* Stage 5c (docs/v2/00_CORRECTION.md): residency hint parsed from
     * the IBFv6 per-tensor blob header. Values match the public
     * inferbit_residency enum (0=AUTO, 1=RAM, 2=DRIVE). Stored as int
     * to keep header decoupling clean. Loader reads the on-disk hint
     * (and may override via IB_RESIDENCY_RAM_LAYERS=N) into this field.
     * v1: diagnostic only; the drive-mode mmap/mlock policy hookup is
     * a follow-up. */
    int residency_hint;
    /* Stage 5b (docs/v2/00_CORRECTION.md): on-disk format for this
     * tensor as resolved at encode time. Values match the public
     * inferbit_convert_format enum (0=INT4, 1=PQV2_FLAT, 2=PQV2_PYRAMID).
     * Populated at load time by inspecting pq->l2_kind (0 = flat,
     * 2 = pyramid; INT4 only appears in legacy v5 / non-pqv2 paths).
     * Diagnostic-grade for v1 — the kernel already dispatches on
     * `pq->l2_kind` per tensor, so the runtime needs nothing else. */
    int tensor_format;
    /* Perf: pre-decoded fp32 scale arrays. The on-disk fp16 scale buffer
     * is a STATIC property of the weight tensor, so we decode it ONCE at
     * model load instead of paying fp16→fp32 conversion on every matmul
     * (~154 calls per decode token on a 28-layer Llama). NULL when the
     * tensor has no scales (pq path, fp16 weights without per-row scale).
     *
     *   scales_fp32       — [M] per-row scales (legacy INT4/INT8 path).
     *   blk32_scales_fp32 — [M * (N/32)] per-block scales (INT4 blk32
     *                       path; only set when t->scale_size > M*2).
     *   norm_fp32         — [N] fully-decoded fp32 norm weight, used by
     *                       rmsnorm_fp16. Only set for norm tensors
     *                       (input_norm / post_attn_norm / output_norm).
     *
     * Owned by the model; freed in inferbit_free. */
    float *scales_fp32;
    float *blk32_scales_fp32;
    float *norm_fp32;
    /* ── Goal H4 — hot-cache instrumentation (scaffolding) ──────────
     *
     * Per-tensor access counter, incremented at every tensor_matmul
     * entry. Used by the future adaptive hot-cache to decide which
     * tensors to promote into the hot pool. Currently diagnostic only;
     * enable the per-call increment + end-of-run top-10 summary by
     * setting IB_TENSOR_HOTSET=1 (otherwise the counter stays zero
     * and the matmul path skips the bump entirely).
     *
     * Lives next to the static scale caches because it's logically a
     * cache-management hint, not part of the on-disk layout. */
    uint64_t access_count;
} ib_tensor_meta;

/* ── Per-layer metadata ─────────────────────────────────────── */

typedef struct {
    ib_tensor_meta q_proj;
    ib_tensor_meta k_proj;
    ib_tensor_meta v_proj;
    ib_tensor_meta o_proj;
    ib_tensor_meta gate_proj;
    ib_tensor_meta up_proj;
    ib_tensor_meta down_proj;
    ib_tensor_meta input_norm;
    ib_tensor_meta post_attn_norm;

    /* Sparsity mask */
    size_t sparsity_mask_offset;
    size_t sparsity_mask_size;

    /* Sparse-FFN cluster record — raw blob {offset,size} from the layer
     * JSON (`ffn_cluster`). Resolved into the ffn_* pointers below once
     * model->weight_data is mapped (see ib_resolve_ffn_clusters). Both 0
     * when the layer ships no record — the disabled default. Mirrors the
     * sparsity_mask offset/size staging idiom above. */
    size_t ffn_cluster_blob_offset;
    size_t ffn_cluster_blob_size;

    /* ── Stage 3a — MoME (docs/v2/00_CORRECTION.md) ──────────────────
     *
     * Post-hoc Mix-of-Mini-Experts router metadata. v1 scaffolding:
     * the FFN block is conceptually split into K mini-experts via a
     * trivial row-range split of gate/up/down. Active when the
     * loader saw `Lk.mlp.<proj>.expert{e}` tensors at load time AND
     * the matching `Lk.mlp.router` raw-fp16 weight.
     *
     *   mome_experts        — K. 1 = no MoME (the default, identical
     *                          to pre-MoME behaviour). >1 = MoME
     *                          active; gate/up/down_proj are
     *                          unused and the *_proj_experts arrays
     *                          hold K stacked sub-tensors.
     *   gate_proj_experts   — array of K ib_tensor_meta, NULL when
     *                          mome_experts == 1. Owned by the
     *                          owning model (free()d in model_free).
     *   up_proj_experts     — same, for up_proj.
     *   down_proj_experts   — same, for down_proj.
     *   router              — [K, hidden] raw fp16 router weight
     *                          (rows = experts, cols = hidden — same
     *                          orientation as a regular FFN weight so
     *                          the runtime fp16 matmul can consume it
     *                          directly). The `pq` field is NULL;
     *                          `offset/size/bits=16` point at the raw
     *                          bytes inside the IBF v6 mmap. When
     *                          mome_experts == 1 this is a zeroed
     *                          struct.
     *
     * v1 correctness invariant: when router is the zero-init
     * placeholder (no calibration done yet), the runtime falls back to
     * "run every expert with uniform softmax(0) weight" which, on the
     * trivial row-split layout, is mathematically identical to the
     * un-split FFN matmul. So a MoME-enabled file always produces the
     * same output as a non-MoME file until a real router is fitted. */
    int mome_experts;
    ib_tensor_meta *gate_proj_experts;
    ib_tensor_meta *up_proj_experts;
    ib_tensor_meta *down_proj_experts;
    ib_tensor_meta router;

    /* ── Training-free sparse-FFN cluster gate (loader + runtime) ─────
     *
     * Optional per-layer "FFN cluster" record. The encoder partitions
     * the FFN intermediate dim into `ffn_n_clusters` contiguous
     * row-ranges (in a permuted inter space) and ships one cheap
     * centroid signature per cluster in hidden(input) space. At runtime
     * sparse_gate_select() scores the incoming hidden vector against the
     * centroids to pick which clusters' FFN rows are worth computing.
     *
     * See src/sparse_gate.h for the on-disk record layout. The loader
     * parses the record when a layer's JSON carries an `ffn_cluster`
     * {offset,size} pointer into the weight-data blob; otherwise these
     * stay zero/NULL and the FFN path is unchanged (byte-identical).
     *
     *   ffn_n_clusters      — cluster count. 0 or 1 => disabled (gate is
     *                         a no-op; default for every existing model).
     *   ffn_inter           — FFN intermediate size from the record.
     *   ffn_hidden          — FFN input/hidden size from the record (=
     *                         centroid dimensionality).
     *   ffn_cluster_offsets — points into the mmap: uint32[ffn_n_clusters
     *                         + 1] contiguous cluster start rows in
     *                         permuted inter space; [0]=0, [N]=ffn_inter.
     *                         NULL when disabled. NOT owned (mmap-backed).
     *   ffn_centroids_fp16  — points into the mmap: fp16[ffn_n_clusters *
     *                         ffn_hidden] per-cluster centroid in hidden
     *                         space. NULL when disabled. The gate
     *                         signatures, kept resident via the mapping.
     *                         NOT owned (mmap-backed, zero-copy).
     *   ffn_inter_perm      — optional uint32[ffn_inter] permutation,
     *                         present only if the record's flags set it.
     *                         Stored for completeness; unused at runtime.
     *                         NULL when absent. NOT owned (mmap-backed). */
    uint32_t        ffn_n_clusters;
    uint32_t        ffn_inter;
    uint32_t        ffn_hidden;
    const uint32_t *ffn_cluster_offsets;   /* [ffn_n_clusters + 1], mmap-backed */
    const uint16_t *ffn_centroids_fp16;    /* [ffn_n_clusters * ffn_hidden], mmap-backed */
    const uint32_t *ffn_inter_perm;        /* [ffn_inter] or NULL, mmap-backed */
} ib_layer_meta;

/* ── KV cache ───────────────────────────────────────────────── */

typedef struct {
    void*  key_data;       /* Quantized key cache */
    void*  value_data;     /* Quantized value cache */
    float* key_scales;     /* Scale factors for keys */
    float* value_scales;   /* Scale factors for values */
    int    length;         /* Current number of tokens in cache */
    int    capacity;       /* Max tokens allocated */
    bool   dynamic;        /* Whether cache grows dynamically */
} ib_kv_cache;

/* ── Burst / cool-down duty-cycle controller (M1) ───────────────
 *
 * Runtime state for the per-decode-step compute-profile controller. The
 * config (`cfg`) is copied in by inferbit_burst_attach; `cur` /
 * since_cooldown / the EMAs are advanced by ib_burst_step_decide. With
 * cfg.enabled == 0 (default) the controller always returns
 * IB_PROFILE_EXACT and never leaves the exact path — the behaviour-
 * preserving invariant. Counters/byte-tallies are diagnostics only.
 * Defined + implemented in src/burst_ctrl.c. */
typedef struct {
    ib_burst_config cfg;
    ib_profile_kind cur;
    int   since_cooldown;
    float ema_accept;
    float last_margin;
    float last_norm;
    unsigned long long burst_steps, cooldown_steps;
    unsigned long long burst_bytes, cooldown_bytes;
} ib_burst_ctrl;

/* ── Model struct ───────────────────────────────────────────── */

struct inferbit_model {
    /* IBF metadata */
    ib_ibf_header header;

    /* Layer metadata */
    ib_layer_meta* layers;       /* Array of num_layers */

    /* Embedding and output head metadata */
    ib_tensor_meta token_embedding;
    ib_tensor_meta output_norm;
    ib_tensor_meta output_head;

    /* Weight data (mmap'd or allocated) */
    void*  weight_data;
    size_t weight_data_size;
    bool   weight_data_mmap;     /* True if mmap'd, false if malloc'd */
    int    mmap_fd;              /* File descriptor if mmap'd */

    /* Set after ib_metal_strip_cpu_mmap: a malloc'd buffer holding the
     * embedding tensor's bytes that survive the original mmap being unmapped.
     * weight_data points into this buffer (with the embedding offset baked in
     * so cpu_embed_lookup keeps working). model_free is responsible for
     * free()-ing this buffer when set. */
    void*  embed_strip_buffer;

    /* IBF v6 PQv2 file backing (NULL for v5 / non-PQv2 models).
     * When set, pq tensor metadata in ib_tensor_meta points into this
     * file's mmap region; freed in inferbit_model_free.
     * Stored as opaque void* to avoid pulling pqv2_format.h here. */
    void*  pqv2_file_backing;

    /* Pre-allocated PQv2 threading scratch — sized for n_threads × max_M.
     * Used by pqv2_threaded_matvec_k256 to avoid aligned_alloc/free on
     * every matvec call (~154 calls per token in the hot path). */
    float *pqv2_thread_acc_pool;
    size_t pqv2_thread_acc_pool_floats;

    /* Companion L2 scratch for pyramid (l2_kind==2) tensors. Same size as
     * pqv2_thread_acc_pool; allocated only when at least one PQv2 tensor
     * has l2_kind==2. NULL on flat models — keeps the existing alloc path
     * as a no-op fallback. Avoids ~88 aligned_alloc/free per decode token
     * on pyramid models (44-176 MB churn per token). */
    float *pqv2_thread_acc_l2_pool;
    size_t pqv2_thread_acc_l2_pool_floats;

    /* KV cache (one per layer) */
    ib_kv_cache* kv_caches;

    /* Activation buffers (reused across layers) */
    float* buf_residual;         /* [hidden_size] */
    float* buf_hidden;           /* [hidden_size] */
    float* buf_attn;             /* [hidden_size] */
    float* buf_mlp;              /* [intermediate_size] */
    float* buf_mlp2;             /* [intermediate_size] */
    float* buf_logits;           /* [vocab_size] */
    float* buf_qkv;              /* Scratch for Q, K, V projections */

    /* Batched-forward scratch (forward_batch, ib_forward_positions).
     * Sized for up to IB_BATCH_MAX parallel positions. Preallocated at
     * model load; reused every batched call to avoid malloc/free in the
     * hot spec-verify loop. NULL when IB_BATCH_MAX == 0. */
    float*  bb_x;                /* [B_MAX * hidden] */
    float*  bb_xb;               /* [B_MAX * hidden] */
    float*  bb_xb2;              /* [B_MAX * hidden] */
    float*  bb_q;                /* [B_MAX * hidden] */
    float*  bb_k;                /* [B_MAX * kv_dim] */
    float*  bb_v;                /* [B_MAX * kv_dim] */
    float*  bb_hb;               /* [B_MAX * intermediate] */
    float*  bb_hb2;              /* [B_MAX * intermediate] */
    float*  bb_scale;            /* [max(hidden,intermediate,vocab)] */
    float*  bb_att;              /* [num_heads * max_ctx] (per-position scratch, reused) */
    int8_t* bb_qscratch;         /* [B_MAX * max(hidden,intermediate)] */
    float*  bb_sa;               /* [B_MAX * groups_max] */
    int*    bb_positions;        /* [B_MAX] */

    /* Precomputed RoPE cos/sin tables (perf: avoids per-token sinf/cosf).
     * Sized [rope_table_ctx * (head_dim/2)] each. Populated by
     * ib_alloc_buffers at load using header.rope_theta + head_dim. Optional:
     * when NULL the kernels fall back to live sinf/cosf, so older models or
     * fast bringup paths still work. Freed in inferbit_model_free. */
    float* rope_cos;
    float* rope_sin;
    int    rope_table_ctx;       /* capacity (positions) of the tables; 0 = none */

    /* Speculative decoding */
    inferbit_model* draft_model;
    int             draft_tokens;

    /* Prompt-lookup speculation: draft k tokens by n-gram match over the
     * running history. Enabled when lookup_ngram > 0. Mutually exclusive with
     * external-draft spec decoding. */
    int             lookup_ngram;      /* size of suffix to match (e.g. 3) */
    int             lookup_k;          /* draft length (e.g. 6) */

    /* Threading */
    int num_threads;
    struct ib_thread_pool* thread_pool;

    /* Residency mode (Path D, doc 23/24, executed via doc 32). Controls
     * whether PQv2 indices pages stay RAM-resident or are streamed from
     * disk per matmul.
     *   0 = RAM (default): full model mmap'd; OS holds it hot.
     *   1 = DRIVE: file fd is F_NOCACHE on Darwin / POSIX_FADV_RANDOM on
     *       Linux. Before each PQv2 matmul we pread() the indices into
     *       a single page-aligned scratch buffer; the kernel reads from
     *       there. Peak indices residency = max-matmul-indices,
     *       independent of model size.
     *
     * Off by default. Enable via IB_RESIDENCY_MODE=drive at model load. */
    int    residency_mode;

    /* Rotating KV-cache window (doc 36 phase 2.2), copied from config at
     * load. 0 = full causal cache. >0 = each layer's KV cache is a ring
     * of `kv_window` physical slots; logical position p lives at slot
     * p % kv_window. Attention only reads the most recent kv_window
     * positions. Peak KV RAM becomes O(kv_window) instead of O(seq_len). */
    int    kv_window;

    /* Lazily-created Metal context + uploaded GPU buffers (doc 36 phase
     * 4.1). Allocated on the first inferbit_forward_with_hiddens call and
     * cached for reuse; freed in inferbit_free. Opaque void* so this
     * header stays free of metal_runtime.h. NULL until first use. */
    void  *metal_ctx;
    void  *metal_bufs;
    int    metal_route_failed;  /* 1 = ib_metal_upload_model failed or backend forced CPU; never retry */
    int    drive_fd;                  /* fd of the IBF, F_NOCACHE set on Darwin */
    void  *drive_indices_scratch;     /* page-aligned shared buffer (slot 0) */
    void  *drive_indices_scratch2;    /* page-aligned shared buffer (slot 1) — prefetch ring */
    size_t drive_indices_scratch_size;
    /* Goal C3 — pyramid drive-mode L2 indices redirect. Parallel ring
     * of two scratch buffers sized for the largest L2 indices region
     * across all PQv2 pyramid tensors. The prefetch worker fills the L2
     * slot alongside the L1 slot (same slot index), so each active-slot
     * scratch tuple is (L1, L2) for one tensor. Allocated only when at
     * least one drive-mode tensor carries l2_kind == 2; left NULL when
     * the model has no L2 pyramid tensors (skips overhead). */
    void  *drive_l2_indices_scratch;
    void  *drive_l2_indices_scratch2;
    size_t drive_l2_indices_scratch_size;
    /* ── Peak-RAM page cap (drive mode) ──────────────────────────────
     *
     * Upper bound on the in-focus L1 index bytes per scratch slot. Set
     * at drive-mode init from env IB_DRIVE_PAGE_MB (MB → bytes; default
     * IB_DRIVE_PAGE_MB_DEFAULT when unset). The L1 (and matching L2)
     * scratch slots are sized to min(largest-tensor-bytes, cap), so a
     * tensor whose total index bytes exceed the cap is streamed in
     * contiguous (chunk,subchunk) lane-groups — each group ≤ cap — with
     * the kernel accumulating partial dot-products into a small acc[M]
     * across groups (see forward.c paged matmul path). Worst-case
     * in-focus index RAM = 2 × cap (the 2-slot prefetch ring). 0 means
     * "no cap" (legacy whole-tensor scratch).
     *
     * The paged matmul (forward.c::drive_paged_matvec) accumulates each
     * tensor's lane-group partials into the existing model-scope
     * pqv2_thread_acc_pool / pqv2_thread_acc_l2_pool (sized n_threads×
     * max_M ≥ max_M, freed in model.c). Those pools are idle between
     * matmuls and the paged path is single-threaded, so no extra
     * teardown-managed buffer is needed here. */
    size_t drive_page_bytes;
    /* 2-slot prefetch ring (perf fix). The scratch buffers above act as a
     * double-buffer: while the kernel reads from one slot, a worker thread
     * preads the NEXT tensor's indices into the other slot. drive_pf_state
     * is an opaque pointer to the prefetcher's runtime state (lazy init).
     * drive_pq_order[] is the static decode-order list (Q,K,V,O,gate,up,
     * down per layer; last entry = output_head) used to predict the next
     * pread target. Built at drive-mode init. */
    void  *drive_pf_state;
    const ib_tensor_meta **drive_pq_order;
    int     drive_pq_order_len;
    /* Pre-transposed sidecar (doc 35 feature 3). Built once at drive-
     * mode init from the original [c][s][m] indices, stores them in
     * kernel-native [m][total] layout for GPU drive-mode preads —
     * eliminates the per-matmul transpose that was the drive-mode
     * CPU-bottleneck floor. CPU drive path continues to use drive_fd
     * (chunk-major). -1 = no sidecar (CPU-only drive or build failed). */
    int    drive_fd_pretransposed;

    /* ── Stage 5d — hybrid CPU/GPU forward (docs/v2/00_CORRECTION.md) ──
     *
     * Lazily-allocated Metal-shared staging buffers for the per-matmul
     * hybrid hook (forward.c::tensor_matmul_hybrid). When a CPU-routed
     * forward hits an FFN matmul whose `preferred_backend == METAL`,
     * the hook copies the fp32 input into hybrid_x_buf, runs one Metal
     * matmul that writes into hybrid_y_buf, and copies the result back
     * to the caller's CPU buffer. Buffers are sized to hold the largest
     * FFN input (= hidden_size) and largest FFN output (= intermediate_
     * size), so they can serve every gate/up/down dispatch.
     *
     * Allocated on first hybrid dispatch via ib_metal_alloc; both pointers
     * are host-visible AND Metal-buffer-backed (unified memory). NULL
     * when no hybrid call has happened yet; freed in inferbit_free
     * alongside the metal_ctx. NOT created when metal_route_failed=1. */
    void  *hybrid_x_buf;            /* Metal-shared fp32 input scratch */
    void  *hybrid_y_buf;            /* Metal-shared fp32 output scratch */
    size_t hybrid_x_buf_floats;     /* element capacity (NOT bytes) */
    size_t hybrid_y_buf_floats;
    int    hybrid_tags_applied;     /* 0 until preferred_backend has been seeded once */

    /* ── Phase 4 — DFlash hybrid orchestrator (doc 36) ──────────────
     *
     * Attached via inferbit_dflash_attach. When dflash_cfg is non-NULL,
     * ib_forward() routes single-token CPU decode through the orchestrator
     * (see dflash_orchestrator.c). During every full forward, the
     * post-residual hidden state at layer dflash_cfg->early_exit_layer is
     * written into dflash_capture_buf (via the 3-line hook inside
     * forward_single_ex), and its L2-norm into dflash_last_norm. The
     * orchestrator uses last_norm + warmup to gate the next step.
     *
     * dflash_full_count / dflash_early_count are reset per generate()
     * (by inferbit_dflash_attach) and incremented by the orchestrator. */
    inferbit_dflash_config* dflash_cfg;
    float* dflash_capture_buf;          /* [hidden_size] — last captured early-layer hidden */
    float  dflash_last_norm;            /* L2-norm of dflash_capture_buf, or 0 before first capture */
    int    dflash_decode_step;          /* number of decode steps observed since attach */
    int    dflash_full_count;
    int    dflash_early_count;

    /* ── Goal H4 — adaptive hot-cache (scaffolding only) ─────────────
     *
     * Small RAM-resident pool intended to hold frequently-touched
     * tensors/codebooks during a generation. Today this is JUST the
     * framework: the allocation, the lookup/promote API entry points,
     * and the per-tensor access counters. The adaptive promotion
     * policy that decides which tensors to copy in (and when to
     * evict) is a follow-up patch — see ib_hot_lookup / ib_hot_promote
     * below for the stub semantics.
     *
     *   hot_pool         : malloc'd region; default 32 MB, sized via
     *                       IB_HOT_POOL_MB (set to 0 to disable). NULL
     *                       when disabled or alloc failed (non-fatal).
     *   hot_pool_bytes   : actual byte capacity of hot_pool.
     *   hot_pool_entries : number of tensors currently held (always 0
     *                       in the scaffolding — ib_hot_promote is a
     *                       no-op until the adaptive logic lands). */
    void  *hot_pool;
    size_t hot_pool_bytes;
    int    hot_pool_entries;

    /* ── Burst / cool-down duty cycle (M1) ───────────────────────────
     *
     * `burst` is the controller state (config + counters). `active_profile`
     * points at the dials the CURRENT step runs with — it is
     * &burst.cfg.burst / .cooldown when the duty cycle picks a profile, and
     * &g_profile_exact (a file-static all-zero EXACT profile in
     * burst_ctrl.c) whenever the feature is disabled. `active_skip_thresh_
     * ratio` is the cached activation-skip ratio for the active profile,
     * read by the threaded matmul instead of getenv() per call. At attach
     * time it is seeded from IB_PQV2_SKIP (if set) so the default,
     * burst-disabled run reproduces today's env-driven behaviour exactly. */
    ib_burst_ctrl              burst;
    const ib_compute_profile  *active_profile;
    float                      active_skip_thresh_ratio;
};

/* ── Config struct ──────────────────────────────────────────── */

struct inferbit_config {
    int  threads;
    int  context_length;
    bool kv_dynamic;
    bool native_parse;
    int  native_bits;
    /* Rotating KV-cache window (doc 36 phase 2.2). 0 = full causal cache
     * (default). >0 = ring buffer of `kv_window` token slots; logical
     * position p maps to physical slot p % kv_window. Bounds KV RAM at
     * long context for the (acceptable) cost of a sliding-window
     * attention horizon. */
    int  kv_window;
    /* KV-cache storage format (Stage 3b of docs/v2/00_CORRECTION.md).
     * Public-API value is inferbit_kv_format; stored as int to keep the
     * struct layout decoupled from the public-enum width.
     * 0 = FP16 (default), 1 = INT8, 2 = PQ8 (v1 falls back to INT8). */
    int  kv_format;
};

/* ── SIMD dispatch ──────────────────────────────────────────── */

typedef enum {
    IB_SIMD_NONE   = 0,
    IB_SIMD_AVX2   = 1,
    IB_SIMD_AVX512 = 2,
    IB_SIMD_NEON   = 3,
} ib_simd_level;

ib_simd_level ib_detect_simd(void);

/* ── Kernel function pointers (set at init based on SIMD) ──── */

typedef struct {
    /* INT4 matmul: out[M] = weights[M, N] @ input[N] */
    void (*matmul_int4)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* INT8 matmul */
    void (*matmul_int8)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* INT2 ternary matmul: weights are {-1, 0, +1}, 4 per byte */
    void (*matmul_int2)(
        float* out, const void* weights, const float* scales,
        const float* input, int M, int N
    );

    /* RMSNorm: out[N] = rmsnorm(input[N], weight[N], eps) */
    void (*rmsnorm)(
        float* out, const float* input, const float* weight,
        float eps, int N
    );

    /* RoPE: apply rotary position encoding in-place.
     * If cos_tab/sin_tab are non-NULL, they are looked up as
     *   cos_tab[pos * (head_dim/2) + i/2], sin_tab[pos * (head_dim/2) + i/2]
     * to avoid the per-call sinf/cosf cost (precomputed at model load).
     * Passing NULL for both falls back to live sinf/cosf computation. */
    void (*rope)(
        float* q, float* k, int head_dim, int pos, float theta,
        const float* cos_tab, const float* sin_tab
    );

    /* Softmax: in-place softmax over N elements */
    void (*softmax)(float* data, int N);

    /* Element-wise: out = a * b (SiLU gate) */
    void (*silu_mul)(float* out, const float* gate, const float* up, int N);

    /* W4A8 matmul: INT4 weights × INT8 activation, grouped activation scale.
     *
     * Activation is quantized in groups of IB_W4A8_GROUP=128 elements, each
     * with its own FP32 scale. Output[i] = sum over groups g of
     *   (weights[i,g] · input[g]) * scales_a[g] * scales_w[i].
     *
     * N must be a multiple of IB_W4A8_GROUP (128). Uses ARM sdot / x86 VNNI
     * when available. */
    void (*matmul_w4a8)(
        float* out, const void* weights, const float* scales_w,
        const int8_t* input, const float* scales_a, int M, int N
    );

    /* W4A8 with per-32-element block scales on the WEIGHT side.
     *
     * scales_w has length M*(N/32) (one fp32 per 32 weight elements per row),
     * vs the per-row scales_w[M] used by matmul_w4a8 above. Activation
     * grouping is unchanged (per-IB_W4A8_GROUP=128). N must be a multiple
     * of 32 (and IB_W4A8_GROUP must be a multiple of 32 — currently 128/32=4).
     *
     * Used to close the per-row outlier-clipping quality gap on Llama-3-class
     * models where late-layer outliers spoil per-row scaling. May be NULL
     * on backends that don't yet support it; callers should fall back. */
    void (*matmul_w4a8_blk32)(
        float* out, const void* weights, const float* scales_w_per_block,
        const int8_t* input, const float* scales_a, int M, int N
    );

    /* W4A8 batched matmul: same as above, but amortizes weight loads across
     * B independent activation vectors.
     *
     *   out         [B * M_stride]  row-major, out[b*M_stride + i]
     *   weights     [M * N/2] INT4 packed (shared across batch). M rows
     *               processed; rows physically span the first M*(N/2) bytes.
     *   scales_w    [M]                     (shared across batch)
     *   input       [B * N]  INT8 row-major, input[b*N + j]
     *   scales_a    [B * (N/IB_W4A8_GROUP)] FP32 (one scale per batch×group)
     *   M_stride    Output column stride. Pass M_stride==M for a contiguous
     *               B-by-M output. The thread-parallel wrapper passes the
     *               global M (so per-thread row slices write into their
     *               correct positions in the shared output), with weights
     *               + scales_w pre-offset to the slice's first row.
     *
     * Caller guarantees N % IB_W4A8_GROUP == 0. Typical B is 2-8 (spec
     * decoding verify width). */
    void (*matmul_w4a8_batch)(
        float* out, const void* weights, const float* scales_w,
        const int8_t* input, const float* scales_a,
        int M, int N, int B, int M_stride
    );

    /* INT8 weight × FP32 activation batched matmul. Same layout contract as
     * matmul_w4a8_batch (output strided by M_stride). */
    void (*matmul_int8_batch)(
        float* out, const void* weights, const float* scales_w,
        const float* input, int M, int N, int B, int M_stride
    );
} ib_kernels;

/* Activation quantization group size for W4A8. Chosen to fit all transformer
 * hidden dims used in practice (multiples of 128). */
#define IB_W4A8_GROUP 128

/* Max parallel positions handled by forward_batch without falling back to
 * malloc. Matches the candidates[32] cap in spec verify. Lowering this
 * shrinks per-model preallocated scratch; raising it is only useful for
 * wider speculation schemes. */
#define IB_BATCH_MAX 32

/* Global kernel dispatch table */
extern ib_kernels ib_kern;

void ib_init_kernels(ib_simd_level level);

/* ── Prompt-lookup speculation helper ───────────────────────── */

/* Search `history[0..hist_len-1]` for the earliest occurrence of its own
 * final `ngram` tokens (the "suffix"). On hit, copies up to `k` tokens that
 * follow the match into `out_candidates` and returns the count. On miss,
 * returns 0. A match whose position `i` leaves fewer than one follower token
 * before overlapping the suffix is skipped. */
int ib_prompt_lookup_search(const int32_t* history, int hist_len,
                            int ngram, int k, int32_t* out_candidates);

/* Multi-candidate (tree) prompt-lookup drafter — see speculative.c. Gathers up
 * to max_branches distinct n-gram continuations into one flat out_tokens buffer
 * (total positions capped at total_cap so the whole tree fits one batched
 * verify), writing per-branch offsets/lengths. Returns the branch count. */
int ib_prompt_lookup_search_tree(const int32_t* history, int hist_len,
                                 int ngram, int per_branch_k, int max_branches,
                                 int total_cap,
                                 int32_t* out_tokens,
                                 int* branch_off, int* branch_len);

/* Spec-tuning env helpers (speculative.c). All default to OFF / fallback so the
 * default decode path is byte-identical when the env vars are unset. */
int ib_spec_k_override(int fallback);   /* IB_SPEC_K, clamped to IB_BATCH_MAX */
int ib_spec_tree_enabled(void);         /* IB_SPEC_TREE=1 */
int ib_spec_log_enabled(void);          /* IB_SPEC_LOG=1 */

/* ── Safetensors parser ─────────────────────────────────────── */

typedef struct ib_safetensors ib_safetensors;

ib_safetensors* ib_st_open(const char* path);
void            ib_st_close(ib_safetensors* sf);
const void*     ib_st_tensor_data(const ib_safetensors* sf, int index);
size_t          ib_st_tensor_size(const ib_safetensors* sf, int index);
int             ib_st_find(const ib_safetensors* sf, const char* name);
int             ib_st_find_suffix(const ib_safetensors* sf, const char* suffix);
int             ib_st_num_tensors(const ib_safetensors* sf);
const char*     ib_st_tensor_name_at(const ib_safetensors* sf, int index);
const char*     ib_st_tensor_dtype_at(const ib_safetensors* sf, int index);
int             ib_st_tensor_ndim_at(const ib_safetensors* sf, int index);
int             ib_st_tensor_shape_at(const ib_safetensors* sf, int index, int dim);

/* ── Multi-shard safetensors ─────────────────────────────────── */

typedef struct ib_safetensors_multi ib_safetensors_multi;

ib_safetensors_multi* ib_st_multi_open(const char* dir_path);
void                  ib_st_multi_close(ib_safetensors_multi* multi);
int  ib_st_multi_find(const ib_safetensors_multi* multi, const char* name,
                      int* out_shard, int* out_tensor);
int  ib_st_multi_find_suffix(const ib_safetensors_multi* multi, const char* suffix,
                             int* out_shard, int* out_tensor);
const void* ib_st_multi_tensor_data(const ib_safetensors_multi* multi, int shard, int tensor);
const char* ib_st_multi_tensor_dtype(const ib_safetensors_multi* multi, int shard, int tensor);
int         ib_st_multi_tensor_shape(const ib_safetensors_multi* multi, int shard, int tensor, int dim);
int         ib_st_multi_num_shards(const ib_safetensors_multi* multi);

/* ── GGUF parser ────────────────────────────────────────────── */

typedef struct ib_gguf ib_gguf;

ib_gguf*    ib_gguf_open(const char* path);
void        ib_gguf_close(ib_gguf* gg);
int         ib_gguf_num_tensors(const ib_gguf* gg);
int         ib_gguf_find(const ib_gguf* gg, const char* name);
int         ib_gguf_find_suffix(const ib_gguf* gg, const char* suffix);
const void* ib_gguf_tensor_data(const ib_gguf* gg, int index);
size_t      ib_gguf_tensor_size(const ib_gguf* gg, int index);
int         ib_gguf_tensor_type(const ib_gguf* gg, int index);
int         ib_gguf_tensor_shape(const ib_gguf* gg, int index, int dim);
int         ib_gguf_tensor_ndim(const ib_gguf* gg, int index);
const char* ib_gguf_tensor_name(const ib_gguf* gg, int index);
int         ib_gguf_meta_int(const ib_gguf* gg, const char* key, int def);
float       ib_gguf_meta_float(const ib_gguf* gg, const char* key, float def);
const char* ib_gguf_meta_string(const ib_gguf* gg, const char* key);
/* ib_gguf_get_config declared after ib_model_config below */

/* ── Tensor source (unified single/multi-shard access) ──────── */

typedef struct ib_tensor_source ib_tensor_source;

ib_tensor_source* ib_ts_open(const char* path);  /* file or directory */
void              ib_ts_close(ib_tensor_source* ts);
int  ib_ts_find(const ib_tensor_source* ts, const char* name,
                int* out_shard, int* out_tensor);
int  ib_ts_find_suffix(const ib_tensor_source* ts, const char* suffix,
                       int* out_shard, int* out_tensor);
const void* ib_ts_tensor_data(const ib_tensor_source* ts, int shard, int tensor);
const char* ib_ts_tensor_dtype(const ib_tensor_source* ts, int shard, int tensor);
int         ib_ts_tensor_shape(const ib_tensor_source* ts, int shard, int tensor, int dim);

/* ── Config JSON parser ─────────────────────────────────────── */

typedef struct {
    char  arch[64];
    int   num_layers;
    int   hidden_size;
    int   num_heads;
    int   num_kv_heads;
    int   head_dim;
    int   intermediate_size;
    int   vocab_size;
    int   max_context_length;
    float rope_theta;
    float norm_epsilon;
    char  norm_type[16];
    char  activation[16];
    int   tie_word_embeddings;
    int   bos_token_id;
    int   eos_token_id;
} ib_model_config;

int ib_parse_config_json(const char* path, ib_model_config* cfg);
int ib_gguf_get_config(const ib_gguf* gg, ib_model_config* cfg);

/* ── GGUF converter ─────────────────────────────────────────── */

int ib_convert_gguf(const char* input_path, const char* output_path,
                    const inferbit_convert_config* cfg);

/* ── Quantization ───────────────────────────────────────────── */

void ib_quantize_int8(int8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
void ib_quantize_int4(uint8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
/* INT4 with per-32-element block scales. cols must be a multiple of 32.
 * Output scale array is [rows * (cols/32)] fp16. */
void ib_quantize_int4_blk32(uint8_t* out, uint16_t* scales, const void* src,
                             const char* dtype, int rows, int cols);
void ib_quantize_int2(uint8_t* out, uint16_t* scales, const void* src,
                      const char* dtype, int rows, int cols);
void ib_copy_norm_fp16(uint16_t* out, const void* src, const char* dtype, int size);

/* ── Forward pass ───────────────────────────────────────────── */

int ib_forward(inferbit_model* model, const int32_t* tokens, int num_tokens, float* out_logits);

/* Apply the final RMSNorm (in-place over `hidden`) followed by the LM-head
 * matmul to produce logits. Factored out of forward_single_ex so the DFlash
 * orchestrator (dflash_orchestrator.c) can reuse the same kernels on the
 * early-exit projection path without duplicating tensor_data lookups and
 * scale-buffer plumbing.
 *
 *   model    : the model owning output_norm + output_head.
 *   hidden_io: [hidden_size] — RMSNorm is applied IN PLACE, so this buffer
 *              is clobbered. Caller pre-fills with the source hidden state
 *              (either the final post-stack residual on the full path, or
 *              the captured early-layer residual on the DFlash early-exit
 *              path).
 *   logits_out: [vocab_size] output.
 *   scale_buf : [max(hidden, intermediate, vocab)] scratch for matmul.
 */
void ib_apply_lm_head_finalize(const inferbit_model* model,
                               float* hidden_io,
                               float* logits_out,
                               float* scale_buf);

/* Single-token decode entrypoint. Non-static wrapper around forward_single_ex
 * so the DFlash orchestrator can dispatch a full-forward decode step without
 * going back through ib_forward()'s routing (which is what called the
 * orchestrator in the first place — recursion would loop). */
int ib_forward_single(inferbit_model* model, int token_id, int pos,
                      float* out_logits);

/* DFlash orchestrator entrypoint. Called from the top of ib_forward() when
 * a DFlash config is attached. Sets *handled = 1 if the orchestrator
 * produced logits (caller should return rc immediately); *handled = 0 if
 * the request is not DFlash-applicable (multi-token prefill, Metal-routed,
 * etc.) and the caller should fall through to the existing routing.
 *
 * Implemented in dflash_orchestrator.c. */
int ib_dflash_try_route(inferbit_model* model,
                        const int32_t* tokens,
                        int num_tokens,
                        float* out_logits,
                        int* handled);

/* CPU single-token matmul wrapper exposed for the MoME dispatcher.
 *
 * Behaviourally identical to the static `tensor_matmul` inside
 * forward.c (PQv2 / W4A8 / INT8 / FP16 dispatch). Lives outside that
 * static so src/mome.c can invoke per-expert matmuls without making
 * tensor_matmul globally visible.
 *
 *   out       : [M] caller-allocated output buffer.
 *   input     : [N] caller-allocated input vector.
 *   M, N      : tensor's output rows and input cols.
 *   scale_buf : [≥ max(M, N)] scratch used by INT4/INT8 paths.
 *
 * No threading guarantees beyond what tensor_matmul already does
 * (PQv2 K=256 spawns the existing per-chunk pool internally). */
void ib_tensor_matmul_cpu(const inferbit_model *m, const ib_tensor_meta *t,
                          float *out, const float *input, int M, int N,
                          float *scale_buf);

/* Multi-position forward. Processes num_tokens tokens advancing the KV cache,
 * and writes per-position logits into out_logits[num_tokens * vocab_size].
 *
 * Unlike ib_forward (which only returns the last position's logits as a
 * prefill optimization), this variant costs one LM head matmul per position.
 * Needed for speculative-decoding verify passes, where we need logits at
 * every candidate position to check whether the main model agrees with the
 * draft. Do not use this for long prompt prefill — it's per-token more
 * expensive than ib_forward by one LM head per token. */
int ib_forward_positions(inferbit_model* model, const int32_t* tokens,
                         int num_tokens, float* out_logits);

/* ── Burst / cool-down duty-cycle controller (M1) ───────────────
 *
 * Implemented in src/burst_ctrl.c. These are the controller's own helpers
 * (the kernels that consume a profile live in their owning files). When the
 * feature is disabled they all behave as the EXACT path. */

/* Pick the profile for this decode step, advancing the controller's
 * counters/EMAs and switching m->active_profile via
 * inferbit_set_compute_profile. Returns IB_PROFILE_EXACT (and forces the
 * exact profile) whenever the duty cycle is disabled. */
ib_profile_kind ib_burst_step_decide(inferbit_model* m);

/* Feed one speculative-verify result into the accept-rate EMA. `accepted`
 * tokens out of `drafted`; drafted==0 leaves the EMA unchanged. */
void ib_burst_feed_accept(inferbit_model* m, int accepted, int drafted);

/* The dials the current step runs with (m->active_profile, or the exact
 * profile when NULL / disabled). */
const ib_compute_profile* ib_active_profile(inferbit_model* m);

/* 1 unless the active profile is L1-only coarse (precision_tier==1). */
int  ib_active_use_l2(inferbit_model* m);

/* -1 (full depth) unless the active profile requests early exit. */
int  ib_active_max_layer(inferbit_model* m);

/* Print a one-line burst/cool-down summary (step counts + accept-rate EMA +
 * last margin/norm) to stderr when IB_BURST_LOG=1; no-op otherwise. Called at
 * generate end from generate.c. */
void ib_burst_log_summary(inferbit_model* m);

/* ── Goal H4 — hot-cache framework (scaffolding only) ───────────
 *
 * The runtime layer that will (eventually) hold a small RAM pool of
 * the hottest tensors during generation. v1 ships ONLY:
 *   • a per-tensor access counter (ib_tensor_meta::access_count),
 *     bumped at every tensor_matmul entry when IB_TENSOR_HOTSET=1.
 *   • a model-owned hot_pool buffer (sized by IB_HOT_POOL_MB, default
 *     32 MB; 0 disables).
 *   • the two API stubs below, intentionally no-ops so callers can
 *     wire them in now without changing observable behaviour.
 *
 * Env knobs:
 *   IB_TENSOR_HOTSET=1  enable per-tensor access-count tracking +
 *                       top-10 summary on inferbit_free.
 *   IB_HOT_POOL_MB=N    hot-pool size in MB (default 32; 0 disables).
 */

/* Returns non-zero iff IB_TENSOR_HOTSET=1 was set when the process
 * started. Cached on first call so the matmul hot path stays branch-
 * predictor-friendly. */
int ib_hotset_enabled(void);

/* Try to find a hot copy of this tensor's bytes. Returns NULL if not
 * in the hot pool. Callers fall through to the regular mmap/drive path. */
const void *ib_hot_lookup(const inferbit_model *m, const ib_tensor_meta *t);

/* Promote a tensor's bytes to the hot pool (memcpy from source).
 * Returns 0 on success, non-zero if the pool is full or the tensor
 * is too large. Caller is responsible for deciding when to promote. */
int ib_hot_promote(inferbit_model *m, const ib_tensor_meta *t);

/* Print a top-N most-accessed-tensors summary to stderr. No-op when
 * IB_TENSOR_HOTSET is unset. Invoked by inferbit_free. */
void ib_hotset_report(const inferbit_model *m);

/* ── Threading ──────────────────────────────────────────────── */

typedef struct ib_thread_pool ib_thread_pool;

ib_thread_pool* ib_pool_create(int n_threads);
void            ib_pool_destroy(ib_thread_pool* tp);
void            ib_pool_run(ib_thread_pool* tp,
                            void (*fn)(void* arg, int thread_id, int start, int end),
                            void* arg, int total, int chunk_size);
void            ib_parallel_matmul(ib_thread_pool* tp, float* out, const void* weights,
                                   const float* scales, const float* input,
                                   int M, int N, int bits);
void            ib_parallel_matmul_w4a8(ib_thread_pool* tp, float* out,
                                        const void* weights, const float* scales_w,
                                        const int8_t* input, const float* scales_a,
                                        int M, int N);
void            ib_parallel_matmul_w4a8_batch(ib_thread_pool* tp, float* out,
                                               const void* weights,
                                               const float* scales_w,
                                               const int8_t* input,
                                               const float* scales_a,
                                               int M, int N, int B);
void            ib_parallel_matmul_int8_batch(ib_thread_pool* tp, float* out,
                                               const void* weights,
                                               const float* scales_w,
                                               const float* input,
                                               int M, int N, int B);
/* Per-group symmetric INT8 quantization. Writes N INT8 values and
 * ceil(N/IB_W4A8_GROUP) FP32 scales. Returns the number of groups written. */
int             ib_quantize_input_int8_g128(const float* input, int8_t* out_q,
                                            float* out_scales, int N);

#endif /* INFERBIT_INTERNAL_H */
