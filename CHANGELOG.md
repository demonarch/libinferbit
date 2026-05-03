# Changelog

## v0.3.0 — INT4 + GPTQ + SDOT pipeline

### Headline

Complete INT4 weight backend stacks AWQ + 0.2% fp16 outlier sidecar +
GPTQ-style error correction, lands at **+0.12% rel PPL** vs HF fp16
on TinyLlama 1.1B at **31 t/s** decode (CPU NEON, T=8, M4). Quality
competitive with or better than llama.cpp Q4_K_M (~+0.5% rel) at the
same bpw target.

### Major additions

- **INT4 weight backend** — packed nibble matmul (2 nibbles/byte +
  fp16 group scales). Detected at `ib_pq_session_open` via raw-tensor
  pair `<base>__i4_q` (uint8 [M, N/2]) + `<base>__i4_s` (fp16 [M, N/G]).
  Optional `<base>__act_scale` (fp32 [N]) wires AWQ pre-scaling at
  runtime. Optional `<base>__i4_outl_idx`/`<base>__i4_outl_val` adds
  fp16 outlier sidecar.
- **Phase 2.w static-act-int8 SDOT kernel** — per-group int8
  quantization of x; ARM dotprod (`vdotq_s32`) for int4×int8 → int32.
  ~3× kernel throughput vs fp32. Default ON; opt out with
  `IB_STATIC_ACT_INT8=0`.
- **Phase F1.c batched-within-matmul** — `int4_matmul_batched_sdot`
  loads W rows once across B input slots; new public API
  `ib_pq_session_matmul_batched`. Default ON in `forward_step_batch`;
  opt out with `IB_F1C_BATCH=0`. Measured 1.27–1.41× prefill speedup.
- **NEON int4 kernel** — 16 cols/iter via vmovl_u8 + vsubq_s16 +
  vcvtq_f32_s32 + vfmaq_f32. Threaded via spin pool; 38% kernel
  speedup, scales to 3.7× at T=8.
- **Speculative decoding primitive** — `ib_pq_speculative_step` with
  caller-supplied drafts (any source: prompt-lookup, n-gram, draft
  model, Medusa heads). Greedy verify, refresh forward on rejection,
  per-slot logits via batched lm_head. Bit-exact output. Peak 1.56×
  on repetitive content at K=8.
- **Phase 7 v2 KV pyramid** — per-layer K/V codebooks (K=256, G=4)
  trained from calibration. 8× compression vs fp32 KV (+0.346 abs /
  +3.6% rel PPL cost). Activated with `IB_PQ_KV_PYRAMID=1`.
- **fp16 raw matmul backend** — third generic backend alongside PQ
  pyramid and INT4. Detected via `<base>__fp16w` raw tensor.
- **Embeddings PQ** — `ib_pq_session_reconstruct_row` lets `embed_lookup`
  decode one row per token from a PQ-encoded embedding table.
- **uint16 raw dtype** (`IB_RAW_U16`) — IBF format extension for
  outlier indices and future >256 codeword tables.

### Critical fixes

- **AWQ scratch sizing** — `session_matmul_with_scratch` silently
  skipped `inv_act` pre-scaling when `N > x_scratch_n`. Affected
  `down_proj` (N=intermediate>hidden) in `forward_step_batch`. Output
  was wrong on tall tensors; now per-slot scratch sized
  `max(hidden, inter)`. Bit-identical to single-token forward.

### Opt-in / experimental

- Phase 6 cross-token Δ cache (`IB_DELTA_CACHE`) — caches per-tensor
  (x_prev, out_prev); sparse Δx incremental update when ‖Δx‖ small.
  Default off (Δx isn't sparse on most tensors).
- Activation-sparse INT4 (`IB_ACT_SPARSE_REL`) — skip groups where
  max|x_chunk| < ε·max|x|. Default off (only down_proj benefits).
- Self-speculative early-exit (`ib_draft_layer_cap`, thread-local) —
  caller-set layer count for draft path. Requires trained early-exit
  head; pretrained models lose to noise here.

### Public API additions

- `ib_pq_session_matmul_batched(s, name, B, x_arr, out_arr)`
- `ib_pq_speculative_step(s, kv, prev_logits, pos, K, draft_tokens,
  out_tokens, n_accepted, next_logits)`
- INT4/PQ/fp16 backends now share `ib_pq_session_matmul` and
  `ib_pq_session_tensor_shape` dispatch.

### Format compatibility

- IBF v5 unchanged.
- Bundles built with v0.2.x continue to work.
- New raw-tensor names: `<base>__i4_q`, `<base>__i4_s`,
  `<base>__act_scale`, `<base>__i4_outl_idx`, `<base>__i4_outl_val`,
  `<base>__fp16w`. Older readers ignore them; loaders ≥ v0.3.0
  detect and route accordingly.

## v0.2.3 (and earlier)

See git history.
