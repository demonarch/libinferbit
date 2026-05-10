/* libinferbit Metal compute shaders.
 * Embedded at build time as a C string and compiled at runtime via
 * [device newLibraryWithSource:].
 */
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

/* Hello-world kernel — out[i] = in[i] * 2.0
 * Smoke test for the build + dispatch path. */
kernel void vec_mul2(
    device const float *in   [[buffer(0)]],
    device       float *out  [[buffer(1)]],
    uint                gid  [[thread_position_in_grid]])
{
    out[gid] = in[gid] * 2.0f;
}

/* ── matmul_w4a8 ────────────────────────────────────────────────────
 *
 * INT4 weight × INT8 activation → fp32 output.
 *
 *   weights:  uint8[M, N/2]  — packed nibbles, low nibble first.
 *   w_scales: half[M]        — per-row weight scale (fp16).
 *   x_q:      int8[N]        — quantized activation values.
 *   x_scales: float[N/IB_W4A8_GROUP] — per-group activation scales.
 *   out:      float[M]       — out[m] = sum_n W[m,n] * x[n].
 *
 * Tiling: one SIMD group (32 threads) per output row. Each thread
 * handles N/32 columns. Parallel reduction via simd_sum. Per-group
 * activation scale applied within each thread's partial sum.
 *
 * Group size 128 matches the CPU w4a8 kernel exactly so quality
 * model is identical. */

constant constexpr int IB_W4A8_GROUP = 128;

kernel void matmul_w4a8(
    device const uchar  *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const char   *x_q       [[buffer(2)]],
    device const float  *x_scales  [[buffer(3)]],
    device       float  *out       [[buffer(4)]],
    constant     uint   &M         [[buffer(5)]],
    constant     uint   &N         [[buffer(6)]],
    uint                 simd_lane  [[thread_index_in_simdgroup]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint                 tg_id      [[threadgroup_position_in_grid]],
    uint                 tg_size    [[threads_per_threadgroup]])
{
    /* Row index = which simdgroup in the whole grid. Threadgroup width
     * is set by host (we use 4 simdgroups per threadgroup → 128 threads). */
    uint simdgroups_per_tg = tg_size / 32u;
    uint m = tg_id * simdgroups_per_tg + simd_id;
    if (m >= M) return;

    device const uchar *row = weights + (size_t)m * (N / 2);

    /* Each lane processes N/32 columns. With group=128 and 32 lanes,
     * each lane handles exactly 4 elements per group, sweeping n_groups
     * groups total. Lane k processes columns [k, k+32, k+64, ...]. */

    /* Accumulate the dot product as float (so we can apply per-group
     * activation scale at group boundaries). */
    float lane_acc = 0.0f;

    int n_groups = (int)((N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP);
    for (int g = 0; g < n_groups; g++) {
        int start = g * IB_W4A8_GROUP;
        int end   = min(start + IB_W4A8_GROUP, (int)N);
        int lane_int = 0;

        /* Each lane processes 4 elements in this group: starting at
         * `start + simd_lane * 4`, with stride IB_W4A8_GROUP if there
         * were multiple per lane (we sized group=128, lanes=32 → 4 each). */
        int n = start + (int)simd_lane * 4;
        if (n + 3 < end) {
            /* Read 2 packed bytes = 4 nibbles. */
            int byte_off = n / 2;
            uchar p0 = row[byte_off];
            uchar p1 = row[byte_off + 1];
            int w0 = (int)(p0 & 0x0F) - 8;
            int w1 = (int)((p0 >> 4) & 0x0F) - 8;
            int w2 = (int)(p1 & 0x0F) - 8;
            int w3 = (int)((p1 >> 4) & 0x0F) - 8;
            lane_int += w0 * (int)x_q[n]
                     + w1 * (int)x_q[n + 1]
                     + w2 * (int)x_q[n + 2]
                     + w3 * (int)x_q[n + 3];
        }
        /* Tail handling — last group might be short, only some lanes work. */
        else {
            for (int t = 0; t < 4 && n + t < end; t++) {
                int byte_off = (n + t) / 2;
                uchar p = row[byte_off];
                int w = ((n + t) & 1) ? ((int)((p >> 4) & 0x0F) - 8)
                                       : ((int)(p & 0x0F) - 8);
                lane_int += w * (int)x_q[n + t];
            }
        }

        /* Reduce across the simdgroup so all 32 lanes have the full
         * group sum, then apply per-group activation scale. */
        int group_int = simd_sum(lane_int);
        lane_acc += (float)group_int * x_scales[g];
    }

    /* Lane 0 writes the result. Per-group reductions already used
     * simd_sum so all lanes have identical lane_acc — just lane 0 stores. */
    if (simd_lane == 0) {
        out[m] = lane_acc * (float)w_scales[m];
    }
}

/* matmul_w4a8 with per-32-element block weight scales.
 *
 * w_scales is M*(N/32) fp16 values (vs M for matmul_w4a8 above).
 * Activation grouping (IB_W4A8_GROUP=128) is unchanged. Each 128-element
 * activation group contains 4 weight blocks of 32; each block has its
 * own scale.
 *
 * Tile layout: 1 SIMD group per output row, 32 lanes per SIMD group. For
 * each group's 4 weight blocks (b=0..3) we have the 32 lanes process 1
 * element each (32 elements = one block exactly), simd_sum reduces to
 * scalar, multiply by w_scale, accumulate as fp32. After 4 blocks:
 * group_partial * a_scale → row accumulator.
 *
 * Same memory layout for weights as matmul_w4a8 (uchar packed nibbles).
 */
kernel void matmul_w4a8_blk32(
    device const uchar  *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],   /* length M*(N/32) */
    device const char   *x_q       [[buffer(2)]],
    device const float  *x_scales  [[buffer(3)]],
    device       float  *out       [[buffer(4)]],
    constant     uint   &M         [[buffer(5)]],
    constant     uint   &N         [[buffer(6)]],
    uint                 simd_lane  [[thread_index_in_simdgroup]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint                 tg_id      [[threadgroup_position_in_grid]],
    uint                 tg_size    [[threads_per_threadgroup]])
{
    uint simdgroups_per_tg = tg_size / 32u;
    uint m = tg_id * simdgroups_per_tg + simd_id;
    if (m >= M) return;

    device const uchar *row = weights + (size_t)m * (N / 2);
    uint n_w_blocks = N / 32u;
    device const half *row_scales = w_scales + (size_t)m * n_w_blocks;

    float lane_acc = 0.0f;
    int n_groups = (int)((N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP);
    for (int g = 0; g < n_groups; g++) {
        int g_start = g * IB_W4A8_GROUP;
        float a_scale = x_scales[g];
        float group_partial = 0.0f;

        /* 4 weight blocks per 128-element activation group. */
        for (int b = 0; b < 4; b++) {
            int blk_start = g_start + b * 32;
            uint wb_idx = (uint)(blk_start / 32);
            float w_scale = (float)row_scales[wb_idx];

            /* Each of 32 lanes processes 1 element of this block. */
            int n = blk_start + (int)simd_lane;
            int lane_int = 0;
            if (n < (int)N) {
                int byte_off = n / 2;
                uchar byte = row[byte_off];
                int w = (n & 1) ? ((int)((byte >> 4) & 0x0F) - 8)
                                 : ((int)(byte & 0x0F) - 8);
                lane_int = w * (int)x_q[n];
            }
            int block_int = simd_sum(lane_int);
            group_partial += (float)block_int * w_scale;
        }
        lane_acc += group_partial * a_scale;
    }

    if (simd_lane == 0) {
        out[m] = lane_acc;
    }
}

/* ── matmul_w4a8_blk32_batched ────────────────────────────────────────
 *
 * Batched prefill variant: takes B activation rows (B = batch size /
 * prefill token count) instead of one. Output shape is [B, M].
 *
 * Layout convention (matches CPU code, row-major, no padding):
 *   x_q       : char[B][N]        (N int8 activations per token)
 *   x_scales  : float[B][N/128]   (per-128-group activation scale)
 *   out       : float[B][M]
 *
 * Tile: one SIMD group per (b, m) output cell. Total dispatched SIMD
 * groups = B * M, organised as B in the .y dimension and M in the .x
 * dimension of the threadgroup grid. Each cell does the same per-row
 * inner loop as `matmul_w4a8_blk32`.
 *
 * Weight reuse across tokens is implicit via Metal's L1/L2 cache: SIMD
 * groups with the same m but different b read the same weight bytes
 * back-to-back. We don't hand-roll threadgroup-memory sharing because
 * the simple version is already memory-bandwidth-bound at the GPU
 * level for typical prefill batch sizes (B=8..64), and explicit
 * sharing adds complexity for marginal gain at this batch range.
 */
kernel void matmul_w4a8_blk32_batched(
    device const uchar  *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const char   *x_q       [[buffer(2)]],
    device const float  *x_scales  [[buffer(3)]],
    device       float  *out       [[buffer(4)]],
    constant     uint   &M         [[buffer(5)]],
    constant     uint   &N         [[buffer(6)]],
    constant     uint   &B         [[buffer(7)]],
    uint                 simd_lane  [[thread_index_in_simdgroup]],
    uint                 simd_id    [[simdgroup_index_in_threadgroup]],
    uint2                tg_id      [[threadgroup_position_in_grid]],
    uint2                tg_size    [[threads_per_threadgroup]])
{
    uint simdgroups_per_tg = tg_size.x / 32u;
    uint m = tg_id.x * simdgroups_per_tg + simd_id;
    uint b = tg_id.y;
    if (m >= M || b >= B) return;

    device const uchar *row = weights + (size_t)m * (N / 2);
    uint n_w_blocks = N / 32u;
    device const half  *row_scales = w_scales + (size_t)m * n_w_blocks;
    device const char  *x_q_row    = x_q      + (size_t)b * N;
    device const float *x_s_row    = x_scales + (size_t)b * (N / 128u);

    float lane_acc = 0.0f;
    int n_groups = (int)((N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP);
    for (int g = 0; g < n_groups; g++) {
        int g_start = g * IB_W4A8_GROUP;
        float a_scale = x_s_row[g];
        float group_partial = 0.0f;

        for (int blk = 0; blk < 4; blk++) {
            int blk_start = g_start + blk * 32;
            uint wb_idx = (uint)(blk_start / 32);
            float w_scale = (float)row_scales[wb_idx];

            int n = blk_start + (int)simd_lane;
            int lane_int = 0;
            if (n < (int)N) {
                int byte_off = n / 2;
                uchar byte = row[byte_off];
                int w = (n & 1) ? ((int)((byte >> 4) & 0x0F) - 8)
                                 : ((int)(byte & 0x0F) - 8);
                lane_int = w * (int)x_q_row[n];
            }
            int block_int = simd_sum(lane_int);
            group_partial += (float)block_int * w_scale;
        }
        lane_acc += group_partial * a_scale;
    }

    if (simd_lane == 0) {
        out[(size_t)b * M + m] = lane_acc;
    }
}

/* matmul_w4a8_blk32_batched_tiled — weight-sharing variant of the
 * batched matmul. Big idea: load the weight row + scales for a single
 * output `m` into threadgroup memory ONCE, then have multiple SIMD
 * groups within the threadgroup re-use those weights for different
 * tokens. Each SIMDgroup handles one (b, m) output cell.
 *
 * Tile geometry:
 *   threadgroup = TILE_B SIMDgroups × 32 lanes  (512 threads at TILE_B=16)
 *   grid        = (M, ceil(B/TILE_B), 1)
 *   threadgroup memory: weight row (N/2 bytes) + scales (N/32 fp16) ≈
 *     up to ~10 KB for the 8B down_proj — well under the 32 KB limit.
 *
 * The weight bandwidth reduction is the main lever: the unbatched +
 * non-tiled batched kernels load the same weight bytes B times (once
 * per (b, m) SIMDgroup). This kernel loads them once per TILE_B
 * SIMDgroups → TILE_B× weight-bandwidth reduction.
 */
#define MMW4A8_TILE_B 16
kernel void matmul_w4a8_blk32_batched_tiled(
    device const uchar  *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const char   *x_q       [[buffer(2)]],
    device const float  *x_scales  [[buffer(3)]],
    device       float  *out       [[buffer(4)]],
    constant     uint   &M         [[buffer(5)]],
    constant     uint   &N         [[buffer(6)]],
    constant     uint   &B         [[buffer(7)]],
    threadgroup uchar   *tg_w      [[threadgroup(0)]],
    threadgroup half    *tg_ws     [[threadgroup(1)]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id   [[simdgroup_index_in_threadgroup]],
    uint2                tg_id     [[threadgroup_position_in_grid]],
    uint2                tid2      [[thread_position_in_threadgroup]])
{
    uint m = tg_id.x;
    uint b = tg_id.y * MMW4A8_TILE_B + simd_id;
    if (m >= M) return;

    uint tid = tid2.x;
    constexpr uint TG_THREADS = MMW4A8_TILE_B * 32u;
    uint w_bytes = N / 2u;
    uint n_w_blocks = N / 32u;

    /* Cooperative load: pull weight row + scales for this m into TG mem. */
    device const uchar *src_w  = weights  + (size_t)m * w_bytes;
    device const half  *src_ws = w_scales + (size_t)m * n_w_blocks;
    for (uint i = tid; i < w_bytes;    i += TG_THREADS) tg_w[i]  = src_w[i];
    for (uint i = tid; i < n_w_blocks; i += TG_THREADS) tg_ws[i] = src_ws[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* SIMD groups beyond B exit (they did help with the cooperative
     * load, so they had to participate). */
    if (b >= B) return;

    device const char  *x_q_row = x_q      + (size_t)b * N;
    device const float *x_s_row = x_scales + (size_t)b * (N / 128u);

    float lane_acc = 0.0f;
    int n_groups = (int)((N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP);
    for (int g = 0; g < n_groups; g++) {
        int g_start = g * IB_W4A8_GROUP;
        float a_scale = x_s_row[g];
        float group_partial = 0.0f;

        for (int blk = 0; blk < 4; blk++) {
            int blk_start = g_start + blk * 32;
            uint wb_idx = (uint)(blk_start / 32);
            float w_scale = (float)tg_ws[wb_idx];

            int n = blk_start + (int)simd_lane;
            int lane_int = 0;
            if (n < (int)N) {
                int byte_off = n / 2;
                uchar byte = tg_w[byte_off];
                int w = (n & 1) ? ((int)((byte >> 4) & 0x0F) - 8)
                                 : ((int)(byte & 0x0F) - 8);
                lane_int = w * (int)x_q_row[n];
            }
            int block_int = simd_sum(lane_int);
            group_partial += (float)block_int * w_scale;
        }
        lane_acc += group_partial * a_scale;
    }

    if (simd_lane == 0) {
        out[(size_t)b * M + m] = lane_acc;
    }
}

/* matmul_w4a8_blk32_batched_simdmat — uses Apple Silicon's
 * simdgroup_matrix_multiply hardware intrinsic.
 *
 * Each threadgroup (1 SIMDgroup = 32 lanes) computes an 8×8 output
 * tile out[b_base..b_base+8][m_base..m_base+8]. The K dimension is
 * processed in chunks of K_TILE=128 (one full activation group):
 *   1) Cooperatively dequant 8×128 weight rows to fp16 in TG memory,
 *      applying per-32-element block scales.
 *   2) Cooperatively dequant 8×128 activation rows to fp16, applying
 *      the per-128-element group scale.
 *   3) Run 16 sub-matmuls of K=8 each via simdgroup_multiply_accumulate.
 *
 * The accumulator stays in fp32 for numerical stability. Output is
 * written back as fp32 to out[B][M] row-major.
 *
 * Grid: (M/8, B/8, 1) threadgroups. M and B must be multiples of 8.
 */
constant constexpr int SDM_M_TILE = 8;
constant constexpr int SDM_B_TILE = 8;
constant constexpr int SDM_K_TILE = 128;   /* IB_W4A8_GROUP */

kernel void matmul_w4a8_blk32_batched_simdmat(
    device const uchar  *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const char   *x_q       [[buffer(2)]],
    device const float  *x_scales  [[buffer(3)]],
    device       float  *out       [[buffer(4)]],
    constant     uint   &M         [[buffer(5)]],
    constant     uint   &N         [[buffer(6)]],
    constant     uint   &B         [[buffer(7)]],
    threadgroup half    *tg_W_fp16 [[threadgroup(0)]],   /* [SDM_M_TILE][SDM_K_TILE] */
    threadgroup half    *tg_A_fp16 [[threadgroup(1)]],   /* [SDM_B_TILE][SDM_K_TILE] */
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint2                tg_id     [[threadgroup_position_in_grid]])
{
    uint m_base = tg_id.x * SDM_M_TILE;
    uint b_base = tg_id.y * SDM_B_TILE;
    if (m_base >= M || b_base >= B) return;

    /* Accumulator: 8x8 fp32 output tile (rows=b, cols=m). */
    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    uint n_groups = N / SDM_K_TILE;
    uint w_row_bytes = N / 2u;
    uint w_scale_per_row = N / 32u;

    for (uint g = 0; g < n_groups; g++) {
        uint k_start = g * SDM_K_TILE;

        /* ── Dequant W tile (8×128 fp16) ── */
        for (uint i = simd_lane; i < SDM_M_TILE * SDM_K_TILE; i += 32u) {
            uint m_local = i / SDM_K_TILE;
            uint k_local = i % SDM_K_TILE;
            uint m_global = m_base + m_local;
            uint k_global = k_start + k_local;

            half v = (half)0;
            if (m_global < M && k_global < N) {
                size_t byte_off = (size_t)m_global * w_row_bytes + (k_global / 2u);
                uchar byte = weights[byte_off];
                int w_int = (k_global & 1u) ? ((int)((byte >> 4) & 0x0F) - 8)
                                              : ((int)(byte & 0x0F) - 8);
                uint wb_idx = k_global / 32u;
                half w_scale = w_scales[m_global * w_scale_per_row + wb_idx];
                v = (half)w_int * w_scale;
            }
            tg_W_fp16[i] = v;
        }

        /* ── Dequant A tile (8×128 fp16) ── */
        for (uint i = simd_lane; i < SDM_B_TILE * SDM_K_TILE; i += 32u) {
            uint b_local = i / SDM_K_TILE;
            uint k_local = i % SDM_K_TILE;
            uint b_global = b_base + b_local;
            uint k_global = k_start + k_local;

            half v = (half)0;
            if (b_global < B && k_global < N) {
                char  a_int   = x_q[(size_t)b_global * N + k_global];
                float a_scale = x_scales[b_global * (N / 128u) + g];
                v = (half)((float)a_int * a_scale);
            }
            tg_A_fp16[i] = v;
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);

        /* ── 16 sub-matmuls of K=8 each, accumulating into C ── */
        for (uint k_sub = 0; k_sub < SDM_K_TILE / 8u; k_sub++) {
            simdgroup_half8x8 A_sub;
            simdgroup_half8x8 W_sub_T;
            /* A_sub: 8 rows × 8 cols, naturally laid out as rows=b, cols=k.
             * Source: tg_A_fp16[b_local][k_sub*8 + k_in]
             * Row stride between b's = SDM_K_TILE. */
            simdgroup_load(A_sub, tg_A_fp16 + k_sub * 8u, SDM_K_TILE);
            /* W_sub_T: need 8 rows × 8 cols with rows=k, cols=m.
             * Source tg_W_fp16 is [m_local][k_local]. Use transpose flag. */
            simdgroup_load(W_sub_T, tg_W_fp16 + k_sub * 8u, SDM_K_TILE,
                            ulong2(0, 0), /*transpose=*/true);
            /* C[b][m] += A_sub[b][k] * W_sub_T[k][m] */
            simdgroup_multiply_accumulate(C, A_sub, W_sub_T, C);
        }

        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Store C → out[b_base..b_base+8][m_base..m_base+8].
     * out is row-major [B][M], row stride M. */
    simdgroup_store(C, out + (size_t)b_base * M + m_base, M);
}

/* Batched activation quantizer: takes B fp32 rows of length N, produces
 * B int8 rows + B float scale rows of length N/128. Tile: each
 * threadgroup handles one (b, group) cell with the same internal
 * structure as quantize_input_int8_g128. */
kernel void quantize_input_int8_g128_batched(
    device const float *x        [[buffer(0)]],
    device       char  *x_q      [[buffer(1)]],
    device       float *x_scales [[buffer(2)]],
    constant     uint  &N        [[buffer(3)]],
    constant     uint  &B        [[buffer(4)]],
    uint2               tg_id    [[threadgroup_position_in_grid]],
    uint                lane     [[thread_index_in_threadgroup]])
{
    uint b = tg_id.y;
    if (b >= B) return;

    device const float *x_row    = x        + (size_t)b * N;
    device       char  *xq_row   = x_q      + (size_t)b * N;
    device       float *xs_row   = x_scales + (size_t)b * (N / 128u);

    int g_start = (int)tg_id.x * 128;
    int g_end   = min(g_start + 128, (int)N);

    int n0 = g_start + (int)lane * 4;
    float v0 = (n0 + 0 < g_end) ? x_row[n0 + 0] : 0.0f;
    float v1 = (n0 + 1 < g_end) ? x_row[n0 + 1] : 0.0f;
    float v2 = (n0 + 2 < g_end) ? x_row[n0 + 2] : 0.0f;
    float v3 = (n0 + 3 < g_end) ? x_row[n0 + 3] : 0.0f;

    float local_max = max(max(fabs(v0), fabs(v1)), max(fabs(v2), fabs(v3)));
    float group_max = simd_max(local_max);
    float scale = (group_max > 1e-30f) ? (group_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (lane == 0) {
        xs_row[tg_id.x] = scale;
    }

    int q0 = clamp((int)round(v0 * inv_scale), -127, 127);
    int q1 = clamp((int)round(v1 * inv_scale), -127, 127);
    int q2 = clamp((int)round(v2 * inv_scale), -127, 127);
    int q3 = clamp((int)round(v3 * inv_scale), -127, 127);
    if (n0 + 0 < g_end) xq_row[n0 + 0] = (char)q0;
    if (n0 + 1 < g_end) xq_row[n0 + 1] = (char)q1;
    if (n0 + 2 < g_end) xq_row[n0 + 2] = (char)q2;
    if (n0 + 3 < g_end) xq_row[n0 + 3] = (char)q3;
}

/* ── matmul_int8 (fp32 input × int8 weights × fp16 row scale) ────────
 *
 * Mirrors CPU `scalar_matmul_int8` exactly:
 *   out[m] = scale[m] * sum_n (weights[m, n] * input[n])
 * with weights int8, input fp32, scale fp16. NO per-group input
 * quantization (different from w4a8 — INT8 weights are wider so the
 * model uses fp32 × int8 directly).
 *
 * Tile: 1 SIMD group per output row. Each lane handles N/32 columns,
 * accumulating in fp32. simd_sum reduction at the end.
 */
kernel void matmul_int8_fp32_in(
    device const char   *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const float  *x         [[buffer(2)]],
    device       float  *out       [[buffer(3)]],
    constant     uint   &M         [[buffer(4)]],
    constant     uint   &N         [[buffer(5)]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id   [[simdgroup_index_in_threadgroup]],
    uint                 tg_id     [[threadgroup_position_in_grid]],
    uint                 tg_size   [[threads_per_threadgroup]])
{
    uint simdgroups_per_tg = tg_size / 32u;
    uint m = tg_id * simdgroups_per_tg + simd_id;
    if (m >= M) return;

    device const char *row = weights + (size_t)m * N;
    float lane_acc = 0.0f;
    /* Lane k handles columns [k, k+32, k+64, ...]. */
    for (uint n = simd_lane; n < N; n += 32u) {
        lane_acc += (float)row[n] * x[n];
    }
    float total = simd_sum(lane_acc);
    if (simd_lane == 0) {
        out[m] = total * (float)w_scales[m];
    }
}

/* Batched variant: takes B fp32 input rows of length N, outputs B fp32
 * rows of length M. Same SIMDgroup-per-(b, m) tile as the blk32 batched
 * matmul. Used when prefill has mixed-precision IBFs (INT8 q/k/v + INT4
 * blk32 FFN) so the INT8 path can also batch. */
kernel void matmul_int8_fp32_in_batched(
    device const char   *weights   [[buffer(0)]],
    device const half   *w_scales  [[buffer(1)]],
    device const float  *x         [[buffer(2)]],
    device       float  *out       [[buffer(3)]],
    constant     uint   &M         [[buffer(4)]],
    constant     uint   &N         [[buffer(5)]],
    constant     uint   &B         [[buffer(6)]],
    uint                 simd_lane [[thread_index_in_simdgroup]],
    uint                 simd_id   [[simdgroup_index_in_threadgroup]],
    uint2                tg_id     [[threadgroup_position_in_grid]],
    uint2                tg_size   [[threads_per_threadgroup]])
{
    uint simdgroups_per_tg = tg_size.x / 32u;
    uint m = tg_id.x * simdgroups_per_tg + simd_id;
    uint b = tg_id.y;
    if (m >= M || b >= B) return;

    device const char  *row   = weights + (size_t)m * N;
    device const float *x_row = x       + (size_t)b * N;

    float lane_acc = 0.0f;
    for (uint n = simd_lane; n < N; n += 32u) {
        lane_acc += (float)row[n] * x_row[n];
    }
    float total = simd_sum(lane_acc);
    if (simd_lane == 0) {
        out[(size_t)b * M + m] = total * (float)w_scales[m];
    }
}

/* ── quantize_input_int8_g128 ────────────────────────────────────────
 *
 * fp32 → int8 quantization with per-group scale. Mirrors the CPU
 * `ib_quantize_input_int8_g128`: groups of IB_W4A8_GROUP=128 elements
 * get one fp32 scale (max|x| / 127), each element rounded into int8.
 *
 *   x:        float[N]      input fp32 activation values.
 *   x_q:      int8[N]       quantized output.
 *   x_scales: float[N/128]  per-group scale (max|x| / 127).
 *
 * Tiling: one threadgroup per group of 128 elements. Each thread of the
 * group handles 4 elements. Reduction via simd_max in two steps for the
 * 128-wide group. Stride 128: with simdgroup width 32, one threadgroup
 * = 32 threads × 4 elements = 128.
 */
kernel void quantize_input_int8_g128(
    device const float *x        [[buffer(0)]],
    device       char  *x_q      [[buffer(1)]],
    device       float *x_scales [[buffer(2)]],
    constant     uint  &N        [[buffer(3)]],
    uint                tg_id    [[threadgroup_position_in_grid]],
    uint                lane     [[thread_index_in_threadgroup]])
{
    /* Group base: this threadgroup handles elements [tg_id*128 .. tg_id*128+128) */
    int g_start = (int)tg_id * 128;
    int g_end   = min(g_start + 128, (int)N);

    /* Each lane handles 4 elements: indices g_start + lane*4 .. g_start + lane*4 + 3 */
    int n0 = g_start + (int)lane * 4;
    float v0 = (n0 + 0 < g_end) ? x[n0 + 0] : 0.0f;
    float v1 = (n0 + 1 < g_end) ? x[n0 + 1] : 0.0f;
    float v2 = (n0 + 2 < g_end) ? x[n0 + 2] : 0.0f;
    float v3 = (n0 + 3 < g_end) ? x[n0 + 3] : 0.0f;

    /* Per-lane local max (4 elements). */
    float local_max = max(max(fabs(v0), fabs(v1)), max(fabs(v2), fabs(v3)));
    /* Reduce across the simdgroup (32 threads). */
    float group_max = simd_max(local_max);

    /* Compute scale (lane 0 writes; all lanes use the same value via broadcast). */
    float scale = (group_max > 1e-30f) ? (group_max / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (lane == 0) {
        x_scales[tg_id] = scale;
    }

    /* Each lane quantizes its 4 elements. */
    int q0 = (int)round(v0 * inv_scale);
    int q1 = (int)round(v1 * inv_scale);
    int q2 = (int)round(v2 * inv_scale);
    int q3 = (int)round(v3 * inv_scale);
    q0 = clamp(q0, -127, 127);
    q1 = clamp(q1, -127, 127);
    q2 = clamp(q2, -127, 127);
    q3 = clamp(q3, -127, 127);
    if (n0 + 0 < g_end) x_q[n0 + 0] = (char)q0;
    if (n0 + 1 < g_end) x_q[n0 + 1] = (char)q1;
    if (n0 + 2 < g_end) x_q[n0 + 2] = (char)q2;
    if (n0 + 3 < g_end) x_q[n0 + 3] = (char)q3;
}

/* ── rmsnorm_fp16 ─────────────────────────────────────────────────────
 *
 * out[i] = x[i] * weight[i] / sqrt(mean(x^2) + eps)
 * with `weight` stored as fp16 (same layout the IBF loader keeps).
 *
 * Single-threadgroup kernel: one dispatch per call (small reduction).
 * Layout: 256 threads = 8 SIMD groups × 32 lanes. Each thread sweeps
 * N/256 elements, accumulating x*x. Two-step reduction:
 *   1) simd_sum within each SIMD group (32 lanes → 1 partial)
 *   2) lane 0 of each SIMD writes to threadgroup memory; first SIMD
 *      reads back, simd_sum across the 8 partials, broadcasts.
 *
 * Designed for N up to ~16384 with one threadgroup; for larger N we
 * could do a two-pass tree reduction, but TinyLlama-class is well
 * under that.
 */
kernel void rmsnorm_fp16(
    device const float *x       [[buffer(0)]],
    device const half  *weight  [[buffer(1)]],
    device       float *out     [[buffer(2)]],
    constant     uint  &N       [[buffer(3)]],
    constant     float &eps     [[buffer(4)]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                simd_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TG_THREADS = 256;
    constexpr uint NUM_SIMDS  = TG_THREADS / 32;

    /* Pass 1: each thread sweeps its strided slice and computes sum-of-squares. */
    float local_ss = 0.0f;
    for (uint i = tid; i < N; i += TG_THREADS) {
        float v = x[i];
        local_ss += v * v;
    }

    /* Reduce across the SIMD group (32 lanes). */
    float simd_ss = simd_sum(local_ss);

    /* Cross-SIMD reduction via threadgroup memory. Lane 0 of each SIMD
     * writes its partial; one SIMD reads back & reduces. */
    threadgroup float partials[NUM_SIMDS];
    if (lane == 0) {
        partials[simd_id] = simd_ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* SIMD 0 takes ownership of the final reduction & broadcast. */
    threadgroup float tg_inv_rms;
    if (simd_id == 0) {
        float v = (lane < NUM_SIMDS) ? partials[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0) {
            float mean = total / (float)N;
            tg_inv_rms = 1.0f / sqrt(mean + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    /* Pass 2: scale + multiply by fp16 weight. */
    for (uint i = tid; i < N; i += TG_THREADS) {
        out[i] = x[i] * inv_rms * (float)weight[i];
    }
}

/* Batched RMSNorm: one threadgroup per row, B rows total. Each row uses
 * its own x[b][:] and out[b][:] slice; weight[:] is shared. */
kernel void rmsnorm_fp16_batched(
    device const float *x       [[buffer(0)]],
    device const half  *weight  [[buffer(1)]],
    device       float *out     [[buffer(2)]],
    constant     uint  &N       [[buffer(3)]],
    constant     float &eps     [[buffer(4)]],
    uint2               tid2    [[thread_position_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                simd_id [[simdgroup_index_in_threadgroup]],
    uint2               tg_id   [[threadgroup_position_in_grid]])
{
    constexpr uint TG_THREADS = 256;
    constexpr uint NUM_SIMDS  = TG_THREADS / 32;
    uint tid = tid2.x;
    uint b   = tg_id.y;

    device const float *x_row   = x   + (size_t)b * N;
    device       float *out_row = out + (size_t)b * N;

    float local_ss = 0.0f;
    for (uint i = tid; i < N; i += TG_THREADS) {
        float v = x_row[i];
        local_ss += v * v;
    }
    float simd_ss = simd_sum(local_ss);

    threadgroup float partials[NUM_SIMDS];
    if (lane == 0) partials[simd_id] = simd_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float tg_inv_rms;
    if (simd_id == 0) {
        float v = (lane < NUM_SIMDS) ? partials[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0) {
            float mean = total / (float)N;
            tg_inv_rms = 1.0f / sqrt(mean + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    for (uint i = tid; i < N; i += TG_THREADS) {
        out_row[i] = x_row[i] * inv_rms * (float)weight[i];
    }
}

/* ── silu_mul ─────────────────────────────────────────────────────────
 *
 * out[i] = silu(gate[i]) * up[i]   where   silu(x) = x / (1 + exp(-x))
 *
 * Element-wise; one thread per element. Used in the FFN block:
 *   ffn(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )
 * On TinyLlama the intermediate dim is 5632.
 */
kernel void silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device       float *out  [[buffer(2)]],
    constant     uint  &N    [[buffer(3)]],
    uint                gid  [[thread_position_in_grid]])
{
    if (gid >= N) return;
    float x = gate[gid];
    float s = x / (1.0f + exp(-x));
    out[gid] = s * up[gid];
}

/* Batched silu_mul. Grid: (N_chunked, B, 1) threadgroups. */
kernel void silu_mul_batched(
    device const float *gate  [[buffer(0)]],
    device const float *up    [[buffer(1)]],
    device       float *out   [[buffer(2)]],
    constant     uint  &N     [[buffer(3)]],
    uint2               gid2  [[thread_position_in_grid]])
{
    uint n = gid2.x;
    uint b = gid2.y;
    if (n >= N) return;
    size_t i = (size_t)b * N + n;
    float x = gate[i];
    float s = x / (1.0f + exp(-x));
    out[i] = s * up[i];
}

/* Batched residual_add: a[b][:] += b_in[b][:] for B rows of length N. */
kernel void residual_add_batched(
    device       float *a     [[buffer(0)]],
    device const float *b_in  [[buffer(1)]],
    constant     uint  &N     [[buffer(2)]],
    uint2               gid2  [[thread_position_in_grid]])
{
    uint n = gid2.x;
    uint bb = gid2.y;
    if (n >= N) return;
    size_t i = (size_t)bb * N + n;
    a[i] += b_in[i];
}

/* ── rope_inplace ─────────────────────────────────────────────────────
 *
 * Llama-style interleaved RoPE applied IN-PLACE to a tensor laid out
 * as [n_heads, head_dim]. Each pair (t[2i], t[2i+1]) inside a head is
 * rotated by angle = pos / theta^(2i/head_dim).
 *
 *   tensor    [n_heads * head_dim]  fp32, in-place
 *   pos                              token position
 *   theta                            10000.0 for Llama
 *
 * One thread per pair: grid = n_heads * (head_dim / 2). Each thread
 * loads its pair, rotates, writes back.
 *
 * NOTE: this kernel matches the CPU `scalar_rope` semantics exactly —
 * the i in `2i/head_dim` is the index of the *pair*, so freq(pair i)
 * = 1 / theta^(2i / head_dim). The CPU loop strides by 2 over a
 * "i = 0..head_dim" range, which is the same thing.
 */
kernel void rope_inplace(
    device       float *tensor   [[buffer(0)]],
    constant     uint  &n_heads  [[buffer(1)]],
    constant     uint  &head_dim [[buffer(2)]],
    constant     uint  &pos      [[buffer(3)]],
    constant     float &theta    [[buffer(4)]],
    uint                gid      [[thread_position_in_grid]])
{
    uint half_hd = head_dim / 2u;
    uint total = n_heads * half_hd;
    if (gid >= total) return;
    uint h    = gid / half_hd;
    uint pair = gid - h * half_hd;
    /* freq = 1 / theta^( (2*pair) / head_dim ) */
    float exponent = (float)(2u * pair) / (float)head_dim;
    float freq = pow(theta, -exponent);
    float angle = (float)pos * freq;
    float c = cos(angle), s = sin(angle);

    uint base = h * head_dim + 2u * pair;
    float v0 = tensor[base];
    float v1 = tensor[base + 1u];
    tensor[base]      = v0 * c - v1 * s;
    tensor[base + 1u] = v0 * s + v1 * c;
}

/* Batched RoPE: each row b at absolute position start_pos + b. Grid:
 * (n_pairs, B, 1). tensor is laid out as [B][n_heads * head_dim]. */
kernel void rope_inplace_batched(
    device       float *tensor    [[buffer(0)]],
    constant     uint  &n_heads   [[buffer(1)]],
    constant     uint  &head_dim  [[buffer(2)]],
    constant     uint  &start_pos [[buffer(3)]],
    constant     float &theta     [[buffer(4)]],
    uint2               gid2      [[thread_position_in_grid]])
{
    uint half_hd = head_dim / 2u;
    uint total_pairs = n_heads * half_hd;
    uint pair_idx = gid2.x;
    uint b        = gid2.y;
    if (pair_idx >= total_pairs) return;

    uint h    = pair_idx / half_hd;
    uint pair = pair_idx - h * half_hd;
    float exponent = (float)(2u * pair) / (float)head_dim;
    float freq = pow(theta, -exponent);
    float angle = (float)(start_pos + b) * freq;
    float c = cos(angle), s = sin(angle);

    size_t row_off = (size_t)b * (size_t)n_heads * (size_t)head_dim;
    size_t base = row_off + (size_t)(h * head_dim + 2u * pair);
    float v0 = tensor[base];
    float v1 = tensor[base + 1u];
    tensor[base]      = v0 * c - v1 * s;
    tensor[base + 1u] = v0 * s + v1 * c;
}

/* ── softmax_row ──────────────────────────────────────────────────────
 *
 * In-place softmax over a single row of N elements. One threadgroup per
 * call (256 threads). Three-pass via threadgroup memory:
 *   1) per-thread local max → simd_max → cross-SIMD max
 *   2) per-thread sum(exp(x - max)) → simd_sum → cross-SIMD sum
 *   3) per-thread divide by sum
 * Numerically stable thanks to the max subtraction.
 *
 * For batches (e.g. attention scores: n_heads × seq_len rows), use
 * `softmax_rows` below with grid = (n_rows, 1, 1).
 */
kernel void softmax_rows(
    device       float *data    [[buffer(0)]],
    constant     uint  &row_len [[buffer(1)]],
    uint                tg_id   [[threadgroup_position_in_grid]],
    uint                tid     [[thread_position_in_threadgroup]],
    uint                lane    [[thread_index_in_simdgroup]],
    uint                simd_id [[simdgroup_index_in_threadgroup]])
{
    constexpr uint TG_THREADS = 256;
    constexpr uint NUM_SIMDS  = TG_THREADS / 32;

    device float *row = data + (size_t)tg_id * row_len;

    /* Pass 1: max */
    float local_max = -INFINITY;
    for (uint i = tid; i < row_len; i += TG_THREADS) {
        float v = row[i];
        if (v > local_max) local_max = v;
    }
    float simd_m = simd_max(local_max);
    threadgroup float partials[NUM_SIMDS];
    if (lane == 0) partials[simd_id] = simd_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float tg_max;
    if (simd_id == 0) {
        float v = (lane < NUM_SIMDS) ? partials[lane] : -INFINITY;
        float total = simd_max(v);
        if (lane == 0) tg_max = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = tg_max;

    /* Pass 2: sum(exp(x - max)) — write exp results back into row first
     * to avoid recomputing exp() in pass 3. */
    float local_sum = 0.0f;
    for (uint i = tid; i < row_len; i += TG_THREADS) {
        float e = exp(row[i] - row_max);
        row[i] = e;
        local_sum += e;
    }
    float simd_s = simd_sum(local_sum);
    if (lane == 0) partials[simd_id] = simd_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float tg_inv_sum;
    if (simd_id == 0) {
        float v = (lane < NUM_SIMDS) ? partials[lane] : 0.0f;
        float total = simd_sum(v);
        if (lane == 0) tg_inv_sum = 1.0f / total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_sum = tg_inv_sum;

    /* Pass 3: normalize */
    for (uint i = tid; i < row_len; i += TG_THREADS) {
        row[i] *= inv_sum;
    }
}

/* ── embed_lookup_fp16 ────────────────────────────────────────────────
 *
 * Copy a row from the embedding matrix into the hidden buffer, with
 * fp16→fp32 conversion. One thread per element of the hidden dim.
 *
 *   embeddings  [vocab, hidden]  fp16
 *   token                          token id
 *   out         [hidden]         fp32
 */
kernel void embed_lookup_fp16(
    device const half  *embeddings [[buffer(0)]],
    constant     uint  &token      [[buffer(1)]],
    constant     uint  &hidden     [[buffer(2)]],
    device       float *out        [[buffer(3)]],
    uint                gid        [[thread_position_in_grid]])
{
    if (gid >= hidden) return;
    out[gid] = (float)embeddings[(size_t)token * hidden + gid];
}

/* ── KV cache + attention kernels (fp16 KV cache) ────────────────────
 *
 * KV cache layout: [seq_len, kv_dim] fp16 for both keys and values,
 * where kv_dim = n_kv_heads * head_dim. Same layout the CPU
 * kv_cache_write uses with kv_bits=16. Phase 5d will add INT8/INT4 KV.
 */

/* Write K, V at position `pos` into the cache. The libinferbit CPU
 * stores kv_bits=16 as fp32 (NOT fp16), so we match exactly: just
 * memcpy via per-thread copy. */
kernel void kv_cache_write_fp16(
    device const float *k       [[buffer(0)]],
    device const float *v       [[buffer(1)]],
    device       float *k_cache [[buffer(2)]],
    device       float *v_cache [[buffer(3)]],
    constant     uint  &pos     [[buffer(4)]],
    constant     uint  &kv_dim  [[buffer(5)]],
    uint                gid     [[thread_position_in_grid]])
{
    if (gid >= kv_dim) return;
    size_t off = (size_t)pos * kv_dim + gid;
    k_cache[off] = k[gid];
    v_cache[off] = v[gid];
}

/* Attention scores: scores[h, t] = (Q[h] · K_cache[t, kv_h]) * scale. */
kernel void attn_scores_qk(
    device const float *q              [[buffer(0)]],
    device const float *k_cache        [[buffer(1)]],
    device       float *scores         [[buffer(2)]],
    constant     uint  &n_heads        [[buffer(3)]],
    constant     uint  &n_kv_heads     [[buffer(4)]],
    constant     uint  &head_dim       [[buffer(5)]],
    constant     uint  &seq_pos_p1     [[buffer(6)]],
    constant     float &scale          [[buffer(7)]],
    uint                gid            [[thread_position_in_grid]])
{
    uint total = n_heads * seq_pos_p1;
    if (gid >= total) return;
    uint h = gid / seq_pos_p1;
    uint t = gid - h * seq_pos_p1;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint kv_dim = n_kv_heads * head_dim;

    device const float *q_h = q + h * head_dim;
    device const float *k_t = k_cache + (size_t)t * kv_dim + kv_h * head_dim;
    float s = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        s += q_h[d] * k_t[d];
    }
    scores[(size_t)h * seq_pos_p1 + t] = s * scale;
}

/* ── INT8 KV cache (matches libinferbit kv_bits=8) ────────────────────
 *
 * Layout (mirrors forward.c::kv_cache_write_int8):
 *   key_cache   int8[seq_len, n_kv_heads, head_dim]
 *   value_cache int8[seq_len, n_kv_heads, head_dim]
 *   key_scales  float[seq_len, n_kv_heads]
 *   value_scales float[seq_len, n_kv_heads]
 *
 * Per-head per-position scale: scale = max(|x|) / 127, floor 1e-8.
 */

/* Write K, V at row `pos` with per-head INT8 quantization.
 *
 * One threadgroup per kv_head (grid=(n_kv_heads,1,1)). 32 threads/SIMD;
 * each thread handles head_dim/32 elements (exact for head_dim multiple
 * of 32 — the typical 64/96/128/256). simd_max across the SIMD gives
 * the per-head max in one step.
 */
kernel void kv_cache_write_int8(
    device const float *k          [[buffer(0)]],
    device const float *v          [[buffer(1)]],
    device       char  *k_cache    [[buffer(2)]],
    device       char  *v_cache    [[buffer(3)]],
    device       float *k_scales   [[buffer(4)]],
    device       float *v_scales   [[buffer(5)]],
    constant     uint  &pos        [[buffer(6)]],
    constant     uint  &n_kv_heads [[buffer(7)]],
    constant     uint  &head_dim   [[buffer(8)]],
    uint                tg_id      [[threadgroup_position_in_grid]],
    uint                lane       [[thread_index_in_simdgroup]])
{
    uint h = tg_id;
    if (h >= n_kv_heads) return;

    uint per_lane = (head_dim + 31u) / 32u;
    /* Pass 1: each lane finds its own max over per_lane elements. */
    float k_lmax = 0.0f, v_lmax = 0.0f;
    for (uint i = 0; i < per_lane; i++) {
        uint d = lane + i * 32u;
        if (d < head_dim) {
            float kv = k[h * head_dim + d];
            float vv = v[h * head_dim + d];
            float ka = fabs(kv); if (ka > k_lmax) k_lmax = ka;
            float va = fabs(vv); if (va > v_lmax) v_lmax = va;
        }
    }
    float k_max = simd_max(k_lmax);
    float v_max = simd_max(v_lmax);
    float k_scale = (k_max > 1e-8f) ? (k_max / 127.0f) : 1e-8f;
    float v_scale = (v_max > 1e-8f) ? (v_max / 127.0f) : 1e-8f;
    if (lane == 0) {
        k_scales[(size_t)pos * n_kv_heads + h] = k_scale;
        v_scales[(size_t)pos * n_kv_heads + h] = v_scale;
    }
    /* Pass 2: each lane quantizes its elements. */
    uint kv_dim = n_kv_heads * head_dim;
    float k_inv = 1.0f / k_scale;
    float v_inv = 1.0f / v_scale;
    for (uint i = 0; i < per_lane; i++) {
        uint d = lane + i * 32u;
        if (d < head_dim) {
            float kv = k[h * head_dim + d];
            float vv = v[h * head_dim + d];
            int kq = (int)round(kv * k_inv);
            int vq = (int)round(vv * v_inv);
            kq = clamp(kq, -127, 127);
            vq = clamp(vq, -127, 127);
            size_t off = (size_t)pos * kv_dim + h * head_dim + d;
            k_cache[off] = (char)kq;
            v_cache[off] = (char)vq;
        }
    }
}

/* INT8 attention scores: scores[h, t] = (Q[h] · K_cache[t, kv_h]) * scale,
 * where K is int8 with per-head scale. Same dispatch shape as
 * attn_scores_qk (one thread per (h, t)).
 */
kernel void attn_scores_qk_int8(
    device const float *q              [[buffer(0)]],
    device const char  *k_cache        [[buffer(1)]],
    device const float *k_scales       [[buffer(2)]],
    device       float *scores         [[buffer(3)]],
    constant     uint  &n_heads        [[buffer(4)]],
    constant     uint  &n_kv_heads     [[buffer(5)]],
    constant     uint  &head_dim       [[buffer(6)]],
    constant     uint  &seq_pos_p1     [[buffer(7)]],
    constant     float &scale          [[buffer(8)]],
    uint                gid            [[thread_position_in_grid]])
{
    uint total = n_heads * seq_pos_p1;
    if (gid >= total) return;
    uint h = gid / seq_pos_p1;
    uint t = gid - h * seq_pos_p1;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint kv_dim = n_kv_heads * head_dim;

    device const float *q_h = q + h * head_dim;
    device const char  *k_t = k_cache + (size_t)t * kv_dim + kv_h * head_dim;
    float k_scale = k_scales[(size_t)t * n_kv_heads + kv_h];
    int int_acc = 0;
    /* Sum int8 first (fits in int while head_dim <= ~256K), apply scale at end. */
    for (uint d = 0; d < head_dim; d++) {
        /* q is fp32; we can't accumulate as int. Do float accumulation. */
        /* fall back to float dot, scaled at end. */
    }
    (void)int_acc;
    float fl_acc = 0.0f;
    for (uint d = 0; d < head_dim; d++) {
        fl_acc += q_h[d] * (float)k_t[d];
    }
    scores[(size_t)h * seq_pos_p1 + t] = fl_acc * k_scale * scale;
}

/* INT8 weighted V: attn_out[h, d] = sum_t scores[h, t] * v_scales[t, kv_h] * V_cache[t, kv_h, d]. */
kernel void attn_weighted_v_int8(
    device const float *scores       [[buffer(0)]],
    device const char  *v_cache      [[buffer(1)]],
    device const float *v_scales     [[buffer(2)]],
    device       float *attn_out     [[buffer(3)]],
    constant     uint  &n_heads      [[buffer(4)]],
    constant     uint  &n_kv_heads   [[buffer(5)]],
    constant     uint  &head_dim     [[buffer(6)]],
    constant     uint  &seq_pos_p1   [[buffer(7)]],
    uint                gid          [[thread_position_in_grid]])
{
    uint total = n_heads * head_dim;
    if (gid >= total) return;
    uint h = gid / head_dim;
    uint d = gid - h * head_dim;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint kv_dim = n_kv_heads * head_dim;

    device const float *s_row = scores + (size_t)h * seq_pos_p1;
    float acc = 0.0f;
    for (uint t = 0; t < seq_pos_p1; t++) {
        float vs = v_scales[(size_t)t * n_kv_heads + kv_h];
        char v_q = v_cache[(size_t)t * kv_dim + kv_h * head_dim + d];
        acc += s_row[t] * vs * (float)v_q;
    }
    attn_out[(size_t)h * head_dim + d] = acc;
}

/* Element-wise add: a[i] += b[i]. Used for residual connections. */
kernel void residual_add(
    device       float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant     uint  &N [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    if (gid >= N) return;
    a[gid] += b[gid];
}

/* Attention weighted V: attn_out[h, d] = sum_t scores[h, t] * V_cache[t, kv_h, d]. */
kernel void attn_weighted_v(
    device const float *scores      [[buffer(0)]],
    device const float *v_cache     [[buffer(1)]],
    device       float *attn_out    [[buffer(2)]],
    constant     uint  &n_heads     [[buffer(3)]],
    constant     uint  &n_kv_heads  [[buffer(4)]],
    constant     uint  &head_dim    [[buffer(5)]],
    constant     uint  &seq_pos_p1  [[buffer(6)]],
    uint                gid         [[thread_position_in_grid]])
{
    uint total = n_heads * head_dim;
    if (gid >= total) return;
    uint h = gid / head_dim;
    uint d = gid - h * head_dim;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_h = h / heads_per_kv;
    uint kv_dim = n_kv_heads * head_dim;

    device const float *s_row = scores + (size_t)h * seq_pos_p1;
    float acc = 0.0f;
    for (uint t = 0; t < seq_pos_p1; t++) {
        acc += s_row[t] * v_cache[(size_t)t * kv_dim + kv_h * head_dim + d];
    }
    attn_out[(size_t)h * head_dim + d] = acc;
}
