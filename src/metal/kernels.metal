/* libinferbit Metal compute shaders.
 * Embedded at build time as a C string and compiled at runtime via
 * [device newLibraryWithSource:].
 */
#include <metal_stdlib>
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
