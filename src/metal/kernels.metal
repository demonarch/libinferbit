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
