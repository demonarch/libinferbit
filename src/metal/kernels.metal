/* libinferbit Metal compute shaders.
 * Compiled by `xcrun metal` → .air → `xcrun metallib` → .metallib at build time.
 * Loaded at runtime by metal_runtime.mm.
 */
#include <metal_stdlib>
using namespace metal;

/* Hello-world kernel — out[i] = in[i] * 2.0
 * Validates the build pipeline (.metal → .metallib) and the C dispatch
 * path. Replaceable later with real GEMM / attention kernels. */
kernel void vec_mul2(
    device const float *in   [[buffer(0)]],
    device       float *out  [[buffer(1)]],
    uint                gid  [[thread_position_in_grid]])
{
    out[gid] = in[gid] * 2.0f;
}
