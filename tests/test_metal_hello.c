/* test_metal_hello — validates the Metal build pipeline.
 * Allocates a unified-memory buffer, runs the vec_mul2 kernel, checks
 * output. If this passes, the .metal → .metallib → C dispatch chain is
 * working and we can build real kernels on top of it.
 */
#include "../src/metal/metal_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    if (!ib_metal_available()) {
        printf("Metal not available on this system\n");
        return 1;
    }
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) {
        printf("ib_metal_create failed\n");
        return 1;
    }
    printf("Metal device: %s\n", ib_metal_device_name(ctx));

    int n = 1024;
    float host_in[1024];
    for (int i = 0; i < n; i++) host_in[i] = (float)i * 0.5f;

    float *gpu_in  = ib_metal_alloc(ctx, n * sizeof(float), host_in);
    float *gpu_out = ib_metal_alloc(ctx, n * sizeof(float), NULL);
    if (!gpu_in || !gpu_out) {
        printf("ib_metal_alloc failed\n");
        return 1;
    }

    int rc = ib_metal_vec_mul2(ctx, gpu_in, gpu_out, n);
    if (rc != 0) {
        printf("ib_metal_vec_mul2 failed (rc=%d)\n", rc);
        return 1;
    }

    int errors = 0;
    for (int i = 0; i < n; i++) {
        float expected = host_in[i] * 2.0f;
        if (fabsf(gpu_out[i] - expected) > 1e-5f) {
            if (errors < 5) {
                printf("  mismatch at %d: got %f, expected %f\n",
                       i, gpu_out[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS — Metal vec_mul2 kernel correctness verified on %d elements\n", n);
    } else {
        printf("FAIL — %d mismatches out of %d\n", errors, n);
    }

    ib_metal_free(ctx, gpu_in);
    ib_metal_free(ctx, gpu_out);
    ib_metal_destroy(ctx);
    return errors == 0 ? 0 : 1;
}
