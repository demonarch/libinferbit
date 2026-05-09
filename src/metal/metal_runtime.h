/* Metal runtime — C interface for libinferbit's Apple Metal GPU backend.
 *
 * All functions are no-ops / return NULL on non-Apple builds. Apple Silicon
 * has unified memory: ib_metal_alloc returns a pointer that's accessible
 * from BOTH CPU and GPU (storage mode shared), with no explicit copy
 * needed before/after a kernel dispatch.
 *
 * Usage:
 *   ib_metal_ctx *ctx = ib_metal_create();
 *   if (!ctx) { fallback to CPU path; }
 *   float *gpu_in  = ib_metal_alloc(ctx, n * sizeof(float), host_data);
 *   float *gpu_out = ib_metal_alloc(ctx, n * sizeof(float), NULL);
 *   ib_metal_vec_mul2(ctx, gpu_in, gpu_out, n);
 *   // gpu_out contents are visible directly (unified memory)
 *   ib_metal_free(ctx, gpu_in);
 *   ib_metal_free(ctx, gpu_out);
 *   ib_metal_destroy(ctx);
 */
#ifndef IB_METAL_RUNTIME_H
#define IB_METAL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context — wraps MTLDevice + MTLCommandQueue + MTLLibrary. */
typedef struct ib_metal_ctx ib_metal_ctx;

/* Create a Metal context using the system default device. Returns NULL
 * if Metal is unavailable, no compatible device is present, or kernel
 * library load fails. */
ib_metal_ctx *ib_metal_create(void);
void ib_metal_destroy(ib_metal_ctx *ctx);

/* Returns 1 if Metal is available + working, 0 otherwise. */
int ib_metal_available(void);

/* Returns a human-readable device name (e.g. "Apple M2"). NULL on error.
 * Caller does not free; lifetime tied to ctx. */
const char *ib_metal_device_name(ib_metal_ctx *ctx);

/* Allocate a unified-memory buffer (MTLResourceStorageModeShared). The
 * returned pointer is BOTH a host-visible pointer AND backed by an
 * MTLBuffer the kernels can use. Caller frees with ib_metal_free.
 *
 * If init is non-NULL, copies `bytes` from init into the buffer.
 * On failure returns NULL. */
void *ib_metal_alloc(ib_metal_ctx *ctx, size_t bytes, const void *init);
void  ib_metal_free(ib_metal_ctx *ctx, void *buf);

/* Hello-world test kernel: out[i] = in[i] * 2.0f. Synchronous. */
int ib_metal_vec_mul2(ib_metal_ctx *ctx,
                       const void *gpu_in, void *gpu_out, int n);

#ifdef __cplusplus
}
#endif

#endif /* IB_METAL_RUNTIME_H */
