#ifndef PQV2_FORMAT_H
#define PQV2_FORMAT_H

#include "pqv2_kernel.h"
#include <stdint.h>
#include <stddef.h>

/* IBF v6 / PQv2 file format — multi-tensor container.
 *
 * File layout (little-endian):
 *   [0..7]    magic     = "IBFV6PQ2"
 *   [8..11]   version   = 1
 *   [12..15]  n_tensors
 *   [16..19]  manifest_size_bytes
 *   [20..23]  reserved (0)
 *
 *   manifest[n_tensors]:
 *     uint16  name_len
 *     bytes   name (utf-8, name_len)
 *     uint8   kind  (0 = pqv2, 1 = raw_fp16, 2 = raw_fp32, 3 = raw_int32)
 *     uint8   ndim
 *     uint8   reserved[2]
 *     int32   shape[4]   (unused dims = 1)
 *     uint64  blob_offset (from start of file)
 *     uint64  blob_size
 *
 *   <align to 64>
 *
 *   For each tensor at blob_offset:
 *     If kind == 0 (PQV2): the "PQV2" single-tensor blob from pqv2_kernel.c:
 *       magic "PQV2", header u32×8 (M,N,G,K,n_sub,half,l2_kind,l2_K),
 *       row_scale fp16[M], cb_q[ns*K*half] int8, cb_scale[ns*K] fp16,
 *       indices[M*nc*ns] u8 (transposed [nc, ns, M]),
 *       L2 (if l2_kind==2): l2_cb_q, l2_cb_scale, l2_indices.
 *     If kind == 1/2/3: raw bytes of the tensor in the given dtype.
 *
 * Caller must keep the entire file mapped/loaded for the lifetime of
 * the parsed tensors (pointers index into the file buffer).
 */

typedef enum {
    IB_PQV2_KIND_PQV2 = 0,
    IB_PQV2_KIND_RAW_FP16 = 1,
    IB_PQV2_KIND_RAW_FP32 = 2,
    IB_PQV2_KIND_RAW_INT32 = 3,
} ib_pqv2_kind;

typedef struct {
    char *name;          /* heap-allocated */
    int kind;            /* ib_pqv2_kind */
    int ndim;
    int shape[4];
    /* For PQV2 kind: parsed pqv2_t (pointers index into file buffer) */
    pqv2_t pq;
    /* For RAW kind: raw data pointer (into file buffer) */
    const void *raw_data;
    size_t raw_size;
} ib_pqv2_named_tensor;

typedef struct {
    int n_tensors;
    ib_pqv2_named_tensor *tensors;
    /* File backing: either heap-loaded buffer or mmap'd region. */
    void *_buffer;
    size_t _buffer_size;
    int _is_mmap;        /* 1 if mmap, 0 if malloc */
    int _fd;
} ib_pqv2_file;

/* Load an IBF v6 PQv2 file. Returns 0 on success.
 * Uses mmap when possible (read-only). Caller frees via ib_pqv2_file_free. */
int ib_pqv2_file_load(const char *path, ib_pqv2_file *out);

/* Free file + all tensor metadata. Tensor data pointers become invalid. */
void ib_pqv2_file_free(ib_pqv2_file *f);

/* Look up a tensor by name. Returns NULL if not found. */
const ib_pqv2_named_tensor *ib_pqv2_find(const ib_pqv2_file *f, const char *name);

#endif
