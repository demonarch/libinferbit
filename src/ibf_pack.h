/*
 * ibf_pack.h — Distribution-time compression for .ibf files.
 *
 * Per Stage 5i of docs/v2/00_CORRECTION.md and the "file-size rule"
 * (user, 2026-05-17): generic compression (zstd) is allowed ONLY at
 * distribution time. The runtime never decompresses on the fly. The
 * flow is:
 *
 *   developer: inferbit pack   model.ibf      -> model.ibf.zst
 *   user:      inferbit unpack model.ibf.zst  -> model.ibf
 *   runtime:   inferbit_load(model.ibf)         (mmap, zero-copy)
 *
 * The library exposes only the pack/unpack primitives. There is no
 * auto-decompress hook in the loader — keeping the read path
 * mmap-only is the whole point of the runtime rule.
 *
 * The implementation lives in src/ibf_pack.c and conditionally
 * links against libzstd (detected at CMake-configure time via
 * pkg-config or find_package). When zstd is not available at build
 * time both functions immediately return -1 with
 * inferbit_last_error() set to "zstd not available at build time".
 */

#ifndef IB_IBF_PACK_H
#define IB_IBF_PACK_H

#include "inferbit.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Compress `ibf_in` -> `zst_out`. `level` is the zstd compression
 * level (1..22; values <=0 are mapped to 19 — high-compression
 * default suited to one-shot distribution builds). Streaming so it
 * works on multi-GB models with a small RAM budget. The encoder
 * is run in long-range mode (windowLog=27, ~128 MiB window) which
 * catches the repeating local-codebook statistics in PQv2 indices.
 *
 * Returns 0 on success, -1 on failure (inferbit_last_error() set). */
int inferbit_pack(const char *ibf_in, const char *zst_out, int level);

/* Decompress `zst_in` -> `ibf_out`. Round-trip bit-identical with
 * inferbit_pack(): the SHA of the round-tripped .ibf matches the
 * source.
 *
 * Returns 0 on success, -1 on failure (inferbit_last_error() set). */
int inferbit_unpack(const char *zst_in, const char *ibf_out);

#ifdef __cplusplus
}
#endif

#endif /* IB_IBF_PACK_H */
