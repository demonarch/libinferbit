/*
 * pq_kmeans.h — Lloyd's k-means in libinferbit.
 *
 * Replaces sklearn KMeans for the PQ codebook fitting stage of v5
 * conversion. Designed to:
 *   - Saturate available CPU cores (uses ib_thread_pool)
 *   - Specialise the inner distance loop for small D (D=2 is the only
 *     case the PQ converter actually uses; the kernel still works for
 *     general D)
 *   - Stay portable: pure C, no platform-specific intrinsics outside
 *     well-guarded SIMD blocks. Compiles to wasm via Emscripten.
 *   - Be deterministic given the same seed + thread count
 *
 * Output equivalence with sklearn: not bit-exact (different RNG, fp32
 * vs sklearn's fp64 distances), but within ~1% relative inertia on
 * realistic data. PPL after PQ encoding is dominated by the K-cell
 * partition rather than the exact centroid coordinates, so this is
 * fine for the conversion pipeline.
 */
#ifndef IB_PQ_KMEANS_H
#define IB_PQ_KMEANS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ib_thread_pool ib_thread_pool;

typedef struct {
    int K;          /* number of clusters */
    int D;          /* dimensionality (2 for canonical PQ) */
    int max_iter;   /* Lloyd iterations, default 20 */
    float tol;      /* convergence threshold on summed centroid shift */
    int n_init;     /* number of random restarts; best by inertia wins */
    int subsample;  /* if > 0 and N > subsample, fit on a random subset */
    uint32_t seed;
    /* Threading: pool may be NULL for single-threaded. */
    ib_thread_pool* pool;
} ib_kmeans_config;

/*
 * Fit centers on `X` (N x D fp32, row-major).
 *
 * Outputs:
 *   centers_out: K x D fp32, caller-allocated.
 *   indices_out: N int32 cluster assignments, caller-allocated.
 *                Pass NULL if you don't need them.
 *   inertia_out: optional pointer for the final SSE objective. May be NULL.
 *
 * Returns 0 on success, < 0 on error.
 *
 * Notes:
 *   - Init is k-means++ on a (sub)sample of the data.
 *   - When n_init > 1, the run with the lowest final inertia wins.
 *   - When subsample > 0 and N > subsample, fitting runs on the
 *     subsample, but `indices_out` (if requested) labels the full N.
 */
int ib_kmeans_fit(const float* X, int N,
                  const ib_kmeans_config* cfg,
                  float* centers_out,
                  int32_t* indices_out,
                  double* inertia_out);

/*
 * Assignment-only: given pre-fit centers, label every point in X.
 * Exposed so the converter can fit on a 30K subsample then label all
 * 50M+ chunks.
 */
int ib_kmeans_assign(const float* X, int N, int D,
                     const float* centers, int K,
                     int32_t* indices_out,
                     ib_thread_pool* pool);

#ifdef __cplusplus
}
#endif
#endif
