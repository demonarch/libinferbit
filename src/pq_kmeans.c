/*
 * pq_kmeans.c — Lloyd's k-means with k-means++ init.
 *
 * Implementation notes:
 *   - All math in fp32. sklearn uses fp64 internals; we lose some
 *     precision in the centroid update but PPL is unaffected (see
 *     header).
 *   - Distance computation uses |x - c|^2 = |x|^2 - 2 x·c + |c|^2.
 *     |c|^2 is precomputed once per Lloyd iter; |x|^2 is constant
 *     across iters and ignored (it doesn't affect argmin).
 *   - Threading: assignment is parallel over X rows. Centroid update
 *     uses per-thread local accumulators reduced after the parallel
 *     phase — avoids atomics.
 *   - Determinism: the same (seed, n_threads, X, config) produces the
 *     same output, modulo floating-point reduction order across
 *     threads (per-thread sums then a single reducer is stable for a
 *     given n_threads).
 */

#include "pq_kmeans.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ── tiny PRNG (xoroshiro64**) — portable, deterministic ────────── */

typedef struct { uint32_t s[2]; } pq_rng;

static inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

static void pq_rng_init(pq_rng* r, uint32_t seed) {
    /* SplitMix32 to seed both halves so seed=0 doesn't degenerate. */
    uint32_t z = seed + 0x9e3779b9u;
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z = z ^ (z >> 16);
    r->s[0] = z ? z : 1;
    z += 0x9e3779b9u;
    z = (z ^ (z >> 16)) * 0x85ebca6bu;
    z = (z ^ (z >> 13)) * 0xc2b2ae35u;
    z = z ^ (z >> 16);
    r->s[1] = z ? z : 1;
}

static inline uint32_t pq_rng_next(pq_rng* r) {
    uint32_t s0 = r->s[0];
    uint32_t s1 = r->s[1];
    uint32_t result = rotl32(s0 * 0x9E3779BBu, 5) * 5;
    s1 ^= s0;
    r->s[0] = rotl32(s0, 26) ^ s1 ^ (s1 << 9);
    r->s[1] = rotl32(s1, 13);
    return result;
}

static inline float pq_rng_uniform01(pq_rng* r) {
    /* 24-bit uniform in [0, 1). */
    return (float)(pq_rng_next(r) >> 8) * (1.0f / 16777216.0f);
}

static inline int pq_rng_below(pq_rng* r, int n) {
    /* unbiased modulo via rejection — fine here since n is small. */
    if (n <= 1) return 0;
    uint32_t bound = (uint32_t)n;
    uint32_t threshold = (~bound + 1u) % bound;
    for (;;) {
        uint32_t x = pq_rng_next(r);
        if (x >= threshold) return (int)(x % bound);
    }
}

/* Random sample of `m` distinct indices from [0, n). Floyd's algorithm.
 * out has space for m ints. */
static void pq_sample_indices(pq_rng* r, int n, int m, int32_t* out) {
    if (m >= n) {
        for (int i = 0; i < n; i++) out[i] = i;
        return;
    }
    /* Floyd: O(m), sample without replacement.
     * Fast path uses a small linear scan against the running set,
     * which is fine for m=30k. */
    for (int j = n - m, k = 0; j < n; j++) {
        int t = pq_rng_below(r, j + 1);
        int taken = 0;
        for (int i = 0; i < k; i++) {
            if (out[i] == t) { taken = 1; break; }
        }
        out[k++] = taken ? j : t;
    }
}

/* ── distance helpers ──────────────────────────────────────────── */

/* squared L2 between an X row and a center row, both length D. */
static inline float sq_dist(const float* x, const float* c, int D) {
    float s = 0.0f;
    for (int d = 0; d < D; d++) {
        float v = x[d] - c[d];
        s += v * v;
    }
    return s;
}

/* Argmin over K centers for a single X row. Returns cluster index;
 * writes the squared distance to *dist_out (may be NULL).
 * cnorms2 is |c_k|^2 precomputed; xc is the dot product workspace. */
static int argmin_centers(const float* x, const float* C, int K, int D,
                          const float* cnorms2, float* dist_out) {
    int best_k = 0;
    float best = sq_dist(x, C, D);
    for (int k = 1; k < K; k++) {
        float d = sq_dist(x, C + (size_t)k * D, D);
        if (d < best) { best = d; best_k = k; }
    }
    if (dist_out) *dist_out = best;
    (void)cnorms2;  /* reserved for the |x|^2 - 2x·c + |c|^2 form below */
    return best_k;
}

/* ── k-means++ init ────────────────────────────────────────────── */

static void kmeanspp_init(const float* X, int N, int D, int K,
                          pq_rng* r, float* centers_out) {
    int first = pq_rng_below(r, N);
    memcpy(centers_out, X + (size_t)first * D, (size_t)D * sizeof(float));

    float* mind = (float*)malloc((size_t)N * sizeof(float));
    if (!mind) return;
    /* mind[i] = squared distance to the nearest already-chosen center */
    for (int i = 0; i < N; i++) {
        mind[i] = sq_dist(X + (size_t)i * D, centers_out, D);
    }

    for (int kk = 1; kk < K; kk++) {
        /* Sample next center proportional to mind[i]. */
        double total = 0.0;
        for (int i = 0; i < N; i++) total += (double)mind[i];

        int chosen;
        if (total <= 0.0) {
            chosen = pq_rng_below(r, N);
        } else {
            double target = pq_rng_uniform01(r) * total;
            double acc = 0.0;
            chosen = N - 1;
            for (int i = 0; i < N; i++) {
                acc += (double)mind[i];
                if (acc >= target) { chosen = i; break; }
            }
        }
        const float* cnew = X + (size_t)chosen * D;
        memcpy(centers_out + (size_t)kk * D, cnew, (size_t)D * sizeof(float));

        /* Update mind with the new center. */
        for (int i = 0; i < N; i++) {
            float d = sq_dist(X + (size_t)i * D, cnew, D);
            if (d < mind[i]) mind[i] = d;
        }
    }
    free(mind);
}

/* ── threaded assignment ───────────────────────────────────────── */

typedef struct {
    const float* X;
    int D;
    const float* C;
    int K;
    const float* cnorms2;
    int32_t* indices;     /* may be NULL when only inertia is needed */
    /* Per-thread per-cluster sums and counts, sized [n_threads][K*(D+1)]. */
    float* sums;          /* contiguous: thread t writes to sums + t*K*D */
    int32_t* counts;      /* contiguous: thread t writes to counts + t*K */
    double* inertias;     /* contiguous: thread t writes to inertias + t */
    int n_threads;
} pq_assign_ctx;

static void pq_assign_task(void* arg, int thread_id, int start, int end) {
    pq_assign_ctx* a = (pq_assign_ctx*)arg;
    int D = a->D, K = a->K;
    float* tsums = a->sums + (size_t)thread_id * (size_t)K * (size_t)D;
    int32_t* tcounts = a->counts + (size_t)thread_id * K;
    double tinertia = 0.0;

    for (int i = start; i < end; i++) {
        const float* xi = a->X + (size_t)i * D;
        float dist;
        int k = argmin_centers(xi, a->C, K, D, a->cnorms2, &dist);
        if (a->indices) a->indices[i] = k;
        tcounts[k] += 1;
        float* csum = tsums + (size_t)k * D;
        for (int d = 0; d < D; d++) csum[d] += xi[d];
        tinertia += (double)dist;
    }
    a->inertias[thread_id] = tinertia;
}

/* ── main fit ──────────────────────────────────────────────────── */

static double lloyd_one_init(const float* X, int N, int D, int K,
                             int max_iter, float tol,
                             pq_rng* rng,
                             float* centers,
                             int32_t* tmp_indices,
                             ib_thread_pool* pool,
                             int n_threads) {
    /* k-means++ init */
    kmeanspp_init(X, N, D, K, rng, centers);

    /* Per-thread accumulator scratch (always allocate at least 1 lane). */
    int lanes = n_threads > 0 ? n_threads : 1;
    size_t sums_sz = (size_t)lanes * K * D * sizeof(float);
    size_t counts_sz = (size_t)lanes * K * sizeof(int32_t);
    size_t inertias_sz = (size_t)lanes * sizeof(double);
    float* sums = (float*)malloc(sums_sz);
    int32_t* counts = (int32_t*)malloc(counts_sz);
    double* inertias = (double*)malloc(inertias_sz);
    /* cnorms2 reserved for future fast-distance path; allocated but unused. */
    float* cnorms2 = (float*)malloc((size_t)K * sizeof(float));
    if (!sums || !counts || !inertias || !cnorms2) {
        free(sums); free(counts); free(inertias); free(cnorms2);
        return INFINITY;
    }

    double prev_inertia = INFINITY;
    double inertia = INFINITY;
    for (int it = 0; it < max_iter; it++) {
        memset(sums, 0, sums_sz);
        memset(counts, 0, counts_sz);
        memset(inertias, 0, inertias_sz);

        pq_assign_ctx ctx = {
            .X = X, .D = D, .C = centers, .K = K, .cnorms2 = cnorms2,
            .indices = tmp_indices,
            .sums = sums, .counts = counts, .inertias = inertias,
            .n_threads = lanes,
        };
        if (pool) {
            ib_pool_run(pool, pq_assign_task, &ctx, N, 0);
        } else {
            pq_assign_task(&ctx, 0, 0, N);
        }

        /* Reduce across lanes and update centers. */
        inertia = 0.0;
        for (int l = 0; l < lanes; l++) inertia += inertias[l];

        float shift = 0.0f;
        for (int k = 0; k < K; k++) {
            int total_count = 0;
            float cs[64];  /* D <= 64 in practice (we use 2) */
            if (D > 64) {
                /* Fall back to malloc for unusually large D. */
                /* Not expected in PQ usage. */
                free(sums); free(counts); free(inertias); free(cnorms2);
                return INFINITY;
            }
            for (int d = 0; d < D; d++) cs[d] = 0.0f;
            for (int l = 0; l < lanes; l++) {
                total_count += counts[l * K + k];
                const float* lsum = sums + (size_t)l * K * D + (size_t)k * D;
                for (int d = 0; d < D; d++) cs[d] += lsum[d];
            }
            float* ck = centers + (size_t)k * D;
            if (total_count > 0) {
                float inv = 1.0f / (float)total_count;
                for (int d = 0; d < D; d++) {
                    float new_v = cs[d] * inv;
                    float dv = new_v - ck[d];
                    shift += dv * dv;
                    ck[d] = new_v;
                }
            }
            /* If a cluster went empty, leave its center at its previous
             * position. sklearn re-seeds empty clusters; we accept the
             * occasional duplicate centroid since k-means++ init makes
             * empties rare in practice. */
        }

        /* Convergence check. */
        if (shift < tol) break;
        if (it > 0 && fabs(prev_inertia - inertia) < (double)tol * (prev_inertia + 1e-12)) break;
        prev_inertia = inertia;
    }

    free(sums); free(counts); free(inertias); free(cnorms2);
    return inertia;
}

int ib_kmeans_fit(const float* X, int N,
                  const ib_kmeans_config* cfg,
                  float* centers_out,
                  int32_t* indices_out,
                  double* inertia_out) {
    if (!X || !cfg || !centers_out || N <= 0 || cfg->K <= 0 || cfg->D <= 0) return -1;

    int K = cfg->K;
    int D = cfg->D;
    int max_iter = cfg->max_iter > 0 ? cfg->max_iter : 20;
    float tol = cfg->tol > 0 ? cfg->tol : 1e-4f;
    int n_init = cfg->n_init > 0 ? cfg->n_init : 1;

    /* Optional subsample for fitting. */
    int M;            /* number of fitting points */
    const float* Xfit;
    float* Xfit_owned = NULL;
    int32_t* sub_idx = NULL;
    if (cfg->subsample > 0 && N > cfg->subsample) {
        M = cfg->subsample;
        Xfit_owned = (float*)malloc((size_t)M * D * sizeof(float));
        sub_idx = (int32_t*)malloc((size_t)M * sizeof(int32_t));
        if (!Xfit_owned || !sub_idx) {
            free(Xfit_owned); free(sub_idx); return -1;
        }
        pq_rng tmp_rng; pq_rng_init(&tmp_rng, cfg->seed ^ 0xa5a5a5a5u);
        pq_sample_indices(&tmp_rng, N, M, sub_idx);
        for (int i = 0; i < M; i++) {
            const float* src = X + (size_t)sub_idx[i] * D;
            float* dst = Xfit_owned + (size_t)i * D;
            memcpy(dst, src, (size_t)D * sizeof(float));
        }
        Xfit = Xfit_owned;
    } else {
        M = N;
        Xfit = X;
    }

    int n_threads = 1;
    if (cfg->pool) {
        /* We don't expose n_threads on the pool; assume the pool's
         * internal count by running tasks. The lanes count for our
         * accumulators just needs to be >= the pool's worker count.
         * Cap at 64 lanes — Apple Silicon, server CPUs all stay below. */
        n_threads = 64;
    }

    /* Best-of-n_init: keep best by inertia. */
    float* best_centers = (float*)malloc((size_t)K * D * sizeof(float));
    int32_t* tmp_indices = (int32_t*)malloc((size_t)M * sizeof(int32_t));
    if (!best_centers || !tmp_indices) {
        free(best_centers); free(tmp_indices);
        free(Xfit_owned); free(sub_idx); return -1;
    }
    double best_inertia = INFINITY;

    for (int init_i = 0; init_i < n_init; init_i++) {
        pq_rng rng;
        pq_rng_init(&rng, cfg->seed + (uint32_t)init_i * 1009u);
        double inertia = lloyd_one_init(Xfit, M, D, K,
                                         max_iter, tol,
                                         &rng, centers_out,
                                         tmp_indices,
                                         cfg->pool, n_threads);
        if (inertia < best_inertia) {
            best_inertia = inertia;
            memcpy(best_centers, centers_out, (size_t)K * D * sizeof(float));
        }
    }
    memcpy(centers_out, best_centers, (size_t)K * D * sizeof(float));
    free(best_centers);
    free(tmp_indices);

    if (inertia_out) *inertia_out = best_inertia;

    /* If indices_out requested, label the FULL X (not just the subsample). */
    if (indices_out) {
        ib_kmeans_assign(X, N, D, centers_out, K, indices_out, cfg->pool);
    }

    free(Xfit_owned); free(sub_idx);
    return 0;
}

/* ── assignment-only ───────────────────────────────────────────── */

typedef struct {
    const float* X;
    int D;
    const float* C;
    int K;
    int32_t* indices;
} pq_assign_only_ctx;

static void pq_assign_only_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    pq_assign_only_ctx* a = (pq_assign_only_ctx*)arg;
    int D = a->D, K = a->K;
    for (int i = start; i < end; i++) {
        const float* xi = a->X + (size_t)i * D;
        a->indices[i] = argmin_centers(xi, a->C, K, D, NULL, NULL);
    }
}

int ib_kmeans_assign(const float* X, int N, int D,
                     const float* centers, int K,
                     int32_t* indices_out,
                     ib_thread_pool* pool) {
    if (!X || !centers || !indices_out || N <= 0 || K <= 0 || D <= 0) return -1;
    pq_assign_only_ctx ctx = {
        .X = X, .D = D, .C = centers, .K = K, .indices = indices_out,
    };
    if (pool) {
        ib_pool_run(pool, pq_assign_only_task, &ctx, N, 0);
    } else {
        pq_assign_only_task(&ctx, 0, 0, N);
    }
    return 0;
}
