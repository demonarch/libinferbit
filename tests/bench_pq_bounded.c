/*
 * bench_pq_bounded — bounded-resident-set streaming inference (#9).
 *
 * Walks the layers of a multi-tensor v5 file running SiLU-gated FFN
 * forward, with an explicit "ring" of K resident layers. Each step:
 *
 *   1. willneed(layer L+1)     // start the I/O for the next layer
 *   2. compute(layer L)
 *   3. dontneed(layer L-K)     // release the layer that fell out of the ring
 *
 * The result: at most K layers' weights are advised resident at any
 * time. On a model that doesn't fit RAM (frontier-scale on consumer
 * hardware), this avoids swap thrashing and gives predictable
 * throughput. On a model that does fit, the OS keeps pages cached
 * anyway and we should see no throughput regression.
 *
 * Bench reports per-K wall and peak RSS so we can verify both:
 *   (a) resident-set is actually bounded
 *   (b) wall time doesn't regress compared to unbounded
 *
 * Usage: bench_pq_bounded <model.ibf> <K> [<repeats>]
 *   K = 0 means unbounded (no eviction, no prefetch — the current default)
 *   K > 0 means keep at most K layers' weights resident
 */
#include "pq_decode.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/resource.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static long maxrss_kib(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}
static inline float silu(float v) { return v / (1.0f + expf(-v)); }

typedef struct { const ib_pq_tensor *gate, *up, *down; } triple;

static void advise_layer(const triple* layer, int dontneed) {
    const ib_pq_tensor* ts[3] = {layer->gate, layer->up, layer->down};
    if (dontneed) ib_pq_advise_dontneed_n(ts, 3);
    else          ib_pq_advise_willneed_n(ts, 3);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: bench_pq_bounded <model.ibf> <K> [<repeats>]\n"
                        "  K=0 -> unbounded; K>0 -> evict beyond the K-layer ring\n");
        return 2;
    }
    int K = atoi(argv[2]);
    int repeats = (argc >= 4) ? atoi(argv[3]) : 3;

    ib_pq_multi m = {0};
    if (ib_pq_open_mmap(argv[1], &m) != 0) {
        fprintf(stderr, "open_mmap failed (file may not be v5)\n");
        return 1;
    }
    int n_layers = 0;
    for (int i = 0; i < m.n; i++)
        if (m.names[i] && m.names[i][0] == 'L') {
            int n = atoi(m.names[i] + 1);
            if (n + 1 > n_layers) n_layers = n + 1;
        }
    triple* layers = (triple*)calloc((size_t)n_layers, sizeof(triple));
    for (int L = 0; L < n_layers; L++) {
        char nm[64];
        snprintf(nm, sizeof(nm), "L%d_gate_proj", L); layers[L].gate = ib_pq_multi_find(&m, nm);
        snprintf(nm, sizeof(nm), "L%d_up_proj",   L); layers[L].up   = ib_pq_multi_find(&m, nm);
        snprintf(nm, sizeof(nm), "L%d_down_proj", L); layers[L].down = ib_pq_multi_find(&m, nm);
        if (!layers[L].gate || !layers[L].up || !layers[L].down) {
            fprintf(stderr, "missing tensors for layer %d\n", L); return 1;
        }
    }
    int hidden = layers[0].gate->N, N_mid = layers[0].gate->M;
    fprintf(stdout, "model: hidden=%d intermediate=%d n_layers=%d K=%d\n",
            hidden, N_mid, n_layers, K);

    /* Warm cache: touch every page once so the first run doesn't pay
     * cold-fault costs (we want to measure the streaming overhead, not
     * the cold-load cost which is well understood from bench_pq_streaming). */
    {
        for (int L = 0; L < n_layers; L++) advise_layer(&layers[L], 0);
        for (volatile int i = 0; i < 1; i++) {}  /* compiler barrier */
    }

    long rss_baseline = maxrss_kib();
    fprintf(stdout, "RSS baseline (after warm-up): %ld KiB\n", rss_baseline);

    ib_thread_pool* pool = ib_pool_create(8);
    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));

    /* If K > 0, pre-evict everything except the first K layers. This
     * sets the "tight RAM" baseline for the bench. */
    if (K > 0 && K < n_layers) {
        for (int L = K; L < n_layers; L++) advise_layer(&layers[L], 1);
        long rss_after_evict = maxrss_kib();
        fprintf(stdout, "RSS after pre-evict (keeping %d layers): %ld KiB\n",
                K, rss_after_evict);
    }

    double best = 1e9;
    long rss_peak = rss_baseline;
    for (int rr = 0; rr < repeats; rr++) {
        for (int j = 0; j < hidden; j++)
            x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;
        double s = now_s();
        for (int L = 0; L < n_layers; L++) {
            /* Prefetch L+1 while we compute L */
            if (K > 0 && L + 1 < n_layers) advise_layer(&layers[L + 1], 0);

            ib_pq_matmul_fp32_threaded(layers[L].gate, x, g, pool);
            ib_pq_matmul_fp32_threaded(layers[L].up,   x, u, pool);
            for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
            ib_pq_matmul_fp32_threaded(layers[L].down, h, y, pool);
            for (int j = 0; j < hidden; j++) x[j] = 0.9f * x[j] + 0.1f * y[j];

            /* Evict the layer that fell out of the ring */
            if (K > 0 && L - K >= 0) advise_layer(&layers[L - K], 1);
        }
        double d = now_s() - s;
        if (d < best) best = d;
        long r = maxrss_kib();
        if (r > rss_peak) rss_peak = r;
    }
    fprintf(stdout, "best wall:    %.1f ms / token   %.2f tok/s   %.2f ms/layer\n",
            best * 1e3, 1.0 / best, best * 1e3 / n_layers);
    fprintf(stdout, "RSS peak:     %ld KiB   (delta vs baseline: %+ld KiB)\n",
            rss_peak, rss_peak - rss_baseline);

    ib_pool_destroy(pool);
    free(x); free(g); free(u); free(h); free(y); free(layers);
    ib_pq_multi_free(&m);
    return 0;
}
