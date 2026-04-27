/*
 * bench_pq_real_model — full-model MLP forward through real PQ tensors.
 *
 * Loads a multi-tensor IBF v5 file produced by
 * scripts/convert/full_llama_mlp_to_v5.py (tensor names of the form
 * "L<i>_gate_proj" / "L<i>_up_proj" / "L<i>_down_proj"). Walks all
 * layers in order, running the SiLU-gated FFN forward through the
 * real PQ-quantised weights. Reports per-layer and per-token
 * latencies at 1 / 4 / 8 threads.
 *
 * usage: bench_pq_real_model <model.ibf> [<repeats>]
 */
#include "pq_decode.h"
#include "inferbit_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

typedef struct {
    const ib_pq_tensor* gate;
    const ib_pq_tensor* up;
    const ib_pq_tensor* down;
} layer_triple;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench_pq_real_model <model.ibf> [<repeats>]\n");
        return 2;
    }
    int repeats = (argc >= 3) ? atoi(argv[2]) : 3;
    if (repeats < 1) repeats = 1;

    ib_pq_multi m = {0};
    if (ib_pq_load_multi(argv[1], &m) != 0) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    fprintf(stdout, "loaded %d tensors from %s\n", m.n, argv[1]);

    /* Discover layers. Tensor names follow "L<i>_<key>". */
    int max_layer = -1;
    for (int i = 0; i < m.n; i++) {
        if (m.names[i] && m.names[i][0] == 'L') {
            int n = atoi(m.names[i] + 1);
            if (n > max_layer) max_layer = n;
        }
    }
    int n_layers = max_layer + 1;
    if (n_layers <= 0) {
        fprintf(stderr, "no L<i>_* tensors found\n"); return 1;
    }

    layer_triple* layers = (layer_triple*)calloc((size_t)n_layers, sizeof(layer_triple));
    if (!layers) return 1;
    for (int L = 0; L < n_layers; L++) {
        char name[64];
        snprintf(name, sizeof(name), "L%d_gate_proj", L); layers[L].gate = ib_pq_multi_find(&m, name);
        snprintf(name, sizeof(name), "L%d_up_proj",   L); layers[L].up   = ib_pq_multi_find(&m, name);
        snprintf(name, sizeof(name), "L%d_down_proj", L); layers[L].down = ib_pq_multi_find(&m, name);
        if (!layers[L].gate || !layers[L].up || !layers[L].down) {
            fprintf(stderr, "missing tensor for layer %d\n", L); return 1;
        }
    }
    int hidden = layers[0].gate->N;
    int N_mid  = layers[0].gate->M;
    fprintf(stdout, "model: hidden=%d intermediate=%d n_layers=%d\n",
            hidden, N_mid, n_layers);

    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));
    if (!x || !g || !u || !h || !y) return 1;

    int n_thread_configs[] = {1, 4, 8};
    for (int p_idx = 0; p_idx < 3; p_idx++) {
        int P = n_thread_configs[p_idx];
        ib_thread_pool* pool = (P == 1) ? NULL : ib_pool_create(P);
        if (P > 1 && !pool) continue;

        double best = 1e9;
        for (int rr = 0; rr < repeats; rr++) {
            for (int j = 0; j < hidden; j++)
                x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;
            double s = now_s();
            for (int L = 0; L < n_layers; L++) {
                if (pool) {
                    ib_pq_matmul_fp32_threaded(layers[L].gate, x, g, pool);
                    ib_pq_matmul_fp32_threaded(layers[L].up,   x, u, pool);
                } else {
                    ib_pq_matmul_fp32(layers[L].gate, x, g);
                    ib_pq_matmul_fp32(layers[L].up,   x, u);
                }
                for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
                if (pool) ib_pq_matmul_fp32_threaded(layers[L].down, h, y, pool);
                else      ib_pq_matmul_fp32(layers[L].down, h, y);
                for (int j = 0; j < hidden; j++) x[j] = 0.9f * x[j] + 0.1f * y[j];
            }
            double d = now_s() - s;
            if (d < best) best = d;
        }
        double per_layer = best / (double)n_layers;
        fprintf(stdout, "  P=%d:  %7.1f ms / token   %.2f tok/s   (%.2f ms / layer)\n",
                P, best * 1e3, 1.0 / best, per_layer * 1e3);
        if (pool) ib_pool_destroy(pool);
    }

    free(x); free(g); free(u); free(h); free(y); free(layers);
    ib_pq_multi_free(&m);
    return 0;
}
