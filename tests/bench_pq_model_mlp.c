/*
 * bench_pq_model_mlp — stacked-FFN bench, simulating a full model's
 * MLP-only forward pass through N PQ-quantized layers.
 *
 * Usage:
 *   bench_pq_model_mlp <ffn_layer.ibf> <hidden_dim> <num_layers> [<repeats>]
 *
 * Loads a single FFN file (gate_proj/up_proj/down_proj) and reuses
 * those tensors `num_layers` times to simulate a stacked model. This
 * is an upper-bound / lower-bound bench (cache effects from real
 * per-layer-different weights would change the absolute number) but
 * gives a good first answer for "how many tok/s do we get on a model
 * of this shape, MLP-only, on this hardware?"
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

static inline float silu(float v) {
    return v / (1.0f + expf(-v));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: bench_pq_model_mlp <ffn.ibf> <hidden> <num_layers> [<repeats>]\n");
        return 2;
    }
    int hidden = atoi(argv[2]);
    int n_layers = atoi(argv[3]);
    int repeats = (argc >= 5) ? atoi(argv[4]) : 3;
    if (repeats < 1) repeats = 1;
    if (n_layers < 1) n_layers = 1;

    ib_pq_multi m = {0};
    if (ib_pq_load_multi(argv[1], &m) != 0) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    const ib_pq_tensor* gate = ib_pq_multi_find(&m, "gate_proj");
    const ib_pq_tensor* up   = ib_pq_multi_find(&m, "up_proj");
    const ib_pq_tensor* down = ib_pq_multi_find(&m, "down_proj");
    if (!gate || !up || !down) {
        fprintf(stderr, "need gate_proj/up_proj/down_proj\n"); return 1;
    }
    if (gate->N != hidden || down->M != hidden) {
        fprintf(stderr, "hidden=%d but gate->N=%d down->M=%d\n",
                hidden, gate->N, down->M);
        return 1;
    }
    int N_mid = gate->M;
    fprintf(stdout, "model-mlp: hidden=%d intermediate=%d n_layers=%d\n",
            hidden, N_mid, n_layers);

    /* Persistent buffers */
    float* x = (float*)malloc((size_t)hidden * sizeof(float));
    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)hidden * sizeof(float));
    if (!x || !g || !u || !h || !y) return 1;
    for (int j = 0; j < hidden; j++)
        x[j] = ((j * 2654435761u) % 1024 - 512) / 512.0f;

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
                    ib_pq_matmul_fp32_threaded(gate, x, g, pool);
                    ib_pq_matmul_fp32_threaded(up,   x, u, pool);
                } else {
                    ib_pq_matmul_fp32(gate, x, g);
                    ib_pq_matmul_fp32(up,   x, u);
                }
                for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
                if (pool) ib_pq_matmul_fp32_threaded(down, h, y, pool);
                else      ib_pq_matmul_fp32(down, h, y);
                /* Residual + tiny non-linear so output drifts: */
                for (int j = 0; j < hidden; j++) x[j] = 0.9f * x[j] + 0.1f * y[j];
            }
            double d = now_s() - s;
            if (d < best) best = d;
        }
        double tok_s = 1.0 / best;
        fprintf(stdout, "  P=%d:  %7.1f ms / token (MLP-only)  =>  %6.2f tok/s\n",
                P, best * 1e3, tok_s);
        if (pool) ib_pool_destroy(pool);
    }

    free(x); free(g); free(u); free(h); free(y);
    ib_pq_multi_free(&m);
    return 0;
}
