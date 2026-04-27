/*
 * bench_pq_ffn — SiLU-gated MLP forward through three PQ tensors.
 *
 * Computes y = down_proj(silu(gate_proj(x)) * up_proj(x)).
 *
 * Usage:
 *   bench_pq_ffn <ffn.ibf> <x.bin> <y.bin> [<repeats>]
 *
 * The IBF v5 file must contain three named tensors: "gate_proj",
 * "up_proj", "down_proj". x.bin is FP32 [N_in], y.bin will be FP32
 * [N_in]. Reports min wall over `repeats` and the equivalent GFLOP/s.
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
        fprintf(stderr,
            "usage: bench_pq_ffn <ffn.ibf> <x.bin> <y.bin> [<repeats>]\n");
        return 2;
    }
    int repeats = (argc >= 5) ? atoi(argv[4]) : 3;
    if (repeats < 1) repeats = 1;

    ib_pq_multi m = {0};
    if (ib_pq_load_multi(argv[1], &m) != 0) {
        fprintf(stderr, "load failed\n"); return 1;
    }
    const ib_pq_tensor* gate = ib_pq_multi_find(&m, "gate_proj");
    const ib_pq_tensor* up   = ib_pq_multi_find(&m, "up_proj");
    const ib_pq_tensor* down = ib_pq_multi_find(&m, "down_proj");
    if (!gate || !up || !down) {
        fprintf(stderr, "missing one of gate_proj/up_proj/down_proj\n"); return 1;
    }
    int N_in  = gate->N;
    int N_mid = gate->M;
    if (up->N != N_in || up->M != N_mid) {
        fprintf(stderr, "up_proj shape mismatch (got %dx%d, expected %dx%d)\n",
                up->M, up->N, N_mid, N_in); return 1;
    }
    if (down->N != N_mid || down->M != N_in) {
        fprintf(stderr, "down_proj shape mismatch\n"); return 1;
    }

    /* Read x */
    FILE* xf = fopen(argv[2], "rb");
    if (!xf) { perror("fopen x"); return 1; }
    float* x = (float*)malloc((size_t)N_in * sizeof(float));
    if (!x) return 1;
    if (fread(x, sizeof(float), (size_t)N_in, xf) != (size_t)N_in) {
        fprintf(stderr, "x short\n"); fclose(xf); return 1;
    }
    fclose(xf);

    float* g = (float*)malloc((size_t)N_mid * sizeof(float));
    float* u = (float*)malloc((size_t)N_mid * sizeof(float));
    float* h = (float*)malloc((size_t)N_mid * sizeof(float));
    float* y = (float*)malloc((size_t)N_in  * sizeof(float));
    if (!g || !u || !h || !y) return 1;

    double t_min = 1e9;
    for (int r = 0; r < repeats; r++) {
        double s = now_s();
        if (ib_pq_matmul_fp32(gate, x, g) != 0) return 1;
        if (ib_pq_matmul_fp32(up,   x, u) != 0) return 1;
        for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
        if (ib_pq_matmul_fp32(down, h, y) != 0) return 1;
        double d = now_s() - s;
        if (d < t_min) t_min = d;
    }

    double flops = 2.0 * ((double)N_mid * N_in * 2.0 + (double)N_in * N_mid);
    fprintf(stdout, "FFN (1 thread):  hidden=%d intermediate=%d  %7.3f ms  %6.2f GFLOP/s\n",
            N_in, N_mid, t_min * 1e3, flops / 1e9 / t_min);

    /* Threaded variant (4 and 8 workers if available) */
    int n_thread_configs[] = {4, 8};
    for (int p_idx = 0; p_idx < 2; p_idx++) {
        int P = n_thread_configs[p_idx];
        ib_thread_pool* pool = ib_pool_create(P);
        if (!pool) continue;
        double t_th = 1e9;
        for (int r = 0; r < repeats; r++) {
            double s = now_s();
            if (ib_pq_matmul_fp32_threaded(gate, x, g, pool) != 0) return 1;
            if (ib_pq_matmul_fp32_threaded(up,   x, u, pool) != 0) return 1;
            for (int i = 0; i < N_mid; i++) h[i] = silu(g[i]) * u[i];
            if (ib_pq_matmul_fp32_threaded(down, h, y, pool) != 0) return 1;
            double d = now_s() - s;
            if (d < t_th) t_th = d;
        }
        fprintf(stdout, "FFN (%d threads):                      %7.3f ms  %6.2f GFLOP/s  (%.2fx vs 1t)\n",
                P, t_th * 1e3, flops / 1e9 / t_th, t_min / t_th);
        ib_pool_destroy(pool);
    }

    /* Write y */
    FILE* of = fopen(argv[3], "wb");
    if (!of) { perror("fopen y"); return 1; }
    int32_t Mout = N_in;
    fwrite(&Mout, sizeof(int32_t), 1, of);
    fwrite(y, sizeof(float), (size_t)N_in, of);
    fclose(of);

    free(x); free(g); free(u); free(h); free(y);
    ib_pq_multi_free(&m);
    return 0;
}
