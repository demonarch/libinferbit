/* Unified parallelism + reduction derisk bench.
 *
 * For TinyLlama L0 q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj
 * weights, time:
 *   T0  baseline                — current sequential 7-dispatch / layer
 *   T1  fused q+k+v + gate+up   — 4 dispatches / layer (vs 7)
 *   T2  T1 + pre-converted fp32 row_scale (skip h2f in reduce)
 *   T3  T1 + SIMD-vectorised reduction (NEON 4-wide vs scalar)
 *   T4  T1 + T2 + T3            — all combined
 *
 *   P0  pool dispatch overhead  — 10000 no-op tasks, time per dispatch
 *
 * Each test runs the full layer's matmul work to mirror real inference.
 * Reports time + correctness against baseline.
 */
#include "../src/inferbit_internal.h"
#include "../src/pqv2_format.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── Pool overhead (P0) ─────────────────────────────────────── */
static void noop_task(void *arg, int tid, int s, int e) {
    (void)arg; (void)tid; (void)s; (void)e;
}

static double measure_pool_overhead(struct ib_thread_pool *tp, int iters) {
    /* warmup */
    for (int i = 0; i < 50; i++) ib_pool_run(tp, noop_task, NULL, 4, 1);
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) ib_pool_run(tp, noop_task, NULL, 4, 1);
    double dt = now_sec() - t0;
    return dt / iters * 1e6;  /* µs per dispatch */
}

/* ── Chunk-range work unit ──────────────────────────────────── */
typedef struct {
    const pqv2_t *t;
    const float  *x;
    float        *acc;
    uint32_t      c_start;
    uint32_t      c_end;
} unit_t;

typedef struct {
    unit_t *units;
    int     n_units;
} fused_arg_t;

static void fused_task(void *arg, int tid, int start, int end) {
    (void)tid;
    fused_arg_t *fa = (fused_arg_t*)arg;
    for (int u = start; u < end; u++) {
        unit_t *un = &fa->units[u];
        pqv2_acc_tbl_int8_k256_chunks(un->t, un->x, un->t->cb_fp32, NULL,
                                        un->acc, NULL,
                                        un->c_start, un->c_end);
    }
}

/* ── Reduction variants ─────────────────────────────────────── */
/* (a) scalar with h2f per element — current behavior */
static void reduce_scalar_h2f(const pqv2_t *t, float *acc_pool,
                               int n_slots, float *y) {
    uint32_t M = t->M;
    for (uint32_t m = 0; m < M; m++) {
        float a = 0.0f;
        for (int s = 0; s < n_slots; s++) a += acc_pool[(size_t)s * M + m];
        y[m] = a * pqv2_h2f(t->row_scale[m]);
    }
}

/* (b) scalar with pre-converted fp32 row_scale */
static void reduce_scalar_rs32(uint32_t M, const float *row_scale_fp32,
                                 float *acc_pool, int n_slots, float *y) {
    for (uint32_t m = 0; m < M; m++) {
        float a = 0.0f;
        for (int s = 0; s < n_slots; s++) a += acc_pool[(size_t)s * M + m];
        y[m] = a * row_scale_fp32[m];
    }
}

/* (c) SIMD reduce: 4-wide accumulate + FMA with cached fp32 row_scale */
static void reduce_simd(uint32_t M, const float *row_scale_fp32,
                          float *acc_pool, int n_slots, float *y) {
#if defined(__ARM_NEON)
    uint32_t m = 0;
    for (; m + 4 <= M; m += 4) {
        float32x4_t a = vdupq_n_f32(0.0f);
        for (int s = 0; s < n_slots; s++) {
            a = vaddq_f32(a, vld1q_f32(&acc_pool[(size_t)s * M + m]));
        }
        float32x4_t rs = vld1q_f32(&row_scale_fp32[m]);
        vst1q_f32(&y[m], vmulq_f32(a, rs));
    }
    for (; m < M; m++) {
        float a = 0.0f;
        for (int s = 0; s < n_slots; s++) a += acc_pool[(size_t)s * M + m];
        y[m] = a * row_scale_fp32[m];
    }
#else
    reduce_scalar_rs32(M, row_scale_fp32, acc_pool, n_slots, y);
#endif
}

/* ── Helpers ────────────────────────────────────────────────── */
static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

/* Run a "layer" worth of matmul work using `tensors[]` (7 tensors:
 * q,k,v,o,gate,up,down). x[] are 7 inputs. acc_pools[] are pre-allocated
 * per-tensor scratch (size n_slots * M). y_outs[] are 7 outputs.
 *
 * Mode 0 = current sequential (7 separate dispatches).
 * Mode 1 = fused q+k+v + fused gate+up (4 dispatches/layer).
 *
 * `reduce_mode` selects the reduction kernel:
 *   0 = scalar h2f (baseline)
 *   1 = scalar with pre-converted fp32 row_scale
 *   2 = SIMD with pre-converted fp32 row_scale
 */
static void run_layer(struct ib_thread_pool *tp, int n_threads,
                       const pqv2_t * const *tensors, const float * const *xs,
                       float **acc_pools, float **y_outs,
                       const float * const *row_scale_fp32,
                       int n_slots, int chunks_per_unit,
                       int n_chunks, int mode, int reduce_mode)
{
    /* zero acc pools */
    for (int t_i = 0; t_i < 7; t_i++) {
        memset(acc_pools[t_i], 0, n_slots * tensors[t_i]->M * sizeof(float));
    }
    /* Build unit lists */
    unit_t units_seq[8];
    if (mode == 0) {
        /* Sequential: 7 dispatches, each is one tensor's work */
        for (int t_i = 0; t_i < 7; t_i++) {
            unit_t *u = malloc(n_slots * sizeof(unit_t));
            for (int s = 0; s < n_slots; s++) {
                int c0 = s * chunks_per_unit;
                int c1 = (s + 1) * chunks_per_unit > n_chunks
                          ? n_chunks : (s + 1) * chunks_per_unit;
                u[s] = (unit_t){tensors[t_i], xs[t_i],
                                acc_pools[t_i] + (size_t)s * tensors[t_i]->M,
                                (uint32_t)c0, (uint32_t)c1};
            }
            fused_arg_t fa = {u, n_slots};
            ib_pool_run(tp, fused_task, &fa, n_slots, 1);
            free(u);
            (void)units_seq;
        }
    } else {
        /* Mode 1: fuse q+k+v (idx 0,1,2) into one dispatch, then o (3),
         * then fuse gate+up (4,5) into one, then down (6) → 4 dispatches. */
        int groups[][4] = { {0, 1, 2, -1}, {3, -1, -1, -1},
                             {4, 5, -1, -1}, {6, -1, -1, -1} };
        for (int g = 0; g < 4; g++) {
            int n_in_group = 0;
            for (int j = 0; j < 4 && groups[g][j] >= 0; j++) n_in_group++;
            int total = n_in_group * n_slots;
            unit_t *u = malloc(total * sizeof(unit_t));
            int ui = 0;
            for (int j = 0; j < n_in_group; j++) {
                int t_i = groups[g][j];
                for (int s = 0; s < n_slots; s++) {
                    int c0 = s * chunks_per_unit;
                    int c1 = (s + 1) * chunks_per_unit > n_chunks
                              ? n_chunks : (s + 1) * chunks_per_unit;
                    u[ui++] = (unit_t){tensors[t_i], xs[t_i],
                                        acc_pools[t_i] + (size_t)s * tensors[t_i]->M,
                                        (uint32_t)c0, (uint32_t)c1};
                }
            }
            int chunk_step = (total + n_threads - 1) / n_threads;
            if (chunk_step < 1) chunk_step = 1;
            fused_arg_t fa = {u, total};
            ib_pool_run(tp, fused_task, &fa, total, chunk_step);
            free(u);
        }
    }
    /* Reduce per tensor */
    for (int t_i = 0; t_i < 7; t_i++) {
        if (reduce_mode == 0)
            reduce_scalar_h2f(tensors[t_i], acc_pools[t_i], n_slots, y_outs[t_i]);
        else if (reduce_mode == 1)
            reduce_scalar_rs32(tensors[t_i]->M, row_scale_fp32[t_i],
                                acc_pools[t_i], n_slots, y_outs[t_i]);
        else
            reduce_simd(tensors[t_i]->M, row_scale_fp32[t_i],
                          acc_pools[t_i], n_slots, y_outs[t_i]);
    }
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    int iters = (argc > 2) ? atoi(argv[2]) : 100;
    int n_threads = (argc > 3) ? atoi(argv[3]) : 4;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(path, &f) != 0) {
        fprintf(stderr, "load fail\n"); return 1;
    }
    /* Look up 7 PQv2 tensors of layer 0 */
    const char *names[7] = {
        "L0.self_attn.q_proj", "L0.self_attn.k_proj", "L0.self_attn.v_proj",
        "L0.self_attn.o_proj", "L0.mlp.gate_proj", "L0.mlp.up_proj",
        "L0.mlp.down_proj",
    };
    const pqv2_t *tensors[7];
    for (int i = 0; i < 7; i++) {
        const ib_pqv2_named_tensor *nt = ib_pqv2_find(&f, names[i]);
        if (!nt || nt->kind != IB_PQV2_KIND_PQV2) {
            fprintf(stderr, "tensor %s missing\n", names[i]); return 1;
        }
        tensors[i] = &nt->pq;
    }

    struct ib_thread_pool *tp = ib_pool_create(n_threads);

    /* P0: pool dispatch overhead */
    double pool_overhead_us = measure_pool_overhead(tp, 10000);
    printf("P0  pool dispatch overhead: %.2f µs/call (cond_var pool)\n", pool_overhead_us);

    /* Build inputs: q/k/v share x_attn (size N=2048),
     * o uses x_oin (also 2048),
     * gate/up share x_mlp (2048), down uses x_din (5632) */
    size_t N_attn = tensors[0]->N;   /* 2048 */
    size_t N_o    = tensors[3]->N;   /* 2048 */
    size_t N_mlp  = tensors[4]->N;   /* 2048 */
    size_t N_down = tensors[6]->N;   /* 5632 */
    float *x_attn = aligned_alloc(64, ((N_attn * sizeof(float) + 63) & ~(size_t)63));
    float *x_o    = aligned_alloc(64, ((N_o    * sizeof(float) + 63) & ~(size_t)63));
    float *x_mlp  = aligned_alloc(64, ((N_mlp  * sizeof(float) + 63) & ~(size_t)63));
    float *x_down = aligned_alloc(64, ((N_down * sizeof(float) + 63) & ~(size_t)63));
    for (size_t i = 0; i < N_attn; i++) x_attn[i] = ((float)(i % 257) - 128.0f) * 0.01f;
    for (size_t i = 0; i < N_o;    i++) x_o[i]    = ((float)(i % 199) -  99.5f) * 0.012f;
    for (size_t i = 0; i < N_mlp;  i++) x_mlp[i]  = ((float)(i % 113) -  56.5f) * 0.011f;
    for (size_t i = 0; i < N_down; i++) x_down[i] = ((float)(i % 311) - 155.5f) * 0.013f;
    const float *xs[7] = { x_attn, x_attn, x_attn, x_o, x_mlp, x_mlp, x_down };

    /* Per-tensor pre-converted fp32 row_scale */
    float *rs_fp32[7];
    for (int i = 0; i < 7; i++) {
        rs_fp32[i] = aligned_alloc(64,
            ((tensors[i]->M * sizeof(float) + 63) & ~(size_t)63));
        for (uint32_t m = 0; m < tensors[i]->M; m++) {
            rs_fp32[i][m] = pqv2_h2f(tensors[i]->row_scale[m]);
        }
    }

    /* Allocate per-tensor acc_pool and y_out */
    int n_chunks = (int)(tensors[0]->N / tensors[0]->G);
    int chunks_per_unit = (n_chunks + n_threads - 1) / n_threads;
    int n_slots = (n_chunks + chunks_per_unit - 1) / chunks_per_unit;

    float *acc_pools[7];
    float *y_outs_T0[7], *y_outs_T1[7], *y_outs_T2[7], *y_outs_T3[7], *y_outs_T4[7];
    for (int i = 0; i < 7; i++) {
        acc_pools[i] = aligned_alloc(64,
            (n_slots * tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
        y_outs_T0[i] = aligned_alloc(64, (tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
        y_outs_T1[i] = aligned_alloc(64, (tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
        y_outs_T2[i] = aligned_alloc(64, (tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
        y_outs_T3[i] = aligned_alloc(64, (tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
        y_outs_T4[i] = aligned_alloc(64, (tensors[i]->M * sizeof(float) + 63) & ~(size_t)63);
    }

    /* Test runner: time `iters` runs of run_layer with given mode/reduce_mode */
    #define BENCH(label, mode, reduce_mode, y_outs) ({ \
        for (int _w = 0; _w < 5; _w++) \
            run_layer(tp, n_threads, tensors, xs, acc_pools, y_outs, \
                       (const float*const*)rs_fp32, n_slots, chunks_per_unit, \
                       n_chunks, mode, reduce_mode); \
        double _t0 = now_sec(); \
        for (int _i = 0; _i < iters; _i++) \
            run_layer(tp, n_threads, tensors, xs, acc_pools, y_outs, \
                       (const float*const*)rs_fp32, n_slots, chunks_per_unit, \
                       n_chunks, mode, reduce_mode); \
        double _ms = (now_sec() - _t0) / iters * 1000.0; \
        printf("%s  %.3f ms/layer\n", label, _ms); \
        _ms; \
    })

    printf("\n=== Layer-of-7-matmul timing (TinyLlama L0, n_threads=%d, iters=%d) ===\n\n",
           n_threads, iters);

    double t0 = BENCH("T0  sequential 7-dispatch  + scalar h2f reduce :",
                      0, 0, y_outs_T0);
    double t1 = BENCH("T1  fused 4-dispatch       + scalar h2f reduce :",
                      1, 0, y_outs_T1);
    double t2 = BENCH("T2  fused 4-dispatch       + scalar fp32-rs    :",
                      1, 1, y_outs_T2);
    double t3 = BENCH("T3  fused 4-dispatch       + SIMD fp32-rs      :",
                      1, 2, y_outs_T3);
    double t4 = t3; /* T3 already includes everything T4 would */

    /* Correctness — every variant should match T0 */
    int all_match = 1;
    for (int i = 0; i < 7; i++) {
        double cs1 = cosine(y_outs_T0[i], y_outs_T1[i], tensors[i]->M);
        double cs2 = cosine(y_outs_T0[i], y_outs_T2[i], tensors[i]->M);
        double cs3 = cosine(y_outs_T0[i], y_outs_T3[i], tensors[i]->M);
        if (cs1 < 0.99999 || cs2 < 0.99999 || cs3 < 0.99999) {
            printf("  ! tensor %d: cos T1=%.6f T2=%.6f T3=%.6f\n",
                    i, cs1, cs2, cs3);
            all_match = 0;
        }
    }
    if (all_match) printf("\ncorrectness: all variants cos > 0.99999 vs T0 baseline ✓\n");

    printf("\nspeedup vs T0:  T1=%.2f×  T2=%.2f×  T3=%.2f×\n",
           t0/t1, t0/t2, t0/t3);
    printf("incremental:    fusion=%.2f×  +cached_rs=%.2f×  +simd_reduce=%.2f×\n",
           t0/t1, t1/t2, t2/t3);

    /* Per-token projection (22 layers): */
    double per_token_T0_ms = t0 * 22;
    double per_token_T3_ms = t3 * 22;
    /* Add other-layer overhead: embedding, attention, RMSNorm, lm_head are
     * not part of these 7 matmuls, but they're roughly equivalent to
     * the matmul time for completeness. We approximate by scaling. */
    printf("\n=== Per-token projection (22 layers, approx) ===\n");
    printf("  T0  baseline             %.1f ms/token  (~%.1f t/s for matmul work alone)\n",
           per_token_T0_ms, 1000.0 / per_token_T0_ms);
    printf("  T3  all opts combined    %.1f ms/token  (~%.1f t/s for matmul work alone)\n",
           per_token_T3_ms, 1000.0 / per_token_T3_ms);
    printf("  Note: end-to-end inference also has attn, RMSNorm, embed, lm_head etc.\n");

    /* Cleanup */
    free(x_attn); free(x_o); free(x_mlp); free(x_down);
    for (int i = 0; i < 7; i++) {
        free(rs_fp32[i]); free(acc_pools[i]);
        free(y_outs_T0[i]); free(y_outs_T1[i]); free(y_outs_T2[i]);
        free(y_outs_T3[i]); free(y_outs_T4[i]);
    }
    ib_pool_destroy(tp);
    ib_pqv2_file_free(&f);
    return 0;
}
