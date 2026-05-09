/* Derisk: q/k/v matmul fusion vs sequential. Same TinyLlama L0
 * q_proj/k_proj/v_proj weights, same input x. Time:
 *   (A) sequential = 3× threaded matvec (current behavior)
 *   (B) fused      = 1× pool dispatch with task-list spanning all 3
 * Compare wall-clock + correctness.
 *
 * If fusion saves >= 10%, the parallelism redesign is worth pursuing.
 */
#include "../src/inferbit_internal.h"
#include "../src/pqv2_format.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* One unit of matvec work: process chunk range [c_start, c_end) of
 * tensor `t`, accumulating into a pre-zeroed acc[M]. */
typedef struct {
    const pqv2_t *t;
    const float  *x;
    float        *acc;     /* pre-zeroed, sized M floats */
    uint32_t      c_start;
    uint32_t      c_end;
} unit_t;

/* Each task entry is one unit. */
typedef struct {
    unit_t *units;
    int     n_units;
} fused_arg_t;

/* The pool task picks units based on its assigned slice of [0, n_units). */
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

/* Run one pool dispatch over a list of units, then reduce per-tensor. */
static void run_fused(struct ib_thread_pool *tp, int n_threads,
                       unit_t *units, int n_units) {
    fused_arg_t fa = { units, n_units };
    int chunk = (n_units + n_threads - 1) / n_threads;
    if (chunk < 1) chunk = 1;
    ib_pool_run(tp, fused_task, &fa, n_units, chunk);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    int iters = (argc > 2) ? atoi(argv[2]) : 200;
    int n_threads = (argc > 3) ? atoi(argv[3]) : 4;

    ib_pqv2_file f = {0};
    if (ib_pqv2_file_load(path, &f) != 0) { fprintf(stderr, "load fail\n"); return 1; }
    const pqv2_t *t_q = &ib_pqv2_find(&f, "L0.self_attn.q_proj")->pq;
    const pqv2_t *t_k = &ib_pqv2_find(&f, "L0.self_attn.k_proj")->pq;
    const pqv2_t *t_v = &ib_pqv2_find(&f, "L0.self_attn.v_proj")->pq;
    if (!t_q || !t_k || !t_v) { fprintf(stderr, "tensors missing\n"); return 1; }
    printf("q: %ux%u  k: %ux%u  v: %ux%u\n",
           t_q->M, t_q->N, t_k->M, t_k->N, t_v->M, t_v->N);

    struct ib_thread_pool *tp = ib_pool_create(n_threads);
    size_t N = t_q->N;
    float *x = aligned_alloc(64, ((N * sizeof(float) + 63) & ~(size_t)63));
    for (size_t i = 0; i < N; i++) x[i] = ((float)(i % 257) - 128.0f) * 0.01f;

    /* Per-tensor outputs */
    float *y_q_seq = aligned_alloc(64, t_q->M * sizeof(float));
    float *y_k_seq = aligned_alloc(64, t_k->M * sizeof(float));
    float *y_v_seq = aligned_alloc(64, t_v->M * sizeof(float));
    float *y_q_fus = aligned_alloc(64, t_q->M * sizeof(float));
    float *y_k_fus = aligned_alloc(64, t_k->M * sizeof(float));
    float *y_v_fus = aligned_alloc(64, t_v->M * sizeof(float));

    /* Per-thread acc pools, one per tensor — pre-allocated. */
    int n_chunks = (int)(t_q->N / t_q->G);
    int chunks_per_unit = (n_chunks + n_threads - 1) / n_threads;
    /* Each tensor needs n_threads slots of size M floats. */
    float *acc_q = aligned_alloc(64, n_threads * t_q->M * sizeof(float));
    float *acc_k = aligned_alloc(64, n_threads * t_k->M * sizeof(float));
    float *acc_v = aligned_alloc(64, n_threads * t_v->M * sizeof(float));

    /* Build one unit list for the fused path: n_threads units per tensor,
     * each covering a chunk slice. Total units = 3 * n_threads. */
    int n_units_per = (n_chunks + chunks_per_unit - 1) / chunks_per_unit;
    int total_units = 3 * n_units_per;
    unit_t *units = calloc(total_units, sizeof(unit_t));
    int u = 0;
    for (int s = 0; s < n_units_per; s++) {
        units[u++] = (unit_t){ t_q, x, acc_q + (size_t)s * t_q->M,
                               (uint32_t)(s * chunks_per_unit),
                               (uint32_t)((s + 1) * chunks_per_unit > n_chunks ?
                                          n_chunks : (s + 1) * chunks_per_unit) };
    }
    for (int s = 0; s < n_units_per; s++) {
        units[u++] = (unit_t){ t_k, x, acc_k + (size_t)s * t_k->M,
                               (uint32_t)(s * chunks_per_unit),
                               (uint32_t)((s + 1) * chunks_per_unit > n_chunks ?
                                          n_chunks : (s + 1) * chunks_per_unit) };
    }
    for (int s = 0; s < n_units_per; s++) {
        units[u++] = (unit_t){ t_v, x, acc_v + (size_t)s * t_v->M,
                               (uint32_t)(s * chunks_per_unit),
                               (uint32_t)((s + 1) * chunks_per_unit > n_chunks ?
                                          n_chunks : (s + 1) * chunks_per_unit) };
    }

    /* === Path A: sequential — 3 separate pool dispatches === */
#define RUN_ONE(TT, ACC_POOL, Y_OUT) do { \
        memset(ACC_POOL, 0, n_threads * (TT)->M * sizeof(float)); \
        unit_t *_local = calloc(n_units_per, sizeof(unit_t)); \
        for (int s = 0; s < n_units_per; s++) { \
            _local[s] = (unit_t){ TT, x, ACC_POOL + (size_t)s * (TT)->M, \
                                  (uint32_t)(s * chunks_per_unit), \
                                  (uint32_t)((s + 1) * chunks_per_unit > n_chunks ? \
                                             n_chunks : (s + 1) * chunks_per_unit) }; \
        } \
        fused_arg_t _fa = { _local, n_units_per }; \
        ib_pool_run(tp, fused_task, &_fa, n_units_per, 1); \
        /* Reduce per row */ \
        for (uint32_t m = 0; m < (TT)->M; m++) { \
            float a = 0.0f; \
            for (int s = 0; s < n_units_per; s++) a += ACC_POOL[(size_t)s * (TT)->M + m]; \
            Y_OUT[m] = a * pqv2_h2f((TT)->row_scale[m]); \
        } \
        free(_local); \
    } while (0)

    /* warmup */
    RUN_ONE(t_q, acc_q, y_q_seq);
    RUN_ONE(t_k, acc_k, y_k_seq);
    RUN_ONE(t_v, acc_v, y_v_seq);

    double t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        RUN_ONE(t_q, acc_q, y_q_seq);
        RUN_ONE(t_k, acc_k, y_k_seq);
        RUN_ONE(t_v, acc_v, y_v_seq);
    }
    double t_seq = (now_sec() - t0) / iters * 1000.0;

    /* === Path B: fused — single pool dispatch over all 3 tensors === */
    /* warmup */
    memset(acc_q, 0, n_threads * t_q->M * sizeof(float));
    memset(acc_k, 0, n_threads * t_k->M * sizeof(float));
    memset(acc_v, 0, n_threads * t_v->M * sizeof(float));
    run_fused(tp, n_threads, units, total_units);
    /* reduce per tensor */
    for (uint32_t m = 0; m < t_q->M; m++) {
        float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_q[(size_t)s * t_q->M + m];
        y_q_fus[m] = a * pqv2_h2f(t_q->row_scale[m]);
    }
    for (uint32_t m = 0; m < t_k->M; m++) {
        float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_k[(size_t)s * t_k->M + m];
        y_k_fus[m] = a * pqv2_h2f(t_k->row_scale[m]);
    }
    for (uint32_t m = 0; m < t_v->M; m++) {
        float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_v[(size_t)s * t_v->M + m];
        y_v_fus[m] = a * pqv2_h2f(t_v->row_scale[m]);
    }

    t0 = now_sec();
    for (int it = 0; it < iters; it++) {
        memset(acc_q, 0, n_threads * t_q->M * sizeof(float));
        memset(acc_k, 0, n_threads * t_k->M * sizeof(float));
        memset(acc_v, 0, n_threads * t_v->M * sizeof(float));
        run_fused(tp, n_threads, units, total_units);
        for (uint32_t m = 0; m < t_q->M; m++) {
            float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_q[(size_t)s * t_q->M + m];
            y_q_fus[m] = a * pqv2_h2f(t_q->row_scale[m]);
        }
        for (uint32_t m = 0; m < t_k->M; m++) {
            float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_k[(size_t)s * t_k->M + m];
            y_k_fus[m] = a * pqv2_h2f(t_k->row_scale[m]);
        }
        for (uint32_t m = 0; m < t_v->M; m++) {
            float a = 0; for (int s = 0; s < n_units_per; s++) a += acc_v[(size_t)s * t_v->M + m];
            y_v_fus[m] = a * pqv2_h2f(t_v->row_scale[m]);
        }
    }
    double t_fus = (now_sec() - t0) / iters * 1000.0;

    /* Correctness */
    double dq = 0, dk = 0, dv = 0;
    for (uint32_t m = 0; m < t_q->M; m++) dq = fmax(dq, fabs(y_q_seq[m] - y_q_fus[m]));
    for (uint32_t m = 0; m < t_k->M; m++) dk = fmax(dk, fabs(y_k_seq[m] - y_k_fus[m]));
    for (uint32_t m = 0; m < t_v->M; m++) dv = fmax(dv, fabs(y_v_seq[m] - y_v_fus[m]));

    printf("\n=== q/k/v fusion derisk (n_threads=%d, iters=%d) ===\n", n_threads, iters);
    printf("max abs diff: q=%.2e k=%.2e v=%.2e\n", dq, dk, dv);
    printf("  sequential (3× pool dispatch): %.3f ms/call\n", t_seq);
    printf("  fused      (1× pool dispatch): %.3f ms/call  ratio=%.3f\n",
            t_fus, t_fus / t_seq);
    if (t_fus < 0.85 * t_seq && dq < 1e-2 && dk < 1e-2 && dv < 1e-2) {
        printf("  VERDICT: fusion is meaningfully faster (%.2f×) and correct\n",
                t_seq / t_fus);
    } else {
        printf("  VERDICT: fusion not a clear win\n");
    }

    free(x); free(y_q_seq); free(y_k_seq); free(y_v_seq);
    free(y_q_fus); free(y_k_fus); free(y_v_fus);
    free(acc_q); free(acc_k); free(acc_v); free(units);
    ib_pool_destroy(tp);
    ib_pqv2_file_free(&f);
    return 0;
}
