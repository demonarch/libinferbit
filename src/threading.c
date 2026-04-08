/*
 * threading.c — Thread pool for parallel matmul and attention
 *
 * Simple pthreads-based thread pool. Workers wait on a condition variable,
 * wake to process a task, then signal completion.
 */

#include "inferbit_internal.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* ── Thread pool ────────────────────────────────────────────── */

typedef void (*ib_task_fn)(void* arg, int thread_id, int start, int end);

typedef struct {
    ib_task_fn   fn;
    void*        arg;
    int          total;     /* Total work items */
    int          n_threads;
    volatile int next_chunk;
    volatile int done_count;
    int          chunk_size;
    pthread_mutex_t mutex;
    pthread_cond_t  cond_work;
    pthread_cond_t  cond_done;
    volatile int    shutdown;
} ib_pool;

typedef struct {
    ib_pool* pool;
    int      thread_id;
} ib_worker_arg;

static void* worker_fn(void* arg) {
    ib_worker_arg* wa = (ib_worker_arg*)arg;
    ib_pool* pool = wa->pool;
    int tid = wa->thread_id;

    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->fn && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }

        /* Grab a chunk of work */
        int start = pool->next_chunk;
        int end = start + pool->chunk_size;
        if (end > pool->total) end = pool->total;
        pool->next_chunk = end;

        if (start >= pool->total) {
            /* No work left, signal done */
            pool->done_count++;
            if (pool->done_count >= pool->n_threads) {
                pthread_cond_signal(&pool->cond_done);
            }
            pthread_mutex_unlock(&pool->mutex);
            continue;
        }
        pthread_mutex_unlock(&pool->mutex);

        /* Do work */
        pool->fn(pool->arg, tid, start, end);

        /* Try to grab more work */
        while (1) {
            pthread_mutex_lock(&pool->mutex);
            start = pool->next_chunk;
            end = start + pool->chunk_size;
            if (end > pool->total) end = pool->total;
            if (start >= pool->total) {
                pool->done_count++;
                if (pool->done_count >= pool->n_threads) {
                    pthread_cond_signal(&pool->cond_done);
                }
                pthread_mutex_unlock(&pool->mutex);
                break;
            }
            pool->next_chunk = end;
            pthread_mutex_unlock(&pool->mutex);

            pool->fn(pool->arg, tid, start, end);
        }
    }

    return NULL;
}

/* ── Public thread pool API (internal to libinferbit) ───────── */

struct ib_thread_pool {
    ib_pool       pool;
    pthread_t*    threads;
    ib_worker_arg* args;
    int           n_threads;
};

ib_thread_pool* ib_pool_create(int n_threads) {
    if (n_threads <= 1) return NULL;  /* No pool needed for single thread */

    ib_thread_pool* tp = calloc(1, sizeof(ib_thread_pool));
    if (!tp) return NULL;

    tp->n_threads = n_threads;
    tp->pool.n_threads = n_threads;
    tp->pool.shutdown = 0;
    tp->pool.fn = NULL;

    pthread_mutex_init(&tp->pool.mutex, NULL);
    pthread_cond_init(&tp->pool.cond_work, NULL);
    pthread_cond_init(&tp->pool.cond_done, NULL);

    tp->threads = calloc(n_threads, sizeof(pthread_t));
    tp->args = calloc(n_threads, sizeof(ib_worker_arg));
    if (!tp->threads || !tp->args) {
        free(tp->threads);
        free(tp->args);
        free(tp);
        return NULL;
    }

    for (int i = 0; i < n_threads; i++) {
        tp->args[i].pool = &tp->pool;
        tp->args[i].thread_id = i;
        pthread_create(&tp->threads[i], NULL, worker_fn, &tp->args[i]);
    }

    return tp;
}

void ib_pool_destroy(ib_thread_pool* tp) {
    if (!tp) return;

    pthread_mutex_lock(&tp->pool.mutex);
    tp->pool.shutdown = 1;
    pthread_cond_broadcast(&tp->pool.cond_work);
    pthread_mutex_unlock(&tp->pool.mutex);

    for (int i = 0; i < tp->n_threads; i++) {
        pthread_join(tp->threads[i], NULL);
    }

    pthread_mutex_destroy(&tp->pool.mutex);
    pthread_cond_destroy(&tp->pool.cond_work);
    pthread_cond_destroy(&tp->pool.cond_done);

    free(tp->threads);
    free(tp->args);
    free(tp);
}

void ib_pool_run(ib_thread_pool* tp, ib_task_fn fn, void* arg, int total, int chunk_size) {
    if (!tp || total <= 0) {
        /* Single-threaded fallback */
        if (fn && total > 0) fn(arg, 0, 0, total);
        return;
    }

    if (chunk_size <= 0) chunk_size = (total + tp->n_threads - 1) / tp->n_threads;

    pthread_mutex_lock(&tp->pool.mutex);
    tp->pool.fn = fn;
    tp->pool.arg = arg;
    tp->pool.total = total;
    tp->pool.chunk_size = chunk_size;
    tp->pool.next_chunk = 0;
    tp->pool.done_count = 0;

    pthread_cond_broadcast(&tp->pool.cond_work);

    /* Wait for all workers to finish */
    while (tp->pool.done_count < tp->n_threads) {
        pthread_cond_wait(&tp->pool.cond_done, &tp->pool.mutex);
    }

    tp->pool.fn = NULL;
    pthread_mutex_unlock(&tp->pool.mutex);
}

/* ── Parallel matmul wrapper ────────────────────────────────── */

typedef struct {
    float*       out;
    const void*  weights;
    const float* scales;
    const float* input;
    int          M;
    int          N;
    int          bits;
} ib_matmul_arg;

static void parallel_matmul_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    ib_matmul_arg* a = (ib_matmul_arg*)arg;
    int N = a->N;

    /* Each thread processes rows [start, end) */
    if (a->bits == 4) {
        const uint8_t* w = (const uint8_t*)a->weights;
        const uint8_t* row_start = w + (size_t)start * (N / 2);
        ib_kern.matmul_int4(a->out + start, row_start, a->scales + start,
                            a->input, end - start, N);
    } else if (a->bits == 8) {
        const int8_t* w = (const int8_t*)a->weights;
        const int8_t* row_start = w + (size_t)start * N;
        ib_kern.matmul_int8(a->out + start, row_start, a->scales + start,
                            a->input, end - start, N);
    }
}

void ib_parallel_matmul(ib_thread_pool* tp, float* out, const void* weights,
                        const float* scales, const float* input,
                        int M, int N, int bits) {
    if (!tp || M < 64) {
        /* Too small to parallelize, or no thread pool */
        if (bits == 4) ib_kern.matmul_int4(out, weights, scales, input, M, N);
        else           ib_kern.matmul_int8(out, weights, scales, input, M, N);
        return;
    }

    ib_matmul_arg arg = { out, weights, scales, input, M, N, bits };
    int chunk = (M + tp->n_threads - 1) / tp->n_threads;
    if (chunk < 16) chunk = 16;  /* Minimum chunk size to avoid overhead */
    ib_pool_run(tp, parallel_matmul_task, &arg, M, chunk);
}
