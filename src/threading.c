/*
 * threading.c — Thread pool for parallel matmul and attention
 *
 * Simple barrier-based design: all workers wake, process chunks,
 * then wait at a barrier. Main thread waits at the same barrier.
 */

#include "inferbit_internal.h"
#include "platform.h"
#include "threading.h"
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

/* C11 stdatomic not reliably available on MSVC — use Interlocked intrinsics */
typedef volatile LONG atomic_int;
#define atomic_store(p, v)     InterlockedExchange((p), (v))
#define atomic_load(p)         InterlockedCompareExchange((p), 0, 0)
#define atomic_fetch_add(p, v) InterlockedExchangeAdd((p), (v))
#define atomic_fetch_sub(p, v) InterlockedExchangeAdd((p), -(v))

/* pthreads-like API on Windows using native threads + SRWLOCK + CONDITION_VARIABLE */
typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;

#define pthread_mutex_init(m, a)     (InitializeSRWLock(m), 0)
#define pthread_mutex_destroy(m)     ((void)0)
#define pthread_mutex_lock(m)        AcquireSRWLockExclusive(m)
#define pthread_mutex_unlock(m)      ReleaseSRWLockExclusive(m)
#define pthread_cond_init(c, a)      (InitializeConditionVariable(c), 0)
#define pthread_cond_destroy(c)      ((void)0)
#define pthread_cond_wait(c, m)      SleepConditionVariableSRW(c, m, INFINITE, 0)
#define pthread_cond_signal(c)       WakeConditionVariable(c)
#define pthread_cond_broadcast(c)    WakeAllConditionVariable(c)

typedef DWORD (WINAPI *win_thread_fn)(LPVOID);

static int pthread_create(pthread_t* t, void* attr, void* (*fn)(void*), void* arg) {
    (void)attr;
    *t = CreateThread(NULL, 0, (win_thread_fn)fn, arg, 0, NULL);
    return (*t == NULL) ? -1 : 0;
}

static int pthread_join(pthread_t t, void** retval) {
    (void)retval;
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
    return 0;
}

#else
#include <pthread.h>
#include <stdatomic.h>
#endif

/* ── Thread pool ────────────────────────────────────────────── */

typedef void (*ib_task_fn)(void* arg, int thread_id, int start, int end);

struct ib_thread_pool {
    int            n_threads;
    pthread_t*     threads;

    /* Current task (protected by mutex) */
    ib_task_fn     fn;
    void*          arg;
    int            total;
    int            chunk_size;

    /* Synchronization */
    pthread_mutex_t mutex;
    pthread_cond_t  cond_work;   /* Workers wait here for new work */
    pthread_cond_t  cond_done;   /* Main waits here for completion */
    atomic_int      next_chunk;  /* Next chunk index to process */
    atomic_int      active;      /* Number of workers still processing */
    int             generation;  /* Incremented each task to avoid spurious wakes */
    int             shutdown;
};

typedef struct {
    ib_thread_pool* pool;
    int             thread_id;
} ib_worker_arg;

static void* worker_fn(void* raw_arg) {
    ib_worker_arg* wa = (ib_worker_arg*)raw_arg;
    ib_thread_pool* pool = wa->pool;
    int tid = wa->thread_id;
    int my_gen = 0;

    while (1) {
        /* Wait for work */
        pthread_mutex_lock(&pool->mutex);
        while (pool->generation == my_gen && !pool->shutdown) {
            pthread_cond_wait(&pool->cond_work, &pool->mutex);
        }
        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        my_gen = pool->generation;
        ib_task_fn fn = pool->fn;
        void* arg = pool->arg;
        int total = pool->total;
        int chunk = pool->chunk_size;
        pthread_mutex_unlock(&pool->mutex);

        /* Process chunks until none remain */
        while (1) {
            int start = atomic_fetch_add(&pool->next_chunk, chunk);
            if (start >= total) break;
            int end = start + chunk;
            if (end > total) end = total;
            fn(arg, tid, start, end);
        }

        /* Signal completion */
        if (atomic_fetch_sub(&pool->active, 1) == 1) {
            /* Last worker done — wake main thread */
            pthread_mutex_lock(&pool->mutex);
            pthread_cond_signal(&pool->cond_done);
            pthread_mutex_unlock(&pool->mutex);
        }
    }

    free(wa);
    return NULL;
}

/* ── Create / destroy ───────────────────────────────────────── */

ib_thread_pool* ib_pool_create(int n_threads) {
    if (n_threads <= 1) return NULL;

    ib_thread_pool* pool = calloc(1, sizeof(ib_thread_pool));
    if (!pool) return NULL;

    pool->n_threads = n_threads;
    pool->generation = 0;
    pool->shutdown = 0;
    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->cond_work, NULL);
    pthread_cond_init(&pool->cond_done, NULL);

    pool->threads = calloc(n_threads, sizeof(pthread_t));

    for (int i = 0; i < n_threads; i++) {
        ib_worker_arg* wa = malloc(sizeof(ib_worker_arg));
        wa->pool = pool;
        wa->thread_id = i;
        pthread_create(&pool->threads[i], NULL, worker_fn, wa);
    }

    return pool;
}

void ib_pool_destroy(ib_thread_pool* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->cond_work);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->cond_work);
    pthread_cond_destroy(&pool->cond_done);
    free(pool->threads);
    free(pool);
}

/* ── Run a parallel task ────────────────────────────────────── */

void ib_pool_run(ib_thread_pool* pool, ib_task_fn fn, void* arg, int total, int chunk_size) {
    if (!pool || total <= 0) {
        if (fn && total > 0) fn(arg, 0, 0, total);
        return;
    }

    if (chunk_size <= 0) {
        chunk_size = (total + pool->n_threads - 1) / pool->n_threads;
    }
    if (chunk_size < 1) chunk_size = 1;

    pthread_mutex_lock(&pool->mutex);
    pool->fn = fn;
    pool->arg = arg;
    pool->total = total;
    pool->chunk_size = chunk_size;
    atomic_store(&pool->next_chunk, 0);
    atomic_store(&pool->active, pool->n_threads);
    pool->generation++;
    pthread_cond_broadcast(&pool->cond_work);

    /* Wait for all workers to finish */
    while (atomic_load(&pool->active) > 0) {
        pthread_cond_wait(&pool->cond_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

/* ── Parallel matmul ────────────────────────────────────────── */

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
    int rows = end - start;

    if (a->bits == 2) {
        const uint8_t* w = (const uint8_t*)a->weights;
        const uint8_t* row_start = w + (size_t)start * (N / 4);
        ib_kern.matmul_int2(a->out + start, row_start, a->scales + start,
                            a->input, rows, N);
    } else if (a->bits == 4) {
        const uint8_t* w = (const uint8_t*)a->weights;
        const uint8_t* row_start = w + (size_t)start * (N / 2);
        ib_kern.matmul_int4(a->out + start, row_start, a->scales + start,
                            a->input, rows, N);
    } else if (a->bits == 8) {
        const int8_t* w = (const int8_t*)a->weights;
        const int8_t* row_start = w + (size_t)start * N;
        ib_kern.matmul_int8(a->out + start, row_start, a->scales + start,
                            a->input, rows, N);
    }
}

void ib_parallel_matmul(ib_thread_pool* tp, float* out, const void* weights,
                        const float* scales, const float* input,
                        int M, int N, int bits) {
    if (!tp || M < 64) {
        if (bits == 2)      ib_kern.matmul_int2(out, weights, scales, input, M, N);
        else if (bits == 4) ib_kern.matmul_int4(out, weights, scales, input, M, N);
        else                ib_kern.matmul_int8(out, weights, scales, input, M, N);
        return;
    }

    ib_matmul_arg arg = { out, weights, scales, input, M, N, bits };
    int chunk = (M + tp->n_threads - 1) / tp->n_threads;
    if (chunk < 16) chunk = 16;
    ib_pool_run(tp, parallel_matmul_task, &arg, M, chunk);
}

/* ── Parallel W4A8 matmul ──────────────────────────────────── */

typedef struct {
    float*        out;
    const void*   weights;
    const float*  scales_w;
    const int8_t* input;
    const float*  scales_a;
    int           M;
    int           N;
} ib_w4a8_arg;

static void parallel_w4a8_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    ib_w4a8_arg* a = (ib_w4a8_arg*)arg;
    int N = a->N;
    int rows = end - start;
    const uint8_t* w = (const uint8_t*)a->weights;
    const uint8_t* row_start = w + (size_t)start * (N / 2);
    ib_kern.matmul_w4a8(a->out + start, row_start, a->scales_w + start,
                        a->input, a->scales_a, rows, N);
}

void ib_parallel_matmul_w4a8(ib_thread_pool* tp, float* out, const void* weights,
                             const float* scales_w, const int8_t* input,
                             const float* scales_a, int M, int N) {
    if (!tp || M < 64) {
        ib_kern.matmul_w4a8(out, weights, scales_w, input, scales_a, M, N);
        return;
    }
    ib_w4a8_arg arg = { out, weights, scales_w, input, scales_a, M, N };
    int chunk = (M + tp->n_threads - 1) / tp->n_threads;
    if (chunk < 16) chunk = 16;
    ib_pool_run(tp, parallel_w4a8_task, &arg, M, chunk);
}

/* ── Parallel batched W4A8 matmul (threaded over rows) ───────
 *
 * Thread pool splits row range. Each worker runs the single-position W4A8
 * kernel B times over its row slice, so the kernel registers stay hot for
 * each batch lane but weight bandwidth is amortized via L2 reuse. This is
 * slightly less effective than calling the batched kernel itself per
 * worker, but avoids the out[b*M + i] layout mismatch that the batched
 * kernel would introduce on a row-sliced output. */
typedef struct {
    float*        out;
    const void*   weights;
    const float*  scales_w;
    const int8_t* input;
    const float*  scales_a;
    int           M;
    int           N;
    int           B;
} ib_w4a8_batch_arg;

static void parallel_w4a8_batch_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    ib_w4a8_batch_arg* a = (ib_w4a8_batch_arg*)arg;
    int rows = end - start;
    const uint8_t* w = (const uint8_t*)a->weights;
    const uint8_t* row_start = w + (size_t)start * (a->N / 2);
    /* Call the BATCHED kernel on this thread's row slice. M_stride==a->M
     * so per-batch outputs land in their correct positions in the global
     * out[b * M_global + i] layout. */
    ib_kern.matmul_w4a8_batch(a->out + start,
                              row_start,
                              a->scales_w + start,
                              a->input,
                              a->scales_a,
                              rows,
                              a->N,
                              a->B,
                              a->M);
}

void ib_parallel_matmul_w4a8_batch(ib_thread_pool* tp, float* out,
                                   const void* weights, const float* scales_w,
                                   const int8_t* input, const float* scales_a,
                                   int M, int N, int B) {
    if (!tp || M < 64) {
        ib_kern.matmul_w4a8_batch(out, weights, scales_w, input, scales_a, M, N, B, M);
        return;
    }
    ib_w4a8_batch_arg arg = { out, weights, scales_w, input, scales_a, M, N, B };
    int chunk = (M + tp->n_threads - 1) / tp->n_threads;
    if (chunk < 16) chunk = 16;
    ib_pool_run(tp, parallel_w4a8_batch_task, &arg, M, chunk);
}

/* ── Parallel batched INT8 matmul (threaded over rows) ────────
 *
 * The per-kernel batched matmul (neon_matmul_int8_batch) is single-thread.
 * For a large output head (vocab × hidden = 128K × 4K for Llama) we need
 * to split rows across the thread pool. Threading the batch axis would
 * fight cache reuse on weights, so we stay row-parallel. */
typedef struct {
    float*       out;
    const void*  weights;
    const float* scales_w;
    const float* input;
    int          M;
    int          N;
    int          B;
} ib_int8_batch_arg;

static void parallel_int8_batch_task(void* arg, int thread_id, int start, int end) {
    (void)thread_id;
    ib_int8_batch_arg* a = (ib_int8_batch_arg*)arg;
    int rows = end - start;
    const int8_t* w = (const int8_t*)a->weights;
    const int8_t* row_start = w + (size_t)start * a->N;
    /* Call the batched kernel on this thread's row slice. M_stride==a->M
     * so each batch's partial output lands at out[b * M_global + start]. */
    ib_kern.matmul_int8_batch(a->out + start,
                              row_start,
                              a->scales_w + start,
                              a->input,
                              rows,
                              a->N,
                              a->B,
                              a->M);
}

void ib_parallel_matmul_int8_batch(ib_thread_pool* tp, float* out,
                                   const void* weights, const float* scales_w,
                                   const float* input, int M, int N, int B) {
    if (!tp || M < 64) {
        ib_kern.matmul_int8_batch(out, weights, scales_w, input, M, N, B, M);
        return;
    }
    ib_int8_batch_arg arg = { out, weights, scales_w, input, M, N, B };
    int chunk = (M + tp->n_threads - 1) / tp->n_threads;
    if (chunk < 16) chunk = 16;
    ib_pool_run(tp, parallel_int8_batch_task, &arg, M, chunk);
}

/* Group-wise symmetric INT8 quantization (G=IB_W4A8_GROUP).
 *
 * Each group of IB_W4A8_GROUP elements gets its own FP32 scale. Tail group
 * (if N % GROUP != 0) gets its own scale covering only the remaining
 * elements. Returns the number of groups written (ceil(N / GROUP)). */
int ib_quantize_input_int8_g128(const float* input, int8_t* out_q,
                                float* out_scales, int N) {
    const int G = IB_W4A8_GROUP;
    int groups = 0;
    for (int base = 0; base < N; base += G) {
        int end = base + G; if (end > N) end = N;

        float max_abs = 0.0f;
        for (int i = base; i < end; i++) {
            float a = input[i] < 0 ? -input[i] : input[i];
            if (a > max_abs) max_abs = a;
        }

        float scale;
        if (max_abs < 1e-10f) {
            scale = 1e-10f;
            for (int i = base; i < end; i++) out_q[i] = 0;
        } else {
            scale = max_abs / 127.0f;
            float inv = 1.0f / scale;
            for (int i = base; i < end; i++) {
                float v = input[i];
                int q = (int)(v * inv + (v >= 0 ? 0.5f : -0.5f));
                if (q < -127) q = -127;
                if (q > 127) q = 127;
                out_q[i] = (int8_t)q;
            }
        }
        out_scales[groups++] = scale;
    }
    return groups;
}

/* ── Single-worker async job queue (ib_async_*) ──────────────────────────
 *
 * ADDITIVE: this is a separate facility from the barrier matmul pool above.
 * It exists so the decode path can overlap blocking I/O (pread) with
 * compute via "submit job → keep computing → wait later", without spawning
 * a thread per request. One dedicated worker drains a FIFO so reads on a
 * given queue land in submission order (required by the drive prefetch
 * ring's slot ping-pong). Use multiple queues for independent concurrent
 * streams (e.g. one for L1 indices, one for L2).
 *
 * The matmul pool above is untouched: these symbols are only reachable if
 * a caller explicitly creates a queue. Reuses the same pthread shims set
 * up at the top of this file (Windows native or POSIX pthreads).
 *
 * Handle lifecycle (the only subtle bit):
 *   - A handle starts with refcount = 2: one ref held by the submitter
 *     (released via ib_async_wait / ib_async_discard) and one by the queue
 *     (released by the worker once fn has run, or by destroy on drain).
 *   - Whoever drops the count to 0 frees it. This makes fire-and-forget
 *     (discard before the job runs) safe: the worker still runs the job
 *     and then frees the handle.
 */

struct ib_async_handle {
    ib_async_fn      fn;
    void*            arg;
    int              result;          /* fn() return value (valid when done) */
    int              done;            /* set by worker after fn() returns */
    int              refcount;        /* 2 at submit; freed at 0 */
    pthread_mutex_t  mu;              /* guards result/done/refcount */
    pthread_cond_t   cv;             /* signalled when done flips to 1 */
    struct ib_async_handle* next;    /* FIFO link (queue-owned) */
};

struct ib_async_queue {
    pthread_t        worker;
    pthread_mutex_t  mu;
    pthread_cond_t   cv;             /* worker waits here for head != NULL */
    ib_async_handle* head;           /* FIFO: dequeue from head */
    ib_async_handle* tail;           /* FIFO: enqueue at tail */
    int              stop;           /* destroy signal (drain then exit) */
    int              started;        /* worker thread successfully created */
};

/* Drop one ref; free when it hits zero. Must NOT hold h->mu on entry. */
static void ib_async_handle_release(ib_async_handle* h) {
    pthread_mutex_lock(&h->mu);
    int rc = --h->refcount;
    pthread_mutex_unlock(&h->mu);
    if (rc == 0) {
        pthread_mutex_destroy(&h->mu);
        pthread_cond_destroy(&h->cv);
        free(h);
    }
}

static void* ib_async_worker_fn(void* raw) {
    ib_async_queue* q = (ib_async_queue*)raw;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        while (!q->head && !q->stop) {
            pthread_cond_wait(&q->cv, &q->mu);
        }
        /* Drain remaining jobs even after stop is set so every submitted
         * handle completes (no caller is left blocked in ib_async_wait). */
        if (!q->head && q->stop) {
            pthread_mutex_unlock(&q->mu);
            break;
        }
        ib_async_handle* h = q->head;
        q->head = h->next;
        if (!q->head) q->tail = NULL;
        pthread_mutex_unlock(&q->mu);

        int r = h->fn ? h->fn(h->arg) : 0;

        pthread_mutex_lock(&h->mu);
        h->result = r;
        h->done = 1;
        pthread_cond_broadcast(&h->cv);
        pthread_mutex_unlock(&h->mu);

        /* Release the queue's ref on this handle. */
        ib_async_handle_release(h);
    }
    return NULL;
}

ib_async_queue* ib_async_queue_create(void) {
    ib_async_queue* q = (ib_async_queue*)calloc(1, sizeof(*q));
    if (!q) return NULL;
    if (pthread_mutex_init(&q->mu, NULL) != 0) { free(q); return NULL; }
    if (pthread_cond_init(&q->cv, NULL) != 0) {
        pthread_mutex_destroy(&q->mu); free(q); return NULL;
    }
    if (pthread_create(&q->worker, NULL, ib_async_worker_fn, q) != 0) {
        pthread_cond_destroy(&q->cv);
        pthread_mutex_destroy(&q->mu);
        free(q);
        return NULL;
    }
    q->started = 1;
    return q;
}

void ib_async_queue_destroy(ib_async_queue* q) {
    if (!q) return;
    pthread_mutex_lock(&q->mu);
    q->stop = 1;
    pthread_cond_broadcast(&q->cv);
    pthread_mutex_unlock(&q->mu);
    if (q->started) pthread_join(q->worker, NULL);
    /* Worker has drained the FIFO and released the queue's ref on every
     * handle; any handle the submitter never waited on was freed by the
     * worker's release (or will be by a pending ib_async_wait/discard). */
    pthread_cond_destroy(&q->cv);
    pthread_mutex_destroy(&q->mu);
    free(q);
}

ib_async_handle* ib_async_submit(ib_async_queue* q, ib_async_fn fn, void* arg) {
    if (!q || !fn) return NULL;
    ib_async_handle* h = (ib_async_handle*)calloc(1, sizeof(*h));
    if (!h) return NULL;
    if (pthread_mutex_init(&h->mu, NULL) != 0) { free(h); return NULL; }
    if (pthread_cond_init(&h->cv, NULL) != 0) {
        pthread_mutex_destroy(&h->mu); free(h); return NULL;
    }
    h->fn = fn;
    h->arg = arg;
    h->result = 0;
    h->done = 0;
    h->refcount = 2;        /* submitter + queue */
    h->next = NULL;

    pthread_mutex_lock(&q->mu);
    if (q->tail) q->tail->next = h; else q->head = h;
    q->tail = h;
    pthread_cond_signal(&q->cv);
    pthread_mutex_unlock(&q->mu);
    return h;
}

int ib_async_wait(ib_async_handle* h) {
    if (!h) return 0;
    pthread_mutex_lock(&h->mu);
    while (!h->done) {
        pthread_cond_wait(&h->cv, &h->mu);
    }
    int r = h->result;
    pthread_mutex_unlock(&h->mu);
    ib_async_handle_release(h);   /* drop submitter's ref */
    return r;
}

int ib_async_poll(ib_async_handle* h, int* out_result) {
    if (!h) { if (out_result) *out_result = 0; return 1; }
    pthread_mutex_lock(&h->mu);
    int done = h->done;
    int r = h->result;
    pthread_mutex_unlock(&h->mu);
    if (done && out_result) *out_result = r;
    return done;   /* handle NOT released here (poll is read-only) */
}

void ib_async_discard(ib_async_handle* h) {
    if (!h) return;
    ib_async_handle_release(h);   /* drop submitter's ref; worker frees rest */
}
