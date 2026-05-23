/*
 * threading.h — Async helpers layered on the existing thread pool.
 *
 * This header is ADDITIVE. The legacy barrier thread-pool API
 * (ib_pool_create / ib_pool_run / ib_parallel_matmul*) continues to live
 * in inferbit_internal.h and is unchanged. Nothing here alters that pool's
 * behaviour; callers that never touch the symbols below are bit-for-bit
 * unaffected.
 *
 * What it adds, for the SYNC→ASYNC decode-path work:
 *
 *   1. ib_async_queue — a single dedicated background worker thread with a
 *      FIFO of submitted jobs. Designed for serialising I/O (pread) so the
 *      caller can "submit + keep computing + later wait", overlapping disk
 *      latency with compute without spinning up a thread per request.
 *
 *   2. ib_async_handle — a one-shot future returned by ib_async_submit.
 *      ib_async_wait() blocks until the job's fn() has run and returns its
 *      result; ib_async_poll() is a non-blocking check. The handle is
 *      reference-free: ib_async_wait()/ib_async_discard() releases it.
 *
 * Both are deliberately minimal and lock-based (no lock-free trickery) so
 * the integrator can reason about correctness easily. The queue preserves
 * strict FIFO order, which matters for the drive prefetch ring where reads
 * must land in submission order to keep the ping-pong slots coherent.
 */

#ifndef IB_THREADING_H
#define IB_THREADING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Single-worker async job queue ──────────────────────────────────────
 *
 * One background thread services jobs in submission (FIFO) order. Each job
 * is a (fn, arg) pair; fn returns an int that ib_async_wait() relays back
 * to the submitter (e.g. 1 = pread ok, 0 = failed). Use one queue per
 * logical I/O stream you want serialised; use several queues to issue
 * independent streams (e.g. L1 and L2, or per-lane-group) concurrently.
 */
typedef struct ib_async_queue  ib_async_queue;
typedef struct ib_async_handle ib_async_handle;

/* Job entry point. Runs on the queue's worker thread. The return value is
 * delivered to ib_async_wait(). Keep it self-contained: it must not touch
 * data the submitter mutates before ib_async_wait() returns. */
typedef int (*ib_async_fn)(void* arg);

/* Create / destroy a queue. ib_async_queue_destroy() drains all pending
 * AND in-flight jobs (so their handles complete normally), then joins the
 * worker. Returns NULL on failure — callers MUST treat NULL as "run the
 * job synchronously yourself", exactly mirroring how ib_pool_create(NULL)
 * degrades the matmul path. */
ib_async_queue* ib_async_queue_create(void);
void            ib_async_queue_destroy(ib_async_queue* q);

/* Submit a job. Returns a handle the caller later waits on, or NULL on
 * allocation failure (caller should then run fn(arg) synchronously). The
 * job is guaranteed to start no earlier than any previously-submitted job
 * on the same queue (FIFO). */
ib_async_handle* ib_async_submit(ib_async_queue* q, ib_async_fn fn, void* arg);

/* Block until the job behind `h` has completed; returns fn()'s result.
 * Releases the handle (do not reuse `h` after this). Passing NULL returns
 * 0 — convenient for "submit returned NULL, treat as failure" call sites. */
int ib_async_wait(ib_async_handle* h);

/* Non-blocking completion check. Writes fn()'s result to *out_result when
 * done and returns 1 (handle NOT released — call ib_async_wait or
 * ib_async_discard to release). Returns 0 if still pending. NULL handle
 * returns 1 with *out_result = 0. */
int ib_async_poll(ib_async_handle* h, int* out_result);

/* Release a handle without caring about its result. If the job hasn't run
 * yet it still runs on the worker; the handle is freed once it finishes.
 * Safe to pass NULL. Use this for fire-and-forget submissions. */
void ib_async_discard(ib_async_handle* h);

#ifdef __cplusplus
}
#endif

#endif /* IB_THREADING_H */
