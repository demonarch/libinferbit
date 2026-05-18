/*
 * forward.c — Transformer forward pass
 *
 * Implements: embedding → [RMSNorm → Attention → Residual → RMSNorm → MLP → Residual] × N → RMSNorm → Output head
 */

#include "inferbit_internal.h"
#include "platform.h"   /* pread + POSIX I/O shims (drive mode) */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#ifdef IB_HAS_METAL
#include "metal/metal_runtime.h"
#endif

#include "mome.h"

/* W4A8 path is on by default. Set IB_W4A8=0 in env to force the FP32
 * activation fallback (used for A/B comparison and debugging). */
static int w4a8_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("IB_W4A8");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached;
}

/* ── Stage 5d — hybrid CPU/GPU dispatch (docs/v2/00_CORRECTION.md) ──
 *
 * v1 ships a single env-var knob:
 *   IB_HYBRID_FFN_GPU=1  → at first ib_forward call, tag every layer's
 *                          gate_proj / up_proj / down_proj as
 *                          INFERBIT_BACKEND_METAL. The CPU forward then
 *                          routes those matmuls through a one-shot
 *                          Metal dispatch while the rest of the layer
 *                          (norms, attention, residuals, embed, lm_head)
 *                          stays on CPU.
 *
 * Default (env unset) leaves every preferred_backend at AUTO (=0) which
 * keeps the existing CPU-or-Metal end-to-end routing bit-identical to
 * pre-Stage-5d behaviour.
 *
 * Lazy tag application + lazy ctx creation. */
static int hybrid_ffn_gpu_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("IB_HYBRID_FFN_GPU");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

static void hybrid_apply_tags(inferbit_model *m) {
    if (!m || m->hybrid_tags_applied) return;
    m->hybrid_tags_applied = 1;
    if (!hybrid_ffn_gpu_enabled()) return;
    /* Tag FFN matmuls only. Attention stays AUTO so it follows the
     * surrounding forward (CPU here). */
    for (int L = 0; L < m->header.num_layers; L++) {
        m->layers[L].gate_proj.preferred_backend = INFERBIT_BACKEND_METAL;
        m->layers[L].up_proj.preferred_backend   = INFERBIT_BACKEND_METAL;
        m->layers[L].down_proj.preferred_backend = INFERBIT_BACKEND_METAL;
    }
}

#ifdef IB_HAS_METAL
/* Forward decl from below: lazy Metal ctx/buf creation. Returns 1 if
 * upload succeeded, 0 if Metal is unavailable or upload failed. */
static int ib_metal_route(inferbit_model* m);

/* Stage 5d helper: lazily create the Metal ctx + upload the model
 * REGARDLESS of IB_BACKEND=cpu (which ib_metal_route honors). The
 * hybrid hook needs the GPU available even when the surrounding
 * forward runs on CPU. Returns 1 on success, 0 if Metal is unavailable
 * or upload fails (caller falls back to CPU dispatch). */
static int hybrid_metal_route(inferbit_model *m) {
    if (m->metal_route_failed) return 0;
    if (m->metal_bufs) return 1;
    ib_metal_ctx *ctx = ib_metal_create();
    if (!ctx) { m->metal_route_failed = 1; return 0; }
    ib_metal_model_buffers *bufs = ib_metal_upload_model(ctx, m);
    if (!bufs) { ib_metal_destroy(ctx); m->metal_route_failed = 1; return 0; }
    m->metal_ctx  = ctx;
    m->metal_bufs = bufs;
    return 1;
}

/* Ensure model->hybrid_x_buf / hybrid_y_buf are Metal-shared and at
 * least n_in / n_out floats long. Grow (re-alloc) if too small. Returns
 * 0 on success; -1 on failure (caller should fall back to CPU dispatch). */
static int hybrid_ensure_buffers(inferbit_model *m, size_t n_in, size_t n_out) {
    ib_metal_ctx *ctx = (ib_metal_ctx*)m->metal_ctx;
    if (!ctx) return -1;
    if (!m->hybrid_x_buf || m->hybrid_x_buf_floats < n_in) {
        if (m->hybrid_x_buf) ib_metal_free(ctx, m->hybrid_x_buf);
        m->hybrid_x_buf = ib_metal_alloc(ctx, n_in * sizeof(float), NULL);
        if (!m->hybrid_x_buf) { m->hybrid_x_buf_floats = 0; return -1; }
        m->hybrid_x_buf_floats = n_in;
    }
    if (!m->hybrid_y_buf || m->hybrid_y_buf_floats < n_out) {
        if (m->hybrid_y_buf) ib_metal_free(ctx, m->hybrid_y_buf);
        m->hybrid_y_buf = ib_metal_alloc(ctx, n_out * sizeof(float), NULL);
        if (!m->hybrid_y_buf) { m->hybrid_y_buf_floats = 0; return -1; }
        m->hybrid_y_buf_floats = n_out;
    }
    return 0;
}

/* Selector lookup: which IB_METAL_TB_* index corresponds to this tensor
 * within a layer. Returns -1 if the pointer isn't one of the known
 * matmul slots of the given layer (in which case the caller falls back
 * to CPU). */
static int hybrid_tensor_which(const ib_layer_meta *L, const ib_tensor_meta *t,
                               int *out_layer_idx, int layer_idx) {
    *out_layer_idx = layer_idx;
    if (t == &L->q_proj)    return IB_METAL_TB_Q_PROJ;
    if (t == &L->k_proj)    return IB_METAL_TB_K_PROJ;
    if (t == &L->v_proj)    return IB_METAL_TB_V_PROJ;
    if (t == &L->o_proj)    return IB_METAL_TB_O_PROJ;
    if (t == &L->gate_proj) return IB_METAL_TB_GATE_PROJ;
    if (t == &L->up_proj)   return IB_METAL_TB_UP_PROJ;
    if (t == &L->down_proj) return IB_METAL_TB_DOWN_PROJ;
    return -1;
}
#endif /* IB_HAS_METAL */

/* Forward decl of the CPU matmul (defined below). */
static void tensor_matmul(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf
);

/* Hybrid-aware matmul dispatcher. If the tensor is METAL-tagged AND the
 * model has (or can lazily acquire) a Metal context AND the layer was
 * uploaded, dispatch this single matmul to the GPU; otherwise fall
 * through to the CPU `tensor_matmul`. layer_idx is the owning layer
 * index for selector resolution. Caller passes M=out_rows, N=in_cols.
 *
 * On any failure the implementation transparently falls back to CPU so
 * the forward pass never crashes — the worst case is a one-time perf
 * regression. */
static void tensor_matmul_hybrid(
    inferbit_model *m, int layer_idx, const ib_tensor_meta *t,
    float *out, const float *input, int M, int N, float *scale_buf
) {
#ifdef IB_HAS_METAL
    if (t->preferred_backend == INFERBIT_BACKEND_METAL) {
        /* Lazy Metal ctx + upload. Use hybrid_metal_route — it ignores
         * IB_BACKEND=cpu (the user explicitly opted into hybrid by
         * tagging this tensor METAL). If Metal genuinely isn't
         * available (no device, unsupported layout), fall back to CPU. */
        if (hybrid_metal_route(m)) {
            if (hybrid_ensure_buffers(m, (size_t)N, (size_t)M) == 0) {
                int li = 0;
                int which = hybrid_tensor_which(&m->layers[layer_idx], t, &li, layer_idx);
                if (which >= 0) {
                    memcpy(m->hybrid_x_buf, input, (size_t)N * sizeof(float));
                    int rc = ib_metal_run_single_matmul(
                        (ib_metal_ctx*)m->metal_ctx, m->metal_bufs,
                        li, which, m->hybrid_x_buf, m->hybrid_y_buf);
                    if (rc == 0) {
                        memcpy(out, m->hybrid_y_buf, (size_t)M * sizeof(float));
                        return;
                    }
                }
            }
        }
        /* fallthrough → CPU */
    }
#else
    (void)layer_idx;
#endif
    tensor_matmul(m, t, out, input, M, N, scale_buf);
}

/* ── Weight data access helpers ─────────────────────────────── */

/* Get pointer to weight data for a tensor */
static inline const void* tensor_data(const inferbit_model* m, const ib_tensor_meta* t) {
    return (const uint8_t*)m->weight_data + t->offset;
}

/* Get pointer to scale factors for a tensor (FP16 stored, we read as half→float) */
static inline const void* tensor_scales_raw(const inferbit_model* m, const ib_tensor_meta* t) {
    if (t->scale_offset == 0 && t->scale_size == 0) return NULL;
    return (const uint8_t*)m->weight_data + t->scale_offset;
}

/* ── Path D drive mode + 2-slot prefetch ring (perf fix) ─────────
 *
 * Background: every PQv2 matmul in drive mode pread()s its indices
 * from disk into a shared scratch buffer the kernel reads from.
 * Per token: 22 layers × 7 matmuls = 154 synchronous preads, each
 * blocking on storage. The matmul kernel and the I/O were strictly
 * serialised → effective decode throughput floored at ~7-10 t/s.
 *
 * Fix: two scratch slots + a single background pread() worker. Before
 * each matmul we kick a prefetch for the NEXT tensor (decode order is
 * static — Q,K,V,O,gate,up,down per layer; output_head at the end). By
 * the time the kernel needs slot N, the worker is already filling slot
 * (N+1 % 2). The current matmul therefore overlaps with the next
 * tensor's I/O, hiding most of the pread() latency behind the kernel
 * compute. The model.c free path stops the worker via the public
 * ib_drive_prefetch_shutdown shim.
 *
 * The kernel reads from `pq->indices`. We MUST repoint `pq->indices`
 * to the slot whose data corresponds to the tensor about to run.
 * Worker writes into the *other* slot, so the active matmul never
 * races with the prefetch. */

#ifdef _WIN32
/* Reuse the Windows pthread shim already defined by threading.c. Including
 * it here would double-define; we replicate the minimal subset we need. */
#include <windows.h>
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
typedef DWORD (WINAPI *win_thread_fn_pf)(LPVOID);
static int pthread_create(pthread_t* t, void* attr, void* (*fn)(void*), void* arg) {
    (void)attr; *t = CreateThread(NULL, 0, (win_thread_fn_pf)fn, arg, 0, NULL);
    return (*t == NULL) ? -1 : 0;
}
static int pthread_join(pthread_t t, void** retval) {
    (void)retval; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0;
}
#else
#include <pthread.h>
#endif

typedef struct ib_drive_pf_state {
    pthread_t      thread;
    pthread_mutex_t mu;
    pthread_cond_t  req_cv;    /* main → worker: a request is pending */
    pthread_cond_t  done_cv;   /* worker → main: request complete */
    /* Request state (protected by mu). */
    const ib_tensor_meta *req_tensor;   /* what to prefetch */
    int             req_slot;           /* which slot to fill (0 or 1) */
    int             req_pending;        /* 1 when worker should service the request */
    int             req_in_flight;      /* 1 between dequeue and completion */
    /* Result of the most recently completed request. */
    const ib_tensor_meta *done_tensor;
    int             done_slot;
    /* Cached pointers for the worker (set once at init). */
    int             fd;
    void           *scratch[2];
    size_t          scratch_size;
    /* Shutdown flag. */
    int             stop;
} ib_drive_pf_state;

static void *ib_drive_pf_worker(void *arg) {
    ib_drive_pf_state *st = (ib_drive_pf_state *)arg;
    pthread_mutex_lock(&st->mu);
    for (;;) {
        while (!st->stop && !st->req_pending) {
            pthread_cond_wait(&st->req_cv, &st->mu);
        }
        if (st->stop) break;
        const ib_tensor_meta *t = st->req_tensor;
        int slot = st->req_slot;
        st->req_pending = 0;
        st->req_in_flight = 1;
        pthread_mutex_unlock(&st->mu);

        int ok = 0;
        if (t && t->pq && slot >= 0 && slot < 2 && st->scratch[slot]) {
            const pqv2_t *pq = t->pq;
            size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
            off_t off = (off_t)pq->indices_file_offset;
            if (bytes > 0 && bytes <= st->scratch_size && off != 0) {
                uint8_t *buf = (uint8_t *)st->scratch[slot];
                size_t done = 0;
                ok = 1;
                while (done < bytes) {
                    ssize_t r = pread(st->fd, buf + done, bytes - done,
                                      off + (off_t)done);
                    if (r <= 0) {
                        if (r == -1 && errno == EINTR) continue;
                        ok = 0;
                        break;
                    }
                    done += (size_t)r;
                }
            }
        }

        pthread_mutex_lock(&st->mu);
        st->req_in_flight = 0;
        st->done_tensor = ok ? t : NULL;
        st->done_slot = ok ? slot : -1;
        pthread_cond_broadcast(&st->done_cv);
    }
    pthread_mutex_unlock(&st->mu);
    return NULL;
}

/* Wait for any in-flight prefetch to finish (called under mu) and clear
 * the result. */
static void pf_wait_idle_locked(ib_drive_pf_state *st) {
    while (st->req_pending || st->req_in_flight) {
        pthread_cond_wait(&st->done_cv, &st->mu);
    }
}

/* Lazily init the prefetcher on the first matmul. Falls back silently to
 * the legacy synchronous pread path on init failure. */
static ib_drive_pf_state *drive_pf_get(const inferbit_model *m) {
    /* We mutate the cached pointer through (inferbit_model*) — the
     * "const" on m is decorative inside this module; matmul callers pass
     * const for read-only weight access, not because m is genuinely
     * immutable (drive scratch buffer is also overwritten). */
    inferbit_model *mm = (inferbit_model *)m;
    if (mm->drive_pf_state) return (ib_drive_pf_state *)mm->drive_pf_state;
    /* Disable prefetcher when IB_DRIVE_PF_OFF=1 — used for A/B baseline
     * comparison. Falls back to legacy synchronous load. */
    {
        const char *off = getenv("IB_DRIVE_PF_OFF");
        if (off && off[0] == '1') return NULL;
    }
    if (!mm->drive_indices_scratch || !mm->drive_indices_scratch2 ||
        !mm->drive_pq_order || mm->drive_pq_order_len <= 0 ||
        mm->drive_fd < 0) {
        return NULL;
    }
    ib_drive_pf_state *st = (ib_drive_pf_state *)calloc(1, sizeof(*st));
    if (!st) return NULL;
    if (pthread_mutex_init(&st->mu, NULL) != 0) { free(st); return NULL; }
    if (pthread_cond_init(&st->req_cv, NULL) != 0) {
        pthread_mutex_destroy(&st->mu); free(st); return NULL;
    }
    if (pthread_cond_init(&st->done_cv, NULL) != 0) {
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu); free(st); return NULL;
    }
    st->fd = mm->drive_fd;
    st->scratch[0] = mm->drive_indices_scratch;
    st->scratch[1] = mm->drive_indices_scratch2;
    st->scratch_size = mm->drive_indices_scratch_size;
    st->done_slot = -1;
    if (pthread_create(&st->thread, NULL, ib_drive_pf_worker, st) != 0) {
        pthread_cond_destroy(&st->done_cv);
        pthread_cond_destroy(&st->req_cv);
        pthread_mutex_destroy(&st->mu);
        free(st);
        return NULL;
    }
    mm->drive_pf_state = st;
    if (getenv("IB_DRIVE_PF_DEBUG")) {
        fprintf(stderr, "[ib drive-pf] prefetcher init: scratch_size=%zu order_len=%d fd=%d\n",
                (size_t)st->scratch_size, mm->drive_pq_order_len, st->fd);
    }
    /* Warm-start: prefetch the very first tensor so the first matmul of
     * the first decode step doesn't have to sync-load (matters only for
     * short generations; negligible for long ones but ~free). */
    if (mm->drive_pq_order && mm->drive_pq_order_len > 0) {
        pthread_mutex_lock(&st->mu);
        st->req_tensor = mm->drive_pq_order[0];
        st->req_slot = 0;
        st->req_pending = 1;
        pthread_cond_signal(&st->req_cv);
        pthread_mutex_unlock(&st->mu);
    }
    return st;
}

/* Public shim called from model.c during inferbit_free. Stops the worker
 * and tears down its sync primitives. Safe to call when no prefetcher
 * was ever initialised. */
void ib_drive_prefetch_shutdown(inferbit_model *m);
void ib_drive_prefetch_shutdown(inferbit_model *m) {
    if (!m || !m->drive_pf_state) return;
    ib_drive_pf_state *st = (ib_drive_pf_state *)m->drive_pf_state;
    pthread_mutex_lock(&st->mu);
    pf_wait_idle_locked(st);
    st->stop = 1;
    pthread_cond_broadcast(&st->req_cv);
    pthread_mutex_unlock(&st->mu);
    pthread_join(st->thread, NULL);
    pthread_cond_destroy(&st->done_cv);
    pthread_cond_destroy(&st->req_cv);
    pthread_mutex_destroy(&st->mu);
    free(st);
    m->drive_pf_state = NULL;
}

/* Find the index of `t` in m->drive_pq_order[] (linear scan over ~150
 * pointers — single cache-line walk on average). Returns -1 if not in
 * the list (sparsity-masked / non-drive / etc.). */
static int drive_order_index(const inferbit_model *m, const ib_tensor_meta *t) {
    int n = m->drive_pq_order_len;
    const ib_tensor_meta **arr = m->drive_pq_order;
    for (int i = 0; i < n; i++) {
        if (arr[i] == t) return i;
    }
    return -1;
}

/* Synchronously pread tensor t's indices into `slot`. Returns 0 on ok. */
static int drive_sync_load_to_slot(const inferbit_model *m,
                                   const ib_tensor_meta *t,
                                   int slot) {
    if (!t || !t->pq) return -1;
    void *dst = (slot == 1) ? m->drive_indices_scratch2 : m->drive_indices_scratch;
    if (!dst) return -1;
    const pqv2_t *pq = t->pq;
    size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
    if (bytes == 0 || bytes > m->drive_indices_scratch_size) return -1;
    off_t off = (off_t)pq->indices_file_offset;
    if (off == 0) return -1;
    uint8_t *buf = (uint8_t *)dst;
    size_t done = 0;
    while (done < bytes) {
        ssize_t r = pread(m->drive_fd, buf + done, bytes - done, off + (off_t)done);
        if (r <= 0) {
            if (r == -1 && errno == EINTR) continue;
            return -1;
        }
        done += (size_t)r;
    }
    return 0;
}

/* Repoint pq->indices for tensor t to the buffer in `slot`. The kernel
 * reads via pq->indices; this is the swap step of the ring. Safe because
 * the kernel is single-threaded per matmul and we only mutate before
 * dispatching. */
static void drive_repoint_indices(const ib_tensor_meta *t, void *slot_buf) {
    pqv2_t *mpq = (pqv2_t *)t->pq;
    mpq->indices = (const uint8_t *)slot_buf;
}

/* Replacement for the old drive_load_indices. Ensures the active scratch
 * slot has tensor t's indices loaded and pq->indices points at it. Then
 * kicks off a background prefetch for the next tensor in decode order
 * (so the next matmul's I/O overlaps with this matmul's compute). */
static int drive_load_indices(const inferbit_model* m, const ib_tensor_meta* t) {
    if (!m || m->residency_mode != 1) return 0;
    if (!t || !t->pq) return 0;
    if (!m->drive_indices_scratch || m->drive_fd < 0) return 0;
    const pqv2_t* pq = t->pq;
    size_t bytes = (size_t)pq->M * (pq->N / pq->G) * pq->n_subchunks;
    if (bytes == 0 || bytes > m->drive_indices_scratch_size) return -1;
    off_t off = (off_t)pq->indices_file_offset;
    if (off == 0) return 0;     /* not redirected; mmap'd path */

    ib_drive_pf_state *st = drive_pf_get(m);
    if (!st) {
        /* Prefetcher unavailable → legacy synchronous path into slot 0. */
        int rc = drive_sync_load_to_slot(m, t, 0);
        if (rc == 0) drive_repoint_indices(t, m->drive_indices_scratch);
#if !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
        (void)posix_fadvise(m->drive_fd, off, (off_t)bytes, POSIX_FADV_DONTNEED);
#endif
        return rc;
    }

    int ready_slot = -1;
    pthread_mutex_lock(&st->mu);
    /* Wait for any pending prefetch to land — it may or may not be for
     * us. We don't preemptively cancel because a partial pread of an
     * unrelated tensor is harmless (just wasted I/O for a single tensor;
     * the case is rare — only on the very first call). */
    pf_wait_idle_locked(st);
    if (st->done_tensor == t && st->done_slot >= 0) {
        ready_slot = st->done_slot;
        st->done_tensor = NULL;
        st->done_slot = -1;
    } else {
        st->done_tensor = NULL;
        st->done_slot = -1;
    }
    pthread_mutex_unlock(&st->mu);

    if (ready_slot < 0) {
        /* Prefetch missed (first matmul, sparsity-masked detour, etc.).
         * Sync-load into slot 0 — slot 1 is now free for the next
         * prefetch kick below. */
        if (drive_sync_load_to_slot(m, t, 0) != 0) return -1;
        ready_slot = 0;
    }
    void *active_buf = (ready_slot == 1)
                       ? m->drive_indices_scratch2
                       : m->drive_indices_scratch;
    drive_repoint_indices(t, active_buf);

    /* Kick the prefetch for the NEXT tensor in decode order, into the
     * OTHER slot. If t isn't in the order list (sparse / output_head
     * tail) we just skip — the next call will sync-load. */
    int idx = drive_order_index(m, t);
    if (idx >= 0) {
        int next = idx + 1;
        if (next >= m->drive_pq_order_len) next = 0;   /* wrap to next decode step */
        const ib_tensor_meta *t_next = m->drive_pq_order[next];
        int next_slot = ready_slot ^ 1;
        pthread_mutex_lock(&st->mu);
        st->req_tensor = t_next;
        st->req_slot = next_slot;
        st->req_pending = 1;
        pthread_cond_signal(&st->req_cv);
        pthread_mutex_unlock(&st->mu);
    }

#if !defined(__APPLE__) && defined(POSIX_FADV_DONTNEED)
    /* Solution 4: on Linux, drop the just-read region from the page
     * cache so subsequent matmuls aren't biased by it. No-op on Darwin. */
    (void)posix_fadvise(m->drive_fd, off, (off_t)bytes, POSIX_FADV_DONTNEED);
#endif
    return 0;
}

/* ── FP16 conversion ────────────────────────────────────────── */

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            /* Zero */
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Subnormal */
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        /* Inf/NaN */
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* Convert FP16 scale array to FP32 (temporary buffer) */
static void scales_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* Read FP16 weight as FP32 (for norms) */
static void fp16_weights_to_fp32(float* out, const void* fp16_data, int count) {
    const uint16_t* src = (const uint16_t*)fp16_data;
    for (int i = 0; i < count; i++) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

/* ── Embedding lookup ───────────────────────────────────────── */

/* Non-static: also used by inferbit_forward_with_hiddens (forward_hiddens.c)
 * to decode token IDs into fp32 embeddings for the Metal prefill path. */
void ib_embedding_lookup(const inferbit_model* m, int token_id, float* out);
void ib_embedding_lookup(const inferbit_model* m, int token_id, float* out) {
    int hidden = m->header.hidden_size;
    const ib_tensor_meta* emb = &m->token_embedding;

    if (emb->pq) {
        /* PQv2 embedding: decode one row. */
        const pqv2_t* pq = emb->pq;
        uint32_t G = pq->G;
        uint32_t ns = pq->n_subchunks;
        uint32_t K = pq->K;
        uint32_t HALF = pq->half;
        uint32_t nc = pq->N / G;
        uint32_t total = nc * ns;
        const int8_t* cb_q = (const int8_t*)pq->cb_q;           /* [ns][K][HALF] */
        const uint16_t* cb_s = (const uint16_t*)pq->cb_scale;   /* [ns][K] */
        float rs = pq->row_scale ? fp16_to_fp32(((const uint16_t*)pq->row_scale)[token_id]) : 1.0f;

        /* Doc-35 feature 1: if the pre-transposed sidecar is built and
         * the embedding has a sidecar entry, pread one ROW (total bytes)
         * from the sidecar instead of mmap-reading `total` widely-strided
         * single bytes. Sidecar layout is [token][total] so a row is
         * one contiguous pread = ~1024 bytes. Keeps the source mmap
         * region cold (cache-eviction-friendly). */
        if (m->residency_mode == 1 && m->drive_fd_pretransposed >= 0
            && pq->indices_pretransposed_offset != 0) {
            uint8_t row_buf[2048];   /* nc*ns ≤ 2048 in practice */
            if (total > sizeof(row_buf)) goto embed_mmap_path;
            off_t off = (off_t)pq->indices_pretransposed_offset
                      + (off_t)token_id * (off_t)total;
            size_t done = 0;
            while (done < total) {
                ssize_t r = pread(m->drive_fd_pretransposed,
                                  row_buf + done, total - done,
                                  off + (off_t)done);
                if (r <= 0) { if (r == -1 && errno == EINTR) continue; goto embed_mmap_path; }
                done += (size_t)r;
            }
            for (uint32_t c = 0; c < nc; c++) {
                for (uint32_t s = 0; s < ns; s++) {
                    uint8_t k = row_buf[c * ns + s];
                    float scl = fp16_to_fp32(cb_s[s * K + k]) * rs;
                    for (uint32_t h = 0; h < HALF; h++) {
                        int8_t q = cb_q[(s * K + k) * HALF + h];
                        out[c * G + s * HALF + h] = (float)q * scl;
                    }
                }
            }
            return;
        }

embed_mmap_path:
        {
            const uint8_t* idx_base = (const uint8_t*)pq->indices;
            /* Bug N16 — when the L1 indices on disk are row-major
             * ([M][n_chunks][n_subchunks], opt-in via IB_PQV2_L1_ROWMAJOR=1),
             * the per-token byte for (c, s) lives at
             *   token_id * total + c * n_sub + s
             * rather than the legacy chunk-major
             *   (c * n_sub + s) * M + token_id
             * Without this branch, embedding lookup reads garbage for every
             * token, poisoning the rest of the forward and blowing PPL up
             * (168450 on TinyLlama). Same logical byte is fetched in both
             * layouts; the indices are byte-equivalent (see encoder
             * pqv2_encode.c::pqv2_encode_slot_worker scatter). */
            if (pq->l1_idx_layout == 1) {
                size_t row_base = (size_t)token_id * (size_t)total;
                for (uint32_t c = 0; c < nc; c++) {
                    for (uint32_t s = 0; s < ns; s++) {
                        uint8_t k = idx_base[row_base + c * ns + s];
                        float scl = fp16_to_fp32(cb_s[s * K + k]) * rs;
                        for (uint32_t h = 0; h < HALF; h++) {
                            int8_t q = cb_q[(s * K + k) * HALF + h];
                            out[c * G + s * HALF + h] = (float)q * scl;
                        }
                    }
                }
            } else {
                /* Legacy chunk-major: idx[(c*ns+s)*M + token_id] */
                for (uint32_t c = 0; c < nc; c++) {
                    for (uint32_t s = 0; s < ns; s++) {
                        uint8_t k = idx_base[((size_t)c * ns + s) * pq->M + token_id];
                        float scl = fp16_to_fp32(cb_s[s * K + k]) * rs;
                        for (uint32_t h = 0; h < HALF; h++) {
                            int8_t q = cb_q[(s * K + k) * HALF + h];
                            out[c * G + s * HALF + h] = (float)q * scl;
                        }
                    }
                }
            }
        }
        return;
    }
    if (emb->bits == 8) {
        /* INT8 embedding: dequantize row */
        const int8_t* data = (const int8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        const int8_t* row = data + (size_t)token_id * hidden;

        if (scales_raw) {
            /* Per-row scale factor */
            const uint16_t* scales_fp16 = (const uint16_t*)scales_raw;
            float scale = fp16_to_fp32(scales_fp16[token_id]);
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i] * scale;
            }
        } else {
            for (int i = 0; i < hidden; i++) {
                out[i] = (float)row[i];
            }
        }
    } else if (emb->bits == 16) {
        /* FP16 embedding */
        const uint16_t* data = (const uint16_t*)tensor_data(m, emb);
        const uint16_t* row = data + (size_t)token_id * hidden;
        for (int i = 0; i < hidden; i++) {
            out[i] = fp16_to_fp32(row[i]);
        }
    } else if (emb->bits == 4) {
        /* INT4 embedding */
        const uint8_t* data = (const uint8_t*)tensor_data(m, emb);
        const void* scales_raw = tensor_scales_raw(m, emb);
        size_t row_bytes = (size_t)hidden / 2;
        const uint8_t* row = data + (size_t)token_id * row_bytes;
        float scale = 1.0f;
        if (scales_raw) {
            scale = fp16_to_fp32(((const uint16_t*)scales_raw)[token_id]);
        }
        for (int i = 0; i < hidden; i += 2) {
            uint8_t byte = row[i / 2];
            out[i]     = (float)((int8_t)(byte & 0x0F) - 8) * scale;
            out[i + 1] = (float)((int8_t)((byte >> 4) & 0x0F) - 8) * scale;
        }
    }
}

/* ── Matmul dispatch ────────────────────────────────────────── */

/*
 * Run quantized matmul for a tensor: out[M] = weights[M,N] @ input[N]
 * Handles bit-width dispatch and scale conversion.
 * `scale_buf` is a caller-provided temporary buffer of at least M floats.
 */
static void pqv2_matvec_dispatch(const pqv2_t *t, const float *x, float *y) {
    if (t->K == 256)      pqv2_matvec_tbl_int8_k256(t, x, y);
    else if (t->K == 128) pqv2_matvec_tbl_int8_k128(t, x, y);
    else if (t->K <= 64)  pqv2_matvec_tbl_int8(t, x, y);
    else                  pqv2_matvec_lut(t, x, y);
}

/* Per-chunk threading: each worker processes a slice of chunks, accumulating
 * into its own thread-local acc[M]. Main thread reduces across workers and
 * applies row_scale + L2 contribution.
 *
 * Why per-chunk and not per-row: the kernel builds an LUT per (chunk, subchunk)
 * that's INDEPENDENT of M but DEPENDS on x. Per-row threading would force
 * each worker to redundantly rebuild every LUT (4× total LUT-build work).
 * Per-chunk threading distributes LUT-build evenly with no redundancy. */
typedef struct {
    const pqv2_t *t;
    const float  *x;
    float        *acc_pool;
    float        *acc_l2_pool;
    uint32_t      M;
    int           chunk_size;
    int           n_slots;
    float         skip_thresh;   /* 0 = no skip */
} ib_pqv2_chunks_arg;

static void ib_pqv2_chunks_task(void *arg, int tid, int start, int end) {
    (void)tid;
    const ib_pqv2_chunks_arg *a = (const ib_pqv2_chunks_arg*)arg;
    int slot = start / a->chunk_size;
    if (slot < 0) slot = 0;
    if (slot >= a->n_slots) slot = a->n_slots - 1;
    float *acc    = a->acc_pool    + (size_t)slot * a->M;
    float *acc_l2 = a->acc_l2_pool ? a->acc_l2_pool + (size_t)slot * a->M : NULL;
    if (a->skip_thresh > 0.0f) {
        pqv2_acc_tbl_int8_k256_chunks_skip(a->t, a->x,
                                              a->t->cb_fp32, a->t->l2_cb_fp32,
                                              acc, acc_l2,
                                              (uint32_t)start, (uint32_t)end,
                                              a->skip_thresh);
    } else {
        pqv2_acc_tbl_int8_k256_chunks(a->t, a->x,
                                        a->t->cb_fp32, a->t->l2_cb_fp32,
                                        acc, acc_l2,
                                        (uint32_t)start, (uint32_t)end);
    }
}

/* Forward decl for the single-position threaded variant (defined below). */
static void pqv2_threaded_matvec_k256(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x, float *y);

/* Batched-aware variant of the per-chunk threading. Same chunk-to-slot
 * mapping as the single-position threaded path so each output position's
 * fp32 summation order is bit-identical between single-token decode and
 * spec verify. acc pool layout: [n_slots, B, M]. acc_l2_pool (same
 * layout, NULL when the tensor has no L2 stage) carries the pyramid
 * residual contribution, folded into y at reduction time. */
typedef struct {
    const pqv2_t *t;
    const float  *x_batch;
    int           B;
    float        *acc_pool;
    float        *acc_l2_pool;
    uint32_t      M;
    int           chunk_size;
    int           n_slots;
} ib_pqv2_chunks_batch_arg;

static void ib_pqv2_chunks_batch_task(void *arg, int tid, int start, int end) {
    (void)tid;
    const ib_pqv2_chunks_batch_arg *a = (const ib_pqv2_chunks_batch_arg*)arg;
    int slot = start / a->chunk_size;
    if (slot < 0) slot = 0;
    if (slot >= a->n_slots) slot = a->n_slots - 1;
    /* Slot owns a [B, M] block in each pool. */
    float *acc    = a->acc_pool    + (size_t)slot * a->B * a->M;
    float *acc_l2 = a->acc_l2_pool ? a->acc_l2_pool + (size_t)slot * a->B * a->M
                                   : NULL;
    pqv2_acc_tbl_int8_k256_chunks_batch(a->t, a->x_batch, a->B,
                                          a->t->cb_fp32, a->t->l2_cb_fp32,
                                          acc, acc_l2,
                                          (uint32_t)start, (uint32_t)end);
}

static void pqv2_threaded_matvec_k256_batch(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x_batch, int B, float *y_batch)
{
    if (B <= 0) return;
    if (B == 1) {
        pqv2_threaded_matvec_k256(m, tp, n_threads, t, x_batch, y_batch);
        return;
    }
    uint32_t M = t->M;
    uint32_t n_chunks = t->N / t->G;
    /* Match the single-position threaded path: L2 is engaged iff l2_kind==2,
     * the fp32 L2 codebook is present, and l2_K <= 64 (kernel constraint).
     * Otherwise the per-position fallback pqv2_matvec_tbl_int8_k256_batch
     * routes through pqv2_matvec_tbl_int8_k256, which already handles L2
     * correctly, so spec verify on pyramid tensors stays bit-identical
     * to single-token decode for any tensor that bails out of threading. */
    if (!tp || n_threads <= 1 || n_chunks < (uint32_t)n_threads ||
        !t->cb_fp32 || t->K != 256 || B > 8) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
        return;
    }
    int has_l2 = (t->l2_kind == 2 && t->l2_cb_fp32 && t->l2_K <= 64);
    int chunks_per_task = ((int)n_chunks + n_threads - 1) / n_threads;
    int n_slots = ((int)n_chunks + chunks_per_task - 1) / chunks_per_task;
    size_t pool_floats = (size_t)n_slots * B * M;
    float *acc_pool = aligned_alloc(64,
        (pool_floats * sizeof(float) + 63) & ~(size_t)63);
    if (!acc_pool) {
        pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
        return;
    }
    memset(acc_pool, 0, pool_floats * sizeof(float));
    float *acc_l2_pool = NULL;
    int acc_l2_owned = 0;
    if (has_l2) {
        /* Reuse model-scope L2 scratch when it fits (typical for B=1
         * decode where pool_floats = n_slots × M ≤ n_threads × max_M).
         * Larger batches (B>1) fall through to a fresh aligned_alloc. */
        if (m && m->pqv2_thread_acc_l2_pool &&
            pool_floats <= m->pqv2_thread_acc_l2_pool_floats) {
            acc_l2_pool = m->pqv2_thread_acc_l2_pool;
        } else {
            acc_l2_pool = aligned_alloc(64,
                (pool_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!acc_l2_pool) {
                /* L2 alloc failed: fall back to the per-position path that
                 * routes through the single-position matvec (which handles
                 * L2 correctly), so we never silently drop the pyramid
                 * residual. */
                free(acc_pool);
                pqv2_matvec_tbl_int8_k256_batch(t, x_batch, B, y_batch);
                return;
            }
            acc_l2_owned = 1;
        }
        memset(acc_l2_pool, 0, pool_floats * sizeof(float));
    }
    ib_pqv2_chunks_batch_arg arg = {
        .t = t, .x_batch = x_batch, .B = B,
        .acc_pool = acc_pool, .acc_l2_pool = acc_l2_pool, .M = M,
        .chunk_size = chunks_per_task, .n_slots = n_slots,
    };
    ib_pool_run(tp, ib_pqv2_chunks_batch_task, &arg,
                 (int)n_chunks, chunks_per_task);

    /* Reduce per-position: y[b,m] = (sum_s acc[s,b,m]) * row_scale[m]
     *                              + sum_s acc_l2[s,b,m].
     * Slot order is fixed (s=0..n_slots-1) so this matches the
     * single-position threaded reduction at forward.c::pqv2_threaded_matvec_k256
     * exactly when B=1. */
    for (int b = 0; b < B; b++) {
        float *yb = y_batch + (size_t)b * M;
        for (uint32_t mm = 0; mm < M; mm++) {
            float a = 0.0f, al2 = 0.0f;
            for (int s = 0; s < n_slots; s++) {
                a += acc_pool[(size_t)s * B * M + (size_t)b * M + mm];
                if (acc_l2_pool) {
                    al2 += acc_l2_pool[(size_t)s * B * M + (size_t)b * M + mm];
                }
            }
            float rs = pqv2_h2f(t->row_scale[mm]);
            yb[mm] = a * rs + al2;
        }
    }
    free(acc_pool);
    if (acc_l2_owned) free(acc_l2_pool);
}

static void pqv2_threaded_matvec_k256(
    const inferbit_model *m,
    struct ib_thread_pool *tp, int n_threads,
    const pqv2_t *t, const float *x, float *y)
{
    uint32_t M = t->M;
    uint32_t n_chunks = t->N / t->G;
    /* Bail out to single-thread when threading wouldn't pay off. */
    if (!tp || n_threads <= 1 || n_chunks < (uint32_t)n_threads ||
        !t->cb_fp32 || t->K != 256) {
        pqv2_matvec_dispatch(t, x, y);
        return;
    }
    int has_l2 = (t->l2_kind == 2 && t->l2_cb_fp32 && t->l2_K <= 64);
    int chunks_per_task = ((int)n_chunks + n_threads - 1) / n_threads;
    int n_slots = ((int)n_chunks + chunks_per_task - 1) / chunks_per_task;
    size_t pool_floats = (size_t)n_slots * M;
    /* Use model-scope scratch to avoid per-call aligned_alloc. The
     * scratch is sized for n_threads × max_M; fall back to a fresh
     * malloc only if (somehow) the request exceeds that budget. */
    float *acc_pool;
    int acc_pool_owned = 0;
    if (m && m->pqv2_thread_acc_pool && pool_floats <= m->pqv2_thread_acc_pool_floats) {
        acc_pool = m->pqv2_thread_acc_pool;
    } else {
        acc_pool = aligned_alloc(64,
            (pool_floats * sizeof(float) + 63) & ~(size_t)63);
        if (!acc_pool) { pqv2_matvec_dispatch(t, x, y); return; }
        acc_pool_owned = 1;
    }
    memset(acc_pool, 0, pool_floats * sizeof(float));
    float *acc_l2_pool = NULL;
    int acc_l2_owned = 0;
    if (has_l2) {
        /* Prefer model-scope L2 scratch (sized n_threads × max_M, same as
         * acc_pool). Falls back to fresh aligned_alloc only if the request
         * somehow exceeds the budget. Eliminates ~88 aligned_alloc/free per
         * decode token on pyramid models. */
        if (m && m->pqv2_thread_acc_l2_pool &&
            pool_floats <= m->pqv2_thread_acc_l2_pool_floats) {
            acc_l2_pool = m->pqv2_thread_acc_l2_pool;
        } else {
            acc_l2_pool = aligned_alloc(64,
                (pool_floats * sizeof(float) + 63) & ~(size_t)63);
            if (!acc_l2_pool) {
                /* L2 alloc failed: do NOT silently run the threaded path with
                 * acc_l2_pool=NULL — the kernel would skip the L2 path entirely
                 * (see ib_pqv2_chunks_task → pqv2_acc_tbl_int8_k256_chunks_inner;
                 * acc_l2==NULL means "no L2"), which DROPS the pyramid residual
                 * and degrades a pyramid (l2_kind==2) model to a flat one
                 * (60% PPL regression observed on tl-pyramid.ibf in RAM mode
                 * where the mmap'd weights leave less headroom for the
                 * per-matmul aligned_alloc; drive mode evicts those pages and
                 * the alloc succeeds, which is why drive PPL was BETTER than
                 * RAM PPL — the bug is RAM-mode-only). Fall back to the
                 * single-threaded matvec, which uses one acc/acc_l2 pair the
                 * size of a single matvec (M floats each) and is allocated
                 * fresh inside pqv2_matvec_tbl_int8_k256. */
                if (acc_pool_owned) free(acc_pool);
                pqv2_matvec_dispatch(t, x, y);
                return;
            }
            acc_l2_owned = 1;
        }
        memset(acc_l2_pool, 0, pool_floats * sizeof(float));
    }
    /* Activation-aware skip: when IB_PQV2_SKIP env is set (e.g. "0.01"),
     * skip (c,s) iters with max|x_slice| < ratio * max|x|. 1% threshold
     * is essentially lossless on transformer activations. Skip rate
     * naturally adapts: outlier-heavy early layers skip a lot, diffuse
     * later layers skip little. */
    float skip_thresh = 0.0f;
    {
        const char *env = getenv("IB_PQV2_SKIP");
        if (env && env[0]) {
            float ratio = (float)atof(env);
            if (ratio > 0.0f && ratio < 1.0f) {
                float xmax = 0.0f;
                for (uint32_t i = 0; i < t->N; i++) {
                    float v = x[i]; if (v < 0) v = -v;
                    if (v > xmax) xmax = v;
                }
                skip_thresh = ratio * xmax;
            }
        }
    }
    ib_pqv2_chunks_arg arg = {
        .t = t, .x = x,
        .acc_pool = acc_pool, .acc_l2_pool = acc_l2_pool,
        .M = M,
        .chunk_size = chunks_per_task,
        .n_slots = n_slots,
        .skip_thresh = skip_thresh,
    };
    ib_pool_run(tp, ib_pqv2_chunks_task, &arg, (int)n_chunks, chunks_per_task);

    /* Reduce: sum across deterministic slot order, then apply row_scale + L2 */
    for (uint32_t m = 0; m < M; m++) {
        float a = 0.0f, al2 = 0.0f;
        for (int s = 0; s < n_slots; s++) {
            a += acc_pool[(size_t)s * M + m];
            if (acc_l2_pool) al2 += acc_l2_pool[(size_t)s * M + m];
        }
        float rs = pqv2_h2f(t->row_scale[m]);
        y[m] = a * rs + al2;
    }
    if (acc_pool_owned) free(acc_pool);
    if (acc_l2_owned) free(acc_l2_pool);
}

static void tensor_matmul(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf
) {
    /* PQv2 dispatch — takes precedence when present. Per-chunk threading
     * for K=256; falls back to single-thread for other K or no pool. */
    if (t->pq) {
        const pqv2_t* pq = t->pq;
        /* Path D drive mode (Solution 5): pread the indices from disk
         * into the model's scratch buffer (which pq->indices was
         * redirected to at load). Kernel then reads from scratch. */
        if (m->residency_mode == 1) {
            (void)drive_load_indices(m, t);
        }
        if (pq->K == 256 && m->thread_pool && m->num_threads > 1) {
            pqv2_threaded_matvec_k256(m, m->thread_pool, m->num_threads,
                                        pq, input, out);
        } else {
            pqv2_matvec_dispatch(pq, input, out);
        }
        return;
    }

    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Detect per-block-32 INT4 scaling: scale_size > rows*2 ⇒ N/32 fp16
     * scales per row instead of one. Triggered by IB_INT4_BLK32 at convert
     * time. The new kernel handles a flat fp32 buffer of M*(N/32) scales.
     *
     * Perf (doc 36): the fp16 scale buffer is a STATIC property of the
     * weight tensor — pre-decoded into t->scales_fp32 / t->blk32_scales_fp32
     * by ib_cache_model_static_fp32() at load time. We pick those up here
     * and skip the per-call fp16→fp32 conversion. The local fallback path
     * stays in place for the (rare) case where the cache wasn't built. */
    int is_blk32_int4 = (t->bits == 4 && t->scale_size > (size_t)M * 2);
    const float* scales_eff = NULL;       /* per-row scales (M) */
    const float* blk32_scales = NULL;     /* M * (N/32) scales */
    float* blk32_owned = NULL;            /* malloc'd fallback only */
    if (is_blk32_int4) {
        if (t->blk32_scales_fp32) {
            blk32_scales = t->blk32_scales_fp32;
        } else {
            size_t total = (size_t)M * (size_t)(N / 32);
            blk32_owned = (float*)malloc(total * sizeof(float));
            if (blk32_owned) {
                scales_to_fp32(blk32_owned, scales_raw, (int)total);
                blk32_scales = blk32_owned;
            } else {
                is_blk32_int4 = 0;   /* fall back if alloc failed */
            }
        }
    }
    /* Per-row scales used by all non-blk32 paths AND as a safety fallback
     * if is_blk32_int4 is set but matmul_w4a8_blk32 is unavailable
     * (matches the original code's fall-through). */
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    if (is_blk32_int4 && ib_kern.matmul_w4a8_blk32) {
        /* Per-block-32 INT4 path: quantize input as usual, dispatch to the
         * blk32-aware kernel. No batched/parallel wrapper for now — the
         * scalar kernel is single-threaded.
         *
         * Hot-path scratch: prefer model-lifetime bb_qscratch/bb_sa (sized
         * for IB_BATCH_MAX*n_max, which always covers a single-position
         * matmul) over per-call malloc. Stack fallback retained for the
         * case where the model wasn't built with batch scratch. */
        int8_t stack_q[4096];
        float  stack_s[4096 / IB_W4A8_GROUP + 1];
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        int8_t* q_buf;
        float*  s_buf;
        int q_buf_owned = 0, s_buf_owned = 0;
        if (m->bb_qscratch && m->bb_sa) {
            q_buf = m->bb_qscratch;
            s_buf = m->bb_sa;
        } else {
            q_buf = (N <= 4096) ? stack_q : (int8_t*)malloc((size_t)N);
            s_buf = (n_groups <= (int)(sizeof stack_s / sizeof *stack_s))
                        ? stack_s
                        : (float*)malloc((size_t)n_groups * sizeof(float));
            q_buf_owned = (q_buf != stack_q);
            s_buf_owned = (s_buf != stack_s);
        }
        ib_quantize_input_int8_g128(input, q_buf, s_buf, N);
        ib_kern.matmul_w4a8_blk32(out, weights, blk32_scales,
                                  q_buf, s_buf, M, N);
        if (q_buf_owned) free(q_buf);
        if (s_buf_owned) free(s_buf);
        if (blk32_owned) free(blk32_owned);
        return;
    }
    if (blk32_owned) free(blk32_owned);

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8) {
        /* Quantize input to INT8 per-group (IB_W4A8_GROUP elements per
         * scale). Prefer model-lifetime scratch (bb_qscratch/bb_sa) so the
         * hot decode loop does no malloc for N > 4096 (e.g. MLP up/gate
         * with N=intermediate). Stack fallback retained for legacy paths
         * where the model isn't initialised with batch scratch. */
        int8_t stack_q[4096];
        float  stack_s[4096 / IB_W4A8_GROUP + 1];
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        int8_t* q_buf;
        float*  s_buf;
        int q_buf_owned = 0, s_buf_owned = 0;
        if (m->bb_qscratch && m->bb_sa) {
            q_buf = m->bb_qscratch;
            s_buf = m->bb_sa;
        } else {
            q_buf = (N <= 4096) ? stack_q : (int8_t*)malloc((size_t)N);
            s_buf = (n_groups <= (int)(sizeof stack_s / sizeof *stack_s))
                        ? stack_s
                        : (float*)malloc((size_t)n_groups * sizeof(float));
            q_buf_owned = (q_buf != stack_q);
            s_buf_owned = (s_buf != stack_s);
        }
        ib_quantize_input_int8_g128(input, q_buf, s_buf, N);
        ib_parallel_matmul_w4a8(m->thread_pool, out, weights, scales_eff,
                                q_buf, s_buf, M, N);
        if (q_buf_owned) free(q_buf);
        if (s_buf_owned) free(s_buf);
    } else if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
        ib_parallel_matmul(m->thread_pool, out, weights, scales_eff, input, M, N, t->bits);
    } else if (t->bits == 16) {
        const uint16_t* w = (const uint16_t*)weights;
        for (int i = 0; i < M; i++) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += fp16_to_fp32(w[i * N + j]) * input[j];
            }
            out[i] = sum;
        }
    }
}

/* Non-static thin wrapper exposing tensor_matmul to other TUs.
 * Declared in inferbit_internal.h. Used by src/mome.c so the MoME
 * dispatcher can run a per-expert matmul without forward.c growing
 * a public PQv2/W4A8/etc. dispatch surface. */
void ib_tensor_matmul_cpu(const inferbit_model *m, const ib_tensor_meta *t,
                          float *out, const float *input, int M, int N,
                          float *scale_buf) {
    tensor_matmul(m, t, out, input, M, N, scale_buf);
}

/* Batched variant of tensor_matmul.
 *
 *   out    [B * M]  row-major, out[b*M + i]
 *   input  [B * N]  row-major, input[b*N + j]
 *   scale_buf: caller-provided, at least M floats (weight scales).
 *   q_scratch: only used for INT4+W4A8 path — B * N int8 bytes for quantized
 *              activations, plus B * ceil(N/IB_W4A8_GROUP) floats for scales.
 *              Caller supplies both to avoid malloc in the hot loop. Pass
 *              NULL for paths that don't need them (INT8, FP16).
 *
 * Same dispatch policy as tensor_matmul: INT4 routes through W4A8 batched
 * kernel when enabled; INT8 uses matmul_int8_batch; FP16 falls back to the
 * sequential FP16 path because we don't have a batched FP16 kernel. */
static void tensor_matmul_batch(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N, int B,
    float* scale_buf, int8_t* q_scratch, float* sa_scratch
) {
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Perf: prefer load-time-cached fp32 scales (see ib_cache_model_static_fp32). */
    const float* scales_eff;
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    /* PQv2 batched path: per-chunk threading shared across B positions.
     * Each chunk slot contributes to ALL B output positions, so weight
     * reads are amortised across B. Same chunk-to-slot partition as
     * the single-position threaded path → identical fp32 sum order. */
    if (t->pq) {
        const pqv2_t* pq = t->pq;
        /* Drive mode: pread indices ONCE for this tensor; the batched
         * kernel below reuses the same scratch for all B positions. */
        if (m->residency_mode == 1) {
            (void)drive_load_indices(m, t);
        }
        if (pq->K == 256 && B >= 1 && B <= 8 &&
            m->thread_pool && m->num_threads > 1) {
            pqv2_threaded_matvec_k256_batch(m, m->thread_pool, m->num_threads,
                                              pq, input, B, out);
            return;
        }
        if (pq->K == 256 && B > 1 && B <= 8) {
            pqv2_matvec_tbl_int8_k256_batch(pq, input, B, out);
            return;
        }
        for (int b = 0; b < B; b++) {
            /* Recursive call will re-pread; could optimize later by
             * not re-loading scratch within the same tensor. */
            tensor_matmul(m, t, out + (size_t)b * M, input + (size_t)b * N,
                          M, N, scale_buf);
        }
        return;
    }

    if (t->bits == 4 && w4a8_enabled() && ib_kern.matmul_w4a8_batch && q_scratch && sa_scratch) {
        int n_groups = (N + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
        for (int b = 0; b < B; b++) {
            ib_quantize_input_int8_g128(input + (size_t)b * N,
                                        q_scratch + (size_t)b * N,
                                        sa_scratch + (size_t)b * n_groups, N);
        }
        ib_parallel_matmul_w4a8_batch(m->thread_pool, out, weights, scales_eff,
                                      q_scratch, sa_scratch, M, N, B);
    } else if (t->bits == 8 && ib_kern.matmul_int8_batch) {
        ib_parallel_matmul_int8_batch(m->thread_pool, out, weights, scales_eff,
                                      input, M, N, B);
    } else {
        /* Fallback: per-position sequential. */
        for (int b = 0; b < B; b++) {
            float* out_b = out + (size_t)b * M;
            const float* in_b = input + (size_t)b * N;
            if (t->bits == 2 || t->bits == 4 || t->bits == 8) {
                ib_parallel_matmul(m->thread_pool, out_b, weights, scales_eff,
                                   in_b, M, N, t->bits);
            } else if (t->bits == 16) {
                const uint16_t* w = (const uint16_t*)weights;
                for (int i = 0; i < M; i++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) {
                        sum += fp16_to_fp32(w[(size_t)i * N + j]) * in_b[j];
                    }
                    out_b[i] = sum;
                }
            }
        }
    }
}

/*
 * Sparse matmul: same as tensor_matmul but skips rows where mask[row] == 0.
 * Outputs zero for skipped rows. mask is a byte array of length M.
 */
static void tensor_matmul_sparse(
    const inferbit_model* m, const ib_tensor_meta* t,
    float* out, const float* input, int M, int N,
    float* scale_buf, const uint8_t* mask
) {
    if (!mask) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        return;
    }

    /* Count active rows and build index */
    int active = 0;
    for (int i = 0; i < M; i++) {
        if (mask[i]) active++;
    }

    /* If most rows are active (>80%), just run dense — overhead of sparse indexing isn't worth it */
    if (active > M * 4 / 5) {
        tensor_matmul(m, t, out, input, M, N, scale_buf);
        /* Zero out masked rows */
        for (int i = 0; i < M; i++) {
            if (!mask[i]) out[i] = 0.0f;
        }
        return;
    }

    /* Sparse path: only compute active rows */
    const void* weights = tensor_data(m, t);
    const void* scales_raw = tensor_scales_raw(m, t);

    /* Perf: prefer the load-time-cached fp32 scales. */
    const float* scales_eff;
    if (t->scales_fp32) {
        scales_eff = t->scales_fp32;
    } else if (scales_raw) {
        scales_to_fp32(scale_buf, scales_raw, M);
        scales_eff = scale_buf;
    } else {
        for (int i = 0; i < M; i++) scale_buf[i] = 1.0f;
        scales_eff = scale_buf;
    }

    /* Zero entire output first */
    memset(out, 0, M * sizeof(float));

    /* Compute only active rows */
    for (int i = 0; i < M; i++) {
        if (!mask[i]) continue;

        float sum = 0.0f;
        if (t->bits == 8) {
            const int8_t* w = (const int8_t*)weights + (size_t)i * N;
            for (int j = 0; j < N; j++) sum += (float)w[j] * input[j];
        } else if (t->bits == 4) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 2);
            for (int j = 0; j < N; j += 2) {
                uint8_t byte = w[j / 2];
                sum += (float)((int8_t)(byte & 0x0F) - 8) * input[j];
                if (j + 1 < N) sum += (float)((int8_t)((byte >> 4) & 0x0F) - 8) * input[j + 1];
            }
        } else if (t->bits == 2) {
            const uint8_t* w = (const uint8_t*)weights + (size_t)i * (N / 4);
            for (int j = 0; j < N; j += 4) {
                uint8_t byte = w[j / 4];
                sum += (float)((byte & 0x03) - 1) * input[j];
                if (j+1 < N) sum += (float)(((byte >> 2) & 0x03) - 1) * input[j+1];
                if (j+2 < N) sum += (float)(((byte >> 4) & 0x03) - 1) * input[j+2];
                if (j+3 < N) sum += (float)(((byte >> 6) & 0x03) - 1) * input[j+3];
            }
        }
        out[i] = sum * scales_eff[i];
    }
}

/* ── RMSNorm with FP16 weights ──────────────────────────────── */

static void rmsnorm_fp16(float* out, const float* input,
                         const void* weight_fp16, float eps, int N,
                         float* weight_buf) {
    fp16_weights_to_fp32(weight_buf, weight_fp16, N);
    ib_kern.rmsnorm(out, input, weight_buf, eps, N);
}

/* Tensor-aware RMSNorm: prefer the load-time-cached fp32 norm weight
 * (t->norm_fp32) and skip the per-call fp16→fp32 conversion. Falls back
 * to the legacy decode path when no cache is present. */
static inline void rmsnorm_fp16_t(float* out, const float* input,
                                  const inferbit_model* m,
                                  const ib_tensor_meta* t,
                                  float eps, int N, float* weight_buf) {
    if (t->norm_fp32) {
        ib_kern.rmsnorm(out, input, t->norm_fp32, eps, N);
    } else {
        rmsnorm_fp16(out, input, tensor_data(m, t), eps, N, weight_buf);
    }
}

/* ── KV cache quantization helpers ──────────────────────────── */

static inline void kv_write_int4_row(uint8_t* dst, const float* src, float scale, int n) {
    float inv = 1.0f / scale;
    for (int i = 0; i < n; i += 2) {
        int q0 = (int)roundf(src[i] * inv);
        int q1 = (i + 1 < n) ? (int)roundf(src[i + 1] * inv) : 0;
        if (q0 < -7) q0 = -7; if (q0 > 7) q0 = 7;
        if (q1 < -7) q1 = -7; if (q1 > 7) q1 = 7;
        uint8_t lo = (uint8_t)(q0 + 8) & 0x0F;
        uint8_t hi = (uint8_t)(q1 + 8) & 0x0F;
        dst[i / 2] = lo | (hi << 4);
    }
}

static inline void kv_read_int4_row(float* out, const uint8_t* src, float scale, int n) {
    for (int i = 0; i < n; i += 2) {
        uint8_t b = src[i / 2];
        out[i] = ((float)((int)(b & 0x0F) - 8)) * scale;
        if (i + 1 < n) out[i + 1] = ((float)((int)((b >> 4) & 0x0F) - 8)) * scale;
    }
}

static void kv_cache_write(ib_kv_cache* kv, int pos,
                           const float* key, const float* value,
                           int kv_dim, int n_kv_heads, int head_dim, int kv_bits) {
    /* Rotating KV window (doc 36 phase 2.2): logical position `pos` lands
     * in physical slot pos % capacity. When not windowed, capacity is the
     * full context so pos < capacity and this is the identity map. */
    int phys = (kv->capacity > 0) ? (pos % kv->capacity) : pos;
    if (kv_bits >= 16) {
        float* k_store = (float*)kv->key_data;
        float* v_store = (float*)kv->value_data;
        memcpy(k_store + (size_t)phys * kv_dim, key, kv_dim * sizeof(float));
        memcpy(v_store + (size_t)phys * kv_dim, value, kv_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        int8_t* k_store = (int8_t*)kv->key_data + (size_t)phys * kv_dim;
        int8_t* v_store = (int8_t*)kv->value_data + (size_t)phys * kv_dim;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 127.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 127.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)phys * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)phys * n_kv_heads + h] = v_scale;
            float k_inv = 1.0f / k_scale;
            float v_inv = 1.0f / v_scale;
            for (int d = 0; d < head_dim; d++) {
                int kq = (int)roundf(k_h[d] * k_inv);
                int vq = (int)roundf(v_h[d] * v_inv);
                if (kq < -127) kq = -127; if (kq > 127) kq = 127;
                if (vq < -127) vq = -127; if (vq > 127) vq = 127;
                k_store[h * head_dim + d] = (int8_t)kq;
                v_store[h * head_dim + d] = (int8_t)vq;
            }
        }
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        uint8_t* k_store = (uint8_t*)kv->key_data + (size_t)phys * row_bytes;
        uint8_t* v_store = (uint8_t*)kv->value_data + (size_t)phys * row_bytes;
        for (int h = 0; h < n_kv_heads; h++) {
            const float* k_h = key + h * head_dim;
            const float* v_h = value + h * head_dim;
            float k_max = 0.0f, v_max = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float ka = fabsf(k_h[d]); if (ka > k_max) k_max = ka;
                float va = fabsf(v_h[d]); if (va > v_max) v_max = va;
            }
            float k_scale = k_max / 7.0f; if (k_scale < 1e-8f) k_scale = 1e-8f;
            float v_scale = v_max / 7.0f; if (v_scale < 1e-8f) v_scale = 1e-8f;
            kv->key_scales[(size_t)phys * n_kv_heads + h] = k_scale;
            kv->value_scales[(size_t)phys * n_kv_heads + h] = v_scale;
            kv_write_int4_row(k_store + (size_t)h * ((head_dim + 1) / 2), k_h, k_scale, head_dim);
            kv_write_int4_row(v_store + (size_t)h * ((head_dim + 1) / 2), v_h, v_scale, head_dim);
        }
        return;
    }
}

static void kv_cache_read_head(const ib_kv_cache* kv, int is_key, int pos, int kv_head,
                               int kv_dim, int n_kv_heads, int head_dim, int kv_bits,
                               float* out_head) {
    /* Rotating KV window: logical position -> physical slot pos % capacity
     * (identity map when not windowed). */
    int phys = (kv->capacity > 0) ? (pos % kv->capacity) : pos;
    if (kv_bits >= 16) {
        const float* src = is_key ? (const float*)kv->key_data : (const float*)kv->value_data;
        const float* row = src + (size_t)phys * kv_dim + kv_head * head_dim;
        memcpy(out_head, row, head_dim * sizeof(float));
        return;
    }

    if (kv_bits == 8) {
        const int8_t* src = is_key ? (const int8_t*)kv->key_data : (const int8_t*)kv->value_data;
        const int8_t* row = src + (size_t)phys * kv_dim + kv_head * head_dim;
        float scale = is_key
            ? kv->key_scales[(size_t)phys * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)phys * n_kv_heads + kv_head];
        for (int d = 0; d < head_dim; d++) out_head[d] = (float)row[d] * scale;
        return;
    }

    if (kv_bits == 4) {
        size_t row_bytes = (size_t)(kv_dim + 1) / 2;
        const uint8_t* src = is_key ? (const uint8_t*)kv->key_data : (const uint8_t*)kv->value_data;
        const uint8_t* row = src + (size_t)phys * row_bytes + (size_t)kv_head * ((head_dim + 1) / 2);
        float scale = is_key
            ? kv->key_scales[(size_t)phys * n_kv_heads + kv_head]
            : kv->value_scales[(size_t)phys * n_kv_heads + kv_head];
        kv_read_int4_row(out_head, row, scale, head_dim);
        return;
    }
}

/* ── Parallel attention task ─────────────────────────────────── */

typedef struct {
    float* q;
    float* att;
    float* xb2;
    ib_kv_cache* kv;
    int head_dim;
    int kv_dim;
    int n_kv_heads;
    int heads_per_kv;
    int pos;
    int kv_bits;
    float scale;
} ib_attn_ctx;

static void ib_attn_head_task(void* arg, int tid, int start, int end) {
    (void)tid;
    ib_attn_ctx* c = (ib_attn_ctx*)arg;
    float k_tmp[256];
    float v_tmp[256];

    /* Rotating KV window (doc 36 phase 2.2): only the most recent
     * `capacity` positions are physically live; older ones were evicted.
     * When not windowed, capacity is the full context so t_lo is 0 and
     * this attends to everything (unchanged behaviour). The att row is
     * compacted into [0, n_valid) so softmax + weighted-V operate on the
     * live window only; logical position t reads physical slot t % cap. */
    int cap = c->kv->capacity;
    int t_lo = (cap > 0 && c->pos + 1 > cap) ? (c->pos + 1 - cap) : 0;
    int n_valid = c->pos + 1 - t_lo;

    for (int h = start; h < end; h++) {
        float* q_h = c->q + h * c->head_dim;
        int kv_h = h / c->heads_per_kv;
        float* att_h = c->att + h * (c->pos + 1);

        for (int t = t_lo; t <= c->pos; t++) {
            int phys = (cap > 0) ? (t % cap) : t;
            float score = 0.0f;
            if (c->kv_bits >= 16) {
                float* k_cache = (float*)c->kv->key_data;
                float* k_t = k_cache + (size_t)phys * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_t[d];
            } else {
                kv_cache_read_head(c->kv, 1, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, k_tmp);
                for (int d = 0; d < c->head_dim; d++) score += q_h[d] * k_tmp[d];
            }
            att_h[t - t_lo] = score * c->scale;
        }

        ib_kern.softmax(att_h, n_valid);

        float* out_h = c->xb2 + h * c->head_dim;
        memset(out_h, 0, c->head_dim * sizeof(float));
        for (int t = t_lo; t <= c->pos; t++) {
            int phys = (cap > 0) ? (t % cap) : t;
            float a = att_h[t - t_lo];
            if (c->kv_bits >= 16) {
                float* v_cache = (float*)c->kv->value_data;
                float* v_t = v_cache + (size_t)phys * c->kv_dim + kv_h * c->head_dim;
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_t[d];
            } else {
                kv_cache_read_head(c->kv, 0, t, kv_h, c->kv_dim, c->n_kv_heads,
                                   c->head_dim, c->kv_bits, v_tmp);
                for (int d = 0; d < c->head_dim; d++) out_h[d] += a * v_tmp[d];
            }
        }
    }
}

/* ── Single-token forward pass ──────────────────────────────── */

/* Extended single-token forward.
 *
 * compute_logits:
 *   0 — skip the final RMSNorm and LM head (prefill path, advances KV only)
 *   1 — compute logits via the LM head (default decode path)
 *
 * hidden_out: if non-NULL, writes the post-final-RMSNorm hidden state into
 *   hidden_out[hidden_size]. Used by the batched verify path to stack B
 *   positions' hidden states before a single batched LM head matmul. When
 *   hidden_out is supplied the final RMSNorm runs regardless of compute_logits. */
static int forward_single_ex(inferbit_model* m, int token_id, int pos,
                             float* logits, int compute_logits,
                             float* hidden_out) {
    int hidden   = m->header.hidden_size;
    int n_layers = m->header.num_layers;
    int n_heads  = m->header.num_heads;
    int n_kv     = m->header.num_kv_heads;
    int head_dim = m->header.head_dim;
    int inter    = m->header.intermediate_size;
    int vocab    = m->header.vocab_size;
    float eps    = m->header.norm_epsilon;
    float theta  = m->header.rope_theta;

    int kv_dim   = n_kv * head_dim;
    int heads_per_kv = n_heads / n_kv;  /* For GQA */

    /* Activation buffers */
    float* x       = m->buf_residual;   /* [hidden] — residual stream */
    float* xb      = m->buf_hidden;     /* [hidden] — after norm */
    float* xb2     = m->buf_attn;       /* [hidden] — scratch */
    float* hb      = m->buf_mlp;        /* [inter]  — MLP scratch */
    float* hb2     = m->buf_mlp2;       /* [inter]  — MLP scratch 2 */
    float* qkv_buf = m->buf_qkv;        /* Scratch for projections + attention scores */

    /* Partition qkv_buf:
     * q:     [hidden]
     * k:     [kv_dim]
     * v:     [kv_dim]
     * att:   [n_heads * (pos+1)] — attention scores
     * scale: [max(hidden, inter, vocab)] — scale factor temp buffer
     */
    float* q     = qkv_buf;
    float* k     = q + hidden;
    float* v     = k + kv_dim;
    float* att   = v + kv_dim;
    int scale_sz = hidden > inter ? hidden : inter;
    if (vocab > scale_sz) scale_sz = vocab;
    float* scale_buf = att + (size_t)n_heads * (pos + 1);

    /* Embedding lookup */
    ib_embedding_lookup(m, token_id, x);

    /* Transformer layers */
    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm before attention */
        rmsnorm_fp16_t(xb, x, m, &layer->input_norm,
                       eps, hidden, scale_buf);

        /* Q/K/V projections */
        tensor_matmul(m, &layer->q_proj, q, xb, hidden, hidden, scale_buf);
        tensor_matmul(m, &layer->k_proj, k, xb, kv_dim, hidden, scale_buf);
        tensor_matmul(m, &layer->v_proj, v, xb, kv_dim, hidden, scale_buf);

        /* RoPE: apply to each Q head paired with its corresponding K head.
         * For GQA, multiple Q heads share one K head. Apply RoPE to each
         * K head only once (on the first Q head that maps to it). */
        /* Precomputed RoPE tables (NULL = kernel falls back to live sinf/cosf). */
        const float* rope_cos_tab = (m->rope_cos && pos < m->rope_table_ctx) ? m->rope_cos : NULL;
        const float* rope_sin_tab = (m->rope_sin && pos < m->rope_table_ctx) ? m->rope_sin : NULL;
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / heads_per_kv;
            int is_first = (h % heads_per_kv == 0);
            if (is_first) {
                ib_kern.rope(q + h * head_dim, k + kv_h * head_dim,
                             head_dim, pos, theta, rope_cos_tab, rope_sin_tab);
            } else {
                /* Apply RoPE to Q only — use a scratch buffer for K */
                float k_scratch[256];
                memcpy(k_scratch, k + kv_h * head_dim, head_dim * sizeof(float));
                ib_kern.rope(q + h * head_dim, k_scratch, head_dim, pos, theta,
                             rope_cos_tab, rope_sin_tab);
                /* Discard k_scratch — K was already rotated */
            }
        }

        /* Write K, V to cache */
        kv_cache_write(kv, pos, k, v, kv_dim, n_kv, head_dim, m->header.kv_bits);
        kv->length = pos + 1;

        /* Multi-head attention (parallelized across heads) */
        float attn_scale = 1.0f / sqrtf((float)head_dim);

        ib_attn_ctx attn_ctx = {
            .q = q, .att = att, .xb2 = xb2,
            .kv = kv,
            .head_dim = head_dim, .kv_dim = kv_dim,
            .n_kv_heads = n_kv,
            .heads_per_kv = heads_per_kv,
            .pos = pos,
            .kv_bits = m->header.kv_bits,
            .scale = attn_scale,
        };

        if (m->thread_pool && n_heads >= 4) {
            ib_pool_run(m->thread_pool, ib_attn_head_task, &attn_ctx, n_heads, 0);
        } else {
            ib_attn_head_task(&attn_ctx, 0, 0, n_heads);
        }

        /* Output projection: xb = O_proj @ xb2 */
        tensor_matmul(m, &layer->o_proj, xb, xb2, hidden, hidden, scale_buf);

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }

        /* RMSNorm before MLP */
        rmsnorm_fp16_t(xb, x, m, &layer->post_attn_norm,
                       eps, hidden, scale_buf);

        /* MLP: gate + up + silu_mul + down
         * With sparsity: skip masked intermediate neurons entirely.
         *
         * Stage 3a — MoME router hook (docs/v2/00_CORRECTION.md). When
         * the layer carries mome_experts > 1 (= the file shipped K
         * expert sub-tensors), branch BEFORE the normal FFN dispatch:
         *
         *   - Router non-zero (calibrated): compute router_logits =
         *     xb @ router_weight, pick top-N indices via
         *     softmax-weighted top-N selection, and run only those
         *     experts through mome_dispatch_ffn. Output goes straight
         *     into xb (overwriting the post-norm input — same role as
         *     the legacy down_proj output).
         *
         *   - Router zero (v1 default, no calibration): run ALL K
         *     experts with weight 1.0. On the trivial row-split that
         *     produces the exact same result as the un-split FFN
         *     matmul — preserving the v1 correctness invariant
         *     ("MoME-enabled file = non-MoME file bit-for-bit, until a
         *     real router lands").
         *
         * Stage 5d hybrid hook: when gate/up/down carry a METAL
         * preferred_backend tag (set via IB_HYBRID_FFN_GPU=1) AND no
         * sparsity mask is active for this layer, route each FFN
         * matmul through tensor_matmul_hybrid which dispatches one
         * GPU matmul and copies the fp32 result back. Falls back
         * transparently to CPU when Metal is unavailable. The sparsity
         * path stays CPU-only — the Metal recorder has no sparse mask
         * variant yet, so sparsity wins when both are configured. */
        {
            const uint8_t* sp_mask = NULL;
            if (layer->sparsity_mask_size > 0) {
                sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            }

            int mome_handled = 0;
            if (!sp_mask && layer->mome_experts > 1 &&
                layer->gate_proj_experts && layer->up_proj_experts &&
                layer->down_proj_experts) {
                const int K_ex = layer->mome_experts;
                if (mome_router_is_nonzero(m, &layer->router)) {
                    /* Calibrated router. Compute logits via the existing
                     * fp16-matmul fast path: layer->router is a raw
                     * fp16 [hidden, K] tensor in IBF v6, so tensor_matmul
                     * handles it via the bits==16 branch.
                     *
                     * Logits buffer lives on the stack since K is small
                     * (≤ IB_MOME_MAX_EXPERTS = 32). */
                    float router_logits[IB_MOME_MAX_EXPERTS];
                    int active[IB_MOME_MAX_TOP_N];
                    ib_tensor_matmul_cpu(m, &layer->router, router_logits,
                                          xb, K_ex, hidden, scale_buf);
                    int top_n = mome_get_top_n(K_ex);
                    mome_top_n(router_logits, K_ex, top_n, active);
                    mome_dispatch_ffn(m, layer, xb, hb, hb2, xb,
                                       router_logits, active, top_n,
                                       scale_buf);
                } else {
                    /* Zero router → honor IB_MOME_TOP_N even without a router.
                     * Pick the first top_n experts; mome_dispatch_ffn scales
                     * each weight by K/n_active (mome.c:178-180) so top_n=K
                     * reconstructs the full FFN exactly. */
                    int top_n = mome_get_top_n(K_ex);
                    int active[IB_MOME_MAX_TOP_N];
                    for (int i = 0; i < top_n; i++) active[i] = i;
                    mome_dispatch_ffn(m, layer, xb, hb, hb2, xb,
                                       /*router_logits=*/NULL,
                                       active,
                                       /*n_active=*/top_n,
                                       scale_buf);
                }
                mome_handled = 1;
            }

            if (!mome_handled) {
                if (sp_mask) {
                    tensor_matmul_sparse(m, &layer->gate_proj, hb, xb, inter, hidden, scale_buf, sp_mask);
                    tensor_matmul_sparse(m, &layer->up_proj, hb2, xb, inter, hidden, scale_buf, sp_mask);
                } else {
                    tensor_matmul_hybrid(m, l, &layer->gate_proj, hb, xb, inter, hidden, scale_buf);
                    tensor_matmul_hybrid(m, l, &layer->up_proj,   hb2, xb, inter, hidden, scale_buf);
                }
                ib_kern.silu_mul(hb, hb, hb2, inter);
                /* down_proj reads from hb which already has zeros for masked rows —
                 * the multiply by zero propagates naturally, no sparse path needed */
                tensor_matmul_hybrid(m, l, &layer->down_proj, xb, hb, hidden, inter, scale_buf);
            }
        }

        /* Residual connection */
        for (int i = 0; i < hidden; i++) {
            x[i] += xb[i];
        }

        /* DFlash early-exit capture hook (Phase 4 / dflash_orchestrator.c).
         * Normally NULL — see struct inferbit_model. When a DFlash config
         * is attached we snapshot the post-residual hidden state at the
         * configured early-exit layer, so the orchestrator can read its
         * L2-norm as a "confidence" signal and (optionally on later steps)
         * project it through the LM head to skip layers > l. The cost when
         * inactive is one cmp+branch per layer. */
        if (m->dflash_capture_buf && m->dflash_cfg
            && l == m->dflash_cfg->early_exit_layer) {
            memcpy(m->dflash_capture_buf, x, (size_t)hidden * sizeof(float));
        }
    }

    if (compute_logits || hidden_out) {
        if (hidden_out) {
            /* Final RMSNorm into x (in place), then snapshot before LM head. */
            rmsnorm_fp16_t(x, x, m, &m->output_norm,
                           eps, hidden, scale_buf);
            memcpy(hidden_out, x, (size_t)hidden * sizeof(float));
            if (compute_logits) {
                tensor_matmul(m, &m->output_head, logits, x, vocab, hidden, scale_buf);
            }
        } else if (compute_logits) {
            /* Common path: factored helper. Behaviorally equivalent to the
             * previous inline RMSNorm + LM-head matmul. */
            ib_apply_lm_head_finalize(m, x, logits, scale_buf);
        }
    }

    return INFERBIT_OK;
}

static int forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    return forward_single_ex(m, token_id, pos, logits, 1, NULL);
}

/* Non-static wrapper. Lets dflash_orchestrator.c dispatch a full-forward
 * decode step without going through ib_forward() (whose routing called the
 * orchestrator in the first place). */
int ib_forward_single(inferbit_model* m, int token_id, int pos, float* logits) {
    return forward_single(m, token_id, pos, logits);
}

/* Factored: final-RMSNorm + LM-head matmul over a single hidden vector.
 * Lifted verbatim from forward_single_ex; behavior-preserving. Used both
 * by the standard decode path and by the DFlash orchestrator's early-exit
 * projection (which feeds an early-layer hidden state through the same
 * final-norm + output-head kernels). */
void ib_apply_lm_head_finalize(const inferbit_model* model,
                               float* hidden_io,
                               float* logits_out,
                               float* scale_buf) {
    int hidden = model->header.hidden_size;
    int vocab  = model->header.vocab_size;
    float eps  = model->header.norm_epsilon;
    /* Cast away const: the helper writes into model-owned scratch via the
     * matmul dispatch path. The model identity itself is unchanged. */
    inferbit_model* m = (inferbit_model*)model;
    rmsnorm_fp16_t(hidden_io, hidden_io, m, &m->output_norm,
                   eps, hidden, scale_buf);
    tensor_matmul(m, &m->output_head, logits_out, hidden_io, vocab, hidden, scale_buf);
}

/* ── Batched forward pass ───────────────────────────────────── */

/* Process B tokens (at contiguous positions positions[0..B-1]) through the
 * transformer, using batched matmul for projections, MLP, and LM head. Each
 * position's attention runs sequentially (each attends to its own prefix of
 * the KV cache, so there's no matmul-shape win from batching attention).
 *
 * If out_logits is non-NULL, writes logits there:
 *   last_logits_only == 0 — per-position logits [B * vocab] row-major.
 *   last_logits_only == 1 — only position B-1's logits, written to
 *                           out_logits[0 .. vocab) (a [vocab]-sized buffer).
 * In both cases the per-layer batched matmuls + KV writes for all B
 * positions still run; last_logits_only only skips the output-head matmul
 * for positions 0..B-2.
 *
 * Invariant: positions[b] = inferbit_kv_length(m) + b on entry (each position
 * gets appended to the KV cache as processed). Caller is responsible for
 * ensuring that's true. */
static int forward_batch(inferbit_model* m, const int32_t* tokens,
                         const int* positions, int B, float* out_logits,
                         int last_logits_only) {
    int hidden   = m->header.hidden_size;
    int n_layers = m->header.num_layers;
    int n_heads  = m->header.num_heads;
    int n_kv     = m->header.num_kv_heads;
    int head_dim = m->header.head_dim;
    int inter    = m->header.intermediate_size;
    int vocab    = m->header.vocab_size;
    float eps    = m->header.norm_epsilon;
    float theta  = m->header.rope_theta;
    int kv_dim   = n_kv * head_dim;
    int heads_per_kv = n_heads / n_kv;

    if (B > IB_BATCH_MAX) {
        ib_set_error("forward_batch B=%d exceeds IB_BATCH_MAX=%d", B, IB_BATCH_MAX);
        return INFERBIT_ERROR_PARAM;
    }

    /* Reuse model-lifetime preallocated scratch. Avoids ~1 MB malloc/free
     * per call in the spec-verify hot loop. Buffers are sized for
     * IB_BATCH_MAX positions; we only touch the first B slots. */
    float*  x          = m->bb_x;
    float*  xb         = m->bb_xb;
    float*  xb2        = m->bb_xb2;
    float*  q          = m->bb_q;
    float*  k          = m->bb_k;
    float*  v          = m->bb_v;
    float*  hb         = m->bb_hb;
    float*  hb2        = m->bb_hb2;
    float*  scale_buf  = m->bb_scale;
    float*  att        = m->bb_att;
    int8_t* q_scratch  = m->bb_qscratch;
    float*  sa_scratch = m->bb_sa;

    size_t x_sz = (size_t)B * hidden;

    /* Embed each token (cheap, per-position). */
    for (int b = 0; b < B; b++) {
        ib_embedding_lookup(m, tokens[b], x + (size_t)b * hidden);
    }

    for (int l = 0; l < n_layers; l++) {
        ib_layer_meta* layer = &m->layers[l];
        ib_kv_cache* kv = &m->kv_caches[l];

        /* RMSNorm per position. Uses rmsnorm_fp16_t to pick up cached fp32 norm weights. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16_t(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                           m, &layer->input_norm,
                           eps, hidden, scale_buf);
        }

        /* Q/K/V projections — batched. */
        tensor_matmul_batch(m, &layer->q_proj, q, xb, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->k_proj, k, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);
        tensor_matmul_batch(m, &layer->v_proj, v, xb, kv_dim, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Per-position: RoPE, KV-cache write, attention, O-proj-input
         * (accumulated per-position into xb2[b]). */
        for (int b = 0; b < B; b++) {
            int pos = positions[b];
            float* qb = q + (size_t)b * hidden;
            float* kb = k + (size_t)b * kv_dim;
            float* vb = v + (size_t)b * kv_dim;

            /* RoPE: same pattern as forward_single_ex. */
            const float* rope_cos_tab = (m->rope_cos && pos < m->rope_table_ctx) ? m->rope_cos : NULL;
            const float* rope_sin_tab = (m->rope_sin && pos < m->rope_table_ctx) ? m->rope_sin : NULL;
            for (int h = 0; h < n_heads; h++) {
                int kv_h = h / heads_per_kv;
                int is_first = (h % heads_per_kv == 0);
                if (is_first) {
                    ib_kern.rope(qb + h * head_dim, kb + kv_h * head_dim,
                                 head_dim, pos, theta, rope_cos_tab, rope_sin_tab);
                } else {
                    float k_scratch[256];
                    memcpy(k_scratch, kb + kv_h * head_dim, head_dim * sizeof(float));
                    ib_kern.rope(qb + h * head_dim, k_scratch, head_dim, pos, theta,
                                 rope_cos_tab, rope_sin_tab);
                }
            }

            kv_cache_write(kv, pos, kb, vb, kv_dim, n_kv, head_dim, m->header.kv_bits);
            kv->length = pos + 1;

            float attn_scale = 1.0f / sqrtf((float)head_dim);
            ib_attn_ctx ctx = {
                .q = qb, .att = att, .xb2 = xb2 + (size_t)b * hidden,
                .kv = kv,
                .head_dim = head_dim, .kv_dim = kv_dim,
                .n_kv_heads = n_kv,
                .heads_per_kv = heads_per_kv,
                .pos = pos,
                .kv_bits = m->header.kv_bits,
                .scale = attn_scale,
            };
            if (m->thread_pool && n_heads >= 4) {
                ib_pool_run(m->thread_pool, ib_attn_head_task, &ctx, n_heads, 0);
            } else {
                ib_attn_head_task(&ctx, 0, 0, n_heads);
            }
        }

        /* O projection — batched. */
        tensor_matmul_batch(m, &layer->o_proj, xb, xb2, hidden, hidden, B,
                            scale_buf, q_scratch, sa_scratch);

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];

        /* RMSNorm before MLP, per position. */
        for (int b = 0; b < B; b++) {
            rmsnorm_fp16_t(xb + (size_t)b * hidden, x + (size_t)b * hidden,
                           m, &layer->post_attn_norm,
                           eps, hidden, scale_buf);
        }

        /* MLP — batched gate/up/down. Sparsity is not applied here; if a
         * layer has sparsity we fall back to the single-position path per
         * batch via tensor_matmul_sparse (rare for Llama, so OK). */
        if (layer->sparsity_mask_size > 0) {
            const uint8_t* sp_mask = (const uint8_t*)m->weight_data + layer->sparsity_mask_offset;
            for (int b = 0; b < B; b++) {
                float* xb_b = xb + (size_t)b * hidden;
                float* hb_b = hb + (size_t)b * inter;
                float* hb2_b = hb2 + (size_t)b * inter;
                float* xb_out = xb + (size_t)b * hidden;  /* overwrite xb[b] with down output */
                tensor_matmul_sparse(m, &layer->gate_proj, hb_b, xb_b, inter, hidden, scale_buf, sp_mask);
                tensor_matmul_sparse(m, &layer->up_proj, hb2_b, xb_b, inter, hidden, scale_buf, sp_mask);
                ib_kern.silu_mul(hb_b, hb_b, hb2_b, inter);
                tensor_matmul(m, &layer->down_proj, xb_out, hb_b, hidden, inter, scale_buf);
            }
        } else {
            tensor_matmul_batch(m, &layer->gate_proj, hb, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            tensor_matmul_batch(m, &layer->up_proj, hb2, xb, inter, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
            for (int b = 0; b < B; b++) {
                ib_kern.silu_mul(hb + (size_t)b * inter,
                                 hb + (size_t)b * inter,
                                 hb2 + (size_t)b * inter, inter);
            }
            tensor_matmul_batch(m, &layer->down_proj, xb, hb, hidden, inter, B,
                                scale_buf, q_scratch, sa_scratch);
        }

        /* Residual add per position. */
        for (size_t i = 0; i < x_sz; i++) x[i] += xb[i];
    }

    /* Final RMSNorm + LM head. */
    if (out_logits) {
        if (last_logits_only) {
            /* Only position B-1's logits are needed — RMSNorm + a single
             * output-head matmul for that one position. Skips B-1 vocab-
             * sized matmuls (the most expensive op) vs the full path. */
            float* x_last = x + (size_t)(B - 1) * hidden;
            rmsnorm_fp16_t(x_last, x_last, m, &m->output_norm,
                           eps, hidden, scale_buf);
            tensor_matmul(m, &m->output_head, out_logits, x_last,
                          vocab, hidden, scale_buf);
        } else {
            for (int b = 0; b < B; b++) {
                rmsnorm_fp16_t(x + (size_t)b * hidden, x + (size_t)b * hidden,
                               m, &m->output_norm,
                               eps, hidden, scale_buf);
            }
            tensor_matmul_batch(m, &m->output_head, out_logits, x, vocab, hidden, B,
                                scale_buf, q_scratch, sa_scratch);
        }
    }

    /* Scratch buffers are model-lifetime; no free here. */
    return INFERBIT_OK;
}

/* ── Metal backend routing ──────────────────────────────────── */

#ifdef IB_HAS_METAL
/* Decide once whether this model runs on the Metal backend, lazily
 * creating + caching the GPU context/buffers on first use. Returns 1 if
 * Metal-routed, 0 for the CPU path. IB_BACKEND=cpu forces CPU. Once a
 * model is Metal-routed it stays Metal-routed for its whole life — we
 * must never silently CPU-fall-back mid-stream, because KV state then
 * lives in metal_bufs and the CPU kv_caches arrays are empty. */
static int ib_metal_route(inferbit_model* m) {
    fprintf(stderr, "[N15] ib_metal_route: enter (model=%p, name=%s)\n",
            (void*)m, m ? m->header.name : "(null)");
    static int forced_cpu = -1;
    if (forced_cpu < 0) {
        const char* e = getenv("IB_BACKEND");
        forced_cpu = (e && strcmp(e, "cpu") == 0) ? 1 : 0;
    }
    if (forced_cpu) {
        fprintf(stderr, "[N15] ib_metal_route: bail — IB_BACKEND=cpu forces CPU path\n");
        return 0;
    }
    if (m->metal_route_failed) {
        fprintf(stderr, "[N15] ib_metal_route: bail — metal_route_failed already set (sticky CPU after prior failure)\n");
        return 0;
    }
    if (m->metal_bufs) {
        fprintf(stderr, "[N15] ib_metal_route: already routed, returning metal_bufs=%p\n", m->metal_bufs);
        return 1;
    }
    fprintf(stderr, "[N15] ib_metal_route: calling ib_metal_create()\n");
    ib_metal_ctx* ctx = ib_metal_create();
    if (!ctx) {
        fprintf(stderr, "[N15] ib_metal_route: bail — ib_metal_create() returned NULL (no Metal device / ctx alloc failed)\n");
        m->metal_route_failed = 1;
        return 0;
    }
    fprintf(stderr, "[N15] ib_metal_route: ctx=%p, calling ib_metal_upload_model()\n", (void*)ctx);
    ib_metal_model_buffers* bufs = ib_metal_upload_model(ctx, m);
    if (!bufs) {
        fprintf(stderr, "[N15] ib_metal_route: bail — ib_metal_upload_model() returned NULL (model unsupported / upload OOM / arch mismatch)\n");
        ib_metal_destroy(ctx);
        m->metal_route_failed = 1;
        return 0;
    }
    fprintf(stderr, "[N15] ib_metal_route: success — bufs=%p, model routed to Metal\n", (void*)bufs);
    m->metal_ctx  = ctx;
    m->metal_bufs = bufs;
    return 1;
}

/* Metal-backed ib_forward: prefill (n_tokens>1, last-token logits) or
 * single-token decode. KV is written into metal_bufs at [kv_pos,
 * kv_pos+n_tokens); we then advance the logical kv_caches[].length
 * counter so inferbit_kv_length stays correct. */
static int ib_forward_metal(inferbit_model* m, const int32_t* tokens,
                            int num_tokens, int kv_pos, float* out_logits) {
    int hidden = m->header.hidden_size;
    float* embeds = (float*)malloc((size_t)num_tokens * hidden * sizeof(float));
    if (!embeds) { ib_set_error("oom: metal embed buffer"); return INFERBIT_ERROR_MEMORY; }
    for (int i = 0; i < num_tokens; i++)
        ib_embedding_lookup(m, tokens[i], embeds + (size_t)i * hidden);

    ib_metal_ctx* ctx = (ib_metal_ctx*)m->metal_ctx;
    ib_metal_model_buffers* bufs = (ib_metal_model_buffers*)m->metal_bufs;
    int rc;
    if (num_tokens == 1) {
        rc = ib_metal_forward_token(ctx, bufs, embeds, kv_pos, out_logits);
    } else {
        rc = ib_metal_forward_prefill(ctx, bufs, embeds, num_tokens, kv_pos, out_logits);
        if (rc == -2) {
            /* Batched prefill layout-incompatible — per-token GPU loop.
             * forward_token writes out_logits each call, so after the
             * loop out_logits holds the LAST token's logits (what prefill
             * callers consume). */
            rc = 0;
            for (int i = 0; i < num_tokens && rc == 0; i++)
                rc = ib_metal_forward_token(ctx, bufs, embeds + (size_t)i * hidden,
                                            kv_pos + i, out_logits);
        }
    }
    free(embeds);
    if (rc != 0) { ib_set_error("metal forward failed (rc=%d)", rc); return INFERBIT_ERROR_INTERNAL; }
    for (int L = 0; L < m->header.num_layers; L++)
        m->kv_caches[L].length = kv_pos + num_tokens;
    return INFERBIT_OK;
}

/* Metal-backed ib_forward_positions: per-position logits for num_tokens
 * tokens at [kv_pos, kv_pos+num_tokens). out_logits is [num_tokens][vocab]. */
static int ib_forward_positions_metal(inferbit_model* m, const int32_t* tokens,
                                      int num_tokens, int kv_pos, float* out_logits) {
    int hidden = m->header.hidden_size;
    int vocab  = m->header.vocab_size;
    float* embeds = (float*)malloc((size_t)num_tokens * hidden * sizeof(float));
    if (!embeds) { ib_set_error("oom: metal embed buffer"); return INFERBIT_ERROR_MEMORY; }
    for (int i = 0; i < num_tokens; i++)
        ib_embedding_lookup(m, tokens[i], embeds + (size_t)i * hidden);

    ib_metal_ctx* ctx = (ib_metal_ctx*)m->metal_ctx;
    ib_metal_model_buffers* bufs = (ib_metal_model_buffers*)m->metal_bufs;
    int rc = ib_metal_forward_prefill_logits_all(ctx, bufs, embeds, num_tokens,
                                                 kv_pos, out_logits);
    if (rc == -2) {
        /* Layout-incompatible — per-token GPU loop, capturing each
         * position's logits into its own out_logits slab. */
        rc = 0;
        for (int i = 0; i < num_tokens && rc == 0; i++)
            rc = ib_metal_forward_token(ctx, bufs, embeds + (size_t)i * hidden,
                                        kv_pos + i, out_logits + (size_t)i * vocab);
    }
    free(embeds);
    if (rc != 0) { ib_set_error("metal forward_positions failed (rc=%d)", rc); return INFERBIT_ERROR_INTERNAL; }
    for (int L = 0; L < m->header.num_layers; L++)
        m->kv_caches[L].length = kv_pos + num_tokens;
    return INFERBIT_OK;
}
#endif /* IB_HAS_METAL */

/* ── Public: backend warmup + introspection ─────────────────── */

#ifdef IB_HAS_METAL
int inferbit_model_warmup(inferbit_model* model) {
    if (!model) return 0;
    /* Resolve routing now — this triggers the (otherwise lazy) GPU
     * upload, moving the TTFT spike here instead of the first forward. */
    ib_metal_route(model);
    return 0;
}

const char* inferbit_model_backend(inferbit_model* model) {
    if (!model) return "cpu";
    return ib_metal_route(model) ? "metal" : "cpu";
}
#else
int inferbit_model_warmup(inferbit_model* model) {
    (void)model;
    return 0;
}

const char* inferbit_model_backend(inferbit_model* model) {
    (void)model;
    return "cpu";
}
#endif /* IB_HAS_METAL */

/* ── Public: forward pass ───────────────────────────────────── */

int ib_forward(inferbit_model* model, const int32_t* tokens, int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward");
        return INFERBIT_ERROR_PARAM;
    }

    /* Stage 5d: lazy-seed per-tensor preferred_backend from env vars
     * (currently just IB_HYBRID_FFN_GPU). One-shot per model. Done here
     * to avoid touching the loader files (pqv2_model.c / ibf_loader.c
     * are in the "do not modify" list for this stage). */
    hybrid_apply_tags(model);

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d", kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }

    /* Validate all tokens first */
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= model->header.vocab_size) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)", tokens[i], model->header.vocab_size);
            return INFERBIT_ERROR_PARAM;
        }
    }

#ifdef IB_HAS_METAL
    if (ib_metal_route(model))
        return ib_forward_metal(model, tokens, num_tokens, kv_pos, out_logits);
#endif

    /* DFlash hybrid orchestrator (Phase 4). CPU-only in v1, single-token
     * decode only. Placed AFTER the Metal-route check so the orchestrator
     * never sees Metal-routed calls. The orchestrator declines (handled=0)
     * for prefill or when no DFlash config is attached; we then fall
     * through to the existing CPU routing. The orchestrator's own
     * dispatch goes via ib_forward_single (non-static wrapper) so there
     * is no recursion through ib_forward. */
    if (model->dflash_cfg) {
        int handled = 0;
        int rc = ib_dflash_try_route(model, tokens, num_tokens, out_logits, &handled);
        if (handled) return rc;
    }

    if (num_tokens == 1) {
        /* Single token — standard decode path */
        return forward_single(model, tokens[0], kv_pos, out_logits);
    }

    /*
     * Batch prefill (CPU fallback path): process the prompt in chunks of
     * IB_BATCH_MAX tokens through the batched forward engine. forward_batch
     * advances kv_caches[].length itself; ib_forward only needs the LAST
     * position's logits, so we pass last_logits_only=1 — each chunk writes
     * just [vocab] into out_logits[0..vocab), and since chunks overwrite,
     * the final chunk's last-position logits are what remain (correct).
     */
    int offset = 0;
    while (offset < num_tokens) {
        int remaining = num_tokens - offset;
        int B = remaining < IB_BATCH_MAX ? remaining : IB_BATCH_MAX;
        /* forward_batch advanced kv_caches[].length on the prior chunk;
         * recompute the base position fresh each iteration. */
        int base = inferbit_kv_length(model);
        int positions[IB_BATCH_MAX];
        for (int j = 0; j < B; j++) positions[j] = base + j;
        int rc = forward_batch(model, tokens + offset, positions, B,
                               out_logits, 1);
        if (rc != INFERBIT_OK) {
            return rc;
        }
        offset += B;
    }
    return INFERBIT_OK;
}

int ib_forward_positions(inferbit_model* model, const int32_t* tokens,
                         int num_tokens, float* out_logits) {
    if (!model || !tokens || !out_logits || num_tokens <= 0) {
        ib_set_error("invalid arguments to ib_forward_positions");
        return INFERBIT_ERROR_PARAM;
    }

    int kv_pos = inferbit_kv_length(model);
    int max_ctx = model->header.max_context_length;
    int vocab   = model->header.vocab_size;

    if (kv_pos + num_tokens > max_ctx) {
        ib_set_error("context length exceeded: %d + %d > %d",
                     kv_pos, num_tokens, max_ctx);
        return INFERBIT_ERROR_CONTEXT;
    }
    for (int i = 0; i < num_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= vocab) {
            ib_set_error("token ID out of range: %d (vocab_size=%d)",
                         tokens[i], vocab);
            return INFERBIT_ERROR_PARAM;
        }
    }

    if (num_tokens > IB_BATCH_MAX) {
        ib_set_error("forward_positions num_tokens=%d exceeds IB_BATCH_MAX=%d",
                     num_tokens, IB_BATCH_MAX);
        return INFERBIT_ERROR_PARAM;
    }

#ifdef IB_HAS_METAL
    if (ib_metal_route(model))
        return ib_forward_positions_metal(model, tokens, num_tokens, kv_pos, out_logits);
#endif

    /* Fill absolute positions in the preallocated scratch buffer. */
    int* positions = model->bb_positions;
    for (int i = 0; i < num_tokens; i++) positions[i] = kv_pos + i;

    return forward_batch(model, tokens, positions, num_tokens, out_logits, 0);
}
