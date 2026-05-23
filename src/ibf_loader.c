/*
 * ibf_loader.c — Parse and load .ibf files
 *
 * File layout:
 *   [0..7]    Magic "INFERBIT"
 *   [8..11]   Format version (uint32 LE)
 *   [12..15]  Header size in bytes (uint32 LE)
 *   [16..19]  Flags (uint32 bitfield)
 *   [20..31]  Reserved (zeros)
 *   [32..32+H) JSON header (UTF-8)
 *   [A..)     Weight data (64-byte aligned)
 */

#include "inferbit_internal.h"
#include "platform.h"
#include "cJSON.h"
#include "pq_decode.h"   /* ib_fp16_to_fp32 — used for static scale caching */
#include "sparse_gate.h" /* FFN cluster record layout (magic/version/header) */
#include "pqv2_format.h" /* ib_pqv2_file / ib_pqv2_find — locate ffn_clusters by name */

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/mman.h>   /* mlock — best-effort hot-pool residency lock */
#endif

#define IBF_MAGIC      "INFERBIT"
#define IBF_MAGIC_SIZE 8
#define IBF_PREAMBLE   32
#define IBF_ALIGNMENT  64
#define IBF_VERSION    1

/* ── Hot-pool threading shim ────────────────────────────────────────
 *
 * The hot pool is read (lookup) and written (promote) from the matmul
 * path, which runs on the thread pool's workers. A single mutex around
 * the small entry table is enough for the existing access pattern (the
 * table is tiny and operations are O(entries) with entries bounded by
 * the MB budget / smallest tensor). Mirror the minimal pthread shim
 * forward.c uses so this stays portable without pulling in threading.c's
 * full Windows layer. */
#ifdef _WIN32
typedef SRWLOCK ib_hp_mutex_t;
#define IB_HP_MUTEX_INIT(m)   InitializeSRWLock(m)
#define IB_HP_MUTEX_LOCK(m)   AcquireSRWLockExclusive(m)
#define IB_HP_MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(m)
#define IB_HP_MUTEX_DESTROY(m) ((void)0)
#else
#include <pthread.h>
typedef pthread_mutex_t ib_hp_mutex_t;
#define IB_HP_MUTEX_INIT(m)   pthread_mutex_init((m), NULL)
#define IB_HP_MUTEX_LOCK(m)   pthread_mutex_lock(m)
#define IB_HP_MUTEX_UNLOCK(m) pthread_mutex_unlock(m)
#define IB_HP_MUTEX_DESTROY(m) pthread_mutex_destroy(m)
#endif

/* ── Hot-pool layout (Goal H4 — adaptive hot-cache) ─────────────────
 *
 * Everything lives inside the single `model->hot_pool` malloc region the
 * loader already allocates from IB_HOT_POOL_MB. That keeps the data
 * structure allocation-light (zero extra mallocs) AND means the existing
 * `free(model->hot_pool)` in model.c frees the whole thing — no new
 * field on inferbit_model and no change to the free path is required.
 *
 * Layout within hot_pool:
 *   [ ib_hot_hdr ][ ib_hot_entry[cap] ][ ... data arena ... ]
 *
 * The control header carries the mutex, the LRU clock, the entry table
 * capacity/count, and the data-arena bump cursor. Entries are a fixed-
 * size array (linear scan); `cap` is derived from the budget so the
 * table can never describe more bytes than the arena holds.
 *
 * Eviction policy: LRU. Each lookup/promote stamps the entry with a
 * monotonically increasing `clock`; when the arena can't fit a new
 * promotion we evict the entry with the smallest stamp until it fits (or
 * until the table is empty, in which case the item is simply too big and
 * we skip it). LRU was chosen over LFU because the burst/cool-down duty
 * cycle wants the *current* burst's working set resident, not whatever
 * was hottest across the whole run. */

#define IB_HOT_TABLE_FRAC 64u   /* ~1/64 of the budget reserved for the
                                 * header+entry table; the rest is arena. */

typedef struct {
    size_t   key;        /* t->offset — stable per-tensor id (0 = empty slot) */
    size_t   data_off;   /* byte offset of the copy within the arena */
    size_t   nbytes;     /* bytes copied */
    uint64_t stamp;      /* last-access LRU clock */
    int      in_use;     /* 0 = free slot, 1 = occupied */
} ib_hot_entry;

typedef struct {
    ib_hp_mutex_t mu;
    int           cap;        /* number of entry slots */
    int           count;      /* occupied slots */
    uint64_t      clock;      /* monotonic LRU stamp source */
    size_t        arena_off;  /* offset of the data arena within hot_pool */
    size_t        arena_cap;  /* arena capacity in bytes */
    size_t        arena_used; /* bytes currently committed (high-water cursor) */
} ib_hot_hdr;

/* Initialise the in-buffer control header + entry table. Called once at
 * load from the IB_HOT_POOL_MB allocation site. `buf`/`bytes` are the raw
 * malloc'd region. If the budget is too small to hold even the header
 * plus one entry, cap is left at 0 and the pool degrades to a no-op
 * (lookup always misses, promote always skips) — never a crash. */
static void ib_hot_pool_init(void *buf, size_t bytes) {
    if (!buf || bytes < sizeof(ib_hot_hdr) + sizeof(ib_hot_entry)) {
        /* Too small to be useful. Zero the header region we can touch so a
         * later lookup/promote sees cap == 0 and bails. */
        if (buf && bytes >= sizeof(ib_hot_hdr)) {
            ib_hot_hdr *h = (ib_hot_hdr *)buf;
            memset(h, 0, sizeof(*h));
            IB_HP_MUTEX_INIT(&h->mu);
        }
        return;
    }
    ib_hot_hdr *h = (ib_hot_hdr *)buf;
    memset(h, 0, sizeof(*h));
    IB_HP_MUTEX_INIT(&h->mu);

    /* Reserve ~1/IB_HOT_TABLE_FRAC of the budget for the entry table
     * (after the header); the remainder is the data arena. At least 1
     * entry; arena gets whatever is left. */
    size_t after_hdr = bytes - sizeof(ib_hot_hdr);
    size_t table_budget = bytes / IB_HOT_TABLE_FRAC;
    if (table_budget > after_hdr) table_budget = after_hdr;
    int cap = (int)(table_budget / sizeof(ib_hot_entry));
    if (cap < 1) cap = 1;
    size_t table_bytes = (size_t)cap * sizeof(ib_hot_entry);

    h->cap        = cap;
    h->count      = 0;
    h->clock      = 0;
    h->arena_off  = sizeof(ib_hot_hdr) + table_bytes;
    h->arena_cap  = bytes - h->arena_off;
    h->arena_used = 0;

    ib_hot_entry *tbl = (ib_hot_entry *)((uint8_t *)buf + sizeof(ib_hot_hdr));
    memset(tbl, 0, table_bytes);
}

/* Convenience accessors that resolve the header / table / arena from the
 * model. Return NULL when the pool is disabled (IB_HOT_POOL_MB unset →
 * hot_pool == NULL) or alloc-failed, which is what makes the default path
 * a strict no-op. */
static inline ib_hot_hdr *ib_hot_hdr_of(const inferbit_model *m) {
    if (!m || !m->hot_pool || m->hot_pool_bytes < sizeof(ib_hot_hdr))
        return NULL;
    return (ib_hot_hdr *)m->hot_pool;
}
static inline ib_hot_entry *ib_hot_table_of(ib_hot_hdr *h) {
    return (ib_hot_entry *)((uint8_t *)h + sizeof(ib_hot_hdr));
}

/* ── Public hot-cache implementation (Goal H4) ──────────────────────
 *
 * NOTE ON LINKAGE: the canonical ib_hot_lookup / ib_hot_promote symbols
 * are still the no-op stubs in forward.c (owned by another agent). To
 * avoid a duplicate-symbol link error these real bodies are named
 * *_impl. The integration step is a two-line redirect in forward.c —
 * see the report's TODO(burst-M2: forward.c). The signatures here are an
 * exact match for the header declarations so the redirect is mechanical.
 *
 * Default-behaviour guarantee: when hot_pool is NULL (IB_HOT_POOL_MB
 * unset) or the table cap is 0, lookup returns NULL and promote returns
 * non-zero ("not promoted") WITHOUT touching any state — byte-identical
 * to today's stubs. */

const void *ib_hot_lookup_key(const inferbit_model *m, size_t key) {
    ib_hot_hdr *h = ib_hot_hdr_of(m);
    if (!h || h->cap <= 0) return NULL;              /* disabled → strict no-op */

    const void *result = NULL;
    IB_HP_MUTEX_LOCK(&h->mu);
    ib_hot_entry *tbl = ib_hot_table_of(h);
    for (int i = 0; i < h->cap; i++) {
        if (tbl[i].in_use && tbl[i].key == key) {
            tbl[i].stamp = ++h->clock;               /* bump LRU recency */
            result = (const uint8_t *)m->hot_pool + h->arena_off + tbl[i].data_off;
            break;
        }
    }
    IB_HP_MUTEX_UNLOCK(&h->mu);
    return result;
}

const void *ib_hot_lookup_impl(const inferbit_model *m, const ib_tensor_meta *t) {
    if (!t) return NULL;
    return ib_hot_lookup_key(m, t->offset);
}

/* Reserve `nbytes` under `key` and return a WRITABLE pointer into the arena
 * (caller fills it — used by the drive path to pread a tensor's indices
 * straight from disk into the pool). Fill-once bump allocator: NO eviction.
 * When the model exceeds the budget the first tensors fill the pool and stay
 * resident; later tensors get NULL and fall back to streaming/paging. This is
 * deliberately thrash-free — eviction on an over-budget model would re-promote
 * the whole working set every token (slower than just paging). Returns the
 * existing pointer if `key` is already resident (idempotent), or NULL if it
 * doesn't fit. Pointers are STABLE for the model's life (no compaction). */
void *ib_hot_reserve(inferbit_model *m, size_t key, size_t nbytes) {
    ib_hot_hdr *h = ib_hot_hdr_of(m);
    if (!h || h->cap <= 0 || nbytes == 0 || nbytes > h->arena_cap) return NULL;

    void *result = NULL;
    IB_HP_MUTEX_LOCK(&h->mu);
    ib_hot_entry *tbl = ib_hot_table_of(h);

    /* Already resident → return existing (idempotent). */
    for (int i = 0; i < h->cap; i++) {
        if (tbl[i].in_use && tbl[i].key == key) {
            result = (uint8_t *)m->hot_pool + h->arena_off + tbl[i].data_off;
            IB_HP_MUTEX_UNLOCK(&h->mu);
            return result;
        }
    }

    /* Fill-once: place at the bump cursor if a slot and room remain. */
    if (h->count < h->cap && h->arena_used + nbytes <= h->arena_cap) {
        int slot = -1;
        for (int i = 0; i < h->cap; i++) { if (!tbl[i].in_use) { slot = i; break; } }
        if (slot >= 0) {
            size_t off = h->arena_used;
            tbl[slot].key      = key;
            tbl[slot].data_off = off;
            tbl[slot].nbytes   = nbytes;
            tbl[slot].stamp    = ++h->clock;
            tbl[slot].in_use   = 1;
            h->count++;
            h->arena_used     += nbytes;
            m->hot_pool_entries = h->count;
            result = (uint8_t *)m->hot_pool + h->arena_off + off;
        }
    }

    IB_HP_MUTEX_UNLOCK(&h->mu);
    return result;
}

/* Promote a byte range [src, src+nbytes) under `key`: reserve + copy. Caches
 * the *streamed index sub-region* (the bytes the matmul reads via pq->indices),
 * byte-identical to the streamed bytes → the RAM throttle is BIT-EXACT (only
 * residency/speed change, never the logits). Returns 0 if resident, nonzero if
 * it didn't fit. */
int ib_hot_promote_bytes(inferbit_model *m, size_t key,
                         const void *src, size_t nbytes) {
    if (!src) return 1;
    void *dst = ib_hot_reserve(m, key, nbytes);
    if (!dst) return 1;
    if (dst != src) memcpy(dst, src, nbytes);
    return 0;
}

/* Whole-tensor convenience wrapper: cache the canonical on-disk blob at
 * m->weight_data + t->offset (t->size bytes), keyed by t->offset. Kept for the
 * generic API surface; the drive path uses ib_hot_promote_bytes directly to
 * cache just the L1 index sub-region it streams. */
int ib_hot_promote_impl(inferbit_model *m, const ib_tensor_meta *t) {
    if (!m || !t || t->size == 0 || !m->weight_data) return 1;
    return ib_hot_promote_bytes(m, t->offset,
                                (const uint8_t *)m->weight_data + t->offset,
                                t->size);
}

/* ── Helpers ────────────────────────────────────────────────── */

static size_t align_up(size_t val, size_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static const char* json_str(const cJSON* obj, const char* key, const char* def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsString(item) && item->valuestring) return item->valuestring;
    return def;
}

static int json_int(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return item->valueint;
    return def;
}

static double json_double(const cJSON* obj, const char* key, double def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return item->valuedouble;
    return def;
}

static int json_bool(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsBool(item)) return cJSON_IsTrue(item);
    return def;
}

static size_t json_size(const cJSON* obj, const char* key, size_t def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (cJSON_IsNumber(item)) return (size_t)item->valuedouble;
    return def;
}

/* ── Parse tensor metadata from JSON ────────────────────────── */

static ib_tensor_meta parse_tensor(const cJSON* obj) {
    ib_tensor_meta t = {0};
    if (!obj || !cJSON_IsObject(obj)) return t;

    t.offset       = json_size(obj, "offset", 0);
    t.size         = json_size(obj, "size", 0);
    t.bits         = json_int(obj, "bits", 16);
    t.scale_offset = json_size(obj, "scale_offset", 0);
    t.scale_size   = json_size(obj, "scale_size", 0);
    t.has_bias     = json_bool(obj, "has_bias", 0);

    cJSON* shape = cJSON_GetObjectItemCaseSensitive(obj, "shape");
    if (cJSON_IsArray(shape)) {
        t.ndim = cJSON_GetArraySize(shape);
        if (t.ndim > 4) t.ndim = 4;
        for (int i = 0; i < t.ndim; i++) {
            cJSON* dim = cJSON_GetArrayItem(shape, i);
            t.shape[i] = cJSON_IsNumber(dim) ? dim->valueint : 0;
        }
    }

    return t;
}

/* ── Parse layer metadata ───────────────────────────────────── */

static ib_layer_meta parse_layer(const cJSON* obj) {
    ib_layer_meta layer = {0};
    if (!obj) return layer;

    cJSON* weights = cJSON_GetObjectItemCaseSensitive(obj, "weights");
    if (weights) {
        layer.q_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "q_proj"));
        layer.k_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "k_proj"));
        layer.v_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "v_proj"));
        layer.o_proj         = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "o_proj"));
        layer.gate_proj      = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "gate_proj"));
        layer.up_proj        = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "up_proj"));
        layer.down_proj      = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "down_proj"));
        layer.input_norm     = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "input_norm"));
        layer.post_attn_norm = parse_tensor(cJSON_GetObjectItemCaseSensitive(weights, "post_attn_norm"));
    }

    cJSON* sparsity = cJSON_GetObjectItemCaseSensitive(obj, "sparsity_mask");
    if (sparsity) {
        layer.sparsity_mask_offset = json_size(sparsity, "offset", 0);
        layer.sparsity_mask_size   = json_size(sparsity, "size", 0);
    }

    /* Optional sparse-FFN cluster record. Presence is signalled purely by
     * the JSON object existing (and carrying a non-zero size). When absent
     * these stay 0 and the gate is disabled — byte-identical to today.
     * The byte layout / magic / version live in src/sparse_gate.h; the
     * record itself is resolved + validated post-mmap in
     * ib_resolve_ffn_clusters(). */
    cJSON* ffn_cl = cJSON_GetObjectItemCaseSensitive(obj, "ffn_cluster");
    if (ffn_cl) {
        layer.ffn_cluster_blob_offset = json_size(ffn_cl, "offset", 0);
        layer.ffn_cluster_blob_size   = json_size(ffn_cl, "size", 0);
    }

    return layer;
}

/* ── Parse IBF header JSON ──────────────────────────────────── */

static int parse_header_json(const char* json_str_buf, size_t json_len,
                             ib_ibf_header* header, ib_layer_meta** out_layers,
                             ib_tensor_meta* token_emb, ib_tensor_meta* out_norm,
                             ib_tensor_meta* out_head) {
    cJSON* root = cJSON_ParseWithLength(json_str_buf, json_len);
    if (!root) {
        ib_set_error("failed to parse IBF JSON header: %s", cJSON_GetErrorPtr());
        return -1;
    }

    /* Model info */
    cJSON* model = cJSON_GetObjectItemCaseSensitive(root, "model");
    if (model) {
        const char* arch = json_str(model, "architecture", "unknown");
        strncpy(header->architecture, arch, sizeof(header->architecture) - 1);
        const char* name = json_str(model, "name", "");
        strncpy(header->name, name, sizeof(header->name) - 1);
    }

    /* Architecture */
    cJSON* arch = cJSON_GetObjectItemCaseSensitive(root, "architecture");
    if (arch) {
        header->num_layers         = json_int(arch, "num_layers", 0);
        header->hidden_size        = json_int(arch, "hidden_size", 0);
        header->num_heads          = json_int(arch, "num_heads", 0);
        header->num_kv_heads       = json_int(arch, "num_kv_heads", header->num_heads);
        header->head_dim           = json_int(arch, "head_dim", header->hidden_size / header->num_heads);
        header->intermediate_size  = json_int(arch, "intermediate_size", 0);
        header->vocab_size         = json_int(arch, "vocab_size", 0);
        header->max_context_length = json_int(arch, "max_context_length", 2048);
        header->rope_theta         = (float)json_double(arch, "rope_theta", 10000.0);
        header->norm_epsilon       = (float)json_double(arch, "norm_epsilon", 1e-5);

        const char* nt = json_str(arch, "norm_type", "rmsnorm");
        strncpy(header->norm_type, nt, sizeof(header->norm_type) - 1);

        const char* act = json_str(arch, "activation", "silu");
        strncpy(header->activation, act, sizeof(header->activation) - 1);

        header->tie_word_embeddings = json_bool(arch, "tie_word_embeddings", 0);
        header->attention_bias      = json_bool(arch, "attention_bias", 0);
        header->mlp_bias            = json_bool(arch, "mlp_bias", 0);
        header->bos_token_id        = json_int(arch, "bos_token_id", 1);
        header->eos_token_id        = json_int(arch, "eos_token_id", 2);
    }

    /* Quantization */
    cJSON* quant = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (quant) {
        header->default_bits   = json_int(quant, "default_bits", 4);
        header->sensitive_bits = json_int(quant, "sensitive_bits", 8);
        header->sparsity       = (float)json_double(quant, "sparsity", 0.0);
        header->block_size     = json_int(quant, "block_size", 8);
    }

    /* KV cache */
    cJSON* kv = cJSON_GetObjectItemCaseSensitive(root, "kv_cache");
    if (kv) {
        header->kv_bits = json_int(kv, "bits", 8);
    }

    /* Data section */
    cJSON* data = cJSON_GetObjectItemCaseSensitive(root, "data");
    if (data) {
        header->weight_data_offset = json_size(data, "weight_data_offset", 0);
        header->weight_data_size   = json_size(data, "weight_data_size", 0);
        header->alignment          = json_int(data, "alignment", 64);
    }

    /* Layers */
    cJSON* layers_arr = cJSON_GetObjectItemCaseSensitive(root, "layers");
    if (cJSON_IsArray(layers_arr) && header->num_layers > 0) {
        *out_layers = calloc(header->num_layers, sizeof(ib_layer_meta));
        if (!*out_layers) {
            ib_set_error("failed to allocate layer metadata");
            cJSON_Delete(root);
            return -1;
        }
        int n = cJSON_GetArraySize(layers_arr);
        if (n > header->num_layers) n = header->num_layers;
        for (int i = 0; i < n; i++) {
            (*out_layers)[i] = parse_layer(cJSON_GetArrayItem(layers_arr, i));
        }
    }

    /* Embeddings */
    cJSON* emb = cJSON_GetObjectItemCaseSensitive(root, "embeddings");
    if (emb) {
        *token_emb = parse_tensor(cJSON_GetObjectItemCaseSensitive(emb, "token_embedding"));
    }

    /* Output */
    cJSON* output = cJSON_GetObjectItemCaseSensitive(root, "output");
    if (output) {
        *out_norm = parse_tensor(cJSON_GetObjectItemCaseSensitive(output, "norm"));
        *out_head = parse_tensor(cJSON_GetObjectItemCaseSensitive(output, "head"));
    }

    cJSON_Delete(root);
    return 0;
}

/* ── KV-format bridge (Stage 3b of docs/v2/00_CORRECTION.md) ──
 *
 * Reconcile the legacy `kv_bits` header field with the new
 * `inferbit_kv_format` config knob and pin both onto the model header
 * before KV allocation. The storage layout downstream is keyed on
 * `header.kv_bits` (see ib_alloc_kv_caches + forward.c::kv_cache_write
 * / kv_cache_read_head); `header.kv_format` is kept for diagnostics
 * and for the future on-line PQ codebook fitter.
 *
 * Rules:
 *   - kv_format == FP16 (default): preserve whatever kv_bits was in
 *     the .ibf header (typically 16). No override.
 *   - kv_format == INT8: force kv_bits = 8.
 *   - kv_format == PQ8: v1 has no codebook fitter wired through the
 *     production K/V write site (forward.c:794), so we cannot store
 *     PQ indices in the cache without losing the K/V data. Fall back
 *     to INT8 storage (kv_bits = 8) and emit a one-shot stderr
 *     warning. The decoder primitives already exist in pq_decode.c
 *     for the ib_pq_* mini-runtime (`storage_pyramid` path,
 *     pq_decode.c:5010-5208) — wiring them through the production
 *     inferbit_model attention path is the follow-up work.
 *
 * Exposed for pqv2_model.c which has its own load path. */
void ib_apply_kv_format(inferbit_model* model, const inferbit_config* config);
void ib_apply_kv_format(inferbit_model* model, const inferbit_config* config) {
    if (!model) return;
    int requested = config ? config->kv_format : (int)INFERBIT_KV_FP16;
    /* Default of zero from a zero-initialised inferbit_config means
     * FP16, which is the no-override path. */
    switch (requested) {
        case (int)INFERBIT_KV_FP16:
            model->header.kv_format = (int)INFERBIT_KV_FP16;
            /* Leave kv_bits as parsed (header default — fp16/fp32 path
             * in forward.c treats anything >=16 the same). */
            break;
        case (int)INFERBIT_KV_INT8:
            model->header.kv_format = (int)INFERBIT_KV_INT8;
            model->header.kv_bits   = 8;
            break;
        case (int)INFERBIT_KV_PQ8: {
            /* v1 fallback: PQ8 storage requires an on-line codebook
             * fitter at the K/V write site (forward.c:794) and a
             * matching decode at the read site (forward.c:861). The
             * pyramid PQ encode/decode primitives live inline in
             * pq_decode.c::forward_step_internal_sc (Phase 7) but
             * are tied to the ib_pq_session raw codebook tensors —
             * they aren't directly reusable for the inferbit_model
             * path yet. Surface the API, fall back to INT8 storage
             * cleanly, and record kv_format on the header for callers
             * who want to detect the fallback after load.
             * TODO: implement per-layer codebook fitting on first N
             * tokens + nearest-neighbor encode at write, decode at
             * read. The PPL invariant per docs/v2/00_CORRECTION.md
             * §3b is "PPL bit-identical to INT8 KV"; meet it by
             * falling back cleanly when the codebook isn't fit yet. */
            static int warned = 0;
            if (!warned) {
                fprintf(stderr,
                        "[inferbit] PQ8 KV cache: v1 falls back to INT8; "
                        "on-line codebook fitting is a future enhancement.\n");
                warned = 1;
            }
            model->header.kv_format = (int)INFERBIT_KV_PQ8;
            model->header.kv_bits   = 8;
            break;
        }
        default:
            /* Unknown values from a zero-init or future enum: stay on
             * the default FP16 path. */
            model->header.kv_format = (int)INFERBIT_KV_FP16;
            break;
    }
}

/* ── Allocate KV caches ─────────────────────────────────────── */

/* Exposed (was static) for PQv2 v6 loader to reuse. */
int ib_alloc_kv_caches(inferbit_model* model, int context_length, int dynamic);
int ib_alloc_kv_caches(inferbit_model* model, int context_length, int dynamic) {
    int num_layers  = model->header.num_layers;
    int num_kv_heads = model->header.num_kv_heads;
    int head_dim    = model->header.head_dim;
    int kv_bits     = model->header.kv_bits;
    int capacity    = context_length > 0 ? context_length : model->header.max_context_length;

    /* Rotating KV window (doc 36 phase 2.2): when set and smaller than
     * the full context, the physical cache is just `kv_window` slots —
     * this is what bounds KV RAM at long context. Logical position p
     * lives at physical slot p % capacity. Clamp the window to the
     * context so p % capacity stays a valid index; a window >= context
     * is a no-op (treated as full causal). */
    if (model->kv_window > 0 && model->kv_window < capacity) {
        capacity = model->kv_window;
    } else {
        model->kv_window = 0;
    }

    model->kv_caches = calloc(num_layers, sizeof(ib_kv_cache));
    if (!model->kv_caches) return -1;

    /* Bytes per token for one layer's K or V cache. */
    size_t kv_dim = (size_t)num_kv_heads * head_dim;
    size_t bytes_per_token;
    if (kv_bits >= 16) {
        bytes_per_token = kv_dim * sizeof(float);
    } else if (kv_bits == 8) {
        bytes_per_token = kv_dim;
    } else if (kv_bits == 4) {
        bytes_per_token = (kv_dim + 1) / 2;
    } else if (kv_bits == 2) {
        bytes_per_token = (kv_dim + 3) / 4;
    } else {
        ib_set_error("unsupported kv_bits=%d", kv_bits);
        return -1;
    }

    for (int i = 0; i < num_layers; i++) {
        ib_kv_cache* kv = &model->kv_caches[i];
        kv->length   = 0;
        kv->dynamic  = (dynamic != 0);

        if (dynamic) {
            /* Start empty, grow later */
            kv->capacity   = 0;
            kv->key_data   = NULL;
            kv->value_data = NULL;
        } else {
            kv->capacity   = capacity;
            kv->key_data   = calloc(capacity, bytes_per_token);
            kv->value_data = calloc(capacity, bytes_per_token);
            if (!kv->key_data || !kv->value_data) return -1;
        }

        /* Scale factors (FP32) for quantized KV */
        if (kv_bits < 16) {
            size_t scale_count = (size_t)capacity * num_kv_heads;
            kv->key_scales   = calloc(scale_count, sizeof(float));
            kv->value_scales = calloc(scale_count, sizeof(float));
            if (!kv->key_scales || !kv->value_scales) return -1;
        }
    }

    return 0;
}

/* ── Static fp32 caches (perf) ───────────────────────────────── */

/* Decode the fp16 on-disk scale (and norm) buffers into fp32 ONCE at load
 * time. forward.c::tensor_matmul + rmsnorm_fp16 then pick up the cached
 * fp32 pointers and skip the per-call fp16→fp32 conversion (~154 calls
 * per decode token on a 28-layer Llama).
 *
 * Bit-identical to the runtime conversion (ib_fp16_to_fp32 and forward.c's
 * static fp16_to_fp32 are both IEEE-correct).
 *
 * Safe to call multiple times — already-cached fields are skipped. */
static void cache_one_tensor(inferbit_model* m, ib_tensor_meta* t,
                             int is_norm) {
    if (!t) return;
    /* No on-disk fp16 scale buffer for PQv2 tensors (bits == -1) or for
     * tensors that simply have no quantization scales. */
    if (t->bits == 0 || t->bits < 0) {
        if (!is_norm) return;
    }
    if (is_norm) {
        /* Norm weights: stored as fp16 in the weight blob (no separate
         * scale buffer; bits == 16, size == N * 2). */
        if (t->norm_fp32 || t->bits != 16 || t->size == 0) return;
        int N = (t->ndim > 0 && t->shape[0] > 0) ? t->shape[0]
                                                  : (int)(t->size / 2);
        if (N <= 0) return;
        float* dst = (float*)malloc((size_t)N * sizeof(float));
        if (!dst) return;
        const uint16_t* src = (const uint16_t*)((const uint8_t*)m->weight_data
                                                + t->offset);
        for (int i = 0; i < N; i++) dst[i] = ib_fp16_to_fp32(src[i]);
        t->norm_fp32 = dst;
        return;
    }
    /* Weight scales: only present for INT2/4/8 quantized tensors. */
    if (t->scale_size == 0 || t->scale_offset == 0) return;
    if (t->scales_fp32 || t->blk32_scales_fp32) return;
    int M = (t->ndim > 0) ? t->shape[0] : 0;
    if (M <= 0) return;
    const uint16_t* src = (const uint16_t*)((const uint8_t*)m->weight_data
                                            + t->scale_offset);
    /* Per-block-32 INT4: scale_size > M*2 bytes (M * (N/32) fp16 entries). */
    if (t->bits == 4 && t->scale_size > (size_t)M * 2) {
        int N = (t->ndim > 1) ? t->shape[1] : 0;
        if (N <= 0 || (N % 32) != 0) return;
        size_t total = (size_t)M * (size_t)(N / 32);
        float* dst = (float*)malloc(total * sizeof(float));
        if (!dst) return;
        for (size_t i = 0; i < total; i++) dst[i] = ib_fp16_to_fp32(src[i]);
        t->blk32_scales_fp32 = dst;
        return;
    }
    /* Per-row scale (legacy INT4/INT8/INT2): M fp16 entries. */
    float* dst = (float*)malloc((size_t)M * sizeof(float));
    if (!dst) return;
    for (int i = 0; i < M; i++) dst[i] = ib_fp16_to_fp32(src[i]);
    t->scales_fp32 = dst;
}

/* ── Sparse-FFN cluster record resolver ──────────────────────────────
 *
 * Wire each layer's ffn_* runtime pointers to its on-disk cluster record.
 *
 * WHERE THE RECORD LIVES (the cross-agent contract — kept byte-for-byte in
 * sync with pqv2_encode.c::push_ffn_cluster_record and src/sparse_gate.h):
 *
 *   • IBF v6 (PQv2) models — the dominant path. The encoder emits the
 *     record as a NAMED RAW_INT32 manifest tensor:
 *
 *         L<li>.mlp.ffn_clusters
 *
 *     (i.e. for layer li, base "L<li>.mlp" + ".ffn_clusters"). We locate
 *     it by that exact name through the model's pqv2 file backing
 *     (ib_pqv2_find), which already parsed every manifest tensor's
 *     {raw_data, raw_size} into the mmap. NO JSON {offset,size} staging is
 *     involved on this path — the tensor name IS the locator.
 *
 *   • Legacy v5 (JSON) models — parse_layer() may stage an `ffn_cluster`
 *     {offset,size} pointer into model->weight_data. If a layer carries
 *     that staging and no pqv2 backing tensor was found, we fall back to
 *     resolving the record at weight_data + offset. (No v5 encoder writes
 *     this today; the path is kept for forward-compat / symmetry.)
 *
 * SELF-DESCRIBING RECORD (identical on both paths; see sparse_gate.h):
 *
 *   [0..7]   magic  "IBFFNCL1"   (IB_FFN_CLUSTER_MAGIC)
 *   [8..11]  u32    version = 1   (IB_FFN_CLUSTER_VERSION)
 *   [12..15] u32    flags         (bit0 = inter_perm present)
 *   [16..19] u32    n_clusters
 *   [20..23] u32    inter
 *   [24..27] u32    hidden        (== sizeof(ib_ffn_cluster_hdr) = 28)
 *   [28..]   u32    cluster_offsets[n_clusters + 1]
 *            fp16   centroids[n_clusters * hidden]
 *            u32    inter_perm[inter]   (iff flags bit0)
 *
 * ROUND-TRIP: convert writes record R for layer L as tensor
 * "L<L>.mlp.ffn_clusters"; this resolver finds R for layer L by that
 * name; hdr.n_clusters / inter / hidden, the offsets table, the centroids
 * and the perm all read back from the same bytes the encoder wrote. When
 * all clusters are active the permuted FFN reduces to the exact
 * non-clustered result (the inter-permutation cancels through down_proj),
 * so the record changes residency/skip behaviour only, never the math.
 *
 * Detection / versioning: the record is self-describing — magic + version
 * gate parsing. Any record that fails magic/version/bounds/consistency is
 * treated as ABSENT (the layer stays disabled) rather than failing the
 * load. A model with NO record (the default) leaves every ffn_* field
 * zero/NULL → byte-identical to today. Pointers index directly into the
 * mmap (zero-copy); the mapping outlives the model, nothing is owned here.
 *
 * Must run AFTER model->weight_data is set. Safe to call when no layer
 * has a record (it then only zeroes the disabled defaults). */
static void ib_resolve_ffn_clusters(inferbit_model* m) {
    if (!m || !m->layers || !m->weight_data) return;
    const uint8_t* base = (const uint8_t*)m->weight_data;
    size_t wsize = m->weight_data_size;
    const ib_pqv2_file* pqf = (const ib_pqv2_file*)m->pqv2_file_backing;

    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta* L = &m->layers[li];

        /* Start disabled; only a fully-valid record flips this on. */
        L->ffn_n_clusters      = 0;
        L->ffn_inter           = 0;
        L->ffn_hidden          = 0;
        L->ffn_cluster_offsets = NULL;
        L->ffn_centroids_fp16  = NULL;
        L->ffn_inter_perm      = NULL;

        /* Resolve {rec, blob} = start + byte length of this layer's record.
         * Prefer the IBF v6 named-tensor locator; fall back to v5 JSON
         * staging. If neither is present, the layer ships no record. */
        const uint8_t* rec = NULL;
        size_t blob = 0;

        if (pqf) {
            char nm[64];
            snprintf(nm, sizeof(nm), "L%d.mlp.ffn_clusters", li);
            const ib_pqv2_named_tensor* nt = ib_pqv2_find(pqf, nm);
            if (nt && nt->raw_data && nt->raw_size > 0) {
                /* raw_data points into the file mmap/buffer (zero-copy). */
                rec  = (const uint8_t*)nt->raw_data;
                blob = nt->raw_size;
            }
        }

        if (!rec) {
            /* Legacy v5 JSON {offset,size} staging into weight_data. */
            size_t off  = L->ffn_cluster_blob_offset;
            size_t jsz  = L->ffn_cluster_blob_size;
            if (off == 0 && jsz == 0) continue;   /* no record staged */
            /* Bounds: the staged span must lie inside the weight blob. */
            if (off > wsize || jsz > wsize - off) continue;
            rec  = base + off;
            blob = jsz;
        }

        /* Header must fit inside the blob. */
        if (blob < sizeof(ib_ffn_cluster_hdr)) continue;

        const ib_ffn_cluster_hdr* hdr = (const ib_ffn_cluster_hdr*)rec;

        if (memcmp(hdr->magic, IB_FFN_CLUSTER_MAGIC,
                   IB_FFN_CLUSTER_MAGIC_SIZE) != 0) continue;
        if (hdr->version != IB_FFN_CLUSTER_VERSION) continue;

        uint32_t nc     = hdr->n_clusters;
        uint32_t inter  = hdr->inter;
        uint32_t hidden = hdr->hidden;

        /* Disabled-by-content: n_clusters 0 or 1 means the encoder shipped
         * a placeholder; keep the gate off (no-op, byte-identical). */
        if (nc <= 1u || inter == 0u || hidden == 0u) continue;

        /* Compute trailing-array byte spans and verify they fit in blob.
         * Use 64-bit math to avoid uint32 overflow on the products. */
        size_t hdr_bytes  = sizeof(ib_ffn_cluster_hdr);
        size_t off_bytes  = ((size_t)nc + 1u) * sizeof(uint32_t);
        size_t cen_bytes  = (size_t)nc * (size_t)hidden * sizeof(uint16_t);
        int    have_perm  = (hdr->flags & IB_FFN_CLUSTER_FLAG_PERM) ? 1 : 0;
        size_t perm_bytes = have_perm ? (size_t)inter * sizeof(uint32_t) : 0u;

        size_t need = hdr_bytes + off_bytes + cen_bytes + perm_bytes;
        if (need > blob) continue;   /* truncated / inconsistent — disable */

        const uint8_t* p = rec + hdr_bytes;
        const uint32_t* offsets   = (const uint32_t*)p;  p += off_bytes;
        const uint16_t* centroids = (const uint16_t*)p;  p += cen_bytes;
        const uint32_t* perm      = have_perm ? (const uint32_t*)p : NULL;

        /* Sanity-check the offsets: monotonic, [0]=0, [nc]=inter. A
         * malformed table disables the layer rather than risking an
         * out-of-range FFN row range at runtime. */
        if (offsets[0] != 0u || offsets[nc] != inter) continue;
        int ok = 1;
        for (uint32_t c = 0; c < nc; c++) {
            if (offsets[c + 1] < offsets[c] || offsets[c + 1] > inter) {
                ok = 0; break;
            }
        }
        if (!ok) continue;

        /* All checks passed — wire the (mmap-backed, zero-copy) pointers. */
        L->ffn_n_clusters      = nc;
        L->ffn_inter           = inter;
        L->ffn_hidden          = hidden;
        L->ffn_cluster_offsets = offsets;
        L->ffn_centroids_fp16  = centroids;
        L->ffn_inter_perm      = perm;
    }
}

/* Walk every tensor slot on the model and pre-decode its static fp32
 * scales/norm buffer. Called once at the end of every load path
 * (ibf_load + pqv2_load_internal) so both legacy v5 and PQv2 v6 models
 * get the optimization. */
void ib_cache_model_static_fp32(inferbit_model* m);
void ib_cache_model_static_fp32(inferbit_model* m) {
    if (!m) return;

    /* Resolve any optional per-layer sparse-FFN cluster records now that
     * model->weight_data (and, for IBF v6, model->pqv2_file_backing) is
     * mapped. This finalizer is the SINGLE seam both load paths (PQv2 v6 +
     * legacy v5) share, so wiring the resolver here makes the cluster
     * fields populate on BOTH paths. No-op (and byte-identical) for every
     * model that ships no `Lk.mlp.ffn_clusters` record. Idempotent: it
     * re-derives the disabled defaults and re-writes the same mmap-backed
     * pointers on each call. */
    ib_resolve_ffn_clusters(m);

    /* Globals: token_embedding has per-row scales when quantized; output
     * head ditto; output_norm is fp16. token_embedding usually doesn't go
     * through tensor_matmul, but caching is cheap. */
    cache_one_tensor(m, &m->token_embedding, 0);
    cache_one_tensor(m, &m->output_head,     0);
    cache_one_tensor(m, &m->output_norm,     1);
    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta* L = &m->layers[li];
        cache_one_tensor(m, &L->q_proj,    0);
        cache_one_tensor(m, &L->k_proj,    0);
        cache_one_tensor(m, &L->v_proj,    0);
        cache_one_tensor(m, &L->o_proj,    0);
        cache_one_tensor(m, &L->gate_proj, 0);
        cache_one_tensor(m, &L->up_proj,   0);
        cache_one_tensor(m, &L->down_proj, 0);
        cache_one_tensor(m, &L->input_norm,     1);
        cache_one_tensor(m, &L->post_attn_norm, 1);
        /* MoME experts (Stage 3a). Skip if absent. */
        if (L->mome_experts > 1) {
            for (int e = 0; e < L->mome_experts; e++) {
                if (L->gate_proj_experts) cache_one_tensor(m, &L->gate_proj_experts[e], 0);
                if (L->up_proj_experts)   cache_one_tensor(m, &L->up_proj_experts[e],   0);
                if (L->down_proj_experts) cache_one_tensor(m, &L->down_proj_experts[e], 0);
            }
        }
    }

    /* Burst / cool-down duty cycle (M1): seed the controller to the EXACT
     * profile and cache the activation-skip ratio from IB_PQV2_SKIP. Both
     * load paths (PQv2 + legacy) call this finalizer, so this is the single
     * seam that guarantees m->active_profile is non-NULL and that the
     * env-driven activation skip keeps working WITHOUT any explicit
     * inferbit_burst_attach call — preserving today's default behaviour. */
    inferbit_burst_attach(m, NULL);
}

/* ── Allocate activation buffers ────────────────────────────── */

int ib_alloc_buffers(inferbit_model* model);
int ib_alloc_buffers(inferbit_model* model) {
    int h = model->header.hidden_size;
    int inter = model->header.intermediate_size;
    int vocab = model->header.vocab_size;
    int num_heads = model->header.num_heads;
    int head_dim = model->header.head_dim;
    int max_ctx = model->header.max_context_length;

    model->buf_residual = calloc(h, sizeof(float));
    model->buf_hidden   = calloc(h, sizeof(float));
    model->buf_attn     = calloc(h, sizeof(float));
    model->buf_mlp      = calloc(inter, sizeof(float));
    model->buf_mlp2     = calloc(inter, sizeof(float));
    model->buf_logits   = calloc(vocab, sizeof(float));

    int num_kv_heads = model->header.num_kv_heads;
    int kv_dim = num_kv_heads * head_dim;

    /* QKV scratch layout (must match forward.c):
     * q:         [hidden]
     * k:         [kv_dim]
     * v:         [kv_dim]
     * att:       [num_heads * max_ctx]
     * scale_buf: [max(hidden, inter, vocab)]
     */
    int scale_sz = h > inter ? h : inter;
    if (vocab > scale_sz) scale_sz = vocab;
    size_t qkv_size = (size_t)h + kv_dim + kv_dim + (size_t)num_heads * max_ctx + scale_sz;
    model->buf_qkv = calloc(qkv_size, sizeof(float));

    if (!model->buf_residual || !model->buf_hidden || !model->buf_attn ||
        !model->buf_mlp || !model->buf_mlp2 || !model->buf_logits || !model->buf_qkv) {
        return -1;
    }

    /* Batched-forward scratch (IB_BATCH_MAX positions). Allocated once so
     * the spec-verify loop doesn't malloc/free ~1 MB per round. */
    int B = IB_BATCH_MAX;
    int n_max = h > inter ? h : inter;
    int groups_max = (n_max + IB_W4A8_GROUP - 1) / IB_W4A8_GROUP;
    int scale_sz_b = h > inter ? h : inter;
    if (vocab > scale_sz_b) scale_sz_b = vocab;

    model->bb_x         = calloc((size_t)B * h,       sizeof(float));
    model->bb_xb        = calloc((size_t)B * h,       sizeof(float));
    model->bb_xb2       = calloc((size_t)B * h,       sizeof(float));
    model->bb_q         = calloc((size_t)B * h,       sizeof(float));
    model->bb_k         = calloc((size_t)B * kv_dim,  sizeof(float));
    model->bb_v         = calloc((size_t)B * kv_dim,  sizeof(float));
    model->bb_hb        = calloc((size_t)B * inter,   sizeof(float));
    model->bb_hb2       = calloc((size_t)B * inter,   sizeof(float));
    model->bb_scale     = calloc((size_t)scale_sz_b,  sizeof(float));
    model->bb_att       = calloc((size_t)num_heads * max_ctx, sizeof(float));
    model->bb_qscratch  = calloc((size_t)B * n_max,   sizeof(int8_t));
    model->bb_sa        = calloc((size_t)B * groups_max, sizeof(float));
    model->bb_positions = calloc((size_t)B,           sizeof(int));

    if (!model->bb_x || !model->bb_xb || !model->bb_xb2 || !model->bb_q ||
        !model->bb_k || !model->bb_v || !model->bb_hb || !model->bb_hb2 ||
        !model->bb_scale || !model->bb_att || !model->bb_qscratch ||
        !model->bb_sa || !model->bb_positions) {
        return -1;
    }

    /* Precomputed RoPE cos/sin tables (perf: removes per-token sinf/cosf
     * cost from the attention path). Table layout is
     *   tab[pos * (head_dim/2) + i/2]  for i in {0, 2, 4, ..., head_dim-2}
     * matching the kernel rotation step. theta comes from header.rope_theta
     * (10000.0 for Llama-2/TinyLlama, 500000.0 for Llama-3). Allocation
     * failure is non-fatal: kernels fall back to live sinf/cosf when
     * rope_cos/rope_sin are NULL. */
    if (head_dim > 1 && max_ctx > 0) {
        int half = head_dim / 2;
        size_t cells = (size_t)max_ctx * (size_t)half;
        model->rope_cos = calloc(cells, sizeof(float));
        model->rope_sin = calloc(cells, sizeof(float));
        if (model->rope_cos && model->rope_sin) {
            float theta = model->header.rope_theta > 0.0f
                          ? model->header.rope_theta : 10000.0f;
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(theta,
                                         (float)(2 * i) / (float)head_dim);
                for (int p = 0; p < max_ctx; p++) {
                    float angle = (float)p * freq;
                    model->rope_cos[(size_t)p * half + i] = cosf(angle);
                    model->rope_sin[(size_t)p * half + i] = sinf(angle);
                }
            }
            model->rope_table_ctx = max_ctx;
        } else {
            /* OOM is non-fatal — kernels handle NULL via sinf/cosf fallback. */
            free(model->rope_cos); model->rope_cos = NULL;
            free(model->rope_sin); model->rope_sin = NULL;
            model->rope_table_ctx = 0;
        }
    }

    /* ── Goal H4 — hot-tensor pool (scaffolding) ────────────────
     *
     * Allocate a small RAM-resident scratch region that the future
     * adaptive promotion logic will use to keep the hottest tensors
     * decode-resident. Today nothing populates this pool — see
     * ib_hot_promote in forward.c, which is a deliberate no-op until
     * the access-count instrumentation has produced enough data to
     * drive promotion decisions.
     *
     * Sized via IB_HOT_POOL_MB (default 32 MB; 0 disables). OOM is
     * non-fatal — model load still succeeds, ib_hot_lookup just keeps
     * returning NULL. */
    {
        size_t hot_mb = 32;
        const char *hp_env = getenv("IB_HOT_POOL_MB");
        if (hp_env && hp_env[0]) {
            long v = strtol(hp_env, NULL, 10);
            if (v >= 0) hot_mb = (size_t)v;
        }
        if (hot_mb > 0) {
            size_t bytes = hot_mb * (size_t)1024 * (size_t)1024;
            model->hot_pool = malloc(bytes);
            if (model->hot_pool) {
                model->hot_pool_bytes = bytes;
                /* Carve the control header + entry table out of the front
                 * of the same buffer so model.c's free(hot_pool) reclaims
                 * everything. Lay out: [hdr][entry[cap]][arena]. */
                ib_hot_pool_init(model->hot_pool, bytes);
                /* Best-effort residency lock: keep the hot weights pinned in
                 * RAM so they aren't paged out under memory pressure. POSIX
                 * mlock can fail without privilege / over RLIMIT_MEMLOCK —
                 * that's fine, it's purely a lossless stability win, never
                 * fatal. The pages stay locked until free(hot_pool) (munlock
                 * is implicit on unmap/free). */
#ifndef _WIN32
                (void)mlock(model->hot_pool, bytes);
#endif
            } else {
                model->hot_pool_bytes = 0;
            }
        }
        model->hot_pool_entries = 0;
    }

    return 0;
}

/* ── Public: load .ibf file ─────────────────────────────────── */

inferbit_model* ibf_load(const char* path, const inferbit_config* config) {
    /* Open file */
    int fd = ib_open(path, O_RDONLY);
    if (fd < 0) {
        ib_set_error("failed to open %s: %s", path, strerror(errno));
        return NULL;
    }

    /* Get file size */
    ib_struct_stat st;
    if (ib_fstat(fd, &st) < 0) {
        ib_set_error("failed to stat %s: %s", path, strerror(errno));
        ib_close(fd);
        return NULL;
    }
    size_t file_size = (size_t)st.st_size;

    if (file_size < IBF_PREAMBLE) {
        ib_set_error("file too small to be IBF: %zu bytes", file_size);
        ib_close(fd);
        return NULL;
    }

    /* Read preamble (32 bytes) */
    uint8_t preamble[IBF_PREAMBLE];
    if (ib_read(fd, preamble, IBF_PREAMBLE) != IBF_PREAMBLE) {
        ib_set_error("failed to read IBF preamble");
        ib_close(fd);
        return NULL;
    }

    /* Validate magic */
    if (memcmp(preamble, IBF_MAGIC, IBF_MAGIC_SIZE) != 0) {
        ib_set_error("invalid IBF magic number");
        ib_close(fd);
        return NULL;
    }

    /* Read version (uint32 LE) */
    uint32_t version;
    memcpy(&version, preamble + 8, 4);
    if (version > IBF_VERSION) {
        ib_set_error("unsupported IBF version: %u (max supported: %u)", version, IBF_VERSION);
        ib_close(fd);
        return NULL;
    }

    /* Read header size */
    uint32_t header_size;
    memcpy(&header_size, preamble + 12, 4);

    if (IBF_PREAMBLE + header_size > file_size) {
        ib_set_error("IBF header size exceeds file size");
        ib_close(fd);
        return NULL;
    }

    /* Read flags */
    uint32_t flags;
    memcpy(&flags, preamble + 16, 4);
    (void)flags;  /* TODO: use flags for sparsity_masks, calibration_data */

    /* Read JSON header */
    char* json_buf = malloc(header_size + 1);
    if (!json_buf) {
        ib_set_error("failed to allocate JSON header buffer");
        ib_close(fd);
        return NULL;
    }
    int nread = (int)ib_read(fd, json_buf, (unsigned int)header_size);
    if (nread < 0 || (unsigned int)nread != header_size) {
        ib_set_error("failed to read IBF JSON header");
        free(json_buf);
        ib_close(fd);
        return NULL;
    }
    json_buf[header_size] = '\0';

    /* Allocate model */
    inferbit_model* model = calloc(1, sizeof(inferbit_model));
    if (!model) {
        ib_set_error("failed to allocate model");
        free(json_buf);
        ib_close(fd);
        return NULL;
    }
    /* Drive-mode sidecar fd defaults to -1 (no sidecar). */
    model->drive_fd_pretransposed = -1;

    /* Parse JSON header */
    if (parse_header_json(json_buf, header_size, &model->header,
                          &model->layers, &model->token_embedding,
                          &model->output_norm, &model->output_head) != 0) {
        free(json_buf);
        free(model);
        ib_close(fd);
        return NULL;
    }
    free(json_buf);

    /* Validate required fields */
    if (model->header.num_layers <= 0 || model->header.hidden_size <= 0 ||
        model->header.vocab_size <= 0) {
        ib_set_error("IBF header missing required architecture fields");
        free(model->layers);
        free(model);
        ib_close(fd);
        return NULL;
    }

    /* Compute weight data offset (aligned) */
    size_t weight_offset = align_up(IBF_PREAMBLE + header_size, IBF_ALIGNMENT);
    if (model->header.weight_data_offset > 0) {
        weight_offset = model->header.weight_data_offset;
    }

    size_t weight_size = file_size - weight_offset;
    if (model->header.weight_data_size > 0 && model->header.weight_data_size < weight_size) {
        weight_size = model->header.weight_data_size;
    }

    /* Memory-map weight data */
    void* mapped = ib_mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        ib_set_error("failed to mmap %s: %s", path, strerror(errno));
        free(model->layers);
        free(model);
        ib_close(fd);
        return NULL;
    }

    /* Ask the kernel to back this region with huge pages when possible.
     *
     * Linux: MADV_HUGEPAGE promotes to transparent 2MB pages (if THP is
     * enabled) and cuts the TLB miss rate dramatically on multi-GB weight
     * sets. MADV_WILLNEED + MADV_SEQUENTIAL tell the prefetcher that the
     * full region will be walked forward.
     *
     * macOS: MADV_HUGEPAGE isn't supported, but the kernel automatically
     * promotes to 16KB pages (which is already 4× the Linux default). We
     * still hint WILLNEED so the first-touch faults don't stall decode.
     *
     * Failures here are benign — they just mean we run at default page
     * size. We don't check the return. */
#ifdef MADV_HUGEPAGE
    madvise(mapped, file_size, MADV_HUGEPAGE);
#endif
#ifdef MADV_WILLNEED
    madvise(mapped, file_size, MADV_WILLNEED);
#endif

    model->weight_data      = (uint8_t*)mapped + weight_offset;
    model->weight_data_size = weight_size;
    model->weight_data_mmap = true;
    model->mmap_fd          = fd;
    /* Note: we keep fd open while mmap is active (some systems need it) */

    /* Apply config */
    int ctx_len = 0;
    int kv_dynamic = 0;
    int threads = ib_hardware_concurrency();
    if (config) {
        ctx_len    = config->context_length;
        kv_dynamic = config->kv_dynamic;
        threads    = config->threads > 0 ? config->threads : ib_hardware_concurrency();
        model->kv_window = config->kv_window;
    }
    model->num_threads = threads;

    /* Stage 3b: resolve kv_format → kv_bits before KV alloc so the
     * storage layout matches the chosen format (PQ8 falls back to
     * INT8 here; FP16 keeps the header default). */
    ib_apply_kv_format(model, config);

    /* Allocate KV caches */
    if (ib_alloc_kv_caches(model, ctx_len, kv_dynamic) != 0) {
        ib_set_error("failed to allocate KV caches");
        ib_munmap(mapped, file_size);
        ib_close(fd);
        free(model->layers);
        free(model);
        return NULL;
    }

    /* Allocate activation buffers */
    if (ib_alloc_buffers(model) != 0) {
        ib_set_error("failed to allocate activation buffers");
        /* TODO: proper cleanup of kv_caches */
        ib_munmap(mapped, file_size);
        ib_close(fd);
        free(model->layers);
        free(model);
        return NULL;
    }

    /* Initialize kernel dispatch */
    ib_simd_level simd = ib_detect_simd();
    ib_init_kernels(simd);

    /* Create thread pool (NULL if single-threaded) */
    model->thread_pool = ib_pool_create(threads);

    /* Perf: pre-decode fp16 weight-scales and norm weights into fp32 so
     * forward.c::tensor_matmul + rmsnorm_fp16 skip per-call conversion.
     * This finalizer also resolves any optional sparse-FFN cluster records
     * (ib_resolve_ffn_clusters), the single seam shared with the PQv2 v6
     * load path — so both paths populate the layer cluster fields. */
    ib_cache_model_static_fp32(model);

    return model;
}
