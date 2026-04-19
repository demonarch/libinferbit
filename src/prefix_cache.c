/*
 * prefix_cache.c — On-disk KV cache for repeated prompt prefixes.
 *
 * v1: exact-match only. Activated by IB_PREFIX_CACHE_DIR. On a cache hit,
 * restores KV state for the first N-1 input tokens and lets the caller run
 * a single-token forward to obtain logits for the Nth position. On a miss,
 * saves the KV state for the first N-1 positions after prefill completes.
 *
 * File format (little-endian):
 *   magic        "IBKV"
 *   version      u32 = 1
 *   fingerprint  u64   (identifies the model)
 *   prefix_len   u32   (number of token positions stored)
 *   num_layers   u32
 *   kv_bits      u32
 *   num_kv_heads u32
 *   head_dim     u32
 *   reserved     u32 * 3
 *   per-layer {
 *     length     u32  (== prefix_len)
 *     _pad       u32
 *     key_data   prefix_len * bytes_per_token
 *     value_data prefix_len * bytes_per_token
 *     key_scales prefix_len * num_kv_heads * 4   (FP32)
 *     value_scales prefix_len * num_kv_heads * 4 (FP32)
 *   }
 */

#include "inferbit_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define IBKV_MAGIC 0x564B4249u  /* "IBKV" little-endian */
#define IBKV_VERSION 1u

/* ── FNV-1a-64 ──────────────────────────────────────────────── */

static uint64_t fnv1a_64_init(void) { return 0xcbf29ce484222325ull; }

static uint64_t fnv1a_64(uint64_t h, const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

/* ── KV-cache sizing helpers ────────────────────────────────── */

static size_t bytes_per_token(int kv_bits, int num_kv_heads, int head_dim) {
    size_t kv_dim = (size_t)num_kv_heads * head_dim;
    if (kv_bits >= 16) return kv_dim * sizeof(float);
    if (kv_bits == 8)  return kv_dim;
    if (kv_bits == 4)  return (kv_dim + 1) / 2;
    if (kv_bits == 2)  return (kv_dim + 3) / 4;
    return 0;
}

/* ── Model fingerprint ──────────────────────────────────────── */

static uint64_t model_fingerprint(const inferbit_model* model) {
    uint64_t h = fnv1a_64_init();

    /* Header identity. Architecture params + special tokens + quant config. */
    const ib_ibf_header* hd = &model->header;
    h = fnv1a_64(h, hd->architecture, sizeof(hd->architecture));
    h = fnv1a_64(h, &hd->num_layers, sizeof(hd->num_layers));
    h = fnv1a_64(h, &hd->hidden_size, sizeof(hd->hidden_size));
    h = fnv1a_64(h, &hd->num_heads, sizeof(hd->num_heads));
    h = fnv1a_64(h, &hd->num_kv_heads, sizeof(hd->num_kv_heads));
    h = fnv1a_64(h, &hd->head_dim, sizeof(hd->head_dim));
    h = fnv1a_64(h, &hd->intermediate_size, sizeof(hd->intermediate_size));
    h = fnv1a_64(h, &hd->vocab_size, sizeof(hd->vocab_size));
    h = fnv1a_64(h, &hd->max_context_length, sizeof(hd->max_context_length));
    h = fnv1a_64(h, &hd->default_bits, sizeof(hd->default_bits));
    h = fnv1a_64(h, &hd->kv_bits, sizeof(hd->kv_bits));
    h = fnv1a_64(h, &hd->weight_data_size, sizeof(hd->weight_data_size));

    /* Mix in a weight-data prefix so two models with identical headers but
     * different weights (e.g. base vs finetune) do not collide. 64 KiB is
     * cheap (microseconds) and gives strong distinction. */
    if (model->weight_data && model->weight_data_size > 0) {
        size_t n = model->weight_data_size < 65536 ? model->weight_data_size : 65536;
        h = fnv1a_64(h, model->weight_data, n);
    }
    return h;
}

/* ── Cache path construction ────────────────────────────────── */

static int build_cache_path(const char* dir, uint64_t key, char* out, size_t out_sz) {
    int n = snprintf(out, out_sz, "%s/%016llx.ibkv", dir, (unsigned long long)key);
    return (n > 0 && (size_t)n < out_sz) ? 0 : -1;
}

static int ensure_dir(const char* dir) {
    struct stat st;
    if (stat(dir, &st) == 0) return S_ISDIR(st.st_mode) ? 0 : -1;
    if (errno != ENOENT) return -1;
    /* Try to create. Fail silently if someone created it first. */
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return -1;
    return 0;
}

/* ── Lookup key ─────────────────────────────────────────────── */

static uint64_t compute_lookup_key(const inferbit_model* model,
                                   const int32_t* tokens, int n_tokens) {
    uint64_t fp = model_fingerprint(model);
    uint64_t h = fnv1a_64_init();
    h = fnv1a_64(h, &fp, sizeof(fp));
    h = fnv1a_64(h, &n_tokens, sizeof(n_tokens));
    h = fnv1a_64(h, tokens, (size_t)n_tokens * sizeof(int32_t));
    return h;
}

/* ── Gate ───────────────────────────────────────────────────── */

typedef struct {
    const char* dir;
    int         min_tokens;
    int         enabled;
} ib_prefix_cache_cfg;

static ib_prefix_cache_cfg read_cfg(void) {
    ib_prefix_cache_cfg c = { NULL, 32, 0 };
    const char* dir = getenv("IB_PREFIX_CACHE_DIR");
    if (dir && dir[0]) {
        c.dir = dir;
        c.enabled = 1;
    }
    const char* mt = getenv("IB_PREFIX_CACHE_MIN_TOKENS");
    if (mt && mt[0]) {
        int v = atoi(mt);
        if (v >= 1) c.min_tokens = v;
    }
    return c;
}

/* ── File I/O helpers ───────────────────────────────────────── */

static int read_exact(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n ? 0 : -1;
}

static int write_exact(FILE* f, const void* buf, size_t n) {
    return fwrite(buf, 1, n, f) == n ? 0 : -1;
}

/* ── Restore ────────────────────────────────────────────────── */

/* Returns restored prefix length on success (>=1), 0 on miss, -1 on error
 * (error is non-fatal; caller falls back to normal prefill). */
int ib_prefix_cache_try_restore(inferbit_model* model,
                                const int32_t* tokens, int n_tokens) {
    ib_prefix_cache_cfg cfg = read_cfg();
    if (!cfg.enabled || !model || !tokens || n_tokens < cfg.min_tokens) return 0;
    if (!model->kv_caches) return 0;

    /* v1 supports only non-dynamic caches (capacity must already be large
     * enough to hold the restored prefix). */
    int prefix_len = n_tokens - 1;
    if (prefix_len <= 0) return 0;
    if (model->kv_caches[0].capacity < prefix_len) return 0;

    /* Only usable when KV is empty — don't stomp on an in-flight continuation. */
    if (model->kv_caches[0].length != 0) return 0;

    uint64_t key = compute_lookup_key(model, tokens, n_tokens);
    char path[4096];
    if (build_cache_path(cfg.dir, key, path, sizeof(path)) != 0) return 0;

    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t magic = 0, version = 0;
    uint64_t fp = 0;
    uint32_t stored_len = 0, layers = 0, kv_bits = 0, kvh = 0, hd = 0, resv[3] = {0};
    if (read_exact(f, &magic, 4) || magic != IBKV_MAGIC) { fclose(f); return 0; }
    if (read_exact(f, &version, 4) || version != IBKV_VERSION) { fclose(f); return 0; }
    if (read_exact(f, &fp, 8)) { fclose(f); return 0; }
    if (read_exact(f, &stored_len, 4) || (int)stored_len != prefix_len) { fclose(f); return 0; }
    if (read_exact(f, &layers, 4) || (int)layers != model->header.num_layers) { fclose(f); return 0; }
    if (read_exact(f, &kv_bits, 4) || (int)kv_bits != model->header.kv_bits) { fclose(f); return 0; }
    if (read_exact(f, &kvh, 4) || (int)kvh != model->header.num_kv_heads) { fclose(f); return 0; }
    if (read_exact(f, &hd, 4) || (int)hd != model->header.head_dim) { fclose(f); return 0; }
    if (read_exact(f, resv, sizeof(resv))) { fclose(f); return 0; }

    size_t bpt = bytes_per_token((int)kv_bits, (int)kvh, (int)hd);
    if (bpt == 0) { fclose(f); return 0; }

    size_t kv_bytes = bpt * (size_t)prefix_len;
    size_t scale_bytes = (size_t)kvh * (size_t)prefix_len * sizeof(float);

    int has_scales = (kv_bits < 16);
    for (int L = 0; L < (int)layers; L++) {
        ib_kv_cache* kv = &model->kv_caches[L];
        uint32_t layer_len = 0, pad = 0;
        if (read_exact(f, &layer_len, 4) || (int)layer_len != prefix_len) goto fail;
        if (read_exact(f, &pad, 4)) goto fail;
        if (!kv->key_data || !kv->value_data) goto fail;
        if (kv->capacity < prefix_len) goto fail;
        if (read_exact(f, kv->key_data,   kv_bytes)) goto fail;
        if (read_exact(f, kv->value_data, kv_bytes)) goto fail;
        if (has_scales) {
            if (!kv->key_scales || !kv->value_scales) goto fail;
            if (read_exact(f, kv->key_scales,   scale_bytes)) goto fail;
            if (read_exact(f, kv->value_scales, scale_bytes)) goto fail;
        }
        kv->length = prefix_len;
    }
    fclose(f);
    return prefix_len;

fail:
    /* Partial restore — reset so the caller gets a clean slate. */
    for (int L = 0; L < model->header.num_layers; L++) model->kv_caches[L].length = 0;
    fclose(f);
    return -1;
}

/* ── Save ───────────────────────────────────────────────────── */

/* Call after a successful prefill of n_tokens. Stores KV for positions
 * 0..n_tokens-2 (i.e., n_tokens-1 positions). Silent no-op on failure. */
int ib_prefix_cache_save(const inferbit_model* model,
                         const int32_t* tokens, int n_tokens) {
    ib_prefix_cache_cfg cfg = read_cfg();
    if (!cfg.enabled || !model || !tokens || n_tokens < cfg.min_tokens) return 0;
    if (!model->kv_caches) return 0;

    int prefix_len = n_tokens - 1;
    if (prefix_len <= 0) return 0;

    /* Require that prefill already wrote at least prefix_len positions. */
    if (model->kv_caches[0].length < prefix_len) return 0;

    if (ensure_dir(cfg.dir) != 0) return 0;

    uint64_t key = compute_lookup_key(model, tokens, n_tokens);
    char path[4096];
    if (build_cache_path(cfg.dir, key, path, sizeof(path)) != 0) return 0;

    /* Write to .tmp then rename for atomicity. */
    char tmp[4200];
    int tn = snprintf(tmp, sizeof(tmp), "%s.tmp.%d", path, (int)getpid());
    if (tn <= 0 || (size_t)tn >= sizeof(tmp)) return 0;

    FILE* f = fopen(tmp, "wb");
    if (!f) return 0;

    uint32_t magic = IBKV_MAGIC, version = IBKV_VERSION;
    uint64_t fp = model_fingerprint(model);
    uint32_t stored_len = (uint32_t)prefix_len;
    uint32_t layers = (uint32_t)model->header.num_layers;
    uint32_t kv_bits = (uint32_t)model->header.kv_bits;
    uint32_t kvh = (uint32_t)model->header.num_kv_heads;
    uint32_t hd = (uint32_t)model->header.head_dim;
    uint32_t resv[3] = {0, 0, 0};

    if (write_exact(f, &magic, 4)) goto fail;
    if (write_exact(f, &version, 4)) goto fail;
    if (write_exact(f, &fp, 8)) goto fail;
    if (write_exact(f, &stored_len, 4)) goto fail;
    if (write_exact(f, &layers, 4)) goto fail;
    if (write_exact(f, &kv_bits, 4)) goto fail;
    if (write_exact(f, &kvh, 4)) goto fail;
    if (write_exact(f, &hd, 4)) goto fail;
    if (write_exact(f, resv, sizeof(resv))) goto fail;

    size_t bpt = bytes_per_token((int)kv_bits, (int)kvh, (int)hd);
    size_t kv_bytes = bpt * (size_t)prefix_len;
    size_t scale_bytes = (size_t)kvh * (size_t)prefix_len * sizeof(float);

    int has_scales = ((int)kv_bits < 16);
    for (int L = 0; L < (int)layers; L++) {
        const ib_kv_cache* kv = &model->kv_caches[L];
        uint32_t layer_len = (uint32_t)prefix_len, pad = 0;
        if (write_exact(f, &layer_len, 4)) goto fail;
        if (write_exact(f, &pad, 4)) goto fail;
        if (write_exact(f, kv->key_data,   kv_bytes)) goto fail;
        if (write_exact(f, kv->value_data, kv_bytes)) goto fail;
        if (has_scales) {
            if (!kv->key_scales || !kv->value_scales) goto fail;
            if (write_exact(f, kv->key_scales,   scale_bytes)) goto fail;
            if (write_exact(f, kv->value_scales, scale_bytes)) goto fail;
        }
    }
    fclose(f);
    if (rename(tmp, path) != 0) { unlink(tmp); return 0; }
    return 1;

fail:
    fclose(f);
    unlink(tmp);
    return 0;
}
