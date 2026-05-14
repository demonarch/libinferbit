/*
 * test_hidden_states_capture.c — validates Phase 3.1 API
 *   ib_metal_forward_prefill_logits_all_ex
 *
 * Loads a real PQv2 IBF, runs prefill twice (no-capture vs capture),
 * and checks:
 *  - logits match between the two paths (proves capture doesn't corrupt
 *    the forward path)
 *  - all captured hidden states are finite
 *  - first/last layer hidden states have non-zero norm (sanity)
 *
 * Usage:  test_hidden_states_capture <model.ibf> [n_tokens=8]
 */
#include "inferbit.h"
#include "metal/metal_runtime.h"
#include "inferbit_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Internal: peek at hidden_size and num_layers without leaking the full
 * model struct. Use the public accessor pattern. */
static int model_hidden(inferbit_model *m) {
    /* hidden_size is in the header; expose via a public helper would be
     * cleaner, but the existing tests use header.* directly via the
     * "inferbit_internal.h" path. Keep it minimal: probe via embedding
     * lookup len. We'll just hardcode by reading from a forward call's
     * output shape, but simplest is to declare the headers and read. */
    return m->header.hidden_size;
}

static int model_layers(inferbit_model *m) {
    return m->header.num_layers;
}

static void cpu_embed_lookup_into(inferbit_model *m, int tok, float *out) {
    /* Use the internal helper if available; for portability we let the
     * model auto-embed via inferbit_forward. To keep the test isolated,
     * just zero-fill a fake embedding — the forward path will see
     * something and the hidden states will be consistent across both
     * runs since the input is the same. */
    int H = model_hidden(m);
    for (int i = 0; i < H; i++) out[i] = ((float)((tok + i) % 13)) * 0.01f;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.ibf> [n_tokens=8]\n", argv[0]);
        return 1;
    }
    const char *ibf_path = argv[1];
    int n_tokens = (argc > 2) ? atoi(argv[2]) : 8;

    inferbit_config *cfg = inferbit_config_create();
    inferbit_config_set_context_length(cfg, 256);
    inferbit_model *m = inferbit_load(ibf_path, cfg);
    if (!m) {
        fprintf(stderr, "load failed: %s\n", inferbit_last_error());
        return 2;
    }

    int hidden = model_hidden(m);
    int n_layers = model_layers(m);
    int vocab   = m->header.vocab_size;
    printf("loaded %s: hidden=%d layers=%d vocab=%d\n",
           ibf_path, hidden, n_layers, vocab);

    ib_metal_ctx *ctx = ib_metal_create();
    ib_metal_model_buffers *bufs = ib_metal_upload_model(ctx, m);
    if (!bufs) { fprintf(stderr, "upload failed\n"); return 3; }

    /* Build n_tokens fake embeddings. */
    size_t embs_floats = (size_t)n_tokens * hidden;
    float *embs = calloc(embs_floats, sizeof(float));
    for (int t = 0; t < n_tokens; t++) {
        cpu_embed_lookup_into(m, t + 1, embs + (size_t)t * hidden);
    }

    /* No-capture path. */
    size_t logits_floats = (size_t)n_tokens * vocab;
    float *logits_a = malloc(logits_floats * sizeof(float));
    ib_metal_reset_kv(bufs);
    int rc = ib_metal_forward_prefill_logits_all(ctx, bufs, embs, n_tokens, 0, logits_a);
    if (rc != 0) { fprintf(stderr, "no-capture forward failed rc=%d\n", rc); return 4; }

    /* Capture path: allocate per-layer buffers + collect last-layer norm. */
    float **hs = calloc(n_layers, sizeof(float *));
    for (int L = 0; L < n_layers; L++) {
        hs[L] = calloc((size_t)n_tokens * hidden, sizeof(float));
    }
    float *logits_b = malloc(logits_floats * sizeof(float));
    ib_metal_reset_kv(bufs);
    rc = ib_metal_forward_prefill_logits_all_ex(ctx, bufs, embs, n_tokens, 0, logits_b, hs);
    if (rc != 0) { fprintf(stderr, "capture forward failed rc=%d\n", rc); return 5; }

    /* Validate logits match. Capture path uses extra checkpoints which
     * shouldn't affect numerics. */
    float max_diff = 0.0f;
    for (size_t i = 0; i < logits_floats; i++) {
        float d = logits_a[i] - logits_b[i];
        if (d < 0) d = -d;
        if (d > max_diff) max_diff = d;
    }
    printf("logits max|diff| no-capture vs capture = %.6e\n", max_diff);
    assert(max_diff < 1e-3f);

    /* Validate hidden states finite + non-trivial norms. */
    int bad = 0;
    for (int L = 0; L < n_layers; L++) {
        double sumsq = 0.0;
        for (size_t i = 0; i < embs_floats; i++) {
            float v = hs[L][i];
            if (!isfinite(v)) { bad++; continue; }
            sumsq += (double)v * v;
        }
        double norm = sqrt(sumsq / (double)embs_floats);
        if (L == 0 || L == n_layers - 1 || L == n_layers / 2) {
            printf("  L%-3d rms = %.6f  finite=%s\n", L, norm,
                   (bad == 0) ? "yes" : "NO");
        }
        assert(bad == 0);
        assert(norm > 1e-6);  /* non-trivial */
    }
    printf("all %d layers: hidden states finite + non-zero\n", n_layers);
    printf("PASS\n");

    for (int L = 0; L < n_layers; L++) free(hs[L]);
    free(hs); free(embs); free(logits_a); free(logits_b);
    ib_metal_release_model(ctx, bufs);
    ib_metal_destroy(ctx);
    inferbit_free(m);
    inferbit_config_free(cfg);
    return 0;
}
