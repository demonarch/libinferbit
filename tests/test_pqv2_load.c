/* test_pqv2_load — smoke-test inferbit_load on an IBF v6 PQv2 file.
 * Validates that the public API correctly dispatches to the PQv2 loader
 * and populates tensor slots with pq pointers.
 */
#include "../include/inferbit.h"
#include "../src/inferbit_internal.h"
#include "../src/pqv2_kernel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *path = (argc > 1) ? argv[1] : "/tmp/tinyllama_pqv2.ibf";
    printf("loading %s ...\n", path);

    inferbit_config cfg = {0};
    inferbit_model *m = inferbit_load(path, &cfg);
    if (!m) {
        fprintf(stderr, "inferbit_load failed: %s\n", inferbit_last_error());
        return 1;
    }

    printf("model loaded: %d layers, hidden=%d, vocab=%d\n",
           m->header.num_layers, m->header.hidden_size, m->header.vocab_size);

    /* Inspect a few tensors */
    int pq_count = 0, raw_count = 0;
    for (int li = 0; li < m->header.num_layers; li++) {
        ib_layer_meta *L = &m->layers[li];
        if (L->q_proj.pq) pq_count++;
        if (L->k_proj.pq) pq_count++;
        if (L->v_proj.pq) pq_count++;
        if (L->o_proj.pq) pq_count++;
        if (L->gate_proj.pq) pq_count++;
        if (L->up_proj.pq) pq_count++;
        if (L->down_proj.pq) pq_count++;
        if (L->input_norm.bits == 16 && L->input_norm.size > 0) raw_count++;
        if (L->post_attn_norm.bits == 16 && L->post_attn_norm.size > 0) raw_count++;
    }
    if (m->token_embedding.size > 0) raw_count++;
    if (m->output_norm.size > 0) raw_count++;
    if (m->output_head.size > 0) raw_count++;

    printf("PQv2 weight tensors:    %d (expected 154)\n", pq_count);
    printf("Raw fp16 tensors:       %d (expected 47)\n", raw_count);

    /* Sanity: spot-check L0 q_proj */
    const pqv2_t *qp = m->layers[0].q_proj.pq;
    if (qp) {
        printf("L0.q_proj: M=%u N=%u K=%u half=%u (expect 2048x2048 K=256 half=2)\n",
               qp->M, qp->N, qp->K, qp->half);
    } else {
        fprintf(stderr, "FAIL: L0.q_proj.pq is NULL\n");
        inferbit_free(m);
        return 1;
    }

    if (pq_count != 154) {
        fprintf(stderr, "FAIL: expected 154 PQv2 tensors, got %d\n", pq_count);
        inferbit_free(m);
        return 1;
    }

    /* Runtime-state sanity */
    if (!m->thread_pool && m->num_threads > 1) {
        fprintf(stderr, "WARN: thread_pool is NULL with num_threads=%d\n",
                m->num_threads);
    }
    if (!m->kv_caches) {
        fprintf(stderr, "FAIL: kv_caches not allocated\n");
        inferbit_free(m);
        return 1;
    }
    if (!m->buf_residual || !m->buf_hidden || !m->buf_logits) {
        fprintf(stderr, "FAIL: activation buffers not allocated\n");
        inferbit_free(m);
        return 1;
    }
    printf("Runtime: kv_caches=allocated, buffers=allocated, threads=%d\n",
           m->num_threads);

    /* End-to-end forward pass: feed a few tokens, get logits, verify shape + sanity. */
    int32_t prompt[] = {1, 450, 4996, 17354, 1701};   /* "<s>The quick brown fox" approx */
    int n_tok = sizeof(prompt) / sizeof(prompt[0]);
    int V = m->header.vocab_size;
    float *logits = calloc((size_t)V, sizeof(float));
    printf("running forward on %d tokens ...\n", n_tok);
    int rc = inferbit_forward(m, prompt, n_tok, logits, V);
    if (rc != 0) {
        fprintf(stderr, "FAIL: inferbit_forward returned %d: %s\n",
                rc, inferbit_last_error());
        free(logits); inferbit_free(m); return 1;
    }
    /* Find argmax */
    int best = 0;
    float best_v = logits[0];
    int n_finite = 0;
    for (int i = 0; i < V; i++) {
        if (logits[i] == logits[i]) n_finite++;  /* NaN check */
        if (logits[i] > best_v) { best_v = logits[i]; best = i; }
    }
    printf("forward returned: argmax token=%d, logit=%.3f, finite=%d/%d\n",
           best, best_v, n_finite, V);
    if (n_finite != V) {
        fprintf(stderr, "FAIL: %d NaN logits\n", V - n_finite);
        free(logits); inferbit_free(m); return 1;
    }

    free(logits);
    inferbit_free(m);
    printf("PASS — full PQv2 inference path operational\n");
    return 0;
}
