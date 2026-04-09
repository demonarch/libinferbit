/*
 * test_forward.c — Test forward pass and generation
 *
 * Creates a tiny .ibf model with known weights and verifies
 * the forward pass runs without crashing, produces finite logits,
 * and generation produces valid token IDs.
 */

#include "inferbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-40s ", #name); \
    name(); \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

/* ── Create a tiny .ibf with actual weight data ─────────────── */

/*
 * Tiny model: 2 layers, hidden=64, heads=2, kv_heads=2, head_dim=32,
 * inter=128, vocab=256, max_ctx=64
 *
 * All weights INT8 for simplicity, scales = 1.0 (FP16 0x3C00)
 * Norms FP16 = 1.0 for all elements
 */

static const char* TINY_IBF = "/tmp/test_inferbit_forward.ibf";

/* Per-layer weight sizes (INT8 = 1 byte per element):
 * q_proj:  64*64  = 4096
 * k_proj:  64*64  = 4096
 * v_proj:  64*64  = 4096
 * o_proj:  64*64  = 4096
 * gate:    128*64 = 8192
 * up:      128*64 = 8192
 * down:    64*128 = 8192
 * input_norm:  64 * 2 = 128 (FP16)
 * post_norm:   64 * 2 = 128 (FP16)
 * scales for each proj (64 or 128 scales * 2 bytes FP16)
 *
 * Total per layer ~45k. We'll pack them sequentially with scale factors after each.
 */

/* Simplification: use one big flat weight region. All INT8 weights = small random,
 * all scales = FP16 1.0, all norms = FP16 1.0. */

#define HIDDEN 64
#define HEADS 2
#define KV_HEADS 2
#define HEAD_DIM 32
#define INTER 128
#define VOCAB 256
#define MAX_CTX 64
#define NUM_LAYERS 2

/* Offsets within weight data (we'll compute them manually) */
/* Embedding: VOCAB * HIDDEN = 16384 INT8 bytes */
/* Emb scales: VOCAB * 2 = 512 bytes FP16 */

static size_t write_int8_block(FILE* f, int rows, int cols, size_t* offset) {
    size_t start = *offset;
    size_t size = (size_t)rows * cols;
    /* Write small INT8 values */
    for (size_t i = 0; i < size; i++) {
        int8_t val = (int8_t)((i % 5) - 2);  /* -2, -1, 0, 1, 2 repeating */
        fwrite(&val, 1, 1, f);
    }
    *offset += size;
    return start;
}

static size_t write_scales(FILE* f, int count, size_t* offset) {
    size_t start = *offset;
    uint16_t one = 0x3C00;  /* FP16 1.0 */
    for (int i = 0; i < count; i++) {
        fwrite(&one, 2, 1, f);
    }
    *offset += (size_t)count * 2;
    return start;
}

static size_t write_norm(FILE* f, int size, size_t* offset) {
    size_t start = *offset;
    uint16_t one = 0x3C00;  /* FP16 1.0 */
    for (int i = 0; i < size; i++) {
        fwrite(&one, 2, 1, f);
    }
    *offset += (size_t)size * 2;
    return start;
}

/* Pad to 64-byte alignment */
static void pad_to_64(FILE* f, size_t* offset) {
    size_t aligned = ((*offset + 63) / 64) * 64;
    size_t pad = aligned - *offset;
    uint8_t zero = 0;
    for (size_t i = 0; i < pad; i++) fwrite(&zero, 1, 1, f);
    *offset = aligned;
}

typedef struct {
    size_t offset;
    size_t size;
    size_t scale_offset;
    size_t scale_size;
    int rows;
    int cols;
    int bits;
} tensor_loc;

static void write_tiny_ibf(void) {
    /* First pass: compute all offsets by writing weight data to a temp buffer */
    FILE* wf = tmpfile();
    assert(wf);
    size_t woff = 0;

    /* Embedding */
    tensor_loc emb;
    emb.offset = write_int8_block(wf, VOCAB, HIDDEN, &woff);
    emb.size = (size_t)VOCAB * HIDDEN;
    emb.scale_offset = write_scales(wf, VOCAB, &woff);
    emb.scale_size = (size_t)VOCAB * 2;
    emb.rows = VOCAB; emb.cols = HIDDEN; emb.bits = 8;

    /* Layers */
    tensor_loc layers[NUM_LAYERS][9];  /* q,k,v,o,gate,up,down,in_norm,post_norm */
    for (int l = 0; l < NUM_LAYERS; l++) {
        /* q_proj */
        layers[l][0].offset = write_int8_block(wf, HIDDEN, HIDDEN, &woff);
        layers[l][0].size = (size_t)HIDDEN * HIDDEN;
        layers[l][0].scale_offset = write_scales(wf, HIDDEN, &woff);
        layers[l][0].scale_size = HIDDEN * 2;
        layers[l][0].rows = HIDDEN; layers[l][0].cols = HIDDEN; layers[l][0].bits = 8;

        /* k_proj */
        layers[l][1].offset = write_int8_block(wf, KV_HEADS * HEAD_DIM, HIDDEN, &woff);
        layers[l][1].size = (size_t)(KV_HEADS * HEAD_DIM) * HIDDEN;
        layers[l][1].scale_offset = write_scales(wf, KV_HEADS * HEAD_DIM, &woff);
        layers[l][1].scale_size = KV_HEADS * HEAD_DIM * 2;
        layers[l][1].rows = KV_HEADS * HEAD_DIM; layers[l][1].cols = HIDDEN; layers[l][1].bits = 8;

        /* v_proj */
        layers[l][2].offset = write_int8_block(wf, KV_HEADS * HEAD_DIM, HIDDEN, &woff);
        layers[l][2].size = (size_t)(KV_HEADS * HEAD_DIM) * HIDDEN;
        layers[l][2].scale_offset = write_scales(wf, KV_HEADS * HEAD_DIM, &woff);
        layers[l][2].scale_size = KV_HEADS * HEAD_DIM * 2;
        layers[l][2].rows = KV_HEADS * HEAD_DIM; layers[l][2].cols = HIDDEN; layers[l][2].bits = 8;

        /* o_proj */
        layers[l][3].offset = write_int8_block(wf, HIDDEN, HIDDEN, &woff);
        layers[l][3].size = (size_t)HIDDEN * HIDDEN;
        layers[l][3].scale_offset = write_scales(wf, HIDDEN, &woff);
        layers[l][3].scale_size = HIDDEN * 2;
        layers[l][3].rows = HIDDEN; layers[l][3].cols = HIDDEN; layers[l][3].bits = 8;

        /* gate_proj */
        layers[l][4].offset = write_int8_block(wf, INTER, HIDDEN, &woff);
        layers[l][4].size = (size_t)INTER * HIDDEN;
        layers[l][4].scale_offset = write_scales(wf, INTER, &woff);
        layers[l][4].scale_size = INTER * 2;
        layers[l][4].rows = INTER; layers[l][4].cols = HIDDEN; layers[l][4].bits = 8;

        /* up_proj */
        layers[l][5].offset = write_int8_block(wf, INTER, HIDDEN, &woff);
        layers[l][5].size = (size_t)INTER * HIDDEN;
        layers[l][5].scale_offset = write_scales(wf, INTER, &woff);
        layers[l][5].scale_size = INTER * 2;
        layers[l][5].rows = INTER; layers[l][5].cols = HIDDEN; layers[l][5].bits = 8;

        /* down_proj */
        layers[l][6].offset = write_int8_block(wf, HIDDEN, INTER, &woff);
        layers[l][6].size = (size_t)HIDDEN * INTER;
        layers[l][6].scale_offset = write_scales(wf, HIDDEN, &woff);
        layers[l][6].scale_size = HIDDEN * 2;
        layers[l][6].rows = HIDDEN; layers[l][6].cols = INTER; layers[l][6].bits = 8;

        /* input_norm (FP16, no scales) */
        layers[l][7].offset = write_norm(wf, HIDDEN, &woff);
        layers[l][7].size = HIDDEN * 2;
        layers[l][7].scale_offset = 0; layers[l][7].scale_size = 0;
        layers[l][7].rows = HIDDEN; layers[l][7].cols = 0; layers[l][7].bits = 16;

        /* post_attn_norm */
        layers[l][8].offset = write_norm(wf, HIDDEN, &woff);
        layers[l][8].size = HIDDEN * 2;
        layers[l][8].scale_offset = 0; layers[l][8].scale_size = 0;
        layers[l][8].rows = HIDDEN; layers[l][8].cols = 0; layers[l][8].bits = 16;
    }

    /* Output norm */
    tensor_loc out_norm;
    out_norm.offset = write_norm(wf, HIDDEN, &woff);
    out_norm.size = HIDDEN * 2;
    out_norm.scale_offset = 0; out_norm.scale_size = 0;
    out_norm.bits = 16;

    /* Output head */
    tensor_loc out_head;
    out_head.offset = write_int8_block(wf, VOCAB, HIDDEN, &woff);
    out_head.size = (size_t)VOCAB * HIDDEN;
    out_head.scale_offset = write_scales(wf, VOCAB, &woff);
    out_head.scale_size = VOCAB * 2;
    out_head.bits = 8;

    size_t total_weight_size = woff;

    /* Read weight data from tmpfile */
    fseek(wf, 0, SEEK_SET);
    uint8_t* weight_blob = malloc(total_weight_size);
    assert(weight_blob);
    fread(weight_blob, 1, total_weight_size, wf);
    fclose(wf);

    /* Build JSON header */
    char json[16384];
    int n = 0;
    n += snprintf(json + n, sizeof(json) - n, "{");
    n += snprintf(json + n, sizeof(json) - n, "\"version\":1,");
    n += snprintf(json + n, sizeof(json) - n, "\"model\":{\"architecture\":\"llama\",\"name\":\"tiny-test\"},");
    n += snprintf(json + n, sizeof(json) - n, "\"architecture\":{");
    n += snprintf(json + n, sizeof(json) - n, "\"num_layers\":%d,", NUM_LAYERS);
    n += snprintf(json + n, sizeof(json) - n, "\"hidden_size\":%d,", HIDDEN);
    n += snprintf(json + n, sizeof(json) - n, "\"num_heads\":%d,", HEADS);
    n += snprintf(json + n, sizeof(json) - n, "\"num_kv_heads\":%d,", KV_HEADS);
    n += snprintf(json + n, sizeof(json) - n, "\"head_dim\":%d,", HEAD_DIM);
    n += snprintf(json + n, sizeof(json) - n, "\"intermediate_size\":%d,", INTER);
    n += snprintf(json + n, sizeof(json) - n, "\"vocab_size\":%d,", VOCAB);
    n += snprintf(json + n, sizeof(json) - n, "\"max_context_length\":%d,", MAX_CTX);
    n += snprintf(json + n, sizeof(json) - n, "\"rope_theta\":10000.0,");
    n += snprintf(json + n, sizeof(json) - n, "\"norm_epsilon\":1e-5,");
    n += snprintf(json + n, sizeof(json) - n, "\"norm_type\":\"rmsnorm\",");
    n += snprintf(json + n, sizeof(json) - n, "\"activation\":\"silu\",");
    n += snprintf(json + n, sizeof(json) - n, "\"tie_word_embeddings\":false");
    n += snprintf(json + n, sizeof(json) - n, "},");
    n += snprintf(json + n, sizeof(json) - n, "\"quantization\":{\"default_bits\":8,\"sensitive_bits\":8,\"sparsity\":0.0,\"block_size\":8},");
    n += snprintf(json + n, sizeof(json) - n, "\"kv_cache\":{\"bits\":16},");

    /* Compute weight data offset */
    size_t preamble_and_json = 32 + (size_t)n + 200;  /* Rough estimate, will finalize */
    /* We'll use a fixed offset that's definitely aligned */
    size_t weight_data_offset = 8192;  /* Plenty of room for header */

    n += snprintf(json + n, sizeof(json) - n, "\"data\":{\"weight_data_offset\":%zu,\"weight_data_size\":%zu,\"alignment\":64},",
                  weight_data_offset, total_weight_size);

    /* Layers */
    n += snprintf(json + n, sizeof(json) - n, "\"layers\":[");
    for (int l = 0; l < NUM_LAYERS; l++) {
        if (l > 0) n += snprintf(json + n, sizeof(json) - n, ",");
        n += snprintf(json + n, sizeof(json) - n, "{\"index\":%d,\"weights\":{", l);

        const char* names[] = {"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","input_norm","post_attn_norm"};
        for (int t = 0; t < 9; t++) {
            if (t > 0) n += snprintf(json + n, sizeof(json) - n, ",");
            tensor_loc* tl = &layers[l][t];
            if (tl->cols > 0) {
                n += snprintf(json + n, sizeof(json) - n,
                    "\"%s\":{\"offset\":%zu,\"size\":%zu,\"shape\":[%d,%d],\"bits\":%d,\"scale_offset\":%zu,\"scale_size\":%zu,\"has_bias\":false}",
                    names[t], tl->offset, tl->size, tl->rows, tl->cols, tl->bits, tl->scale_offset, tl->scale_size);
            } else {
                n += snprintf(json + n, sizeof(json) - n,
                    "\"%s\":{\"offset\":%zu,\"size\":%zu,\"shape\":[%d],\"bits\":%d,\"scale_offset\":0,\"scale_size\":0,\"has_bias\":false}",
                    names[t], tl->offset, tl->size, tl->rows, tl->bits);
            }
        }
        n += snprintf(json + n, sizeof(json) - n, "},\"sparsity_mask\":{\"offset\":0,\"size\":0}}");
    }
    n += snprintf(json + n, sizeof(json) - n, "],");

    /* Embeddings */
    n += snprintf(json + n, sizeof(json) - n,
        "\"embeddings\":{\"token_embedding\":{\"offset\":%zu,\"size\":%zu,\"shape\":[%d,%d],\"bits\":8,\"scale_offset\":%zu,\"scale_size\":%zu,\"has_bias\":false}},",
        emb.offset, emb.size, VOCAB, HIDDEN, emb.scale_offset, emb.scale_size);

    /* Output */
    n += snprintf(json + n, sizeof(json) - n,
        "\"output\":{\"norm\":{\"offset\":%zu,\"size\":%zu,\"shape\":[%d],\"bits\":16,\"scale_offset\":0,\"scale_size\":0,\"has_bias\":false},",
        out_norm.offset, out_norm.size, HIDDEN);
    n += snprintf(json + n, sizeof(json) - n,
        "\"head\":{\"offset\":%zu,\"size\":%zu,\"shape\":[%d,%d],\"bits\":8,\"scale_offset\":%zu,\"scale_size\":%zu,\"has_bias\":false}}",
        out_head.offset, out_head.size, VOCAB, HIDDEN, out_head.scale_offset, out_head.scale_size);

    n += snprintf(json + n, sizeof(json) - n, "}");

    /* Write final file */
    FILE* f = fopen(TINY_IBF, "wb");
    assert(f);

    /* Preamble */
    fwrite("INFERBIT", 1, 8, f);
    uint32_t ver = 1;
    fwrite(&ver, 4, 1, f);
    uint32_t hsize = (uint32_t)n;
    fwrite(&hsize, 4, 1, f);
    uint32_t flags = 0;
    fwrite(&flags, 4, 1, f);
    uint8_t reserved[12] = {0};
    fwrite(reserved, 1, 12, f);

    /* JSON header */
    fwrite(json, 1, n, f);

    /* Pad to weight_data_offset */
    size_t current = 32 + n;
    assert(current <= weight_data_offset);
    size_t pad = weight_data_offset - current;
    uint8_t zero = 0;
    for (size_t i = 0; i < pad; i++) fwrite(&zero, 1, 1, f);

    /* Weight data */
    fwrite(weight_blob, 1, total_weight_size, f);

    fclose(f);
    free(weight_blob);
}

/* ── Tests ──────────────────────────────────────────────────── */

void test_forward_basic(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    if (!model) {
        printf("FAIL: load error: %s\n", inferbit_last_error());
        assert(0);
    }

    /* Run forward pass with a single token */
    float logits[VOCAB];
    int32_t tokens[] = {1};
    int rc = inferbit_forward(model, tokens, 1, logits, VOCAB);
    assert(rc == INFERBIT_OK);

    /* Logits should be finite */
    for (int i = 0; i < VOCAB; i++) {
        assert(isfinite(logits[i]));
    }

    /* KV cache should have 1 token */
    assert(inferbit_kv_length(model) == 1);

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_forward_multi_token(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    assert(model != NULL);

    float logits[VOCAB];
    int32_t tokens[] = {1, 5, 10};
    int rc = inferbit_forward(model, tokens, 3, logits, VOCAB);
    assert(rc == INFERBIT_OK);
    assert(inferbit_kv_length(model) == 3);

    for (int i = 0; i < VOCAB; i++) {
        assert(isfinite(logits[i]));
    }

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_forward_context_limit(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    assert(model != NULL);

    /* Fill context to near max */
    float logits[VOCAB];
    for (int i = 0; i < MAX_CTX - 1; i++) {
        int32_t tok = (int32_t)(i % VOCAB);
        int rc = inferbit_forward(model, &tok, 1, logits, VOCAB);
        assert(rc == INFERBIT_OK);
    }
    assert(inferbit_kv_length(model) == MAX_CTX - 1);

    /* One more should work */
    int32_t tok = 42;
    int rc = inferbit_forward(model, &tok, 1, logits, VOCAB);
    assert(rc == INFERBIT_OK);
    assert(inferbit_kv_length(model) == MAX_CTX);

    /* Next should fail — context full */
    rc = inferbit_forward(model, &tok, 1, logits, VOCAB);
    assert(rc == INFERBIT_ERROR_CONTEXT);

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_generate_basic(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    assert(model != NULL);

    int32_t input[] = {1, 2, 3};
    int32_t output[10];
    inferbit_sample_params p = inferbit_default_sample_params();
    p.max_tokens = 10;
    p.temperature = 0.0f;  /* Greedy */
    p.seed = 42;

    int n = inferbit_generate(model, input, 3, output, 10, p);
    assert(n > 0);

    /* All output tokens should be valid */
    for (int i = 0; i < n; i++) {
        assert(output[i] >= 0 && output[i] < VOCAB);
    }

    inferbit_free(model);
    inferbit_config_free(cfg);
}

static int stream_count = 0;
static int32_t stream_tokens[64];

static int stream_cb(int32_t token, void* ctx) {
    (void)ctx;
    if (stream_count < 64) {
        stream_tokens[stream_count++] = token;
    }
    if (stream_count >= 5) return 0;  /* Stop after 5 */
    return 1;
}

void test_generate_stream(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    assert(model != NULL);

    stream_count = 0;
    int32_t input[] = {1, 2};
    inferbit_sample_params p = inferbit_default_sample_params();
    p.max_tokens = 20;
    p.temperature = 0.5f;
    p.seed = 123;

    int n = inferbit_generate_stream(model, input, 2, stream_cb, NULL, p);
    assert(n > 0);
    assert(stream_count <= 5);

    for (int i = 0; i < stream_count; i++) {
        assert(stream_tokens[i] >= 0 && stream_tokens[i] < VOCAB);
    }

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_kv_clear_and_reuse(void) {
    write_tiny_ibf();
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(TINY_IBF, cfg);
    assert(model != NULL);

    float logits[VOCAB];
    int32_t tokens[] = {1, 2, 3};
    int rc = inferbit_forward(model, tokens, 3, logits, VOCAB);
    assert(rc == INFERBIT_OK);
    assert(inferbit_kv_length(model) == 3);

    /* Clear and run again */
    inferbit_kv_clear(model);
    assert(inferbit_kv_length(model) == 0);

    rc = inferbit_forward(model, tokens, 3, logits, VOCAB);
    assert(rc == INFERBIT_OK);
    assert(inferbit_kv_length(model) == 3);

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_forward_deterministic(void) {
    write_tiny_ibf();

    /* Run the same forward pass twice and verify logits match */
    float logits1[VOCAB], logits2[VOCAB];
    int32_t tokens[] = {5, 10};

    inferbit_config* cfg = inferbit_config_create();

    inferbit_model* m1 = inferbit_load(TINY_IBF, cfg);
    assert(m1 != NULL);
    int rc = inferbit_forward(m1, tokens, 2, logits1, VOCAB);
    assert(rc == INFERBIT_OK);

    inferbit_model* m2 = inferbit_load(TINY_IBF, cfg);
    assert(m2 != NULL);
    rc = inferbit_forward(m2, tokens, 2, logits2, VOCAB);
    assert(rc == INFERBIT_OK);

    /* Logits should be very close (threading may cause minor FP rounding differences) */
    for (int i = 0; i < VOCAB; i++) {
        float diff = fabsf(logits1[i] - logits2[i]);
        float mag = fabsf(logits1[i]) + fabsf(logits2[i]) + 1e-10f;
        assert(diff / mag < 1e-4f);
    }

    inferbit_free(m1);
    inferbit_free(m2);
    inferbit_config_free(cfg);
    unlink(TINY_IBF);
}

/* ── Main ───────────────────────────────────────────────────── */

int main(void) {
    printf("libinferbit forward pass tests\n");
    printf("─────────────────────────────────────────────\n");

    TEST(test_forward_basic);
    TEST(test_forward_multi_token);
    TEST(test_forward_context_limit);
    TEST(test_generate_basic);
    TEST(test_generate_stream);
    TEST(test_kv_clear_and_reuse);
    TEST(test_forward_deterministic);

    printf("─────────────────────────────────────────────\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return tests_passed == tests_run ? 0 : 1;
}
