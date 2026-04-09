/*
 * test_convert.c — Test safetensors → .ibf conversion
 *
 * Creates a fake safetensors file with LLaMA-style tensor names,
 * converts it to .ibf, then loads it and verifies correctness.
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

/* ── Write a fake safetensors file ──────────────────────────── */

/*
 * Tiny LLaMA-style model:
 *   hidden=64, heads=2, kv_heads=2, head_dim=32, inter=128, vocab=256, 2 layers
 *   All weights FP16
 */

#define HIDDEN 64
#define HEADS 2
#define KV_HEADS 2
#define HEAD_DIM 32
#define INTER 128
#define VOCAB 256
#define NUM_LAYERS 2

static const char* FAKE_ST = "/tmp/test_inferbit_fake.safetensors";
static const char* OUTPUT_IBF = "/tmp/test_inferbit_converted.ibf";

/* FP16 representation of a small value */
static uint16_t f32_to_fp16(float f) {
    uint32_t b;
    memcpy(&b, &f, 4);
    uint16_t sign = (uint16_t)((b >> 16) & 0x8000);
    int32_t exp = (int32_t)((b >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = b & 0x7FFFFF;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7C00;
    return sign | (uint16_t)(exp << 10) | (uint16_t)(mant >> 13);
}

typedef struct {
    const char* name;
    int shape[2];
    int ndim;
    size_t data_start;
    size_t data_end;
} fake_tensor;

static void write_fake_safetensors(void) {
    /* Build tensor list */
    fake_tensor tensors[128];
    int nt = 0;

    /* Helper to add a tensor */
    #define ADD(n, r, c) do { \
        tensors[nt].name = n; \
        tensors[nt].shape[0] = r; tensors[nt].shape[1] = c; \
        tensors[nt].ndim = (c > 0) ? 2 : 1; \
        nt++; \
    } while(0)

    ADD("model.embed_tokens.weight", VOCAB, HIDDEN);

    for (int l = 0; l < NUM_LAYERS; l++) {
        char buf[256];
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.q_proj.weight", l);
        ADD(strdup(buf), HIDDEN, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.k_proj.weight", l);
        ADD(strdup(buf), KV_HEADS * HEAD_DIM, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.v_proj.weight", l);
        ADD(strdup(buf), KV_HEADS * HEAD_DIM, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.o_proj.weight", l);
        ADD(strdup(buf), HIDDEN, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_proj.weight", l);
        ADD(strdup(buf), INTER, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.up_proj.weight", l);
        ADD(strdup(buf), INTER, HIDDEN);
        snprintf(buf, sizeof(buf), "model.layers.%d.mlp.down_proj.weight", l);
        ADD(strdup(buf), HIDDEN, INTER);
        snprintf(buf, sizeof(buf), "model.layers.%d.input_layernorm.weight", l);
        ADD(strdup(buf), HIDDEN, 0);
        snprintf(buf, sizeof(buf), "model.layers.%d.post_attention_layernorm.weight", l);
        ADD(strdup(buf), HIDDEN, 0);
    }

    ADD("model.norm.weight", HIDDEN, 0);
    ADD("lm_head.weight", VOCAB, HIDDEN);

    #undef ADD

    /* Compute data offsets and sizes */
    size_t data_offset = 0;
    for (int i = 0; i < nt; i++) {
        int elems = tensors[i].shape[0] * (tensors[i].ndim == 2 ? tensors[i].shape[1] : 1);
        size_t size = (size_t)elems * 2;  /* FP16 = 2 bytes */
        tensors[i].data_start = data_offset;
        tensors[i].data_end = data_offset + size;
        data_offset += size;
    }
    size_t total_data = data_offset;

    /* Build JSON header */
    char json[32768];
    int n = 0;
    n += snprintf(json + n, sizeof(json) - n, "{");
    for (int i = 0; i < nt; i++) {
        if (i > 0) n += snprintf(json + n, sizeof(json) - n, ",");
        n += snprintf(json + n, sizeof(json) - n, "\"%s\":{\"dtype\":\"F16\",\"shape\":[%d",
                      tensors[i].name, tensors[i].shape[0]);
        if (tensors[i].ndim == 2) {
            n += snprintf(json + n, sizeof(json) - n, ",%d", tensors[i].shape[1]);
        }
        n += snprintf(json + n, sizeof(json) - n, "],\"data_offsets\":[%zu,%zu]}",
                      tensors[i].data_start, tensors[i].data_end);
    }
    n += snprintf(json + n, sizeof(json) - n, "}");

    /* Write file */
    FILE* f = fopen(FAKE_ST, "wb");
    assert(f);

    /* Header size (uint64 LE) */
    uint64_t header_size = (uint64_t)n;
    fwrite(&header_size, 8, 1, f);

    /* JSON header */
    fwrite(json, 1, n, f);

    /* Tensor data: fill with small FP16 values */
    for (size_t i = 0; i < total_data / 2; i++) {
        float val = 0.01f * ((float)(i % 100) - 50.0f);
        uint16_t fp16 = f32_to_fp16(val);
        fwrite(&fp16, 2, 1, f);
    }

    fclose(f);
}

/* ── Track progress ─────────────────────────────────────────── */

static float last_progress = -1.0f;
static void test_progress(float pct, const char* stage, void* ctx) {
    (void)ctx;
    (void)stage;
    last_progress = pct;
}

/* ── Tests ──────────────────────────────────────────────────── */

void test_detect_format(void) {
    write_fake_safetensors();
    assert(inferbit_detect_format(FAKE_ST) == INFERBIT_FORMAT_SAFETENSORS);
    assert(inferbit_detect_format("/nonexistent") == INFERBIT_FORMAT_UNKNOWN);
}

void test_convert_basic(void) {
    write_fake_safetensors();

    inferbit_convert_config cfg = inferbit_default_convert_config();
    cfg.default_bits = 4;
    cfg.sensitive_bits = 8;
    cfg.progress = test_progress;
    last_progress = -1.0f;

    int rc = inferbit_convert(FAKE_ST, OUTPUT_IBF, &cfg);
    if (rc != INFERBIT_OK) {
        printf("FAIL: %s\n", inferbit_last_error());
    }
    assert(rc == INFERBIT_OK);
    assert(last_progress >= 0.99f);  /* Progress reached ~1.0 */
}

void test_convert_and_load(void) {
    write_fake_safetensors();

    inferbit_convert_config ccfg = inferbit_default_convert_config();
    ccfg.default_bits = 4;
    ccfg.sensitive_bits = 8;
    int rc = inferbit_convert(FAKE_ST, OUTPUT_IBF, &ccfg);
    assert(rc == INFERBIT_OK);

    /* Load the converted file */
    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(OUTPUT_IBF, cfg);
    if (!model) {
        printf("FAIL: load: %s\n", inferbit_last_error());
    }
    assert(model != NULL);

    /* Verify architecture */
    assert(strcmp(inferbit_model_architecture(model), "llama") == 0);
    assert(inferbit_model_num_layers(model) == NUM_LAYERS);
    assert(inferbit_model_hidden_size(model) == HIDDEN);
    assert(inferbit_model_vocab_size(model) == VOCAB);
    assert(inferbit_model_default_bits(model) == 4);

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_convert_and_forward(void) {
    write_fake_safetensors();

    inferbit_convert_config ccfg = inferbit_default_convert_config();
    ccfg.default_bits = 8;  /* All INT8 for simpler debugging */
    ccfg.sensitive_bits = 8;
    int rc = inferbit_convert(FAKE_ST, OUTPUT_IBF, &ccfg);
    assert(rc == INFERBIT_OK);

    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(OUTPUT_IBF, cfg);
    assert(model != NULL);

    /* Run forward pass */
    float logits[VOCAB];
    int32_t tokens[] = {1, 5};
    rc = inferbit_forward(model, tokens, 2, logits, VOCAB);
    assert(rc == INFERBIT_OK);

    /* Logits should be finite */
    for (int i = 0; i < VOCAB; i++) {
        assert(isfinite(logits[i]));
    }

    inferbit_free(model);
    inferbit_config_free(cfg);
}

void test_convert_and_generate(void) {
    write_fake_safetensors();

    inferbit_convert_config ccfg = inferbit_default_convert_config();
    int rc = inferbit_convert(FAKE_ST, OUTPUT_IBF, &ccfg);
    assert(rc == INFERBIT_OK);

    inferbit_config* cfg = inferbit_config_create();
    inferbit_model* model = inferbit_load(OUTPUT_IBF, cfg);
    assert(model != NULL);

    int32_t input[] = {1, 2, 3};
    int32_t output[5];
    inferbit_sample_params p = inferbit_default_sample_params();
    p.temperature = 0.0f;
    p.max_tokens = 5;

    int n = inferbit_generate(model, input, 3, output, 5, p);
    assert(n > 0);
    for (int i = 0; i < n; i++) {
        assert(output[i] >= 0 && output[i] < VOCAB);
    }

    inferbit_free(model);
    inferbit_config_free(cfg);
    unlink(FAKE_ST);
    unlink(OUTPUT_IBF);
}

void test_convert_null_args(void) {
    int rc = inferbit_convert(NULL, OUTPUT_IBF, NULL);
    assert(rc == INFERBIT_ERROR_PARAM);

    rc = inferbit_convert(FAKE_ST, NULL, NULL);
    assert(rc == INFERBIT_ERROR_PARAM);
}

/* ── Main ───────────────────────────────────────────────────── */

int main(void) {
    printf("libinferbit converter tests\n");
    printf("─────────────────────────────────────────────\n");

    TEST(test_detect_format);
    TEST(test_convert_basic);
    TEST(test_convert_and_load);
    TEST(test_convert_and_forward);
    TEST(test_convert_and_generate);
    TEST(test_convert_null_args);

    printf("─────────────────────────────────────────────\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return tests_passed == tests_run ? 0 : 1;
}
