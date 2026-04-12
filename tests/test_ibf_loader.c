/*
 * test_ibf_loader.c — Test IBF file loading
 *
 * Creates a minimal .ibf file in /tmp and verifies loading works.
 */

#include "inferbit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void get_temp_file(char* buf, size_t buflen, const char* name) {
    const char* tmp = getenv("TEMP");
    if (!tmp) tmp = getenv("TMP");
    if (!tmp) tmp = "/tmp";
    snprintf(buf, buflen, "%s/%s", tmp, name);
}
#ifdef _WIN32
#include <io.h>
#define unlink _unlink
#else
#include <unistd.h>
#endif

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-40s ", #name); \
    name(); \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

/* ── Helper: write a minimal .ibf file ──────────────────────── */

static char MINI_IBF_PATH[512];

static const char* MINI_HEADER_JSON =
    "{"
    "  \"version\": 1,"
    "  \"model\": {"
    "    \"architecture\": \"llama\","
    "    \"name\": \"test-mini\""
    "  },"
    "  \"architecture\": {"
    "    \"num_layers\": 2,"
    "    \"hidden_size\": 64,"
    "    \"num_heads\": 2,"
    "    \"num_kv_heads\": 2,"
    "    \"head_dim\": 32,"
    "    \"intermediate_size\": 128,"
    "    \"vocab_size\": 256,"
    "    \"max_context_length\": 64,"
    "    \"rope_theta\": 10000.0,"
    "    \"norm_epsilon\": 1e-5,"
    "    \"norm_type\": \"rmsnorm\","
    "    \"activation\": \"silu\","
    "    \"tie_word_embeddings\": false,"
    "    \"attention_bias\": false,"
    "    \"mlp_bias\": false"
    "  },"
    "  \"quantization\": {"
    "    \"default_bits\": 4,"
    "    \"sensitive_bits\": 8,"
    "    \"sparsity\": 0.0,"
    "    \"block_size\": 8"
    "  },"
    "  \"kv_cache\": {"
    "    \"bits\": 8"
    "  },"
    "  \"data\": {"
    "    \"weight_data_offset\": 0,"
    "    \"weight_data_size\": 4096,"
    "    \"alignment\": 64"
    "  },"
    "  \"layers\": ["
    "    {"
    "      \"index\": 0,"
    "      \"weights\": {"
    "        \"q_proj\":    { \"offset\": 0, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 128, \"scale_size\": 128, \"has_bias\": false },"
    "        \"k_proj\":    { \"offset\": 256, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 384, \"scale_size\": 128, \"has_bias\": false },"
    "        \"v_proj\":    { \"offset\": 512, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 640, \"scale_size\": 128, \"has_bias\": false },"
    "        \"o_proj\":    { \"offset\": 768, \"size\": 64, \"shape\": [64, 64], \"bits\": 4, \"scale_offset\": 832, \"scale_size\": 128, \"has_bias\": false },"
    "        \"gate_proj\": { \"offset\": 960, \"size\": 128, \"shape\": [64, 128], \"bits\": 4, \"scale_offset\": 1088, \"scale_size\": 128, \"has_bias\": false },"
    "        \"up_proj\":   { \"offset\": 1216, \"size\": 128, \"shape\": [64, 128], \"bits\": 4, \"scale_offset\": 1344, \"scale_size\": 128, \"has_bias\": false },"
    "        \"down_proj\": { \"offset\": 1472, \"size\": 128, \"shape\": [128, 64], \"bits\": 4, \"scale_offset\": 1600, \"scale_size\": 256, \"has_bias\": false },"
    "        \"input_norm\":     { \"offset\": 1856, \"size\": 128, \"shape\": [64], \"bits\": 16, \"scale_offset\": 0, \"scale_size\": 0, \"has_bias\": false },"
    "        \"post_attn_norm\": { \"offset\": 1984, \"size\": 128, \"shape\": [64], \"bits\": 16, \"scale_offset\": 0, \"scale_size\": 0, \"has_bias\": false }"
    "      },"
    "      \"sparsity_mask\": { \"offset\": 0, \"size\": 0 }"
    "    },"
    "    {"
    "      \"index\": 1,"
    "      \"weights\": {"
    "        \"q_proj\":    { \"offset\": 2112, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 2240, \"scale_size\": 128, \"has_bias\": false },"
    "        \"k_proj\":    { \"offset\": 2368, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 2496, \"scale_size\": 128, \"has_bias\": false },"
    "        \"v_proj\":    { \"offset\": 2624, \"size\": 128, \"shape\": [64, 64], \"bits\": 8, \"scale_offset\": 2752, \"scale_size\": 128, \"has_bias\": false },"
    "        \"o_proj\":    { \"offset\": 2880, \"size\": 64, \"shape\": [64, 64], \"bits\": 4, \"scale_offset\": 2944, \"scale_size\": 128, \"has_bias\": false },"
    "        \"gate_proj\": { \"offset\": 3072, \"size\": 128, \"shape\": [64, 128], \"bits\": 4, \"scale_offset\": 3200, \"scale_size\": 128, \"has_bias\": false },"
    "        \"up_proj\":   { \"offset\": 3328, \"size\": 128, \"shape\": [64, 128], \"bits\": 4, \"scale_offset\": 3456, \"scale_size\": 128, \"has_bias\": false },"
    "        \"down_proj\": { \"offset\": 3584, \"size\": 128, \"shape\": [128, 64], \"bits\": 4, \"scale_offset\": 3712, \"scale_size\": 256, \"has_bias\": false },"
    "        \"input_norm\":     { \"offset\": 3968, \"size\": 128, \"shape\": [64], \"bits\": 16, \"scale_offset\": 0, \"scale_size\": 0, \"has_bias\": false },"
    "        \"post_attn_norm\": { \"offset\": 4096, \"size\": 128, \"shape\": [64], \"bits\": 16, \"scale_offset\": 0, \"scale_size\": 0, \"has_bias\": false }"
    "      },"
    "      \"sparsity_mask\": { \"offset\": 0, \"size\": 0 }"
    "    }"
    "  ],"
    "  \"embeddings\": {"
    "    \"token_embedding\": { \"offset\": 0, \"size\": 512, \"shape\": [256, 64], \"bits\": 8, \"scale_offset\": 512, \"scale_size\": 512, \"has_bias\": false }"
    "  },"
    "  \"output\": {"
    "    \"norm\": { \"offset\": 0, \"size\": 128, \"shape\": [64], \"bits\": 16, \"scale_offset\": 0, \"scale_size\": 0, \"has_bias\": false },"
    "    \"head\": { \"offset\": 128, \"size\": 512, \"shape\": [256, 64], \"bits\": 8, \"scale_offset\": 640, \"scale_size\": 512, \"has_bias\": false }"
    "  }"
    "}";

static void write_mini_ibf(void) {
    FILE* f = fopen(MINI_IBF_PATH, "wb");
    assert(f != NULL);

    size_t json_len = strlen(MINI_HEADER_JSON);

    /* Preamble: 32 bytes */
    /* Magic (8) */
    fwrite("INFERBIT", 1, 8, f);
    /* Version (4, uint32 LE) */
    uint32_t version = 1;
    fwrite(&version, 4, 1, f);
    /* Header size (4, uint32 LE) */
    uint32_t hsize = (uint32_t)json_len;
    fwrite(&hsize, 4, 1, f);
    /* Flags (4) */
    uint32_t flags = 0;
    fwrite(&flags, 4, 1, f);
    /* Reserved (12) */
    uint8_t reserved[12] = {0};
    fwrite(reserved, 1, 12, f);

    /* JSON header */
    fwrite(MINI_HEADER_JSON, 1, json_len, f);

    /* Padding to 64-byte alignment */
    size_t pos = 32 + json_len;
    size_t aligned = ((pos + 63) / 64) * 64;
    size_t pad = aligned - pos;
    if (pad > 0) {
        uint8_t zeros[64] = {0};
        fwrite(zeros, 1, pad, f);
    }

    /* Weight data: 4096 bytes of dummy data */
    uint8_t* weight_data = calloc(4096, 1);
    assert(weight_data != NULL);
    /* Fill with some non-zero pattern for testing */
    for (int i = 0; i < 4096; i++) {
        weight_data[i] = (uint8_t)(i & 0xFF);
    }
    fwrite(weight_data, 1, 4096, f);
    free(weight_data);

    fclose(f);
}

/* ── Tests ──────────────────────────────────────────────────── */

void test_load_ibf_basic(void) {
    write_mini_ibf();

    inferbit_config* config = inferbit_config_create();
    inferbit_model* model = inferbit_load(MINI_IBF_PATH, config);

    if (!model) {
        printf("FAIL: %s\n", inferbit_last_error());
        assert(0);
    }

    assert(strcmp(inferbit_model_architecture(model), "llama") == 0);
    assert(inferbit_model_num_layers(model) == 2);
    assert(inferbit_model_hidden_size(model) == 64);
    assert(inferbit_model_vocab_size(model) == 256);
    assert(inferbit_model_max_context(model) == 64);
    assert(inferbit_model_default_bits(model) == 4);

    inferbit_free(model);
    inferbit_config_free(config);
    unlink(MINI_IBF_PATH);
}

void test_load_ibf_kv_cache(void) {
    write_mini_ibf();

    inferbit_config* config = inferbit_config_create();
    inferbit_model* model = inferbit_load(MINI_IBF_PATH, config);
    assert(model != NULL);

    /* KV cache should start empty */
    assert(inferbit_kv_length(model) == 0);

    /* KV memory should be non-zero (pre-allocated) */
    assert(inferbit_model_kv_memory(model) > 0);

    inferbit_free(model);
    inferbit_config_free(config);
    unlink(MINI_IBF_PATH);
}

void test_load_ibf_dynamic_kv(void) {
    write_mini_ibf();

    inferbit_config* config = inferbit_config_create();
    inferbit_config_set_kv_cache_dynamic(config, 1);
    inferbit_model* model = inferbit_load(MINI_IBF_PATH, config);
    assert(model != NULL);

    assert(inferbit_kv_length(model) == 0);

    inferbit_free(model);
    inferbit_config_free(config);
    unlink(MINI_IBF_PATH);
}

void test_load_ibf_bad_magic(void) {
    FILE* f = fopen(MINI_IBF_PATH, "wb");
    fwrite("NOTVALID", 1, 8, f);
    uint8_t rest[56] = {0};
    fwrite(rest, 1, 56, f);
    fclose(f);

    inferbit_model* model = inferbit_load(MINI_IBF_PATH, NULL);
    assert(model == NULL);
    assert(inferbit_last_error() != NULL);

    unlink(MINI_IBF_PATH);
}

void test_load_ibf_too_small(void) {
    FILE* f = fopen(MINI_IBF_PATH, "wb");
    fwrite("INF", 1, 3, f);
    fclose(f);

    inferbit_model* model = inferbit_load(MINI_IBF_PATH, NULL);
    assert(model == NULL);

    unlink(MINI_IBF_PATH);
}

void test_load_wrong_extension(void) {
    char tmp_st[512];
    get_temp_file(tmp_st, sizeof(tmp_st), "model.safetensors");
    inferbit_model* model = inferbit_load(tmp_st, NULL);
    assert(model == NULL);
    const char* err = inferbit_last_error();
    assert(err != NULL);
    assert(strstr(err, ".ibf") != NULL);
}

void test_load_ibf_weight_data(void) {
    write_mini_ibf();

    inferbit_config* config = inferbit_config_create();
    inferbit_model* model = inferbit_load(MINI_IBF_PATH, config);
    assert(model != NULL);

    /* Weight data should be mapped */
    assert(inferbit_model_weight_memory(model) > 0);

    inferbit_free(model);
    inferbit_config_free(config);
    unlink(MINI_IBF_PATH);
}

void test_load_ibf_threads(void) {
    write_mini_ibf();

    inferbit_config* config = inferbit_config_create();
    inferbit_config_set_threads(config, 16);
    inferbit_model* model = inferbit_load(MINI_IBF_PATH, config);
    assert(model != NULL);

    /* Can't directly query thread count from public API, but it shouldn't crash */
    inferbit_free(model);
    inferbit_config_free(config);
    unlink(MINI_IBF_PATH);
}

/* ── Main ───────────────────────────────────────────────────── */

int main(void) {
    get_temp_file(MINI_IBF_PATH, sizeof(MINI_IBF_PATH), "test_inferbit_mini.ibf");

    printf("libinferbit IBF loader tests\n");
    printf("─────────────────────────────────────────────\n");

    TEST(test_load_ibf_basic);
    TEST(test_load_ibf_kv_cache);
    TEST(test_load_ibf_dynamic_kv);
    TEST(test_load_ibf_bad_magic);
    TEST(test_load_ibf_too_small);
    TEST(test_load_wrong_extension);
    TEST(test_load_ibf_weight_data);
    TEST(test_load_ibf_threads);

    printf("─────────────────────────────────────────────\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return tests_passed == tests_run ? 0 : 1;
}
