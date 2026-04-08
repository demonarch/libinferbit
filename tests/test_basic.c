#include "inferbit.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-40s ", #name); \
    name(); \
    tests_passed++; \
    printf("PASS\n"); \
} while(0)

/* ── Tests ──────────────────────────────────────────────────── */

void test_version(void) {
    assert(inferbit_version_major() == 0);
    assert(inferbit_version_minor() == 1);
    assert(inferbit_version_patch() == 0);
    assert(strcmp(inferbit_version(), "0.1.0") == 0);
}

void test_config_create_free(void) {
    inferbit_config* config = inferbit_config_create();
    assert(config != NULL);
    inferbit_config_set_threads(config, 8);
    inferbit_config_set_context_length(config, 2048);
    inferbit_config_set_kv_cache_dynamic(config, 1);
    inferbit_config_free(config);
}

void test_config_null_safety(void) {
    /* These should not crash */
    inferbit_config_set_threads(NULL, 8);
    inferbit_config_set_context_length(NULL, 2048);
    inferbit_config_set_kv_cache_dynamic(NULL, 1);
    inferbit_config_set_native_parse(NULL, 1);
    inferbit_config_set_native_bits(NULL, 4);
    inferbit_config_free(NULL);
}

void test_load_null_path(void) {
    inferbit_model* model = inferbit_load(NULL, NULL);
    assert(model == NULL);
    const char* err = inferbit_last_error();
    assert(err != NULL);
    assert(strstr(err, "NULL") != NULL);
}

void test_load_nonexistent(void) {
    inferbit_config* config = inferbit_config_create();
    inferbit_model* model = inferbit_load("/nonexistent/model.ibf", config);
    assert(model == NULL);
    /* Error should be set (either "not implemented" or "file not found") */
    assert(inferbit_last_error() != NULL);
    inferbit_config_free(config);
}

void test_default_sample_params(void) {
    inferbit_sample_params p = inferbit_default_sample_params();
    assert(p.temperature > 0.0f);
    assert(p.top_k > 0);
    assert(p.top_p > 0.0f && p.top_p <= 1.0f);
    assert(p.max_tokens > 0);
}

void test_generate_null_safety(void) {
    int32_t in[] = {1, 2, 3};
    int32_t out[10];
    inferbit_sample_params p = inferbit_default_sample_params();

    int ret = inferbit_generate(NULL, in, 3, out, 10, p);
    assert(ret == INFERBIT_ERROR_PARAM);

    ret = inferbit_generate_stream(NULL, in, 3, NULL, NULL, p);
    assert(ret == INFERBIT_ERROR_PARAM);

    ret = inferbit_forward(NULL, in, 3, NULL, 0);
    assert(ret == INFERBIT_ERROR_PARAM);
}

void test_model_info_null(void) {
    assert(strcmp(inferbit_model_architecture(NULL), "unknown") == 0);
    assert(inferbit_model_num_layers(NULL) == 0);
    assert(inferbit_model_hidden_size(NULL) == 0);
    assert(inferbit_model_vocab_size(NULL) == 0);
    assert(inferbit_model_max_context(NULL) == 0);
    assert(inferbit_model_default_bits(NULL) == 0);
    assert(inferbit_model_weight_memory(NULL) == 0);
    assert(inferbit_model_kv_memory(NULL) == 0);
    assert(inferbit_model_total_memory(NULL) == 0);
}

void test_kv_cache_null(void) {
    /* These should not crash */
    inferbit_kv_clear(NULL);
    inferbit_kv_truncate(NULL, 0);
    assert(inferbit_kv_length(NULL) == 0);
}

void test_speculative_null(void) {
    /* These should not crash */
    inferbit_set_draft_model(NULL, NULL, 4);
    inferbit_unset_draft_model(NULL);
}

void test_free_null(void) {
    inferbit_free(NULL);  /* Should not crash */
}

/* ── Main ───────────────────────────────────────────────────── */

int main(void) {
    printf("libinferbit tests\n");
    printf("─────────────────────────────────────────────\n");

    TEST(test_version);
    TEST(test_config_create_free);
    TEST(test_config_null_safety);
    TEST(test_load_null_path);
    TEST(test_load_nonexistent);
    TEST(test_default_sample_params);
    TEST(test_generate_null_safety);
    TEST(test_model_info_null);
    TEST(test_kv_cache_null);
    TEST(test_speculative_null);
    TEST(test_free_null);

    printf("─────────────────────────────────────────────\n");
    printf("%d/%d tests passed\n", tests_passed, tests_run);

    return tests_passed == tests_run ? 0 : 1;
}
