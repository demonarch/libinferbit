/*
 * config_json.c — Parse HuggingFace config.json for model architecture
 *
 * Reads exact architecture params instead of guessing from tensor shapes.
 * Supports LLaMA, Mistral, Falcon, Phi, Qwen, Gemma naming conventions.
 */

#include "inferbit_internal.h"
#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ib_model_config is defined in inferbit_internal.h */

/* ── JSON helpers ───────────────────────────────────────────── */

static int jint(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsNumber(item) ? item->valueint : def;
}

static double jdbl(const cJSON* obj, const char* key, double def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsNumber(item) ? item->valuedouble : def;
}

static const char* jstr(const cJSON* obj, const char* key, const char* def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return (cJSON_IsString(item) && item->valuestring) ? item->valuestring : def;
}

static int jbool(const cJSON* obj, const char* key, int def) {
    cJSON* item = cJSON_GetObjectItemCaseSensitive(obj, key);
    return cJSON_IsBool(item) ? cJSON_IsTrue(item) : def;
}

/* ── Detect architecture from model_type ────────────────────── */

static void detect_arch_name(const char* model_type, char* out, int out_size) {
    if (!model_type) { strncpy(out, "llama", out_size - 1); return; }

    /* Normalize: lowercase comparison */
    if (strstr(model_type, "llama") || strstr(model_type, "Llama"))
        strncpy(out, "llama", out_size - 1);
    else if (strstr(model_type, "mistral") || strstr(model_type, "Mistral"))
        strncpy(out, "mistral", out_size - 1);
    else if (strstr(model_type, "falcon") || strstr(model_type, "Falcon"))
        strncpy(out, "falcon", out_size - 1);
    else if (strstr(model_type, "phi") || strstr(model_type, "Phi"))
        strncpy(out, "phi", out_size - 1);
    else if (strstr(model_type, "qwen") || strstr(model_type, "Qwen"))
        strncpy(out, "qwen", out_size - 1);
    else if (strstr(model_type, "gemma") || strstr(model_type, "Gemma"))
        strncpy(out, "gemma", out_size - 1);
    else if (strstr(model_type, "gpt_neox") || strstr(model_type, "GPTNeoX"))
        strncpy(out, "gpt_neox", out_size - 1);
    else
        strncpy(out, model_type, out_size - 1);
}

/* ── Parse config.json ──────────────────────────────────────── */

int ib_parse_config_json(const char* path, ib_model_config* cfg) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0 || size > 10 * 1024 * 1024) {
        fclose(f);
        return -1;
    }

    char* buf = malloc(size + 1);
    if (!buf) { fclose(f); return -1; }
    fread(buf, 1, size, f);
    buf[size] = '\0';
    fclose(f);

    cJSON* root = cJSON_Parse(buf);
    free(buf);
    if (!root) return -1;

    memset(cfg, 0, sizeof(*cfg));

    /* Model type / architecture */
    const char* model_type = jstr(root, "model_type", NULL);
    detect_arch_name(model_type, cfg->arch, sizeof(cfg->arch));

    /* Core dimensions */
    cfg->hidden_size       = jint(root, "hidden_size", 0);
    cfg->num_layers        = jint(root, "num_hidden_layers", 0);
    cfg->num_heads         = jint(root, "num_attention_heads", 0);
    cfg->intermediate_size = jint(root, "intermediate_size", 0);
    cfg->vocab_size        = jint(root, "vocab_size", 0);

    /* KV heads — various naming conventions */
    cfg->num_kv_heads = jint(root, "num_key_value_heads",
                         jint(root, "num_kv_heads",
                          jint(root, "multi_query_group_num", cfg->num_heads)));

    /* Head dim — explicit or derived */
    cfg->head_dim = jint(root, "head_dim",
                     cfg->num_heads > 0 ? cfg->hidden_size / cfg->num_heads : 128);

    /* Context length — various naming conventions */
    cfg->max_context_length = jint(root, "max_position_embeddings",
                               jint(root, "max_sequence_length",
                                jint(root, "seq_length",
                                 jint(root, "sliding_window", 4096))));

    /* RoPE */
    cfg->rope_theta = (float)jdbl(root, "rope_theta", 10000.0);

    /* Normalization */
    cfg->norm_epsilon = (float)jdbl(root, "rms_norm_eps",
                         jdbl(root, "layer_norm_eps",
                          jdbl(root, "layer_norm_epsilon", 1e-5)));

    /* Detect norm type */
    /* Most modern models use RMSNorm; check for layernorm indicators */
    if (cJSON_GetObjectItemCaseSensitive(root, "rms_norm_eps")) {
        strncpy(cfg->norm_type, "rmsnorm", sizeof(cfg->norm_type) - 1);
    } else if (cJSON_GetObjectItemCaseSensitive(root, "layer_norm_eps")) {
        strncpy(cfg->norm_type, "layernorm", sizeof(cfg->norm_type) - 1);
    } else {
        strncpy(cfg->norm_type, "rmsnorm", sizeof(cfg->norm_type) - 1);
    }

    /* Activation */
    const char* act = jstr(root, "hidden_act", jstr(root, "activation_function", "silu"));
    strncpy(cfg->activation, act, sizeof(cfg->activation) - 1);

    /* Tied embeddings */
    cfg->tie_word_embeddings = jbool(root, "tie_word_embeddings", 0);

    /* Special tokens */
    cfg->bos_token_id = jint(root, "bos_token_id", 1);
    cfg->eos_token_id = jint(root, "eos_token_id", 2);

    cJSON_Delete(root);
    return 0;
}
