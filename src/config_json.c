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

    /* Model type / architecture (always from the top-level object) */
    const char* model_type = jstr(root, "model_type", NULL);
    detect_arch_name(model_type, cfg->arch, sizeof(cfg->arch));

    /* Multimodal / nested configs (e.g. qwen3_vl_moe) put the LM params under
     * "text_config". Descend into it for the core dims; fall back to root for
     * plain text models. vocab_size may live at either level. */
    cJSON* core = cJSON_GetObjectItemCaseSensitive(root, "text_config");
    if (!cJSON_IsObject(core)) core = root;

    /* Core dimensions */
    cfg->hidden_size       = jint(core, "hidden_size", 0);
    cfg->num_layers        = jint(core, "num_hidden_layers", 0);
    cfg->num_heads         = jint(core, "num_attention_heads", 0);
    cfg->intermediate_size = jint(core, "intermediate_size", 0);
    cfg->vocab_size        = jint(core, "vocab_size", jint(root, "vocab_size", 0));

    /* KV heads — various naming conventions */
    cfg->num_kv_heads = jint(core, "num_key_value_heads",
                         jint(core, "num_kv_heads",
                          jint(core, "multi_query_group_num", cfg->num_heads)));

    /* Head dim — explicit or derived */
    cfg->head_dim = jint(core, "head_dim",
                     cfg->num_heads > 0 ? cfg->hidden_size / cfg->num_heads : 128);

    /* MoE params (qwen3_moe / qwen3_vl_moe). 0 num_experts => dense FFN. */
    cfg->num_experts          = jint(core, "num_experts", 0);
    cfg->num_experts_per_tok  = jint(core, "num_experts_per_tok", 0);
    cfg->moe_intermediate_size = jint(core, "moe_intermediate_size", 0);
    /* Qwen3 family applies per-head RMSNorm to q and k (qk-norm). */
    cfg->qk_norm = (model_type && (strstr(model_type, "qwen3") || strstr(model_type, "Qwen3"))) ? 1 : 0;
    /* M-RoPE iff rope_scaling carries an mrope_section. */
    cJSON* rs = cJSON_GetObjectItemCaseSensitive(core, "rope_scaling");
    cfg->mrope = (cJSON_IsObject(rs) &&
                  cJSON_GetObjectItemCaseSensitive(rs, "mrope_section")) ? 1 : 0;

    /* Context length — various naming conventions */
    cfg->max_context_length = jint(core, "max_position_embeddings",
                               jint(core, "max_sequence_length",
                                jint(core, "seq_length",
                                 jint(core, "sliding_window", 4096))));

    /* RoPE */
    cfg->rope_theta = (float)jdbl(core, "rope_theta", 10000.0);

    /* Normalization */
    cfg->norm_epsilon = (float)jdbl(core, "rms_norm_eps",
                         jdbl(core, "layer_norm_eps",
                          jdbl(core, "layer_norm_epsilon", 1e-5)));

    /* Detect norm type */
    /* Most modern models use RMSNorm; check for layernorm indicators */
    if (cJSON_GetObjectItemCaseSensitive(core, "rms_norm_eps")) {
        strncpy(cfg->norm_type, "rmsnorm", sizeof(cfg->norm_type) - 1);
    } else if (cJSON_GetObjectItemCaseSensitive(core, "layer_norm_eps")) {
        strncpy(cfg->norm_type, "layernorm", sizeof(cfg->norm_type) - 1);
    } else {
        strncpy(cfg->norm_type, "rmsnorm", sizeof(cfg->norm_type) - 1);
    }

    /* Activation */
    const char* act = jstr(core, "hidden_act", jstr(core, "activation_function", "silu"));
    strncpy(cfg->activation, act, sizeof(cfg->activation) - 1);

    /* Tied embeddings (top-level flag, even for VL models) */
    cfg->tie_word_embeddings = jbool(root, "tie_word_embeddings", 0);

    /* Special tokens */
    cfg->bos_token_id = jint(core, "bos_token_id", 1);
    cfg->eos_token_id = jint(core, "eos_token_id", 2);

    cJSON_Delete(root);
    return 0;
}
