#include "inferbit_internal.h"
#include <stdlib.h>
#include <string.h>

inferbit_config* inferbit_config_create(void) {
    inferbit_config* config = calloc(1, sizeof(inferbit_config));
    if (!config) {
        ib_set_error("failed to allocate config");
        return NULL;
    }
    config->threads        = 0;  /* 0 = auto-detect */
    config->context_length = 0;  /* 0 = use model default */
    config->kv_dynamic     = false;
    config->native_parse   = false;
    config->native_bits    = 4;
    return config;
}

void inferbit_config_free(inferbit_config* config) {
    free(config);
}

void inferbit_config_set_threads(inferbit_config* config, int threads) {
    if (config) config->threads = threads;
}

void inferbit_config_set_context_length(inferbit_config* config, int length) {
    if (config) config->context_length = length;
}

void inferbit_config_set_kv_cache_dynamic(inferbit_config* config, int dynamic) {
    if (config) config->kv_dynamic = (dynamic != 0);
}

void inferbit_config_set_native_parse(inferbit_config* config, int enabled) {
    if (config) config->native_parse = (enabled != 0);
}

void inferbit_config_set_native_bits(inferbit_config* config, int bits) {
    if (config) config->native_bits = bits;
}
