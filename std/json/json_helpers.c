/*
 * JSON Helper Functions for Nanolang
 * Wraps cJSON library with nanolang-friendly API
 */

#include "cJSON.h"
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// Value extraction helpers

double nl_cjson_get_number_value(void* item) {
    if (!item) return 0.0;
    cJSON* json = (cJSON*)item;
    if (!cJSON_IsNumber(json)) return 0.0;
    return json->valuedouble;
}

const char* nl_cjson_get_string_value(void* item) {
    if (!item) return "";
    cJSON* json = (cJSON*)item;
    if (!cJSON_IsString(json)) return "";
    return json->valuestring ? json->valuestring : "";
}

bool nl_cjson_get_bool_value(void* item) {
    if (!item) return false;
    cJSON* json = (cJSON*)item;
    if (!cJSON_IsBool(json)) return false;
    return cJSON_IsTrue(json);
}

