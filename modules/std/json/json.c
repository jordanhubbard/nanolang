#define _GNU_SOURCE
#include "json.h"

#include <stdlib.h>
#include <string.h>

#include "cJSON.h"

static char* nl_strdup_or_empty(const char* s) {
    if (!s) s = "";
    char* out = strdup(s);
    return out ? out : strdup("");
}

/* Detect whether this looks like NanoLang-literal-escaped JSON (quotes escaped as \") */
static bool nl_json_needs_literal_unescape(const char *s) {
    if (!s) return false;
    size_t raw_len = strnlen(s, 1024ULL * 1024ULL);

    bool in_string = false;
    int backslashes = 0;

    for (size_t i = 0; i < raw_len; i++) {
        char c = s[i];
        if (!in_string && c == '\\' && (i + 1) < raw_len && s[i + 1] == '"') {
            /* \" outside a JSON string: this is invalid JSON unless unescaped */
            return true;
        }

        if (c == '\\') {
            backslashes++;
        } else {
            if (c == '"' && (backslashes % 2) == 0) {
                in_string = !in_string;
            }
            backslashes = 0;
        }
    }
    return false;
}

/* Unescape NanoLang-literal JSON: turn \" -> " and \\ -> \ everywhere. */
static char* nl_unescape_nanolang_literal_json(const char* s) {
    if (!s) return NULL;
    size_t raw_len = strnlen(s, 1024ULL * 1024ULL);
    char* out = (char*)malloc(raw_len + 1);
    if (!out) return NULL;

    size_t j = 0;
    for (size_t i = 0; i < raw_len; i++) {
        char c = s[i];
        if (c == '\\' && (i + 1) < raw_len) {
            char n = s[i + 1];
            if (n == '"' || n == '\\') {
                out[j++] = n;
                i++;
                continue;
            }
        }
        out[j++] = c;
    }
    out[j] = '\0';
    return out;
}

void* nl_json_parse(const char* text) {
    if (!text) return NULL;
    cJSON* out = NULL;
    if (nl_json_needs_literal_unescape(text)) {
        char* normalized = nl_unescape_nanolang_literal_json(text);
        if (!normalized) return NULL;
        out = cJSON_Parse(normalized);
        free(normalized);
    } else {
        /* Treat as normal JSON text (e.g., from a file) */
        out = cJSON_Parse(text);
    }
    return (void*)out;
}

void nl_json_free(void* json) {
    if (!json) return;
    cJSON_Delete((cJSON*)json);
}

const char* nl_json_stringify(void* json) {
    if (!json) return nl_strdup_or_empty("null");
    /* cJSON_PrintUnformatted returns heap memory */
    char* s = cJSON_PrintUnformatted((cJSON*)json);
    return s ? s : nl_strdup_or_empty("");
}

int64_t nl_json_is_null(void* json) { return json && cJSON_IsNull((cJSON*)json) ? 1 : 0; }
int64_t nl_json_is_bool(void* json) { return json && cJSON_IsBool((cJSON*)json) ? 1 : 0; }
int64_t nl_json_is_number(void* json) { return json && cJSON_IsNumber((cJSON*)json) ? 1 : 0; }
int64_t nl_json_is_string(void* json) { return json && cJSON_IsString((cJSON*)json) ? 1 : 0; }
int64_t nl_json_is_array(void* json) { return json && cJSON_IsArray((cJSON*)json) ? 1 : 0; }
int64_t nl_json_is_object(void* json) { return json && cJSON_IsObject((cJSON*)json) ? 1 : 0; }

int64_t nl_json_as_int(void* json) {
    if (!json) return 0;
    if (cJSON_IsBool((cJSON*)json)) return cJSON_IsTrue((cJSON*)json) ? 1 : 0;
    if (!cJSON_IsNumber((cJSON*)json)) return 0;
    return (int64_t)cJSON_GetNumberValue((cJSON*)json);
}

int64_t nl_json_as_bool(void* json) {
    if (!json) return 0;
    if (cJSON_IsBool((cJSON*)json)) return cJSON_IsTrue((cJSON*)json) ? 1 : 0;
    if (cJSON_IsNumber((cJSON*)json)) return cJSON_GetNumberValue((cJSON*)json) != 0.0 ? 1 : 0;
    if (cJSON_IsString((cJSON*)json)) return cJSON_GetStringValue((cJSON*)json) && cJSON_GetStringValue((cJSON*)json)[0] ? 1 : 0;
    return 0;
}

const char* nl_json_as_string(void* json) {
    if (!json) return nl_strdup_or_empty("");
    if (cJSON_IsString((cJSON*)json)) {
        return nl_strdup_or_empty(cJSON_GetStringValue((cJSON*)json));
    }
    /* fall back to JSON stringification */
    return nl_json_stringify(json);
}

int64_t nl_json_object_has(void* obj, const char* key) {
    if (!obj || !key) return 0;
    if (!cJSON_IsObject((cJSON*)obj)) return 0;
    cJSON* it = cJSON_GetObjectItemCaseSensitive((cJSON*)obj, key);
    return it ? 1 : 0;
}

void* nl_json_get(void* obj, const char* key) {
    if (!obj || !key) return NULL;
    if (!cJSON_IsObject((cJSON*)obj)) return NULL;
    cJSON* it = cJSON_GetObjectItemCaseSensitive((cJSON*)obj, key);
    if (!it) return NULL;
    return (void*)cJSON_Duplicate(it, 1);
}

DynArray* nl_json_object_keys(void* obj) {
    DynArray* out = dyn_array_new(ELEM_STRING);
    if (!obj || !cJSON_IsObject((cJSON*)obj)) return out;

    for (cJSON* it = ((cJSON*)obj)->child; it; it = it->next) {
        if (it->string) {
            /* Copy: key strings are owned by cJSON and become invalid after nl_json_free(obj). */
            dyn_array_push_string_copy(out, it->string);
        }
    }
    return out;
}

int64_t nl_json_array_size(void* arr) {
    if (!arr || !cJSON_IsArray((cJSON*)arr)) return 0;
    return (int64_t)cJSON_GetArraySize((cJSON*)arr);
}

void* nl_json_get_index(void* arr, int64_t idx) {
    if (!arr || !cJSON_IsArray((cJSON*)arr)) return NULL;
    if (idx < 0) return NULL;
    cJSON* it = cJSON_GetArrayItem((cJSON*)arr, (int)idx);
    if (!it) return NULL;
    return (void*)cJSON_Duplicate(it, 1);
}

void* nl_json_new_object(void) { return (void*)cJSON_CreateObject(); }
void* nl_json_new_array(void) { return (void*)cJSON_CreateArray(); }
void* nl_json_new_string(const char* s) { return (void*)cJSON_CreateString(s ? s : ""); }
void* nl_json_new_int(int64_t v) { return (void*)cJSON_CreateNumber((double)v); }
void* nl_json_new_bool(int64_t v) { return (void*)cJSON_CreateBool(v ? 1 : 0); }
void* nl_json_new_null(void) { return (void*)cJSON_CreateNull(); }

int64_t nl_json_object_set(void* obj, const char* key, void* val) {
    if (!obj || !key || !val) return 0;
    if (!cJSON_IsObject((cJSON*)obj)) return 0;
    cJSON* copy = cJSON_Duplicate((cJSON*)val, 1);
    if (!copy) return 0;
    if (!cJSON_AddItemToObject((cJSON*)obj, key, copy)) {
        cJSON_Delete(copy);
        return 0;
    }
    return 1;
}

int64_t nl_json_array_push(void* arr, void* val) {
    if (!arr || !val) return 0;
    if (!cJSON_IsArray((cJSON*)arr)) return 0;
    cJSON* copy = cJSON_Duplicate((cJSON*)val, 1);
    if (!copy) return 0;
    if (!cJSON_AddItemToArray((cJSON*)arr, copy)) {
        cJSON_Delete(copy);
        return 0;
    }
    return 1;
}
