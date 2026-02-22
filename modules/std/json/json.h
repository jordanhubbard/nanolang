#ifndef NANOLANG_STD_JSON_H
#define NANOLANG_STD_JSON_H

#include <stdint.h>

#include "nanolang.h"

/* Opaque JSON handle (cJSON*) */
void* nl_json_parse(const char* text);
void nl_json_free(void* json);

/* Returns a newly allocated string (caller may leak; nanolang strings are char*). */
char* nl_json_stringify(void* json);

/* Type predicates */
int64_t nl_json_is_null(void* json);
int64_t nl_json_is_bool(void* json);
int64_t nl_json_is_number(void* json);
int64_t nl_json_is_string(void* json);
int64_t nl_json_is_array(void* json);
int64_t nl_json_is_object(void* json);

/* Value conversion helpers (best-effort) */
int64_t nl_json_as_int(void* json);
int64_t nl_json_as_bool(void* json);
char* nl_json_as_string(void* json);

/* Object access */
int64_t nl_json_object_has(void* obj, const char* key);
void* nl_json_get(void* obj, const char* key); /* returns duplicated item (owns) or NULL */
DynArray* nl_json_object_keys(void* obj);

/* Array access */
int64_t nl_json_array_size(void* arr);
void* nl_json_get_index(void* arr, int64_t idx); /* returns duplicated item (owns) or NULL */

/* Constructors */
void* nl_json_new_object(void);
void* nl_json_new_array(void);
void* nl_json_new_string(const char* s);
void* nl_json_new_int(int64_t v);
void* nl_json_new_bool(int64_t v);
void* nl_json_new_null(void);

/* Mutators (clone `val` before insertion; does not consume `val`) */
int64_t nl_json_object_set(void* obj, const char* key, void* val);
int64_t nl_json_array_push(void* arr, void* val);

#endif
