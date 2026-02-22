#ifndef NANOLANG_STD_COLLECTIONS_H
#define NANOLANG_STD_COLLECTIONS_H

#include <stdint.h>
#include <stdbool.h>

#include "nanolang.h"

/* ============================ StringBuilder ============================ */

void* nl_sb_new(void);
void* nl_sb_with_capacity(int64_t capacity);
void nl_sb_append(void* sb, const char* text);
void nl_sb_append_char(void* sb, int64_t ch);
void nl_sb_clear(void* sb);
int64_t nl_sb_length(void* sb);
int64_t nl_sb_capacity(void* sb);
char* nl_sb_to_string(void* sb);
void nl_sb_free(void* sb);

/* ============================ HashMap<string,string> ============================ */

void* nl_hm_new(void);
void nl_hm_put(void* hm, const char* key, const char* value);
bool nl_hm_has(void* hm, const char* key);
char* nl_hm_get(void* hm, const char* key);
int64_t nl_hm_size(void* hm);
DynArray* nl_hm_keys(void* hm);
DynArray* nl_hm_values(void* hm);
void nl_hm_remove(void* hm, const char* key);
void nl_hm_clear(void* hm);
void nl_hm_free(void* hm);

/* ============================ Set<string> ============================ */

void* nl_set_new(void);
void nl_set_add(void* set, const char* key);
bool nl_set_has(void* set, const char* key);
int64_t nl_set_size(void* set);
DynArray* nl_set_values(void* set);
void nl_set_remove(void* set, const char* key);
void nl_set_clear(void* set);
void nl_set_free(void* set);

#endif
