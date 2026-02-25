#ifndef HASHMAP_BOOTSTRAP_H
#define HASHMAP_BOOTSTRAP_H

#include <stdint.h>
#include <stdbool.h>

/* Bootstrap HashMap implementation for self-hosted compiler */

/* HashMap<string, int> */
typedef struct HashMap_string_int_Entry {
    uint8_t state; /* 0=empty, 1=filled, 2=tombstone */
    char* key;
    int64_t value;
} HashMap_string_int_Entry;

typedef struct HashMap_string_int {
    int64_t capacity;
    int64_t size;
    int64_t tombstones;
    HashMap_string_int_Entry *entries;
} HashMap_string_int;

HashMap_string_int* nl_hashmap_string_int_alloc(int64_t cap);
void nl_hashmap_string_int_free(HashMap_string_int *hm);
void nl_hashmap_string_int_put(HashMap_string_int *hm, const char* key, int64_t val);
bool nl_hashmap_string_int_has(HashMap_string_int *hm, const char* key);
int64_t nl_hashmap_string_int_get(HashMap_string_int *hm, const char* key);

/* HashMap<string, string> */
typedef struct HashMap_string_string_Entry {
    uint8_t state; /* 0=empty, 1=filled, 2=tombstone */
    char* key;
    char* value;
} HashMap_string_string_Entry;

typedef struct HashMap_string_string {
    int64_t capacity;
    int64_t size;
    int64_t tombstones;
    HashMap_string_string_Entry *entries;
} HashMap_string_string;

HashMap_string_string* nl_hashmap_string_string_alloc(int64_t cap);
void nl_hashmap_string_string_free(HashMap_string_string *hm);
void nl_hashmap_string_string_put(HashMap_string_string *hm, const char* key, const char* val);
bool nl_hashmap_string_string_has(HashMap_string_string *hm, const char* key);
const char* nl_hashmap_string_string_get(HashMap_string_string *hm, const char* key);

/* Type aliases used by transpiled code - use pointers */
typedef HashMap_string_int* nl_HashMap_string_int_;
typedef HashMap_string_string* nl_HashMap_string_string_;
typedef HashMap_string_int* nl_HashMap; /* Generic fallback */

/* Generic wrappers - HACK for bootstrap (not type-safe!) */
/* These assume the first field of both HashMap types is the same layout */

static inline void* nl_map_new_string_int(void) {
    return nl_hashmap_string_int_alloc(16);
}

static inline void* nl_map_new_string_string(void) {
    return nl_hashmap_string_string_alloc(16);
}

/* Transpiler will need to generate type-specific calls,
   or we need to port full HashMap codegen from C compiler */

#endif /* HASHMAP_BOOTSTRAP_H */
