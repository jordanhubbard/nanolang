#ifndef EVAL_HASHMAP_H
#define EVAL_HASHMAP_H

#include "../nanolang.h"
#include <stdint.h>
#include <stdbool.h>

/* HashMap key and value types */
typedef enum {
    NL_HM_KEY_INT,
    NL_HM_KEY_STRING,
} NLHashMapKeyType;

typedef enum {
    NL_HM_VAL_INT,
    NL_HM_VAL_STRING,
} NLHashMapValType;

/* HashMap entry */
typedef struct {
    uint8_t state; /* 0=empty, 1=filled, 2=tombstone */
    union {
        int64_t i;
        char *s;
    } key;
    union {
        int64_t i;
        char *s;
    } value;
} NLHashMapEntry;

/* HashMap core structure */
typedef struct {
    NLHashMapKeyType key_type;
    NLHashMapValType val_type;
    int64_t capacity;
    int64_t size;
    int64_t tombstones;
    NLHashMapEntry *entries;
} NLHashMapCore;

/* HashMap functions */
uint64_t nl_hm_hash_string(const char *s);
uint64_t nl_hm_hash_int(int64_t x);
bool nl_hm_parse_monomorph(const char *mono, NLHashMapKeyType *out_k, NLHashMapValType *out_v);
const char *nl_hm_typeinfo_arg_name(const TypeInfo *ti);
NLHashMapCore *nl_hm_alloc(NLHashMapKeyType kt, NLHashMapValType vt, int64_t capacity);
bool nl_hm_key_equals(const NLHashMapCore *hm, const NLHashMapEntry *e, const Value *key);
uint64_t nl_hm_hash_key(const NLHashMapCore *hm, const Value *key);
int64_t nl_hm_find_slot(const NLHashMapCore *hm, const Value *key, bool *out_found);
void nl_hm_free_entry(NLHashMapCore *hm, NLHashMapEntry *e);
void nl_hm_rehash(NLHashMapCore *hm, int64_t new_cap);
void nl_hm_clear(NLHashMapCore *hm);
void nl_hm_free(NLHashMapCore *hm);

#endif /* EVAL_HASHMAP_H */
