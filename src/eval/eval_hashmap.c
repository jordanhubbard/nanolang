/* eval_hashmap.c - HashMap interpreter implementation
 * Extracted from eval.c for better organization
 */

#define _POSIX_C_SOURCE 200809L

#include "eval_hashmap.h"
#include "../nanolang.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================ Core HashMap<K,V> (Interpreter Implementation) ============================
 *
 * NOTE: HashMap<K,V> is supported in BOTH interpreter and compiled modes:
 * - Interpreter mode: Uses this runtime implementation
 * - Compiled mode: Transpiler generates monomorphized HashMap code
 *
 * The transpiler creates specialized HashMap_K_V types with all operations
 * (new, put, get, has, size, free) for each K,V combination used in the program.
 * ========================================================================================== */

uint64_t nl_hm_hash_string(const char *s) {
    if (!s) return 0;
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        h ^= (uint64_t)(*p);
        h *= 1099511628211ULL;
    }
    return h;
}

uint64_t nl_hm_hash_int(int64_t x) {
    uint64_t z = (uint64_t)x;
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33;
    return z;
}

bool nl_hm_parse_monomorph(const char *mono, NLHashMapKeyType *out_k, NLHashMapValType *out_v) {
    if (out_k) *out_k = NL_HM_KEY_INT;
    if (out_v) *out_v = NL_HM_VAL_INT;
    if (!mono) return false;
    if (strncmp(mono, "HashMap_", 8) != 0) return false;

    const char *rest = mono + 8;
    const char *sep = strchr(rest, '_');
    if (!sep) return false;

    size_t klen = (size_t)(sep - rest);
    const char *vstr = sep + 1;

    NLHashMapKeyType kt;
    if (klen == 3 && strncmp(rest, "int", 3) == 0) {
        kt = NL_HM_KEY_INT;
    } else if (klen == 6 && strncmp(rest, "string", 6) == 0) {
        kt = NL_HM_KEY_STRING;
    } else {
        return false;
    }

    NLHashMapValType vt;
    if (strcmp(vstr, "int") == 0) {
        vt = NL_HM_VAL_INT;
    } else if (strcmp(vstr, "string") == 0) {
        vt = NL_HM_VAL_STRING;
    } else {
        return false;
    }

    if (out_k) *out_k = kt;
    if (out_v) *out_v = vt;
    return true;
}

const char *nl_hm_typeinfo_arg_name(const TypeInfo *ti) {
    if (!ti) return NULL;
    switch (ti->base_type) {
        case TYPE_INT: return "int";
        case TYPE_STRING: return "string";
        default: return NULL;
    }
}

NLHashMapCore *nl_hm_alloc(NLHashMapKeyType kt, NLHashMapValType vt, int64_t capacity) {
    if (capacity < 16) capacity = 16;
    int64_t cap = 1;
    while (cap < capacity) cap <<= 1;

    NLHashMapCore *hm = (NLHashMapCore*)calloc(1, sizeof(NLHashMapCore));
    if (!hm) return NULL;
    hm->key_type = kt;
    hm->val_type = vt;
    hm->capacity = cap;
    hm->entries = (NLHashMapEntry*)calloc((size_t)cap, sizeof(NLHashMapEntry));
    if (!hm->entries) {
        free(hm);
        return NULL;
    }
    return hm;
}

bool nl_hm_key_equals(const NLHashMapCore *hm, const NLHashMapEntry *e, const Value *key) {
    if (!hm || !e || !key) return false;
    if (hm->key_type == NL_HM_KEY_INT) {
        return key->type == VAL_INT && e->key.i == key->as.int_val;
    }
    return key->type == VAL_STRING && e->key.s && key->as.string_val && strcmp(e->key.s, key->as.string_val) == 0;
}

uint64_t nl_hm_hash_key(const NLHashMapCore *hm, const Value *key) {
    if (!hm || !key) return 0;
    if (hm->key_type == NL_HM_KEY_INT) return nl_hm_hash_int(key->as.int_val);
    return nl_hm_hash_string(key->as.string_val);
}

int64_t nl_hm_find_slot(const NLHashMapCore *hm, const Value *key, bool *out_found) {
    if (out_found) *out_found = false;
    if (!hm || !hm->entries || hm->capacity <= 0) return -1;

    uint64_t h = nl_hm_hash_key(hm, key);
    int64_t mask = hm->capacity - 1;
    int64_t idx = (int64_t)(h & (uint64_t)mask);
    int64_t first_tomb = -1;

    for (int64_t probe = 0; probe < hm->capacity; probe++) {
        NLHashMapEntry *e = &hm->entries[idx];
        if (e->state == 0) {
            if (first_tomb != -1) idx = first_tomb;
            return idx;
        }
        if (e->state == 2) {
            if (first_tomb == -1) first_tomb = idx;
        } else if (nl_hm_key_equals(hm, e, key)) {
            if (out_found) *out_found = true;
            return idx;
        }
        idx = (idx + 1) & mask;
    }
    return first_tomb;
}

void nl_hm_free_entry(NLHashMapCore *hm, NLHashMapEntry *e) {
    if (!hm || !e) return;
    if (e->state != 1) return;
    if (hm->key_type == NL_HM_KEY_STRING && e->key.s) free(e->key.s);
    if (hm->val_type == NL_HM_VAL_STRING && e->value.s) free(e->value.s);
    e->key.s = NULL;
    e->value.s = NULL;
}

void nl_hm_rehash(NLHashMapCore *hm, int64_t new_cap) {
    if (!hm) return;
    NLHashMapCore *next = nl_hm_alloc(hm->key_type, hm->val_type, new_cap);
    if (!next) return;

    for (int64_t i = 0; i < hm->capacity; i++) {
        NLHashMapEntry *e = &hm->entries[i];
        if (e->state != 1) continue;

        Value key;
        key.is_return = false; key.is_break = false; key.is_continue = false;
        if (hm->key_type == NL_HM_KEY_INT) {
            key.type = VAL_INT;
            key.as.int_val = e->key.i;
        } else {
            key.type = VAL_STRING;
            key.as.string_val = e->key.s;
        }

        bool found = false;
        int64_t idx = nl_hm_find_slot(next, &key, &found);
        if (idx >= 0) {
            next->entries[idx] = *e; /* move ownership */
            next->entries[idx].state = 1;
            next->size++;
        }
        e->key.s = NULL;
        e->value.s = NULL;
    }

    free(hm->entries);
    hm->entries = next->entries;
    hm->capacity = next->capacity;
    hm->size = next->size;
    hm->tombstones = 0;
    free(next);
}

void nl_hm_clear(NLHashMapCore *hm) {
    if (!hm || !hm->entries) return;
    for (int64_t i = 0; i < hm->capacity; i++) {
        nl_hm_free_entry(hm, &hm->entries[i]);
        hm->entries[i].state = 0;
    }
    hm->size = 0;
    hm->tombstones = 0;
}

void nl_hm_free(NLHashMapCore *hm) {
    if (!hm) return;
    nl_hm_clear(hm);
    free(hm->entries);
    free(hm);
}
