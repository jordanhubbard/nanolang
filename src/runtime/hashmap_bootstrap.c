#include "hashmap_bootstrap.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Simple FNV-1a hash function */
static uint64_t hash_string(const char *s) {
    if (!s) return 0;
    uint64_t hash = 14695981039346656037ULL;
    while (*s) {
        hash ^= (uint8_t)(*s++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

/* HashMap<string, int> implementation */

HashMap_string_int* nl_hashmap_string_int_alloc(int64_t cap) {
    HashMap_string_int *hm = (HashMap_string_int*)malloc(sizeof(HashMap_string_int));
    hm->capacity = cap;
    hm->size = 0;
    hm->tombstones = 0;
    hm->entries = (HashMap_string_int_Entry*)calloc(cap, sizeof(HashMap_string_int_Entry));
    return hm;
}

void nl_hashmap_string_int_free(HashMap_string_int *hm) {
    if (!hm) return;
    free(hm->entries);
    free(hm);
}

static int64_t find_slot_string_int(HashMap_string_int *hm, char* key) {
    uint64_t hash = hash_string(key);
    int64_t idx = hash % hm->capacity;
    int64_t first_tombstone = -1;

    for (int64_t i = 0; i < hm->capacity; i++) {
        int64_t slot = (idx + i) % hm->capacity;
        HashMap_string_int_Entry *e = &hm->entries[slot];

        if (e->state == 0) {
            return first_tombstone >= 0 ? first_tombstone : slot;
        } else if (e->state == 2 && first_tombstone < 0) {
            first_tombstone = slot;
        } else if (e->state == 1 && strcmp(e->key, key) == 0) {
            return slot;
        }
    }
    return first_tombstone >= 0 ? first_tombstone : idx;
}

static void rehash_string_int(HashMap_string_int *hm) {
    int64_t old_cap = hm->capacity;
    HashMap_string_int_Entry *old_entries = hm->entries;

    hm->capacity *= 2;
    hm->entries = (HashMap_string_int_Entry*)calloc(hm->capacity, sizeof(HashMap_string_int_Entry));
    hm->size = 0;
    hm->tombstones = 0;

    for (int64_t i = 0; i < old_cap; i++) {
        if (old_entries[i].state == 1) {
            nl_hashmap_string_int_put(hm, old_entries[i].key, old_entries[i].value);
        }
    }
    free(old_entries);
}

void nl_hashmap_string_int_put(HashMap_string_int *hm, char* key, int64_t val) {
    if (hm->size + hm->tombstones >= hm->capacity * 0.75) {
        rehash_string_int(hm);
    }

    int64_t slot = find_slot_string_int(hm, key);
    HashMap_string_int_Entry *e = &hm->entries[slot];

    if (e->state != 1) {
        hm->size++;
    }
    e->state = 1;
    e->key = key;
    e->value = val;
}

bool nl_hashmap_string_int_has(HashMap_string_int *hm, char* key) {
    int64_t slot = find_slot_string_int(hm, key);
    return hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0;
}

int64_t nl_hashmap_string_int_get(HashMap_string_int *hm, char* key) {
    int64_t slot = find_slot_string_int(hm, key);
    if (hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0) {
        return hm->entries[slot].value;
    }
    return 0; /* Default value */
}

/* HashMap<string, string> implementation */

HashMap_string_string* nl_hashmap_string_string_alloc(int64_t cap) {
    HashMap_string_string *hm = (HashMap_string_string*)malloc(sizeof(HashMap_string_string));
    hm->capacity = cap;
    hm->size = 0;
    hm->tombstones = 0;
    hm->entries = (HashMap_string_string_Entry*)calloc(cap, sizeof(HashMap_string_string_Entry));
    return hm;
}

void nl_hashmap_string_string_free(HashMap_string_string *hm) {
    if (!hm) return;
    free(hm->entries);
    free(hm);
}

static int64_t find_slot_string_string(HashMap_string_string *hm, char* key) {
    uint64_t hash = hash_string(key);
    int64_t idx = hash % hm->capacity;
    int64_t first_tombstone = -1;

    for (int64_t i = 0; i < hm->capacity; i++) {
        int64_t slot = (idx + i) % hm->capacity;
        HashMap_string_string_Entry *e = &hm->entries[slot];

        if (e->state == 0) {
            return first_tombstone >= 0 ? first_tombstone : slot;
        } else if (e->state == 2 && first_tombstone < 0) {
            first_tombstone = slot;
        } else if (e->state == 1 && strcmp(e->key, key) == 0) {
            return slot;
        }
    }
    return first_tombstone >= 0 ? first_tombstone : idx;
}

static void rehash_string_string(HashMap_string_string *hm) {
    int64_t old_cap = hm->capacity;
    HashMap_string_string_Entry *old_entries = hm->entries;

    hm->capacity *= 2;
    hm->entries = (HashMap_string_string_Entry*)calloc(hm->capacity, sizeof(HashMap_string_string_Entry));
    hm->size = 0;
    hm->tombstones = 0;

    for (int64_t i = 0; i < old_cap; i++) {
        if (old_entries[i].state == 1) {
            nl_hashmap_string_string_put(hm, old_entries[i].key, old_entries[i].value);
        }
    }
    free(old_entries);
}

void nl_hashmap_string_string_put(HashMap_string_string *hm, char* key, char* val) {
    if (hm->size + hm->tombstones >= hm->capacity * 0.75) {
        rehash_string_string(hm);
    }

    int64_t slot = find_slot_string_string(hm, key);
    HashMap_string_string_Entry *e = &hm->entries[slot];

    if (e->state != 1) {
        hm->size++;
    }
    e->state = 1;
    e->key = key;
    e->value = val;
}

bool nl_hashmap_string_string_has(HashMap_string_string *hm, char* key) {
    int64_t slot = find_slot_string_string(hm, key);
    return hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0;
}

char* nl_hashmap_string_string_get(HashMap_string_string *hm, char* key) {
    int64_t slot = find_slot_string_string(hm, key);
    if (hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0) {
        return hm->entries[slot].value;
    }
    return ""; /* Default value */
}
