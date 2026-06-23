#include "hashmap_bootstrap.h"
#include <stdlib.h>
#include <string.h>

/* Local strdup to avoid implicit-declaration issues in strict C99 */
static char* hm_strdup(const char* s) {
    if (!s) s = "";
    size_t len = strlen(s);
    char* copy = (char*)malloc(len + 1);
    if (copy) memcpy(copy, s, len + 1);
    return copy;
}

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
    for (int64_t i = 0; i < hm->capacity; i++) {
        if (hm->entries[i].state == 1) {
            free(hm->entries[i].key);
        }
    }
    free(hm->entries);
    free(hm);
}

static int64_t find_slot_string_int(HashMap_string_int *hm, const char* key) {
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
            /* nl_hashmap_string_int_put will strdup the key again */
            nl_hashmap_string_int_put(hm, old_entries[i].key, old_entries[i].value);
            free(old_entries[i].key); /* Free the old strdup'd key */
        }
    }
    free(old_entries);
}

void nl_hashmap_string_int_put(HashMap_string_int *hm, const char* key, int64_t val) {
    if (hm->size + hm->tombstones >= hm->capacity * 0.75) {
        rehash_string_int(hm);
    }

    int64_t slot = find_slot_string_int(hm, key);
    HashMap_string_int_Entry *e = &hm->entries[slot];

    if (e->state == 1) {
        /* Updating existing entry: free old key and strdup new */
        free(e->key);
        e->key = hm_strdup(key);
        e->value = val;
    } else {
        hm->size++;
        e->state = 1;
        e->key = hm_strdup(key);
        e->value = val;
    }
}

bool nl_hashmap_string_int_has(HashMap_string_int *hm, const char* key) {
    int64_t slot = find_slot_string_int(hm, key);
    return hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0;
}

int64_t nl_hashmap_string_int_get(HashMap_string_int *hm, const char* key) {
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
    for (int64_t i = 0; i < hm->capacity; i++) {
        if (hm->entries[i].state == 1) {
            free(hm->entries[i].key);
            free(hm->entries[i].value);
        }
    }
    free(hm->entries);
    free(hm);
}

static int64_t find_slot_string_string(HashMap_string_string *hm, const char* key) {
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
            /* nl_hashmap_string_string_put will strdup key and value again */
            nl_hashmap_string_string_put(hm, old_entries[i].key, old_entries[i].value);
            free(old_entries[i].key);
            free(old_entries[i].value);
        }
    }
    free(old_entries);
}

void nl_hashmap_string_string_put(HashMap_string_string *hm, const char* key, const char* val) {
    if (hm->size + hm->tombstones >= hm->capacity * 0.75) {
        rehash_string_string(hm);
    }

    int64_t slot = find_slot_string_string(hm, key);
    HashMap_string_string_Entry *e = &hm->entries[slot];

    if (e->state == 1) {
        /* Updating existing entry: free old key/value and strdup new */
        free(e->key);
        free(e->value);
        e->key = hm_strdup(key);
        e->value = hm_strdup(val);
    } else {
        hm->size++;
        e->state = 1;
        e->key = hm_strdup(key);
        e->value = hm_strdup(val);
    }
}

bool nl_hashmap_string_string_has(HashMap_string_string *hm, const char* key) {
    int64_t slot = find_slot_string_string(hm, key);
    return hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0;
}

const char* nl_hashmap_string_string_get(HashMap_string_string *hm, const char* key) {
    int64_t slot = find_slot_string_string(hm, key);
    if (hm->entries[slot].state == 1 && strcmp(hm->entries[slot].key, key) == 0) {
        return hm->entries[slot].value;
    }
    return ""; /* Default value */
}
