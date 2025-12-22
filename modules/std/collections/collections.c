#include "collections.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ============================ StringBuilder ============================ */

typedef struct {
    char *buf;
    int64_t len;
    int64_t cap;
} NLStringBuilder;

static NLStringBuilder *sb_alloc(int64_t cap) {
    if (cap < 16) cap = 16;
    NLStringBuilder *sb = (NLStringBuilder*)calloc(1, sizeof(NLStringBuilder));
    if (!sb) return NULL;
    sb->buf = (char*)malloc((size_t)cap);
    if (!sb->buf) {
        free(sb);
        return NULL;
    }
    sb->buf[0] = '\0';
    sb->len = 0;
    sb->cap = cap;
    return sb;
}

static void sb_ensure(NLStringBuilder *sb, int64_t extra) {
    assert(sb);
    if (extra < 0) return;
    int64_t need = sb->len + extra + 1;
    if (need <= sb->cap) return;
    int64_t new_cap = sb->cap;
    while (new_cap < need) {
        if (new_cap > INT64_MAX / 2) {
            new_cap = need;
            break;
        }
        new_cap *= 2;
    }
    char *nb = (char*)realloc(sb->buf, (size_t)new_cap);
    if (!nb) {
        fprintf(stderr, "Error: Out of memory growing StringBuilder\n");
        exit(1);
    }
    sb->buf = nb;
    sb->cap = new_cap;
}

void* nl_sb_new(void) {
    return (void*)sb_alloc(256);
}

void* nl_sb_with_capacity(int64_t capacity) {
    return (void*)sb_alloc(capacity);
}

void nl_sb_append(void* sb_ptr, const char* text) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb || !text) return;
    size_t tlen = strnlen(text, 1024ULL * 1024ULL);
    sb_ensure(sb, (int64_t)tlen);
    memcpy(sb->buf + sb->len, text, tlen);
    sb->len += (int64_t)tlen;
    sb->buf[sb->len] = '\0';
}

void nl_sb_append_char(void* sb_ptr, int64_t ch) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb) return;
    sb_ensure(sb, 1);
    sb->buf[sb->len++] = (char)ch;
    sb->buf[sb->len] = '\0';
}

void nl_sb_clear(void* sb_ptr) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb) return;
    sb->len = 0;
    if (sb->buf) sb->buf[0] = '\0';
}

int64_t nl_sb_length(void* sb_ptr) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb) return 0;
    return sb->len;
}

int64_t nl_sb_capacity(void* sb_ptr) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb) return 0;
    return sb->cap;
}

const char* nl_sb_to_string(void* sb_ptr) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb || !sb->buf) return strdup("");
    char *out = strdup(sb->buf);
    if (!out) out = strdup("");
    return out;
}

void nl_sb_free(void* sb_ptr) {
    NLStringBuilder *sb = (NLStringBuilder*)sb_ptr;
    if (!sb) return;
    free(sb->buf);
    free(sb);
}

/* ============================ HashMap<string,string> ============================ */

typedef struct {
    char *key;     /* NULL = empty */
    char *value;   /* Owned */
    bool tombstone;
} HMEntry;

typedef struct {
    HMEntry *entries;
    int64_t capacity;
    int64_t size;
    int64_t tombstones;
} NLHashMap;

static uint64_t fnv1a(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    for (const unsigned char *p = (const unsigned char*)s; *p; p++) {
        h ^= (uint64_t)(*p);
        h *= 1099511628211ULL;
    }
    return h;
}

static NLHashMap* hm_alloc(int64_t capacity) {
    if (capacity < 16) capacity = 16;
    /* round up to power of 2 */
    int64_t cap = 1;
    while (cap < capacity) cap <<= 1;

    NLHashMap *hm = (NLHashMap*)calloc(1, sizeof(NLHashMap));
    if (!hm) return NULL;
    hm->entries = (HMEntry*)calloc((size_t)cap, sizeof(HMEntry));
    if (!hm->entries) {
        free(hm);
        return NULL;
    }
    hm->capacity = cap;
    hm->size = 0;
    hm->tombstones = 0;
    return hm;
}

static void hm_entry_free(HMEntry *e) {
    if (!e) return;
    free(e->key);
    free(e->value);
    e->key = NULL;
    e->value = NULL;
    e->tombstone = false;
}

static void hm_resize(NLHashMap *hm, int64_t new_cap) {
    NLHashMap *next = hm_alloc(new_cap);
    if (!next) {
        fprintf(stderr, "Error: Out of memory resizing HashMap\n");
        exit(1);
    }

    for (int64_t i = 0; i < hm->capacity; i++) {
        HMEntry *e = &hm->entries[i];
        if (e->key && !e->tombstone) {
            /* Re-insert */
            uint64_t h = fnv1a(e->key);
            int64_t mask = next->capacity - 1;
            int64_t idx = (int64_t)(h & (uint64_t)mask);
            while (true) {
                HMEntry *d = &next->entries[idx];
                if (!d->key) {
                    d->key = e->key;
                    d->value = e->value;
                    d->tombstone = false;
                    next->size++;
                    break;
                }
                idx = (idx + 1) & mask;
            }
            e->key = NULL;
            e->value = NULL;
        }
    }

    free(hm->entries);
    hm->entries = next->entries;
    hm->capacity = next->capacity;
    hm->size = next->size;
    hm->tombstones = 0;
    free(next);
}

static HMEntry* hm_find_slot(NLHashMap *hm, const char *key, bool *found_out) {
    if (found_out) *found_out = false;
    if (!hm || !key) return NULL;
    if ((hm->size + hm->tombstones) * 10 >= hm->capacity * 7) {
        hm_resize(hm, hm->capacity * 2);
    }

    uint64_t h = fnv1a(key);
    int64_t mask = hm->capacity - 1;
    int64_t idx = (int64_t)(h & (uint64_t)mask);
    HMEntry *first_tomb = NULL;

    while (true) {
        HMEntry *e = &hm->entries[idx];
        if (!e->key) {
            if (first_tomb) return first_tomb;
            return e;
        }
        if (e->tombstone) {
            if (!first_tomb) first_tomb = e;
        } else if (strcmp(e->key, key) == 0) {
            if (found_out) *found_out = true;
            return e;
        }
        idx = (idx + 1) & mask;
    }
}

void* nl_hm_new(void) {
    return (void*)hm_alloc(64);
}

void nl_hm_put(void* hm_ptr, const char* key, const char* value) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm || !key) return;
    if (!value) value = "";

    bool found = false;
    HMEntry *e = hm_find_slot(hm, key, &found);
    if (!e) return;

    if (found) {
        free(e->value);
        e->value = strdup(value);
        if (!e->value) e->value = strdup("");
        return;
    }

    if (e->tombstone) {
        hm->tombstones--;
        e->tombstone = false;
    }

    e->key = strdup(key);
    e->value = strdup(value);
    if (!e->key) e->key = strdup("");
    if (!e->value) e->value = strdup("");
    hm->size++;
}

bool nl_hm_has(void* hm_ptr, const char* key) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm || !key) return false;
    bool found = false;
    (void)hm_find_slot(hm, key, &found);
    return found;
}

const char* nl_hm_get(void* hm_ptr, const char* key) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm || !key) return "";
    bool found = false;
    HMEntry *e = hm_find_slot(hm, key, &found);
    if (!found || !e || !e->value) return "";
    return e->value;
}

int64_t nl_hm_size(void* hm_ptr) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm) return 0;
    return hm->size;
}

DynArray* nl_hm_keys(void* hm_ptr) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm) return dyn_array_new(ELEM_STRING);
    DynArray *out = dyn_array_new(ELEM_STRING);
    for (int64_t i = 0; i < hm->capacity; i++) {
        HMEntry *e = &hm->entries[i];
        if (e->key && !e->tombstone) {
            char *k = strdup(e->key);
            dyn_array_push_string(out, k ? k : "");
        }
    }
    return out;
}

DynArray* nl_hm_values(void* hm_ptr) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm) return dyn_array_new(ELEM_STRING);
    DynArray *out = dyn_array_new(ELEM_STRING);
    for (int64_t i = 0; i < hm->capacity; i++) {
        HMEntry *e = &hm->entries[i];
        if (e->key && !e->tombstone) {
            char *v = strdup(e->value ? e->value : "");
            dyn_array_push_string(out, v ? v : "");
        }
    }
    return out;
}

void nl_hm_remove(void* hm_ptr, const char* key) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm || !key) return;
    bool found = false;
    HMEntry *e = hm_find_slot(hm, key, &found);
    if (!found || !e) return;
    free(e->key);
    free(e->value);
    e->key = (char*)"__tomb"; /* non-NULL sentinel */
    e->value = NULL;
    e->tombstone = true;
    hm->size--;
    hm->tombstones++;
}

void nl_hm_clear(void* hm_ptr) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm) return;
    for (int64_t i = 0; i < hm->capacity; i++) {
        HMEntry *e = &hm->entries[i];
        if (e->key && !e->tombstone) {
            hm_entry_free(e);
        } else if (e->tombstone) {
            e->key = NULL;
            e->value = NULL;
            e->tombstone = false;
        }
    }
    hm->size = 0;
    hm->tombstones = 0;
}

void nl_hm_free(void* hm_ptr) {
    NLHashMap *hm = (NLHashMap*)hm_ptr;
    if (!hm) return;
    for (int64_t i = 0; i < hm->capacity; i++) {
        HMEntry *e = &hm->entries[i];
        if (e->key && !e->tombstone) {
            hm_entry_free(e);
        }
    }
    free(hm->entries);
    free(hm);
}

/* ============================ Set<string> (on top of HashMap) ============================ */

typedef struct {
    NLHashMap *hm;
} NLStringSet;

void* nl_set_new(void) {
    NLStringSet *s = (NLStringSet*)calloc(1, sizeof(NLStringSet));
    if (!s) return NULL;
    s->hm = (NLHashMap*)nl_hm_new();
    return s;
}

void nl_set_add(void* set_ptr, const char* key) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm || !key) return;
    nl_hm_put(s->hm, key, "1");
}

bool nl_set_has(void* set_ptr, const char* key) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm || !key) return false;
    return nl_hm_has(s->hm, key);
}

int64_t nl_set_size(void* set_ptr) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm) return 0;
    return nl_hm_size(s->hm);
}

DynArray* nl_set_values(void* set_ptr) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm) return dyn_array_new(ELEM_STRING);
    return nl_hm_keys(s->hm);
}

void nl_set_remove(void* set_ptr, const char* key) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm || !key) return;
    nl_hm_remove(s->hm, key);
}

void nl_set_clear(void* set_ptr) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s || !s->hm) return;
    nl_hm_clear(s->hm);
}

void nl_set_free(void* set_ptr) {
    NLStringSet *s = (NLStringSet*)set_ptr;
    if (!s) return;
    if (s->hm) nl_hm_free(s->hm);
    free(s);
}
