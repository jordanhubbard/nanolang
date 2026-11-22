#define _GNU_SOURCE
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static int fail_at = -1, allocation_index;
static int fail_strdup, strdup_calls;
static int refuse_large;
static size_t refused_large_size;
static long live_allocations;
static pthread_mutex_t probe_lock = PTHREAD_MUTEX_INITIALIZER;
static int refuse_allocation(void) {
    int refused;
    assert(!pthread_mutex_lock(&probe_lock));
    refused = fail_at >= 0 && allocation_index++ == fail_at;
    assert(!pthread_mutex_unlock(&probe_lock));
    return refused;
}
static void record_allocation(void *p) {
    if (p) { assert(!pthread_mutex_lock(&probe_lock)); live_allocations++;
        assert(!pthread_mutex_unlock(&probe_lock)); }
}
static void *probe_malloc(size_t size) {
    if (refuse_large && size > 1024 * 1024) { refused_large_size = size; return NULL; }
    if (refuse_allocation()) return NULL;
    void *p = malloc(size); record_allocation(p); return p;
}
static void *probe_calloc(size_t count, size_t size) {
    if (refuse_allocation()) return NULL;
    void *p = calloc(count, size); record_allocation(p); return p;
}
static void *probe_realloc(void *old, size_t size) {
    if (refuse_allocation()) return NULL;
    int was_null = old == NULL;
    void *p = realloc(old, size); if (was_null) record_allocation(p); return p;
}
static char *probe_strdup(const char *text) {
    if (fail_strdup) { strdup_calls++; return NULL; }
    size_t size = strlen(text) + 1; char *p = probe_malloc(size);
    if (p) memcpy(p, text, size);
    return p;
}
static void probe_free(void *p) {
    if (p) { assert(!pthread_mutex_lock(&probe_lock)); assert(live_allocations > 0);
        live_allocations--; assert(!pthread_mutex_unlock(&probe_lock)); }
    free(p);
}
#define malloc probe_malloc
#define calloc probe_calloc
#define realloc probe_realloc
#define strdup probe_strdup
#define free probe_free
#include "../modules/std/collections/collections.c"
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free

static void text_equals(void *handle, const char *expected) {
    const char *copy = nl_sb_to_string(handle); assert(copy && !strcmp(copy, expected));
    probe_free((void *)copy);
}
static void refused(void *handle) {
    nl_sb_append(handle, "ignored"); nl_sb_append_char(handle, 'x'); nl_sb_clear(handle);
    assert(nl_sb_length(handle) == 0 && nl_sb_capacity(handle) == 0);
    text_equals(handle, ""); nl_sb_free(handle);
}
static void lifetime(void) {
    int other = 0;
    refused(NULL); refused(&other); refused((void *)UINTPTR_MAX);
    void *builder = nl_sb_new(), *map = nl_hm_new(), *set = nl_set_new();
    assert(builder && map && set && builder != map && builder != set);
    void *alias = builder; nl_sb_append(builder, "kept"); text_equals(alias, "kept");
    refused(map); refused(set);
    assert(nl_hm_size(builder) == 0 && nl_set_size(builder) == 0);
    nl_hm_free(builder); nl_set_free(builder); text_equals(builder, "kept");
    const char *snapshot = nl_sb_to_string(builder); assert(snapshot);
    nl_sb_free(alias); refused(builder); nl_sb_free(builder);
    assert(!strcmp(snapshot, "kept")); probe_free((void *)snapshot);
    for (int i = 0; i < 1000; i++) {
        void *fresh = nl_sb_new(); assert(fresh && fresh != builder);
        refused(builder); nl_sb_append(fresh, "new"); text_equals(fresh, "new"); nl_sb_free(fresh);
    }
    nl_hm_free(map); nl_set_free(set); assert(!collection_handles && live_allocations == 0);
}
static void contents(void) {
    for (int64_t capacity = -1; capacity <= 16; capacity++) {
        void *builder = nl_sb_with_capacity(capacity); assert(builder);
        assert(nl_sb_length(builder) == 0 && nl_sb_capacity(builder) == 16);
        nl_sb_append(builder, NULL); nl_sb_append(builder, ""); text_equals(builder, "");
        nl_sb_append(builder, "123456789012345"); assert(nl_sb_capacity(builder) == 16);
        const char *before = nl_sb_to_string(builder); assert(before);
        nl_sb_append_char(builder, 'x'); assert(nl_sb_length(builder) == 16 && nl_sb_capacity(builder) >= 17);
        text_equals(builder, "123456789012345x"); assert(!strcmp(before, "123456789012345"));
        nl_sb_clear(builder); assert(nl_sb_length(builder) == 0); text_equals(builder, "");
        nl_sb_append(builder, "caf\303\251"); assert(nl_sb_length(builder) == 5); text_equals(builder, "caf\303\251");
        nl_sb_free(builder); assert(!strcmp(before, "123456789012345")); probe_free((void *)before);
    }
    void *minimum = nl_sb_with_capacity(INT64_MIN); assert(minimum && nl_sb_capacity(minimum) == 16); nl_sb_free(minimum);
    void *builder = nl_sb_with_capacity(17); assert(builder && nl_sb_capacity(builder) == 17);
    char *large = malloc(1024 * 1024 + 2); assert(large); memset(large, 'a', 1024 * 1024 + 1); large[1024 * 1024 + 1] = 0;
    nl_sb_append(builder, large); assert(nl_sb_length(builder) == 1024 * 1024);
    const char *copy = nl_sb_to_string(builder); assert(copy && strlen(copy) == 1024 * 1024);
    nl_sb_free(builder); assert(copy[0] == 'a' && copy[1024 * 1024 - 1] == 'a'); probe_free((void *)copy); free(large);
    assert(!collection_handles && live_allocations == 0);
}
static void allocations(void) {
    for (int constructor = 0; constructor < 2; constructor++) {
        for (int point = 0; point < 3; point++) {
            fail_at = point; allocation_index = 0;
            void *builder = constructor ? nl_sb_with_capacity(16) : nl_sb_new();
            fail_at = -1; assert(!builder && !collection_handles && live_allocations == 0);
            builder = nl_sb_new(); assert(builder); nl_sb_free(builder);
        }
    }
    refuse_large = 1; refused_large_size = 0;
    assert(!nl_sb_with_capacity(INT64_MAX)); refuse_large = 0;
#if SIZE_MAX >= INT64_MAX
    assert(refused_large_size == (size_t)INT64_MAX);
#endif
    assert(!collection_handles && live_allocations == 0);
    void *builder = nl_sb_new(); assert(builder); nl_sb_append(builder, "owned");
    fail_at = 0; allocation_index = 0;
    const char *fallback = nl_sb_to_string(builder); fail_at = -1;
    assert(fallback && !strcmp(fallback, "")); probe_free((void *)fallback); text_equals(builder, "owned");
    fail_strdup = 1; strdup_calls = 0; assert(!nl_sb_to_string(builder)); assert(strdup_calls == 2);
    strdup_calls = 0; assert(!nl_sb_to_string(NULL)); assert(strdup_calls == 1); fail_strdup = 0;
    text_equals(builder, "owned"); nl_sb_free(builder); assert(!collection_handles && live_allocations == 0);
}
static void *worker(void *unused) {
    (void)unused;
    for (int i = 0; i < 200; i++) {
        void *builder = nl_sb_with_capacity(16); assert(builder);
        nl_sb_append(builder, "independent"); nl_sb_append_char(builder, '!'); text_equals(builder, "independent!");
        nl_sb_free(builder);
    }
    return NULL;
}
static void threads(void) {
    pthread_t pool[4];
    for (int i = 0; i < 4; i++) assert(!pthread_create(&pool[i], NULL, worker, NULL));
    for (int i = 0; i < 4; i++) assert(!pthread_join(pool[i], NULL));
    assert(!collection_handles && live_allocations == 0);
}
static void exhaustion(void) {
    collection_next_identity = UINTPTR_MAX;
    void *last = nl_sb_new(); assert(last && (uintptr_t)last == UINTPTR_MAX);
    assert(!nl_sb_new() && !nl_sb_with_capacity(1) && !nl_hm_new() && !nl_set_new());
    nl_sb_append(last, "last"); text_equals(last, "last"); nl_sb_free(last); refused(last);
    assert(!collection_handles && !live_allocations);
}
static void *growth_refusal_builder;
static void growth_refusal_state(void) {
    NLStringBuilder *payload = collection_lookup(growth_refusal_builder, COLLECTION_BUILDER, false);
    assert(payload && payload->len == 15 && payload->cap == 16);
    assert(!strcmp(payload->buf, "123456789012345"));
    nl_sb_free(growth_refusal_builder); assert(!collection_handles && !live_allocations);
    fputs("I retain prior Builder contents on growth refusal.\n", stderr);
}
int main(int argc, char **argv) {
    if (argc == 2) {
        void *builder = nl_sb_with_capacity(16); assert(builder);
        if (!strcmp(argv[1], "growth-oom")) {
            nl_sb_append(builder, "123456789012345"); growth_refusal_builder = builder;
            assert(!atexit(growth_refusal_state)); fail_at = 0; allocation_index = 0;
            nl_sb_append_char(builder, 'x');
        } else if (!strcmp(argv[1], "overflow")) {
            NLStringBuilder *payload = collection_lookup(builder, COLLECTION_BUILDER, false);
            assert(payload); payload->len = INT64_MAX; nl_sb_append_char(builder, 'x');
        } else return 2;
        return 3;
    }
    lifetime(); contents(); allocations(); threads(); exhaustion();
    puts("I pass checked Builder identity, lifetime, snapshots, growth, allocation, exhaustion and independent-thread controls.");
    return 0;
}
