#ifndef NL_NATIVE_RECORD_LIST_H
#define NL_NATIVE_RECORD_LIST_H

#include "list_capacity.h"
#include <stdbool.h>
#include <string.h>

/* I copy native record values. Their pointer fields retain their existing
 * native lifetime; this container does not acquire ownership of pointees. */
static inline void nl_record_list_fail(void) {
    fprintf(stderr, "I cannot complete this list operation: invalid bounds, storage, or allocation.\n");
    exit(1);
}

/* I check source-width arguments before an existing provider's int ABI. */
static inline int nl_native_list_index(int64_t index, int64_t length, bool insertion) {
    if (length < 0 || length > INT_MAX || index < 0 || index > length ||
        (!insertion && index == length)) nl_record_list_fail();
    return (int)index;
}

static inline int nl_native_list_capacity(int64_t capacity) {
    if (capacity < 0 || capacity > INT_MAX) nl_record_list_fail();
    return (int)capacity;
}

/* The producer supplies an exact declaration key and its complete C type.
 * A preceding typedef permits records to contain references to this list. */
#define NL_DEFINE_RECORD_LIST(KEY, ELEMENT) \
struct List_##KEY { ELEMENT *data; int count; int capacity; }; \
static inline void nl_list_##KEY##_validate(const List_##KEY *list) { \
    if (!list || !list->data || list->count < 0 || \
        list->capacity < 4 || list->count > list->capacity) \
        nl_record_list_fail(); \
    nl_list_validate_capacity(list->capacity, sizeof(ELEMENT)); \
} \
static inline int nl_list_##KEY##_index(const List_##KEY *list, int64_t index, bool insertion) { \
    nl_list_##KEY##_validate(list); \
    if (index < 0 || index > list->count || (!insertion && index == list->count)) \
        nl_record_list_fail(); \
    return (int)index; \
} \
static inline void nl_list_##KEY##_reserve(List_##KEY *list, int required) { \
    nl_list_##KEY##_validate(list); \
    int capacity = nl_list_grown_capacity(list->capacity, required, sizeof(ELEMENT)); \
    if (capacity != list->capacity) { \
        ELEMENT *data = (ELEMENT *)realloc(list->data, (size_t)capacity * sizeof(ELEMENT)); \
        if (!data) nl_record_list_fail(); \
        list->data = data; \
        list->capacity = capacity; \
    } \
} \
List_##KEY *nl_list_##KEY##_new(void) { \
    nl_list_validate_capacity(4, sizeof(ELEMENT)); \
    List_##KEY *list = (List_##KEY *)malloc(sizeof(*list)); \
    if (!list) nl_record_list_fail(); \
    ELEMENT *data = (ELEMENT *)malloc(4 * sizeof(ELEMENT)); \
    if (!data) { free(list); nl_record_list_fail(); } \
    list->data = data; list->count = 0; list->capacity = 4; \
    return list; \
} \
void nl_list_##KEY##_push(List_##KEY *list, ELEMENT value) { \
    nl_list_##KEY##_validate(list); \
    int length = nl_list_next_length(list->count); \
    nl_list_##KEY##_reserve(list, length); \
    list->data[list->count] = value; list->count = length; \
} \
ELEMENT nl_list_##KEY##_get(List_##KEY *list, int64_t index) { \
    int slot = nl_list_##KEY##_index(list, index, false); \
    return list->data[slot]; \
} \
void nl_list_##KEY##_set(List_##KEY *list, int64_t index, ELEMENT value) { \
    int slot = nl_list_##KEY##_index(list, index, false); \
    list->data[slot] = value; \
} \
void nl_list_##KEY##_insert(List_##KEY *list, int64_t index, ELEMENT value) { \
    int slot = nl_list_##KEY##_index(list, index, true); \
    int length = nl_list_next_length(list->count); \
    nl_list_##KEY##_reserve(list, length); \
    memmove(list->data + slot + 1, list->data + slot, \
            (size_t)(list->count - slot) * sizeof(ELEMENT)); \
    list->data[slot] = value; list->count = length; \
} \
ELEMENT nl_list_##KEY##_remove(List_##KEY *list, int64_t index) { \
    int slot = nl_list_##KEY##_index(list, index, false); \
    ELEMENT value = list->data[slot]; \
    memmove(list->data + slot, list->data + slot + 1, \
            (size_t)(list->count - slot - 1) * sizeof(ELEMENT)); \
    list->count--; list->data[list->count] = (ELEMENT){0}; \
    return value; \
} \
ELEMENT nl_list_##KEY##_pop(List_##KEY *list) { \
    nl_list_##KEY##_validate(list); \
    return nl_list_##KEY##_remove(list, (int64_t)list->count - 1); \
} \
int64_t nl_list_##KEY##_length(List_##KEY *list) { \
    nl_list_##KEY##_validate(list); return list->count; \
} \
int64_t nl_list_##KEY##_capacity(List_##KEY *list) { \
    nl_list_##KEY##_validate(list); return list->capacity; \
} \
bool nl_list_##KEY##_is_empty(List_##KEY *list) { \
    nl_list_##KEY##_validate(list); return list->count == 0; \
} \
void nl_list_##KEY##_clear(List_##KEY *list) { \
    nl_list_##KEY##_validate(list); \
    for (int i = 0; i < list->count; i++) list->data[i] = (ELEMENT){0}; \
    list->count = 0; \
} \
void nl_list_##KEY##_free(List_##KEY *list) { \
    if (!list) return; \
    nl_list_##KEY##_validate(list); free(list->data); free(list); \
}

#endif
