#ifndef NL_LIST_CAPACITY_H
#define NL_LIST_CAPACITY_H

#include <limits.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* I retain the list API's fail-fast contract, with checked integer arithmetic. */
static inline void nl_list_validate_capacity(int capacity, size_t element_size) {
    if (capacity < 0 || element_size == 0 || (size_t)capacity > SIZE_MAX / element_size) {
        fprintf(stderr, "I cannot represent this list capacity.\n");
        exit(1);
    }
}

static inline int nl_list_next_length(int length) {
    if (length < 0 || length == INT_MAX) {
        fprintf(stderr, "I cannot grow this list length.\n");
        exit(1);
    }
    return length + 1;
}

static inline int nl_list_grown_capacity(int current, int required, size_t element_size) {
    nl_list_validate_capacity(current, element_size);
    nl_list_validate_capacity(required, element_size);
    if (current >= required) return current;
    int capacity = current ? current : 8;
    if ((size_t)capacity > SIZE_MAX / element_size) capacity = required;
    while (capacity < required) {
        if (capacity > INT_MAX / 2 || (size_t)capacity > (SIZE_MAX / element_size) / 2) {
            capacity = required;
            break;
        }
        capacity *= 2;
    }
    return capacity;
}

#endif
