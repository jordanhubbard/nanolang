#include "list_CompilerDiagnostic.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Ensure capacity for new elements */
static void nl_list_CompilerDiagnostic_ensure_capacity(List_CompilerDiagnostic *list, int required_capacity) {
    if (required_capacity <= list->capacity) {
        return;
    }

    int new_capacity = list->capacity == 0 ? INITIAL_CAPACITY : list->capacity;
    while (new_capacity < required_capacity) {
        new_capacity *= GROWTH_FACTOR;
    }

    struct nl_CompilerDiagnostic *new_data = realloc(list->data, sizeof(struct nl_CompilerDiagnostic) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }

    list->data = new_data;
    list->capacity = new_capacity;
}

List_CompilerDiagnostic* nl_list_CompilerDiagnostic_new(void) {
    List_CompilerDiagnostic *list = malloc(sizeof(List_CompilerDiagnostic));
    list->data = NULL;
    list->length = 0;
    list->capacity = 0;
    return list;
}

List_CompilerDiagnostic* nl_list_CompilerDiagnostic_with_capacity(int capacity) {
    List_CompilerDiagnostic *list = malloc(sizeof(List_CompilerDiagnostic));
    list->data = malloc(sizeof(struct nl_CompilerDiagnostic) * capacity);
    list->length = 0;
    list->capacity = capacity;
    return list;
}

void nl_list_CompilerDiagnostic_push(List_CompilerDiagnostic *list, struct nl_CompilerDiagnostic value) {
    nl_list_CompilerDiagnostic_ensure_capacity(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_pop(List_CompilerDiagnostic *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

void nl_list_CompilerDiagnostic_insert(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    
    nl_list_CompilerDiagnostic_ensure_capacity(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_CompilerDiagnostic) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_remove(List_CompilerDiagnostic *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    
    struct nl_CompilerDiagnostic value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_CompilerDiagnostic) * (list->length - index - 1));
    
    list->length--;
    return value;
}

void nl_list_CompilerDiagnostic_set(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    list->data[index] = value;
}

struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_get(List_CompilerDiagnostic *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    return list->data[index];
}

void nl_list_CompilerDiagnostic_clear(List_CompilerDiagnostic *list) {
    list->length = 0;
}

int nl_list_CompilerDiagnostic_length(List_CompilerDiagnostic *list) {
    return list->length;
}

int nl_list_CompilerDiagnostic_capacity(List_CompilerDiagnostic *list) {
    return list->capacity;
}

bool nl_list_CompilerDiagnostic_is_empty(List_CompilerDiagnostic *list) {
    return list->length == 0;
}

void nl_list_CompilerDiagnostic_free(List_CompilerDiagnostic *list) {
    if (list->data) {
        free(list->data);
    }
    free(list);
}
