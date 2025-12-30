/* Schema type list implementations for self-hosted compiler */
#include "schema_lists.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* ============================================================================
 * List<ASTTupleIndex> Implementation
 * ============================================================================ */

static void nl_list_ASTTupleIndex_ensure_capacity(List_ASTTupleIndex *list, int required_capacity) {
    if (required_capacity <= list->capacity) {
        return;
    }

    int new_capacity = list->capacity == 0 ? INITIAL_CAPACITY : list->capacity;
    while (new_capacity < required_capacity) {
        new_capacity *= GROWTH_FACTOR;
    }

    nl_ASTTupleIndex *new_data = realloc(list->data, sizeof(nl_ASTTupleIndex) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }

    list->data = new_data;
    list->capacity = new_capacity;
}

List_ASTTupleIndex* nl_list_ASTTupleIndex_new(void) {
    List_ASTTupleIndex *list = malloc(sizeof(List_ASTTupleIndex));
    list->data = NULL;
    list->length = 0;
    list->capacity = 0;
    return list;
}

void nl_list_ASTTupleIndex_push(List_ASTTupleIndex *list, nl_ASTTupleIndex value) {
    nl_list_ASTTupleIndex_ensure_capacity(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

nl_ASTTupleIndex nl_list_ASTTupleIndex_get(List_ASTTupleIndex *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    return list->data[index];
}

int nl_list_ASTTupleIndex_length(List_ASTTupleIndex *list) {
    return list->length;
}

void nl_list_ASTTupleIndex_free(List_ASTTupleIndex *list) {
    if (list->data) {
        free(list->data);
    }
    free(list);
}

/* ============================================================================
 * List<CompilerDiagnostic> Implementation
 * ============================================================================ */

static void nl_list_CompilerDiagnostic_ensure_capacity(List_CompilerDiagnostic *list, int required_capacity) {
    if (required_capacity <= list->capacity) {
        return;
    }

    int new_capacity = list->capacity == 0 ? INITIAL_CAPACITY : list->capacity;
    while (new_capacity < required_capacity) {
        new_capacity *= GROWTH_FACTOR;
    }

    nl_CompilerDiagnostic *new_data = realloc(list->data, sizeof(nl_CompilerDiagnostic) * new_capacity);
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

void nl_list_CompilerDiagnostic_push(List_CompilerDiagnostic *list, nl_CompilerDiagnostic value) {
    nl_list_CompilerDiagnostic_ensure_capacity(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

nl_CompilerDiagnostic nl_list_CompilerDiagnostic_get(List_CompilerDiagnostic *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(1);
    }
    return list->data[index];
}

int nl_list_CompilerDiagnostic_length(List_CompilerDiagnostic *list) {
    return list->length;
}

void nl_list_CompilerDiagnostic_free(List_CompilerDiagnostic *list) {
    if (list->data) {
        free(list->data);
    }
    free(list);
}

