#include "list_CompilerDiagnostic_get.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>

/* Note: The actual struct nl_CompilerDiagnostic_get definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_CompilerDiagnostic_get(List_CompilerDiagnostic_get *list, int min_capacity) {
    if (min_capacity < 0 || list->capacity < 0 ||
        (size_t)min_capacity > SIZE_MAX / sizeof(CompilerDiagnostic_get)) {
        fprintf(stderr, "I cannot represent this list capacity.\n");
        exit(1);
    }
    if (list->capacity >= min_capacity) {
        return;
    }
    
    int new_capacity = list->capacity;
    if (new_capacity == 0) {
        new_capacity = INITIAL_CAPACITY;
    }
    if ((size_t)new_capacity > SIZE_MAX / sizeof(CompilerDiagnostic_get)) {
        new_capacity = min_capacity;
    }
    
    while (new_capacity < min_capacity) {
        if (new_capacity > INT_MAX / GROWTH_FACTOR ||
            (size_t)new_capacity > (SIZE_MAX / sizeof(CompilerDiagnostic_get)) / GROWTH_FACTOR) {
            new_capacity = min_capacity;
            break;
        }
        new_capacity *= GROWTH_FACTOR;
    }
    
    CompilerDiagnostic_get *new_data = realloc(list->data, sizeof(CompilerDiagnostic_get) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_CompilerDiagnostic_get* nl_list_CompilerDiagnostic_get_new(void) {
    return nl_list_CompilerDiagnostic_get_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic_get* nl_list_CompilerDiagnostic_get_with_capacity(int capacity) {
    if (capacity < 0 || (size_t)capacity > SIZE_MAX / sizeof(CompilerDiagnostic_get)) {
        fprintf(stderr, "I cannot represent this list capacity.\n");
        exit(1);
    }
    List_CompilerDiagnostic_get *list = malloc(sizeof(List_CompilerDiagnostic_get));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = capacity ? malloc(sizeof(CompilerDiagnostic_get) * (size_t)capacity) : NULL;
    if (capacity && !list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_get_push(List_CompilerDiagnostic_get *list, CompilerDiagnostic_get value) {
    if (list->length < 0 || list->length == INT_MAX) {
        fprintf(stderr, "I cannot grow this list length.\n");
        exit(1);
    }
    ensure_capacity_CompilerDiagnostic_get(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_pop(List_CompilerDiagnostic_get *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_get_insert(List_CompilerDiagnostic_get *list, int index, CompilerDiagnostic_get value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    if (list->length == INT_MAX) {
        fprintf(stderr, "I cannot grow this list length.\n");
        exit(1);
    }
    ensure_capacity_CompilerDiagnostic_get(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(CompilerDiagnostic_get) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_remove(List_CompilerDiagnostic_get *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    CompilerDiagnostic_get value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(CompilerDiagnostic_get) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_get_set(List_CompilerDiagnostic_get *list, int index, CompilerDiagnostic_get value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_get(List_CompilerDiagnostic_get *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_get_clear(List_CompilerDiagnostic_get *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_get_length(List_CompilerDiagnostic_get *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_get_capacity(List_CompilerDiagnostic_get *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_get_is_empty(List_CompilerDiagnostic_get *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_get_free(List_CompilerDiagnostic_get *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
