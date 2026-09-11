#include "list_CompilerSourceLocation.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "list_capacity.h"

/* Note: The actual struct nl_CompilerSourceLocation definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_CompilerSourceLocation(List_CompilerSourceLocation *list, int min_capacity) {
    int new_capacity = nl_list_grown_capacity(list->capacity, min_capacity, sizeof(*list->data));
    if (new_capacity == list->capacity) return;
    
    struct nl_CompilerSourceLocation *new_data = realloc(list->data, sizeof(struct nl_CompilerSourceLocation) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_CompilerSourceLocation* nl_list_CompilerSourceLocation_new(void) {
    return nl_list_CompilerSourceLocation_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_CompilerSourceLocation* nl_list_CompilerSourceLocation_with_capacity(int capacity) {
    nl_list_validate_capacity(capacity, sizeof(*((List_CompilerSourceLocation *)0)->data));
    List_CompilerSourceLocation *list = malloc(sizeof(List_CompilerSourceLocation));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = capacity ? malloc(sizeof(*list->data) * (size_t)capacity) : NULL;
    if (capacity && !list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_CompilerSourceLocation_push(List_CompilerSourceLocation *list, struct nl_CompilerSourceLocation value) {
    ensure_capacity_CompilerSourceLocation(list, nl_list_next_length(list->length));
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_pop(List_CompilerSourceLocation *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_CompilerSourceLocation_insert(List_CompilerSourceLocation *list, int index, struct nl_CompilerSourceLocation value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_CompilerSourceLocation(list, nl_list_next_length(list->length));
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_CompilerSourceLocation) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_remove(List_CompilerSourceLocation *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_CompilerSourceLocation value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_CompilerSourceLocation) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_CompilerSourceLocation_set(List_CompilerSourceLocation *list, int index, struct nl_CompilerSourceLocation value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_get(List_CompilerSourceLocation *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_CompilerSourceLocation_clear(List_CompilerSourceLocation *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_CompilerSourceLocation_length(List_CompilerSourceLocation *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_CompilerSourceLocation_capacity(List_CompilerSourceLocation *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_CompilerSourceLocation_is_empty(List_CompilerSourceLocation *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_CompilerSourceLocation_free(List_CompilerSourceLocation *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
