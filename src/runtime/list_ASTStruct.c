#include "list_ASTStruct.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "list_capacity.h"

/* Note: The actual struct nl_ASTStruct definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTStruct(List_ASTStruct *list, int min_capacity) {
    int new_capacity = nl_list_grown_capacity(list->capacity, min_capacity, sizeof(*list->data));
    if (new_capacity == list->capacity) return;
    
    struct nl_ASTStruct *new_data = realloc(list->data, sizeof(struct nl_ASTStruct) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTStruct* nl_list_ASTStruct_new(void) {
    return nl_list_ASTStruct_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTStruct* nl_list_ASTStruct_with_capacity(int capacity) {
    nl_list_validate_capacity(capacity, sizeof(*((List_ASTStruct *)0)->data));
    List_ASTStruct *list = malloc(sizeof(List_ASTStruct));
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
void nl_list_ASTStruct_push(List_ASTStruct *list, struct nl_ASTStruct value) {
    ensure_capacity_ASTStruct(list, nl_list_next_length(list->length));
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTStruct nl_list_ASTStruct_pop(List_ASTStruct *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTStruct_insert(List_ASTStruct *list, int index, struct nl_ASTStruct value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTStruct(list, nl_list_next_length(list->length));
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTStruct) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTStruct nl_list_ASTStruct_remove(List_ASTStruct *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTStruct value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTStruct) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTStruct_set(List_ASTStruct *list, int index, struct nl_ASTStruct value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTStruct nl_list_ASTStruct_get(List_ASTStruct *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTStruct_clear(List_ASTStruct *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTStruct_length(List_ASTStruct *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTStruct_capacity(List_ASTStruct *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTStruct_is_empty(List_ASTStruct *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTStruct_free(List_ASTStruct *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
