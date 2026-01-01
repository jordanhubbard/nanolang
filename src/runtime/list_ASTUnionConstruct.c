#include "list_ASTUnionConstruct.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_ASTUnionConstruct definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTUnionConstruct(List_ASTUnionConstruct *list, int min_capacity) {
    if (list->capacity >= min_capacity) {
        return;
    }
    
    int new_capacity = list->capacity;
    if (new_capacity == 0) {
        new_capacity = INITIAL_CAPACITY;
    }
    
    while (new_capacity < min_capacity) {
        new_capacity *= GROWTH_FACTOR;
    }
    
    struct nl_ASTUnionConstruct *new_data = realloc(list->data, sizeof(struct nl_ASTUnionConstruct) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTUnionConstruct* nl_list_ASTUnionConstruct_new(void) {
    return nl_list_ASTUnionConstruct_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTUnionConstruct* nl_list_ASTUnionConstruct_with_capacity(int capacity) {
    List_ASTUnionConstruct *list = malloc(sizeof(List_ASTUnionConstruct));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(struct nl_ASTUnionConstruct) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_ASTUnionConstruct_push(List_ASTUnionConstruct *list, struct nl_ASTUnionConstruct value) {
    ensure_capacity_ASTUnionConstruct(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_pop(List_ASTUnionConstruct *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTUnionConstruct_insert(List_ASTUnionConstruct *list, int index, struct nl_ASTUnionConstruct value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTUnionConstruct(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTUnionConstruct) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_remove(List_ASTUnionConstruct *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTUnionConstruct value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTUnionConstruct) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTUnionConstruct_set(List_ASTUnionConstruct *list, int index, struct nl_ASTUnionConstruct value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_get(List_ASTUnionConstruct *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTUnionConstruct_clear(List_ASTUnionConstruct *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTUnionConstruct_length(List_ASTUnionConstruct *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTUnionConstruct_capacity(List_ASTUnionConstruct *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTUnionConstruct_is_empty(List_ASTUnionConstruct *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTUnionConstruct_free(List_ASTUnionConstruct *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
