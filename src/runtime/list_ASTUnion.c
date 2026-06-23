#include "list_ASTUnion.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_ASTUnion definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTUnion(List_ASTUnion *list, int min_capacity) {
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
    
    struct nl_ASTUnion *new_data = realloc(list->data, sizeof(struct nl_ASTUnion) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTUnion* nl_list_ASTUnion_new(void) {
    return nl_list_ASTUnion_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTUnion* nl_list_ASTUnion_with_capacity(int capacity) {
    List_ASTUnion *list = malloc(sizeof(List_ASTUnion));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(struct nl_ASTUnion) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_ASTUnion_push(List_ASTUnion *list, struct nl_ASTUnion value) {
    ensure_capacity_ASTUnion(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTUnion nl_list_ASTUnion_pop(List_ASTUnion *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTUnion_insert(List_ASTUnion *list, int index, struct nl_ASTUnion value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTUnion(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTUnion) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTUnion nl_list_ASTUnion_remove(List_ASTUnion *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTUnion value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTUnion) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTUnion_set(List_ASTUnion *list, int index, struct nl_ASTUnion value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTUnion nl_list_ASTUnion_get(List_ASTUnion *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTUnion_clear(List_ASTUnion *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTUnion_length(List_ASTUnion *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTUnion_capacity(List_ASTUnion *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTUnion_is_empty(List_ASTUnion *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTUnion_free(List_ASTUnion *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
