#include "list_ASTShadow.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_ASTShadow definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTShadow(List_ASTShadow *list, int min_capacity) {
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
    
    struct nl_ASTShadow *new_data = realloc(list->data, sizeof(struct nl_ASTShadow) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTShadow* nl_list_ASTShadow_new(void) {
    return nl_list_ASTShadow_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTShadow* nl_list_ASTShadow_with_capacity(int capacity) {
    List_ASTShadow *list = malloc(sizeof(List_ASTShadow));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(struct nl_ASTShadow) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_ASTShadow_push(List_ASTShadow *list, struct nl_ASTShadow value) {
    ensure_capacity_ASTShadow(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTShadow nl_list_ASTShadow_pop(List_ASTShadow *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTShadow_insert(List_ASTShadow *list, int index, struct nl_ASTShadow value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTShadow(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTShadow) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTShadow nl_list_ASTShadow_remove(List_ASTShadow *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTShadow value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTShadow) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTShadow_set(List_ASTShadow *list, int index, struct nl_ASTShadow value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTShadow nl_list_ASTShadow_get(List_ASTShadow *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTShadow_clear(List_ASTShadow *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTShadow_length(List_ASTShadow *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTShadow_capacity(List_ASTShadow *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTShadow_is_empty(List_ASTShadow *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTShadow_free(List_ASTShadow *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
