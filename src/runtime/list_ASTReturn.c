#include "list_ASTReturn.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_ASTReturn definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTReturn(List_ASTReturn *list, int min_capacity) {
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
    
    struct nl_ASTReturn *new_data = realloc(list->data, sizeof(struct nl_ASTReturn) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTReturn* nl_list_ASTReturn_new(void) {
    return nl_list_ASTReturn_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTReturn* nl_list_ASTReturn_with_capacity(int capacity) {
    List_ASTReturn *list = malloc(sizeof(List_ASTReturn));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(struct nl_ASTReturn) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_ASTReturn_push(List_ASTReturn *list, struct nl_ASTReturn value) {
    ensure_capacity_ASTReturn(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTReturn nl_list_ASTReturn_pop(List_ASTReturn *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTReturn_insert(List_ASTReturn *list, int index, struct nl_ASTReturn value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTReturn(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTReturn) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTReturn nl_list_ASTReturn_remove(List_ASTReturn *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTReturn value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTReturn) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTReturn_set(List_ASTReturn *list, int index, struct nl_ASTReturn value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTReturn nl_list_ASTReturn_get(List_ASTReturn *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTReturn_clear(List_ASTReturn *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTReturn_length(List_ASTReturn *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTReturn_capacity(List_ASTReturn *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTReturn_is_empty(List_ASTReturn *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTReturn_free(List_ASTReturn *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
