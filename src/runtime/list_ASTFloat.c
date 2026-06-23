#include "list_ASTFloat.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_ASTFloat definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTFloat(List_ASTFloat *list, int min_capacity) {
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
    
    struct nl_ASTFloat *new_data = realloc(list->data, sizeof(struct nl_ASTFloat) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTFloat* nl_list_ASTFloat_new(void) {
    return nl_list_ASTFloat_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTFloat* nl_list_ASTFloat_with_capacity(int capacity) {
    List_ASTFloat *list = malloc(sizeof(List_ASTFloat));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(struct nl_ASTFloat) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_ASTFloat_push(List_ASTFloat *list, struct nl_ASTFloat value) {
    ensure_capacity_ASTFloat(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTFloat nl_list_ASTFloat_pop(List_ASTFloat *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTFloat_insert(List_ASTFloat *list, int index, struct nl_ASTFloat value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTFloat(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTFloat) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTFloat nl_list_ASTFloat_remove(List_ASTFloat *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTFloat value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTFloat) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTFloat_set(List_ASTFloat *list, int index, struct nl_ASTFloat value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTFloat nl_list_ASTFloat_get(List_ASTFloat *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTFloat_clear(List_ASTFloat *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTFloat_length(List_ASTFloat *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTFloat_capacity(List_ASTFloat *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTFloat_is_empty(List_ASTFloat *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTFloat_free(List_ASTFloat *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
