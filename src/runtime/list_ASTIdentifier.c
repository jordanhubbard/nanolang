#include "list_ASTIdentifier.h"
#include "../generated/compiler_schema.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "list_capacity.h"

/* Note: The actual struct nl_ASTIdentifier definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_ASTIdentifier(List_ASTIdentifier *list, int min_capacity) {
    int new_capacity = nl_list_grown_capacity(list->capacity, min_capacity, sizeof(*list->data));
    if (new_capacity == list->capacity) return;
    
    struct nl_ASTIdentifier *new_data = realloc(list->data, sizeof(struct nl_ASTIdentifier) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_ASTIdentifier* nl_list_ASTIdentifier_new(void) {
    return nl_list_ASTIdentifier_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_ASTIdentifier* nl_list_ASTIdentifier_with_capacity(int capacity) {
    nl_list_validate_capacity(capacity, sizeof(*((List_ASTIdentifier *)0)->data));
    List_ASTIdentifier *list = malloc(sizeof(List_ASTIdentifier));
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
void nl_list_ASTIdentifier_push(List_ASTIdentifier *list, struct nl_ASTIdentifier value) {
    ensure_capacity_ASTIdentifier(list, nl_list_next_length(list->length));
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
struct nl_ASTIdentifier nl_list_ASTIdentifier_pop(List_ASTIdentifier *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_ASTIdentifier_insert(List_ASTIdentifier *list, int index, struct nl_ASTIdentifier value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_ASTIdentifier(list, nl_list_next_length(list->length));
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(struct nl_ASTIdentifier) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
struct nl_ASTIdentifier nl_list_ASTIdentifier_remove(List_ASTIdentifier *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    struct nl_ASTIdentifier value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(struct nl_ASTIdentifier) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_ASTIdentifier_set(List_ASTIdentifier *list, int index, struct nl_ASTIdentifier value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
struct nl_ASTIdentifier nl_list_ASTIdentifier_get(List_ASTIdentifier *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_ASTIdentifier_clear(List_ASTIdentifier *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_ASTIdentifier_length(List_ASTIdentifier *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_ASTIdentifier_capacity(List_ASTIdentifier *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_ASTIdentifier_is_empty(List_ASTIdentifier *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_ASTIdentifier_free(List_ASTIdentifier *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
