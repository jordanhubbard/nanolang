#include "list_int.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity(List_int *list, int min_capacity) {
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
    
    int64_t *new_data = realloc(list->data, sizeof(int64_t) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_int* list_int_new(void) {
    return list_int_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_int* list_int_with_capacity(int capacity) {
    List_int *list = malloc(sizeof(List_int));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(int64_t) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void list_int_push(List_int *list, int64_t value) {
    ensure_capacity(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
int64_t list_int_pop(List_int *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void list_int_insert(List_int *list, int index, int64_t value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(int64_t) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
int64_t list_int_remove(List_int *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    int64_t value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(int64_t) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void list_int_set(List_int *list, int index, int64_t value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
int64_t list_int_get(List_int *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void list_int_clear(List_int *list) {
    list->length = 0;
}

/* Get the current length of the list */
int list_int_length(List_int *list) {
    return list->length;
}

/* Get the current capacity of the list */
int list_int_capacity(List_int *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool list_int_is_empty(List_int *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void list_int_free(List_int *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}

