#ifndef LIST_INT_H
#define LIST_INT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of integers (int64_t) */
typedef struct {
    int64_t *data;       /* Array of elements */
    int length;          /* Current number of elements */
    int capacity;        /* Allocated capacity */
} List_int;

/* Create a new empty list */
List_int* list_int_new(void);

/* Create a new list with specified initial capacity */
List_int* list_int_with_capacity(int capacity);

/* Append an element to the end of the list */
void list_int_push(List_int *list, int64_t value);

/* Remove and return the last element */
int64_t list_int_pop(List_int *list);

/* Insert an element at the specified index */
void list_int_insert(List_int *list, int index, int64_t value);

/* Remove and return the element at the specified index */
int64_t list_int_remove(List_int *list, int index);

/* Set the value at the specified index */
void list_int_set(List_int *list, int index, int64_t value);

/* Get the value at the specified index */
int64_t list_int_get(List_int *list, int index);

/* Clear all elements from the list */
void list_int_clear(List_int *list);

/* Get the current length of the list */
int list_int_length(List_int *list);

/* Get the current capacity of the list */
int list_int_capacity(List_int *list);

/* Check if the list is empty */
bool list_int_is_empty(List_int *list);

/* Free the list and all its resources */
void list_int_free(List_int *list);

#endif /* LIST_INT_H */

