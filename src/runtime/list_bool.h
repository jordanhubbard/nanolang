#ifndef LIST_BOOL_H
#define LIST_BOOL_H

#include <stdbool.h>

/* Dynamic list of boolean values (List<bool>) */
typedef struct {
    bool *data;          /* Array of elements */
    int length;          /* Current number of elements */
    int capacity;        /* Allocated capacity */
} List_bool;

/* Create a new empty list */
List_bool* list_bool_new(void);

/* Create a new list with specified initial capacity */
List_bool* list_bool_with_capacity(int capacity);

/* Append an element to the end of the list */
void list_bool_push(List_bool *list, bool value);

/* Remove and return the last element */
bool list_bool_pop(List_bool *list);

/* Insert an element at the specified index */
void list_bool_insert(List_bool *list, int index, bool value);

/* Remove and return the element at the specified index */
bool list_bool_remove(List_bool *list, int index);

/* Set the value at the specified index */
void list_bool_set(List_bool *list, int index, bool value);

/* Get the value at the specified index */
bool list_bool_get(List_bool *list, int index);

/* Clear all elements from the list */
void list_bool_clear(List_bool *list);

/* Get the current length of the list */
int list_bool_length(List_bool *list);

/* Get the current capacity of the list */
int list_bool_capacity(List_bool *list);

/* Check if the list is empty */
bool list_bool_is_empty(List_bool *list);

/* Free the list and all its resources */
void list_bool_free(List_bool *list);

#endif /* LIST_BOOL_H */
