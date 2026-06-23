#ifndef LIST_ASTPRINT_H
#define LIST_ASTPRINT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTPrint */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTPrint
#define FORWARD_DEFINED_List_ASTPrint
typedef struct List_ASTPrint {
    struct nl_ASTPrint *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTPrint;
#endif

/* Create a new empty list */
List_ASTPrint* nl_list_ASTPrint_new(void);

/* Create a new list with specified initial capacity */
List_ASTPrint* nl_list_ASTPrint_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTPrint_push(List_ASTPrint *list, struct nl_ASTPrint value);

/* Remove and return the last element */
struct nl_ASTPrint nl_list_ASTPrint_pop(List_ASTPrint *list);

/* Insert an element at the specified index */
void nl_list_ASTPrint_insert(List_ASTPrint *list, int index, struct nl_ASTPrint value);

/* Remove and return the element at the specified index */
struct nl_ASTPrint nl_list_ASTPrint_remove(List_ASTPrint *list, int index);

/* Set the value at the specified index */
void nl_list_ASTPrint_set(List_ASTPrint *list, int index, struct nl_ASTPrint value);

/* Get the value at the specified index */
struct nl_ASTPrint nl_list_ASTPrint_get(List_ASTPrint *list, int index);

/* Clear all elements from the list */
void nl_list_ASTPrint_clear(List_ASTPrint *list);

/* Get the current length of the list */
int nl_list_ASTPrint_length(List_ASTPrint *list);

/* Get the current capacity of the list */
int nl_list_ASTPrint_capacity(List_ASTPrint *list);

/* Check if the list is empty */
bool nl_list_ASTPrint_is_empty(List_ASTPrint *list);

/* Free the list and all its resources */
void nl_list_ASTPrint_free(List_ASTPrint *list);

#endif /* LIST_ASTPRINT_H */
