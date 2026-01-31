#ifndef LIST_ASTNUMBER_H
#define LIST_ASTNUMBER_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTNumber */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTNumber
#define FORWARD_DEFINED_List_ASTNumber
typedef struct List_ASTNumber {
    struct nl_ASTNumber *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTNumber;
#endif

/* Create a new empty list */
List_ASTNumber* nl_list_ASTNumber_new(void);

/* Create a new list with specified initial capacity */
List_ASTNumber* nl_list_ASTNumber_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTNumber_push(List_ASTNumber *list, struct nl_ASTNumber value);

/* Remove and return the last element */
struct nl_ASTNumber nl_list_ASTNumber_pop(List_ASTNumber *list);

/* Insert an element at the specified index */
void nl_list_ASTNumber_insert(List_ASTNumber *list, int index, struct nl_ASTNumber value);

/* Remove and return the element at the specified index */
struct nl_ASTNumber nl_list_ASTNumber_remove(List_ASTNumber *list, int index);

/* Set the value at the specified index */
void nl_list_ASTNumber_set(List_ASTNumber *list, int index, struct nl_ASTNumber value);

/* Get the value at the specified index */
struct nl_ASTNumber nl_list_ASTNumber_get(List_ASTNumber *list, int index);

/* Clear all elements from the list */
void nl_list_ASTNumber_clear(List_ASTNumber *list);

/* Get the current length of the list */
int nl_list_ASTNumber_length(List_ASTNumber *list);

/* Get the current capacity of the list */
int nl_list_ASTNumber_capacity(List_ASTNumber *list);

/* Check if the list is empty */
bool nl_list_ASTNumber_is_empty(List_ASTNumber *list);

/* Free the list and all its resources */
void nl_list_ASTNumber_free(List_ASTNumber *list);

#endif /* LIST_ASTNUMBER_H */
