#ifndef LIST_ASTNUMBER_PUSH_H
#define LIST_ASTNUMBER_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTNumber_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTNumber_push
#define DEFINED_List_ASTNumber_push
typedef struct List_ASTNumber_push {
    ASTNumber_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTNumber_push;
#endif

/* Create a new empty list */
List_ASTNumber_push* nl_list_ASTNumber_push_new(void);

/* Create a new list with specified initial capacity */
List_ASTNumber_push* nl_list_ASTNumber_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTNumber_push_push(List_ASTNumber_push *list, ASTNumber_push value);

/* Remove and return the last element */
ASTNumber_push nl_list_ASTNumber_push_pop(List_ASTNumber_push *list);

/* Insert an element at the specified index */
void nl_list_ASTNumber_push_insert(List_ASTNumber_push *list, int index, ASTNumber_push value);

/* Remove and return the element at the specified index */
ASTNumber_push nl_list_ASTNumber_push_remove(List_ASTNumber_push *list, int index);

/* Set the value at the specified index */
void nl_list_ASTNumber_push_set(List_ASTNumber_push *list, int index, ASTNumber_push value);

/* Get the value at the specified index */
ASTNumber_push nl_list_ASTNumber_push_get(List_ASTNumber_push *list, int index);

/* Clear all elements from the list */
void nl_list_ASTNumber_push_clear(List_ASTNumber_push *list);

/* Get the current length of the list */
int nl_list_ASTNumber_push_length(List_ASTNumber_push *list);

/* Get the current capacity of the list */
int nl_list_ASTNumber_push_capacity(List_ASTNumber_push *list);

/* Check if the list is empty */
bool nl_list_ASTNumber_push_is_empty(List_ASTNumber_push *list);

/* Free the list and all its resources */
void nl_list_ASTNumber_push_free(List_ASTNumber_push *list);

#endif /* LIST_ASTNUMBER_PUSH_H */
