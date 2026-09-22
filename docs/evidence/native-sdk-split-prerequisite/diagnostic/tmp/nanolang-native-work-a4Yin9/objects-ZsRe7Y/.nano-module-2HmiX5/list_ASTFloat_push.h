#ifndef LIST_ASTFLOAT_PUSH_H
#define LIST_ASTFLOAT_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFloat_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFloat_push
#define DEFINED_List_ASTFloat_push
typedef struct List_ASTFloat_push {
    ASTFloat_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFloat_push;
#endif

/* Create a new empty list */
List_ASTFloat_push* nl_list_ASTFloat_push_new(void);

/* Create a new list with specified initial capacity */
List_ASTFloat_push* nl_list_ASTFloat_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFloat_push_push(List_ASTFloat_push *list, ASTFloat_push value);

/* Remove and return the last element */
ASTFloat_push nl_list_ASTFloat_push_pop(List_ASTFloat_push *list);

/* Insert an element at the specified index */
void nl_list_ASTFloat_push_insert(List_ASTFloat_push *list, int index, ASTFloat_push value);

/* Remove and return the element at the specified index */
ASTFloat_push nl_list_ASTFloat_push_remove(List_ASTFloat_push *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFloat_push_set(List_ASTFloat_push *list, int index, ASTFloat_push value);

/* Get the value at the specified index */
ASTFloat_push nl_list_ASTFloat_push_get(List_ASTFloat_push *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFloat_push_clear(List_ASTFloat_push *list);

/* Get the current length of the list */
int nl_list_ASTFloat_push_length(List_ASTFloat_push *list);

/* Get the current capacity of the list */
int nl_list_ASTFloat_push_capacity(List_ASTFloat_push *list);

/* Check if the list is empty */
bool nl_list_ASTFloat_push_is_empty(List_ASTFloat_push *list);

/* Free the list and all its resources */
void nl_list_ASTFloat_push_free(List_ASTFloat_push *list);

#endif /* LIST_ASTFLOAT_PUSH_H */
