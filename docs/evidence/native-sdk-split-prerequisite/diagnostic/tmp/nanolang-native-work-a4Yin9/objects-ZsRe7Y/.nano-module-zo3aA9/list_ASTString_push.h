#ifndef LIST_ASTSTRING_PUSH_H
#define LIST_ASTSTRING_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTString_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTString_push
#define DEFINED_List_ASTString_push
typedef struct List_ASTString_push {
    ASTString_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTString_push;
#endif

/* Create a new empty list */
List_ASTString_push* nl_list_ASTString_push_new(void);

/* Create a new list with specified initial capacity */
List_ASTString_push* nl_list_ASTString_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTString_push_push(List_ASTString_push *list, ASTString_push value);

/* Remove and return the last element */
ASTString_push nl_list_ASTString_push_pop(List_ASTString_push *list);

/* Insert an element at the specified index */
void nl_list_ASTString_push_insert(List_ASTString_push *list, int index, ASTString_push value);

/* Remove and return the element at the specified index */
ASTString_push nl_list_ASTString_push_remove(List_ASTString_push *list, int index);

/* Set the value at the specified index */
void nl_list_ASTString_push_set(List_ASTString_push *list, int index, ASTString_push value);

/* Get the value at the specified index */
ASTString_push nl_list_ASTString_push_get(List_ASTString_push *list, int index);

/* Clear all elements from the list */
void nl_list_ASTString_push_clear(List_ASTString_push *list);

/* Get the current length of the list */
int nl_list_ASTString_push_length(List_ASTString_push *list);

/* Get the current capacity of the list */
int nl_list_ASTString_push_capacity(List_ASTString_push *list);

/* Check if the list is empty */
bool nl_list_ASTString_push_is_empty(List_ASTString_push *list);

/* Free the list and all its resources */
void nl_list_ASTString_push_free(List_ASTString_push *list);

#endif /* LIST_ASTSTRING_PUSH_H */
