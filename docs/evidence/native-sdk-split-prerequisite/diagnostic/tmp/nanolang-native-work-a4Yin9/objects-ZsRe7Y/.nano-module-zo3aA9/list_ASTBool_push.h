#ifndef LIST_ASTBOOL_PUSH_H
#define LIST_ASTBOOL_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBool_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTBool_push
#define DEFINED_List_ASTBool_push
typedef struct List_ASTBool_push {
    ASTBool_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBool_push;
#endif

/* Create a new empty list */
List_ASTBool_push* nl_list_ASTBool_push_new(void);

/* Create a new list with specified initial capacity */
List_ASTBool_push* nl_list_ASTBool_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBool_push_push(List_ASTBool_push *list, ASTBool_push value);

/* Remove and return the last element */
ASTBool_push nl_list_ASTBool_push_pop(List_ASTBool_push *list);

/* Insert an element at the specified index */
void nl_list_ASTBool_push_insert(List_ASTBool_push *list, int index, ASTBool_push value);

/* Remove and return the element at the specified index */
ASTBool_push nl_list_ASTBool_push_remove(List_ASTBool_push *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBool_push_set(List_ASTBool_push *list, int index, ASTBool_push value);

/* Get the value at the specified index */
ASTBool_push nl_list_ASTBool_push_get(List_ASTBool_push *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBool_push_clear(List_ASTBool_push *list);

/* Get the current length of the list */
int nl_list_ASTBool_push_length(List_ASTBool_push *list);

/* Get the current capacity of the list */
int nl_list_ASTBool_push_capacity(List_ASTBool_push *list);

/* Check if the list is empty */
bool nl_list_ASTBool_push_is_empty(List_ASTBool_push *list);

/* Free the list and all its resources */
void nl_list_ASTBool_push_free(List_ASTBool_push *list);

#endif /* LIST_ASTBOOL_PUSH_H */
