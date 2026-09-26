#ifndef LIST_ASTSTMTREF_PUSH_H
#define LIST_ASTSTMTREF_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStmtRef_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStmtRef_push
#define DEFINED_List_ASTStmtRef_push
typedef struct List_ASTStmtRef_push {
    ASTStmtRef_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStmtRef_push;
#endif

/* Create a new empty list */
List_ASTStmtRef_push* nl_list_ASTStmtRef_push_new(void);

/* Create a new list with specified initial capacity */
List_ASTStmtRef_push* nl_list_ASTStmtRef_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStmtRef_push_push(List_ASTStmtRef_push *list, ASTStmtRef_push value);

/* Remove and return the last element */
ASTStmtRef_push nl_list_ASTStmtRef_push_pop(List_ASTStmtRef_push *list);

/* Insert an element at the specified index */
void nl_list_ASTStmtRef_push_insert(List_ASTStmtRef_push *list, int index, ASTStmtRef_push value);

/* Remove and return the element at the specified index */
ASTStmtRef_push nl_list_ASTStmtRef_push_remove(List_ASTStmtRef_push *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStmtRef_push_set(List_ASTStmtRef_push *list, int index, ASTStmtRef_push value);

/* Get the value at the specified index */
ASTStmtRef_push nl_list_ASTStmtRef_push_get(List_ASTStmtRef_push *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStmtRef_push_clear(List_ASTStmtRef_push *list);

/* Get the current length of the list */
int nl_list_ASTStmtRef_push_length(List_ASTStmtRef_push *list);

/* Get the current capacity of the list */
int nl_list_ASTStmtRef_push_capacity(List_ASTStmtRef_push *list);

/* Check if the list is empty */
bool nl_list_ASTStmtRef_push_is_empty(List_ASTStmtRef_push *list);

/* Free the list and all its resources */
void nl_list_ASTStmtRef_push_free(List_ASTStmtRef_push *list);

#endif /* LIST_ASTSTMTREF_PUSH_H */
