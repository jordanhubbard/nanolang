#ifndef LIST_ASTSTMTREF_H
#define LIST_ASTSTMTREF_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStmtRef */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTStmtRef
#define FORWARD_DEFINED_List_ASTStmtRef
typedef struct List_ASTStmtRef {
    struct nl_ASTStmtRef *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStmtRef;
#endif

/* Create a new empty list */
List_ASTStmtRef* nl_list_ASTStmtRef_new(void);

/* Create a new list with specified initial capacity */
List_ASTStmtRef* nl_list_ASTStmtRef_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStmtRef_push(List_ASTStmtRef *list, struct nl_ASTStmtRef value);

/* Remove and return the last element */
struct nl_ASTStmtRef nl_list_ASTStmtRef_pop(List_ASTStmtRef *list);

/* Insert an element at the specified index */
void nl_list_ASTStmtRef_insert(List_ASTStmtRef *list, int index, struct nl_ASTStmtRef value);

/* Remove and return the element at the specified index */
struct nl_ASTStmtRef nl_list_ASTStmtRef_remove(List_ASTStmtRef *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStmtRef_set(List_ASTStmtRef *list, int index, struct nl_ASTStmtRef value);

/* Get the value at the specified index */
struct nl_ASTStmtRef nl_list_ASTStmtRef_get(List_ASTStmtRef *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStmtRef_clear(List_ASTStmtRef *list);

/* Get the current length of the list */
int nl_list_ASTStmtRef_length(List_ASTStmtRef *list);

/* Get the current capacity of the list */
int nl_list_ASTStmtRef_capacity(List_ASTStmtRef *list);

/* Check if the list is empty */
bool nl_list_ASTStmtRef_is_empty(List_ASTStmtRef *list);

/* Free the list and all its resources */
void nl_list_ASTStmtRef_free(List_ASTStmtRef *list);

#endif /* LIST_ASTSTMTREF_H */
