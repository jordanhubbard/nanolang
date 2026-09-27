#ifndef LIST_ASTSTMTREF_GET_H
#define LIST_ASTSTMTREF_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStmtRef_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStmtRef_get
#define DEFINED_List_ASTStmtRef_get
typedef struct List_ASTStmtRef_get {
    ASTStmtRef_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStmtRef_get;
#endif

/* Create a new empty list */
List_ASTStmtRef_get* nl_list_ASTStmtRef_get_new(void);

/* Create a new list with specified initial capacity */
List_ASTStmtRef_get* nl_list_ASTStmtRef_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStmtRef_get_push(List_ASTStmtRef_get *list, ASTStmtRef_get value);

/* Remove and return the last element */
ASTStmtRef_get nl_list_ASTStmtRef_get_pop(List_ASTStmtRef_get *list);

/* Insert an element at the specified index */
void nl_list_ASTStmtRef_get_insert(List_ASTStmtRef_get *list, int index, ASTStmtRef_get value);

/* Remove and return the element at the specified index */
ASTStmtRef_get nl_list_ASTStmtRef_get_remove(List_ASTStmtRef_get *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStmtRef_get_set(List_ASTStmtRef_get *list, int index, ASTStmtRef_get value);

/* Get the value at the specified index */
ASTStmtRef_get nl_list_ASTStmtRef_get_get(List_ASTStmtRef_get *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStmtRef_get_clear(List_ASTStmtRef_get *list);

/* Get the current length of the list */
int nl_list_ASTStmtRef_get_length(List_ASTStmtRef_get *list);

/* Get the current capacity of the list */
int nl_list_ASTStmtRef_get_capacity(List_ASTStmtRef_get *list);

/* Check if the list is empty */
bool nl_list_ASTStmtRef_get_is_empty(List_ASTStmtRef_get *list);

/* Free the list and all its resources */
void nl_list_ASTStmtRef_get_free(List_ASTStmtRef_get *list);

#endif /* LIST_ASTSTMTREF_GET_H */
