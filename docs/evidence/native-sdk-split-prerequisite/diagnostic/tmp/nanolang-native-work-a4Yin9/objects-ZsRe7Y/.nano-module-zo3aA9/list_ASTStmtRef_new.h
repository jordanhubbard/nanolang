#ifndef LIST_ASTSTMTREF_NEW_H
#define LIST_ASTSTMTREF_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStmtRef_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStmtRef_new
#define DEFINED_List_ASTStmtRef_new
typedef struct List_ASTStmtRef_new {
    ASTStmtRef_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStmtRef_new;
#endif

/* Create a new empty list */
List_ASTStmtRef_new* nl_list_ASTStmtRef_new_new(void);

/* Create a new list with specified initial capacity */
List_ASTStmtRef_new* nl_list_ASTStmtRef_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStmtRef_new_push(List_ASTStmtRef_new *list, ASTStmtRef_new value);

/* Remove and return the last element */
ASTStmtRef_new nl_list_ASTStmtRef_new_pop(List_ASTStmtRef_new *list);

/* Insert an element at the specified index */
void nl_list_ASTStmtRef_new_insert(List_ASTStmtRef_new *list, int index, ASTStmtRef_new value);

/* Remove and return the element at the specified index */
ASTStmtRef_new nl_list_ASTStmtRef_new_remove(List_ASTStmtRef_new *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStmtRef_new_set(List_ASTStmtRef_new *list, int index, ASTStmtRef_new value);

/* Get the value at the specified index */
ASTStmtRef_new nl_list_ASTStmtRef_new_get(List_ASTStmtRef_new *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStmtRef_new_clear(List_ASTStmtRef_new *list);

/* Get the current length of the list */
int nl_list_ASTStmtRef_new_length(List_ASTStmtRef_new *list);

/* Get the current capacity of the list */
int nl_list_ASTStmtRef_new_capacity(List_ASTStmtRef_new *list);

/* Check if the list is empty */
bool nl_list_ASTStmtRef_new_is_empty(List_ASTStmtRef_new *list);

/* Free the list and all its resources */
void nl_list_ASTStmtRef_new_free(List_ASTStmtRef_new *list);

#endif /* LIST_ASTSTMTREF_NEW_H */
