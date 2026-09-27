#ifndef LIST_ASTSTMTREF_LENGTH_H
#define LIST_ASTSTMTREF_LENGTH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStmtRef_length */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStmtRef_length
#define DEFINED_List_ASTStmtRef_length
typedef struct List_ASTStmtRef_length {
    ASTStmtRef_length *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStmtRef_length;
#endif

/* Create a new empty list */
List_ASTStmtRef_length* nl_list_ASTStmtRef_length_new(void);

/* Create a new list with specified initial capacity */
List_ASTStmtRef_length* nl_list_ASTStmtRef_length_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStmtRef_length_push(List_ASTStmtRef_length *list, ASTStmtRef_length value);

/* Remove and return the last element */
ASTStmtRef_length nl_list_ASTStmtRef_length_pop(List_ASTStmtRef_length *list);

/* Insert an element at the specified index */
void nl_list_ASTStmtRef_length_insert(List_ASTStmtRef_length *list, int index, ASTStmtRef_length value);

/* Remove and return the element at the specified index */
ASTStmtRef_length nl_list_ASTStmtRef_length_remove(List_ASTStmtRef_length *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStmtRef_length_set(List_ASTStmtRef_length *list, int index, ASTStmtRef_length value);

/* Get the value at the specified index */
ASTStmtRef_length nl_list_ASTStmtRef_length_get(List_ASTStmtRef_length *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStmtRef_length_clear(List_ASTStmtRef_length *list);

/* Get the current length of the list */
int nl_list_ASTStmtRef_length_length(List_ASTStmtRef_length *list);

/* Get the current capacity of the list */
int nl_list_ASTStmtRef_length_capacity(List_ASTStmtRef_length *list);

/* Check if the list is empty */
bool nl_list_ASTStmtRef_length_is_empty(List_ASTStmtRef_length *list);

/* Free the list and all its resources */
void nl_list_ASTStmtRef_length_free(List_ASTStmtRef_length *list);

#endif /* LIST_ASTSTMTREF_LENGTH_H */
