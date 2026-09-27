#ifndef LIST_ASTBOOL_GET_H
#define LIST_ASTBOOL_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBool_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTBool_get
#define DEFINED_List_ASTBool_get
typedef struct List_ASTBool_get {
    ASTBool_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBool_get;
#endif

/* Create a new empty list */
List_ASTBool_get* nl_list_ASTBool_get_new(void);

/* Create a new list with specified initial capacity */
List_ASTBool_get* nl_list_ASTBool_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBool_get_push(List_ASTBool_get *list, ASTBool_get value);

/* Remove and return the last element */
ASTBool_get nl_list_ASTBool_get_pop(List_ASTBool_get *list);

/* Insert an element at the specified index */
void nl_list_ASTBool_get_insert(List_ASTBool_get *list, int index, ASTBool_get value);

/* Remove and return the element at the specified index */
ASTBool_get nl_list_ASTBool_get_remove(List_ASTBool_get *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBool_get_set(List_ASTBool_get *list, int index, ASTBool_get value);

/* Get the value at the specified index */
ASTBool_get nl_list_ASTBool_get_get(List_ASTBool_get *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBool_get_clear(List_ASTBool_get *list);

/* Get the current length of the list */
int nl_list_ASTBool_get_length(List_ASTBool_get *list);

/* Get the current capacity of the list */
int nl_list_ASTBool_get_capacity(List_ASTBool_get *list);

/* Check if the list is empty */
bool nl_list_ASTBool_get_is_empty(List_ASTBool_get *list);

/* Free the list and all its resources */
void nl_list_ASTBool_get_free(List_ASTBool_get *list);

#endif /* LIST_ASTBOOL_GET_H */
