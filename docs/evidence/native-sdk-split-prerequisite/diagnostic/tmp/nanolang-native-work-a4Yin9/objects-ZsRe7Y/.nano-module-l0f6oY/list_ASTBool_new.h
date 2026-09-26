#ifndef LIST_ASTBOOL_NEW_H
#define LIST_ASTBOOL_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBool_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTBool_new
#define DEFINED_List_ASTBool_new
typedef struct List_ASTBool_new {
    ASTBool_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBool_new;
#endif

/* Create a new empty list */
List_ASTBool_new* nl_list_ASTBool_new_new(void);

/* Create a new list with specified initial capacity */
List_ASTBool_new* nl_list_ASTBool_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBool_new_push(List_ASTBool_new *list, ASTBool_new value);

/* Remove and return the last element */
ASTBool_new nl_list_ASTBool_new_pop(List_ASTBool_new *list);

/* Insert an element at the specified index */
void nl_list_ASTBool_new_insert(List_ASTBool_new *list, int index, ASTBool_new value);

/* Remove and return the element at the specified index */
ASTBool_new nl_list_ASTBool_new_remove(List_ASTBool_new *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBool_new_set(List_ASTBool_new *list, int index, ASTBool_new value);

/* Get the value at the specified index */
ASTBool_new nl_list_ASTBool_new_get(List_ASTBool_new *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBool_new_clear(List_ASTBool_new *list);

/* Get the current length of the list */
int nl_list_ASTBool_new_length(List_ASTBool_new *list);

/* Get the current capacity of the list */
int nl_list_ASTBool_new_capacity(List_ASTBool_new *list);

/* Check if the list is empty */
bool nl_list_ASTBool_new_is_empty(List_ASTBool_new *list);

/* Free the list and all its resources */
void nl_list_ASTBool_new_free(List_ASTBool_new *list);

#endif /* LIST_ASTBOOL_NEW_H */
