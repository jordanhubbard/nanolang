#ifndef LIST_ASTFLOAT_GET_H
#define LIST_ASTFLOAT_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFloat_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFloat_get
#define DEFINED_List_ASTFloat_get
typedef struct List_ASTFloat_get {
    ASTFloat_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFloat_get;
#endif

/* Create a new empty list */
List_ASTFloat_get* nl_list_ASTFloat_get_new(void);

/* Create a new list with specified initial capacity */
List_ASTFloat_get* nl_list_ASTFloat_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFloat_get_push(List_ASTFloat_get *list, ASTFloat_get value);

/* Remove and return the last element */
ASTFloat_get nl_list_ASTFloat_get_pop(List_ASTFloat_get *list);

/* Insert an element at the specified index */
void nl_list_ASTFloat_get_insert(List_ASTFloat_get *list, int index, ASTFloat_get value);

/* Remove and return the element at the specified index */
ASTFloat_get nl_list_ASTFloat_get_remove(List_ASTFloat_get *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFloat_get_set(List_ASTFloat_get *list, int index, ASTFloat_get value);

/* Get the value at the specified index */
ASTFloat_get nl_list_ASTFloat_get_get(List_ASTFloat_get *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFloat_get_clear(List_ASTFloat_get *list);

/* Get the current length of the list */
int nl_list_ASTFloat_get_length(List_ASTFloat_get *list);

/* Get the current capacity of the list */
int nl_list_ASTFloat_get_capacity(List_ASTFloat_get *list);

/* Check if the list is empty */
bool nl_list_ASTFloat_get_is_empty(List_ASTFloat_get *list);

/* Free the list and all its resources */
void nl_list_ASTFloat_get_free(List_ASTFloat_get *list);

#endif /* LIST_ASTFLOAT_GET_H */
