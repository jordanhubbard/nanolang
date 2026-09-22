#ifndef LIST_ASTFLOAT_NEW_H
#define LIST_ASTFLOAT_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFloat_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFloat_new
#define DEFINED_List_ASTFloat_new
typedef struct List_ASTFloat_new {
    ASTFloat_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFloat_new;
#endif

/* Create a new empty list */
List_ASTFloat_new* nl_list_ASTFloat_new_new(void);

/* Create a new list with specified initial capacity */
List_ASTFloat_new* nl_list_ASTFloat_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFloat_new_push(List_ASTFloat_new *list, ASTFloat_new value);

/* Remove and return the last element */
ASTFloat_new nl_list_ASTFloat_new_pop(List_ASTFloat_new *list);

/* Insert an element at the specified index */
void nl_list_ASTFloat_new_insert(List_ASTFloat_new *list, int index, ASTFloat_new value);

/* Remove and return the element at the specified index */
ASTFloat_new nl_list_ASTFloat_new_remove(List_ASTFloat_new *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFloat_new_set(List_ASTFloat_new *list, int index, ASTFloat_new value);

/* Get the value at the specified index */
ASTFloat_new nl_list_ASTFloat_new_get(List_ASTFloat_new *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFloat_new_clear(List_ASTFloat_new *list);

/* Get the current length of the list */
int nl_list_ASTFloat_new_length(List_ASTFloat_new *list);

/* Get the current capacity of the list */
int nl_list_ASTFloat_new_capacity(List_ASTFloat_new *list);

/* Check if the list is empty */
bool nl_list_ASTFloat_new_is_empty(List_ASTFloat_new *list);

/* Free the list and all its resources */
void nl_list_ASTFloat_new_free(List_ASTFloat_new *list);

#endif /* LIST_ASTFLOAT_NEW_H */
