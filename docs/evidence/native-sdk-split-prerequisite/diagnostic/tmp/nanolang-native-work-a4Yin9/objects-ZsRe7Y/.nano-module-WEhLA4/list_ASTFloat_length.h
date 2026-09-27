#ifndef LIST_ASTFLOAT_LENGTH_H
#define LIST_ASTFLOAT_LENGTH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFloat_length */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFloat_length
#define DEFINED_List_ASTFloat_length
typedef struct List_ASTFloat_length {
    ASTFloat_length *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFloat_length;
#endif

/* Create a new empty list */
List_ASTFloat_length* nl_list_ASTFloat_length_new(void);

/* Create a new list with specified initial capacity */
List_ASTFloat_length* nl_list_ASTFloat_length_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFloat_length_push(List_ASTFloat_length *list, ASTFloat_length value);

/* Remove and return the last element */
ASTFloat_length nl_list_ASTFloat_length_pop(List_ASTFloat_length *list);

/* Insert an element at the specified index */
void nl_list_ASTFloat_length_insert(List_ASTFloat_length *list, int index, ASTFloat_length value);

/* Remove and return the element at the specified index */
ASTFloat_length nl_list_ASTFloat_length_remove(List_ASTFloat_length *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFloat_length_set(List_ASTFloat_length *list, int index, ASTFloat_length value);

/* Get the value at the specified index */
ASTFloat_length nl_list_ASTFloat_length_get(List_ASTFloat_length *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFloat_length_clear(List_ASTFloat_length *list);

/* Get the current length of the list */
int nl_list_ASTFloat_length_length(List_ASTFloat_length *list);

/* Get the current capacity of the list */
int nl_list_ASTFloat_length_capacity(List_ASTFloat_length *list);

/* Check if the list is empty */
bool nl_list_ASTFloat_length_is_empty(List_ASTFloat_length *list);

/* Free the list and all its resources */
void nl_list_ASTFloat_length_free(List_ASTFloat_length *list);

#endif /* LIST_ASTFLOAT_LENGTH_H */
