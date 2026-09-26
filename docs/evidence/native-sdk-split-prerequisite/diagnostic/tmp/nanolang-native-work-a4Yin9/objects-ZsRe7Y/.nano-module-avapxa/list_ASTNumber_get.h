#ifndef LIST_ASTNUMBER_GET_H
#define LIST_ASTNUMBER_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTNumber_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTNumber_get
#define DEFINED_List_ASTNumber_get
typedef struct List_ASTNumber_get {
    ASTNumber_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTNumber_get;
#endif

/* Create a new empty list */
List_ASTNumber_get* nl_list_ASTNumber_get_new(void);

/* Create a new list with specified initial capacity */
List_ASTNumber_get* nl_list_ASTNumber_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTNumber_get_push(List_ASTNumber_get *list, ASTNumber_get value);

/* Remove and return the last element */
ASTNumber_get nl_list_ASTNumber_get_pop(List_ASTNumber_get *list);

/* Insert an element at the specified index */
void nl_list_ASTNumber_get_insert(List_ASTNumber_get *list, int index, ASTNumber_get value);

/* Remove and return the element at the specified index */
ASTNumber_get nl_list_ASTNumber_get_remove(List_ASTNumber_get *list, int index);

/* Set the value at the specified index */
void nl_list_ASTNumber_get_set(List_ASTNumber_get *list, int index, ASTNumber_get value);

/* Get the value at the specified index */
ASTNumber_get nl_list_ASTNumber_get_get(List_ASTNumber_get *list, int index);

/* Clear all elements from the list */
void nl_list_ASTNumber_get_clear(List_ASTNumber_get *list);

/* Get the current length of the list */
int nl_list_ASTNumber_get_length(List_ASTNumber_get *list);

/* Get the current capacity of the list */
int nl_list_ASTNumber_get_capacity(List_ASTNumber_get *list);

/* Check if the list is empty */
bool nl_list_ASTNumber_get_is_empty(List_ASTNumber_get *list);

/* Free the list and all its resources */
void nl_list_ASTNumber_get_free(List_ASTNumber_get *list);

#endif /* LIST_ASTNUMBER_GET_H */
