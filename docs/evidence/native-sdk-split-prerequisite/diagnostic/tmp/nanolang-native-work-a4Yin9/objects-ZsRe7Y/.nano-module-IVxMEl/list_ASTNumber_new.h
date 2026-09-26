#ifndef LIST_ASTNUMBER_NEW_H
#define LIST_ASTNUMBER_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTNumber_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTNumber_new
#define DEFINED_List_ASTNumber_new
typedef struct List_ASTNumber_new {
    ASTNumber_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTNumber_new;
#endif

/* Create a new empty list */
List_ASTNumber_new* nl_list_ASTNumber_new_new(void);

/* Create a new list with specified initial capacity */
List_ASTNumber_new* nl_list_ASTNumber_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTNumber_new_push(List_ASTNumber_new *list, ASTNumber_new value);

/* Remove and return the last element */
ASTNumber_new nl_list_ASTNumber_new_pop(List_ASTNumber_new *list);

/* Insert an element at the specified index */
void nl_list_ASTNumber_new_insert(List_ASTNumber_new *list, int index, ASTNumber_new value);

/* Remove and return the element at the specified index */
ASTNumber_new nl_list_ASTNumber_new_remove(List_ASTNumber_new *list, int index);

/* Set the value at the specified index */
void nl_list_ASTNumber_new_set(List_ASTNumber_new *list, int index, ASTNumber_new value);

/* Get the value at the specified index */
ASTNumber_new nl_list_ASTNumber_new_get(List_ASTNumber_new *list, int index);

/* Clear all elements from the list */
void nl_list_ASTNumber_new_clear(List_ASTNumber_new *list);

/* Get the current length of the list */
int nl_list_ASTNumber_new_length(List_ASTNumber_new *list);

/* Get the current capacity of the list */
int nl_list_ASTNumber_new_capacity(List_ASTNumber_new *list);

/* Check if the list is empty */
bool nl_list_ASTNumber_new_is_empty(List_ASTNumber_new *list);

/* Free the list and all its resources */
void nl_list_ASTNumber_new_free(List_ASTNumber_new *list);

#endif /* LIST_ASTNUMBER_NEW_H */
