#ifndef LIST_ASTSTRING_NEW_H
#define LIST_ASTSTRING_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTString_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTString_new
#define DEFINED_List_ASTString_new
typedef struct List_ASTString_new {
    ASTString_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTString_new;
#endif

/* Create a new empty list */
List_ASTString_new* nl_list_ASTString_new_new(void);

/* Create a new list with specified initial capacity */
List_ASTString_new* nl_list_ASTString_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTString_new_push(List_ASTString_new *list, ASTString_new value);

/* Remove and return the last element */
ASTString_new nl_list_ASTString_new_pop(List_ASTString_new *list);

/* Insert an element at the specified index */
void nl_list_ASTString_new_insert(List_ASTString_new *list, int index, ASTString_new value);

/* Remove and return the element at the specified index */
ASTString_new nl_list_ASTString_new_remove(List_ASTString_new *list, int index);

/* Set the value at the specified index */
void nl_list_ASTString_new_set(List_ASTString_new *list, int index, ASTString_new value);

/* Get the value at the specified index */
ASTString_new nl_list_ASTString_new_get(List_ASTString_new *list, int index);

/* Clear all elements from the list */
void nl_list_ASTString_new_clear(List_ASTString_new *list);

/* Get the current length of the list */
int nl_list_ASTString_new_length(List_ASTString_new *list);

/* Get the current capacity of the list */
int nl_list_ASTString_new_capacity(List_ASTString_new *list);

/* Check if the list is empty */
bool nl_list_ASTString_new_is_empty(List_ASTString_new *list);

/* Free the list and all its resources */
void nl_list_ASTString_new_free(List_ASTString_new *list);

#endif /* LIST_ASTSTRING_NEW_H */
