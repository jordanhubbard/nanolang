#ifndef LIST_ASTSHADOW_H
#define LIST_ASTSHADOW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTShadow */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTShadow
#define DEFINED_List_ASTShadow
typedef struct List_ASTShadow {
    struct nl_ASTShadow *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTShadow;
#endif

/* Create a new empty list */
List_ASTShadow* nl_list_ASTShadow_new(void);

/* Create a new list with specified initial capacity */
List_ASTShadow* nl_list_ASTShadow_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTShadow_push(List_ASTShadow *list, struct nl_ASTShadow value);

/* Remove and return the last element */
struct nl_ASTShadow nl_list_ASTShadow_pop(List_ASTShadow *list);

/* Insert an element at the specified index */
void nl_list_ASTShadow_insert(List_ASTShadow *list, int index, struct nl_ASTShadow value);

/* Remove and return the element at the specified index */
struct nl_ASTShadow nl_list_ASTShadow_remove(List_ASTShadow *list, int index);

/* Set the value at the specified index */
void nl_list_ASTShadow_set(List_ASTShadow *list, int index, struct nl_ASTShadow value);

/* Get the value at the specified index */
struct nl_ASTShadow nl_list_ASTShadow_get(List_ASTShadow *list, int index);

/* Clear all elements from the list */
void nl_list_ASTShadow_clear(List_ASTShadow *list);

/* Get the current length of the list */
int nl_list_ASTShadow_length(List_ASTShadow *list);

/* Get the current capacity of the list */
int nl_list_ASTShadow_capacity(List_ASTShadow *list);

/* Check if the list is empty */
bool nl_list_ASTShadow_is_empty(List_ASTShadow *list);

/* Free the list and all its resources */
void nl_list_ASTShadow_free(List_ASTShadow *list);

#endif /* LIST_ASTSHADOW_H */
