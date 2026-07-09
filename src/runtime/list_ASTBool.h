#ifndef LIST_ASTBOOL_H
#define LIST_ASTBOOL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBool */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTBool
#define FORWARD_DEFINED_List_ASTBool
typedef struct List_ASTBool {
    struct nl_ASTBool *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBool;
#endif

/* Create a new empty list */
List_ASTBool* nl_list_ASTBool_new(void);

/* Create a new list with specified initial capacity */
List_ASTBool* nl_list_ASTBool_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBool_push(List_ASTBool *list, struct nl_ASTBool value);

/* Remove and return the last element */
struct nl_ASTBool nl_list_ASTBool_pop(List_ASTBool *list);

/* Insert an element at the specified index */
void nl_list_ASTBool_insert(List_ASTBool *list, int index, struct nl_ASTBool value);

/* Remove and return the element at the specified index */
struct nl_ASTBool nl_list_ASTBool_remove(List_ASTBool *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBool_set(List_ASTBool *list, int index, struct nl_ASTBool value);

/* Get the value at the specified index */
struct nl_ASTBool nl_list_ASTBool_get(List_ASTBool *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBool_clear(List_ASTBool *list);

/* Get the current length of the list */
int nl_list_ASTBool_length(List_ASTBool *list);

/* Get the current capacity of the list */
int nl_list_ASTBool_capacity(List_ASTBool *list);

/* Check if the list is empty */
bool nl_list_ASTBool_is_empty(List_ASTBool *list);

/* Free the list and all its resources */
void nl_list_ASTBool_free(List_ASTBool *list);

#endif /* LIST_ASTBOOL_H */
