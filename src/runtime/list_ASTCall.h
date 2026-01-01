#ifndef LIST_ASTCALL_H
#define LIST_ASTCALL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTCall */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTCall
#define FORWARD_DEFINED_List_ASTCall
typedef struct List_ASTCall {
    struct nl_ASTCall *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTCall;
#endif

/* Create a new empty list */
List_ASTCall* nl_list_ASTCall_new(void);

/* Create a new list with specified initial capacity */
List_ASTCall* nl_list_ASTCall_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTCall_push(List_ASTCall *list, struct nl_ASTCall value);

/* Remove and return the last element */
struct nl_ASTCall nl_list_ASTCall_pop(List_ASTCall *list);

/* Insert an element at the specified index */
void nl_list_ASTCall_insert(List_ASTCall *list, int index, struct nl_ASTCall value);

/* Remove and return the element at the specified index */
struct nl_ASTCall nl_list_ASTCall_remove(List_ASTCall *list, int index);

/* Set the value at the specified index */
void nl_list_ASTCall_set(List_ASTCall *list, int index, struct nl_ASTCall value);

/* Get the value at the specified index */
struct nl_ASTCall nl_list_ASTCall_get(List_ASTCall *list, int index);

/* Clear all elements from the list */
void nl_list_ASTCall_clear(List_ASTCall *list);

/* Get the current length of the list */
int nl_list_ASTCall_length(List_ASTCall *list);

/* Get the current capacity of the list */
int nl_list_ASTCall_capacity(List_ASTCall *list);

/* Check if the list is empty */
bool nl_list_ASTCall_is_empty(List_ASTCall *list);

/* Free the list and all its resources */
void nl_list_ASTCall_free(List_ASTCall *list);

#endif /* LIST_ASTCALL_H */
