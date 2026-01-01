#ifndef LIST_ASTUNION_H
#define LIST_ASTUNION_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTUnion */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTUnion
#define DEFINED_List_ASTUnion
typedef struct List_ASTUnion {
    struct nl_ASTUnion *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTUnion;
#endif

/* Create a new empty list */
List_ASTUnion* nl_list_ASTUnion_new(void);

/* Create a new list with specified initial capacity */
List_ASTUnion* nl_list_ASTUnion_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTUnion_push(List_ASTUnion *list, struct nl_ASTUnion value);

/* Remove and return the last element */
struct nl_ASTUnion nl_list_ASTUnion_pop(List_ASTUnion *list);

/* Insert an element at the specified index */
void nl_list_ASTUnion_insert(List_ASTUnion *list, int index, struct nl_ASTUnion value);

/* Remove and return the element at the specified index */
struct nl_ASTUnion nl_list_ASTUnion_remove(List_ASTUnion *list, int index);

/* Set the value at the specified index */
void nl_list_ASTUnion_set(List_ASTUnion *list, int index, struct nl_ASTUnion value);

/* Get the value at the specified index */
struct nl_ASTUnion nl_list_ASTUnion_get(List_ASTUnion *list, int index);

/* Clear all elements from the list */
void nl_list_ASTUnion_clear(List_ASTUnion *list);

/* Get the current length of the list */
int nl_list_ASTUnion_length(List_ASTUnion *list);

/* Get the current capacity of the list */
int nl_list_ASTUnion_capacity(List_ASTUnion *list);

/* Check if the list is empty */
bool nl_list_ASTUnion_is_empty(List_ASTUnion *list);

/* Free the list and all its resources */
void nl_list_ASTUnion_free(List_ASTUnion *list);

#endif /* LIST_ASTUNION_H */
