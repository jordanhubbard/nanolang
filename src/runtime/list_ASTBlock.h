#ifndef LIST_ASTBLOCK_H
#define LIST_ASTBLOCK_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBlock */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTBlock
#define DEFINED_List_ASTBlock
typedef struct List_ASTBlock {
    struct nl_ASTBlock *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBlock;
#endif

/* Create a new empty list */
List_ASTBlock* nl_list_ASTBlock_new(void);

/* Create a new list with specified initial capacity */
List_ASTBlock* nl_list_ASTBlock_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBlock_push(List_ASTBlock *list, struct nl_ASTBlock value);

/* Remove and return the last element */
struct nl_ASTBlock nl_list_ASTBlock_pop(List_ASTBlock *list);

/* Insert an element at the specified index */
void nl_list_ASTBlock_insert(List_ASTBlock *list, int index, struct nl_ASTBlock value);

/* Remove and return the element at the specified index */
struct nl_ASTBlock nl_list_ASTBlock_remove(List_ASTBlock *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBlock_set(List_ASTBlock *list, int index, struct nl_ASTBlock value);

/* Get the value at the specified index */
struct nl_ASTBlock nl_list_ASTBlock_get(List_ASTBlock *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBlock_clear(List_ASTBlock *list);

/* Get the current length of the list */
int nl_list_ASTBlock_length(List_ASTBlock *list);

/* Get the current capacity of the list */
int nl_list_ASTBlock_capacity(List_ASTBlock *list);

/* Check if the list is empty */
bool nl_list_ASTBlock_is_empty(List_ASTBlock *list);

/* Free the list and all its resources */
void nl_list_ASTBlock_free(List_ASTBlock *list);

#endif /* LIST_ASTBLOCK_H */
