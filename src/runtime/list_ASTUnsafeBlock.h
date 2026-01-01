#ifndef LIST_ASTUNSAFEBLOCK_H
#define LIST_ASTUNSAFEBLOCK_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTUnsafeBlock */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTUnsafeBlock
#define DEFINED_List_ASTUnsafeBlock
typedef struct List_ASTUnsafeBlock {
    struct nl_ASTUnsafeBlock *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTUnsafeBlock;
#endif

/* Create a new empty list */
List_ASTUnsafeBlock* nl_list_ASTUnsafeBlock_new(void);

/* Create a new list with specified initial capacity */
List_ASTUnsafeBlock* nl_list_ASTUnsafeBlock_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTUnsafeBlock_push(List_ASTUnsafeBlock *list, struct nl_ASTUnsafeBlock value);

/* Remove and return the last element */
struct nl_ASTUnsafeBlock nl_list_ASTUnsafeBlock_pop(List_ASTUnsafeBlock *list);

/* Insert an element at the specified index */
void nl_list_ASTUnsafeBlock_insert(List_ASTUnsafeBlock *list, int index, struct nl_ASTUnsafeBlock value);

/* Remove and return the element at the specified index */
struct nl_ASTUnsafeBlock nl_list_ASTUnsafeBlock_remove(List_ASTUnsafeBlock *list, int index);

/* Set the value at the specified index */
void nl_list_ASTUnsafeBlock_set(List_ASTUnsafeBlock *list, int index, struct nl_ASTUnsafeBlock value);

/* Get the value at the specified index */
struct nl_ASTUnsafeBlock nl_list_ASTUnsafeBlock_get(List_ASTUnsafeBlock *list, int index);

/* Clear all elements from the list */
void nl_list_ASTUnsafeBlock_clear(List_ASTUnsafeBlock *list);

/* Get the current length of the list */
int nl_list_ASTUnsafeBlock_length(List_ASTUnsafeBlock *list);

/* Get the current capacity of the list */
int nl_list_ASTUnsafeBlock_capacity(List_ASTUnsafeBlock *list);

/* Check if the list is empty */
bool nl_list_ASTUnsafeBlock_is_empty(List_ASTUnsafeBlock *list);

/* Free the list and all its resources */
void nl_list_ASTUnsafeBlock_free(List_ASTUnsafeBlock *list);

#endif /* LIST_ASTUNSAFEBLOCK_H */
