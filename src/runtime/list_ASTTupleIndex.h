#ifndef LIST_ASTTUPLEINDEX_H
#define LIST_ASTTUPLEINDEX_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTTupleIndex */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTTupleIndex
#define DEFINED_List_ASTTupleIndex
typedef struct List_ASTTupleIndex {
    struct nl_ASTTupleIndex *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTTupleIndex;
#endif

/* Create a new empty list */
List_ASTTupleIndex* nl_list_ASTTupleIndex_new(void);

/* Create a new list with specified initial capacity */
List_ASTTupleIndex* nl_list_ASTTupleIndex_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTTupleIndex_push(List_ASTTupleIndex *list, struct nl_ASTTupleIndex value);

/* Remove and return the last element */
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_pop(List_ASTTupleIndex *list);

/* Insert an element at the specified index */
void nl_list_ASTTupleIndex_insert(List_ASTTupleIndex *list, int index, struct nl_ASTTupleIndex value);

/* Remove and return the element at the specified index */
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_remove(List_ASTTupleIndex *list, int index);

/* Set the value at the specified index */
void nl_list_ASTTupleIndex_set(List_ASTTupleIndex *list, int index, struct nl_ASTTupleIndex value);

/* Get the value at the specified index */
struct nl_ASTTupleIndex nl_list_ASTTupleIndex_get(List_ASTTupleIndex *list, int index);

/* Clear all elements from the list */
void nl_list_ASTTupleIndex_clear(List_ASTTupleIndex *list);

/* Get the current length of the list */
int nl_list_ASTTupleIndex_length(List_ASTTupleIndex *list);

/* Get the current capacity of the list */
int nl_list_ASTTupleIndex_capacity(List_ASTTupleIndex *list);

/* Check if the list is empty */
bool nl_list_ASTTupleIndex_is_empty(List_ASTTupleIndex *list);

/* Free the list and all its resources */
void nl_list_ASTTupleIndex_free(List_ASTTupleIndex *list);

#endif /* LIST_ASTTUPLEINDEX_H */
