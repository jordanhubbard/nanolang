#ifndef LIST_ASTSET_H
#define LIST_ASTSET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTSet */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTSet
#define DEFINED_List_ASTSet
typedef struct List_ASTSet {
    struct nl_ASTSet *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTSet;
#endif

/* Create a new empty list */
List_ASTSet* nl_list_ASTSet_new(void);

/* Create a new list with specified initial capacity */
List_ASTSet* nl_list_ASTSet_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTSet_push(List_ASTSet *list, struct nl_ASTSet value);

/* Remove and return the last element */
struct nl_ASTSet nl_list_ASTSet_pop(List_ASTSet *list);

/* Insert an element at the specified index */
void nl_list_ASTSet_insert(List_ASTSet *list, int index, struct nl_ASTSet value);

/* Remove and return the element at the specified index */
struct nl_ASTSet nl_list_ASTSet_remove(List_ASTSet *list, int index);

/* Set the value at the specified index */
void nl_list_ASTSet_set(List_ASTSet *list, int index, struct nl_ASTSet value);

/* Get the value at the specified index */
struct nl_ASTSet nl_list_ASTSet_get(List_ASTSet *list, int index);

/* Clear all elements from the list */
void nl_list_ASTSet_clear(List_ASTSet *list);

/* Get the current length of the list */
int nl_list_ASTSet_length(List_ASTSet *list);

/* Get the current capacity of the list */
int nl_list_ASTSet_capacity(List_ASTSet *list);

/* Check if the list is empty */
bool nl_list_ASTSet_is_empty(List_ASTSet *list);

/* Free the list and all its resources */
void nl_list_ASTSet_free(List_ASTSet *list);

#endif /* LIST_ASTSET_H */
