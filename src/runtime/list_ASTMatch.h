#ifndef LIST_ASTMATCH_H
#define LIST_ASTMATCH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTMatch */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTMatch
#define DEFINED_List_ASTMatch
typedef struct List_ASTMatch {
    struct nl_ASTMatch *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTMatch;
#endif

/* Create a new empty list */
List_ASTMatch* nl_list_ASTMatch_new(void);

/* Create a new list with specified initial capacity */
List_ASTMatch* nl_list_ASTMatch_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTMatch_push(List_ASTMatch *list, struct nl_ASTMatch value);

/* Remove and return the last element */
struct nl_ASTMatch nl_list_ASTMatch_pop(List_ASTMatch *list);

/* Insert an element at the specified index */
void nl_list_ASTMatch_insert(List_ASTMatch *list, int index, struct nl_ASTMatch value);

/* Remove and return the element at the specified index */
struct nl_ASTMatch nl_list_ASTMatch_remove(List_ASTMatch *list, int index);

/* Set the value at the specified index */
void nl_list_ASTMatch_set(List_ASTMatch *list, int index, struct nl_ASTMatch value);

/* Get the value at the specified index */
struct nl_ASTMatch nl_list_ASTMatch_get(List_ASTMatch *list, int index);

/* Clear all elements from the list */
void nl_list_ASTMatch_clear(List_ASTMatch *list);

/* Get the current length of the list */
int nl_list_ASTMatch_length(List_ASTMatch *list);

/* Get the current capacity of the list */
int nl_list_ASTMatch_capacity(List_ASTMatch *list);

/* Check if the list is empty */
bool nl_list_ASTMatch_is_empty(List_ASTMatch *list);

/* Free the list and all its resources */
void nl_list_ASTMatch_free(List_ASTMatch *list);

#endif /* LIST_ASTMATCH_H */
