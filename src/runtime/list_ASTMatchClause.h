#ifndef LIST_ASTMATCHCLAUSE_H
#define LIST_ASTMATCHCLAUSE_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTMatchClause */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTMatchClause
#define FORWARD_DEFINED_List_ASTMatchClause
typedef struct List_ASTMatchClause {
    struct nl_ASTMatchClause *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTMatchClause;
#endif

/* Create a new empty list */
List_ASTMatchClause* nl_list_ASTMatchClause_new(void);

/* Create a new list with specified initial capacity */
List_ASTMatchClause* nl_list_ASTMatchClause_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTMatchClause_push(List_ASTMatchClause *list, struct nl_ASTMatchClause value);

/* Remove and return the last element */
struct nl_ASTMatchClause nl_list_ASTMatchClause_pop(List_ASTMatchClause *list);

/* Insert an element at the specified index */
void nl_list_ASTMatchClause_insert(List_ASTMatchClause *list, int index, struct nl_ASTMatchClause value);

/* Remove and return the element at the specified index */
struct nl_ASTMatchClause nl_list_ASTMatchClause_remove(List_ASTMatchClause *list, int index);

/* Set the value at the specified index */
void nl_list_ASTMatchClause_set(List_ASTMatchClause *list, int index, struct nl_ASTMatchClause value);

/* Get the value at the specified index */
struct nl_ASTMatchClause nl_list_ASTMatchClause_get(List_ASTMatchClause *list, int index);

/* Clear all elements from the list */
void nl_list_ASTMatchClause_clear(List_ASTMatchClause *list);

/* Get the current length of the list */
int nl_list_ASTMatchClause_length(List_ASTMatchClause *list);

/* Get the current capacity of the list */
int nl_list_ASTMatchClause_capacity(List_ASTMatchClause *list);

/* Check if the list is empty */
bool nl_list_ASTMatchClause_is_empty(List_ASTMatchClause *list);

/* Free the list and all its resources */
void nl_list_ASTMatchClause_free(List_ASTMatchClause *list);

#endif /* LIST_ASTMATCHCLAUSE_H */
