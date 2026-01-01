#ifndef LIST_ASTTUPLELITERAL_H
#define LIST_ASTTUPLELITERAL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTTupleLiteral */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTTupleLiteral
#define DEFINED_List_ASTTupleLiteral
typedef struct List_ASTTupleLiteral {
    struct nl_ASTTupleLiteral *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTTupleLiteral;
#endif

/* Create a new empty list */
List_ASTTupleLiteral* nl_list_ASTTupleLiteral_new(void);

/* Create a new list with specified initial capacity */
List_ASTTupleLiteral* nl_list_ASTTupleLiteral_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTTupleLiteral_push(List_ASTTupleLiteral *list, struct nl_ASTTupleLiteral value);

/* Remove and return the last element */
struct nl_ASTTupleLiteral nl_list_ASTTupleLiteral_pop(List_ASTTupleLiteral *list);

/* Insert an element at the specified index */
void nl_list_ASTTupleLiteral_insert(List_ASTTupleLiteral *list, int index, struct nl_ASTTupleLiteral value);

/* Remove and return the element at the specified index */
struct nl_ASTTupleLiteral nl_list_ASTTupleLiteral_remove(List_ASTTupleLiteral *list, int index);

/* Set the value at the specified index */
void nl_list_ASTTupleLiteral_set(List_ASTTupleLiteral *list, int index, struct nl_ASTTupleLiteral value);

/* Get the value at the specified index */
struct nl_ASTTupleLiteral nl_list_ASTTupleLiteral_get(List_ASTTupleLiteral *list, int index);

/* Clear all elements from the list */
void nl_list_ASTTupleLiteral_clear(List_ASTTupleLiteral *list);

/* Get the current length of the list */
int nl_list_ASTTupleLiteral_length(List_ASTTupleLiteral *list);

/* Get the current capacity of the list */
int nl_list_ASTTupleLiteral_capacity(List_ASTTupleLiteral *list);

/* Check if the list is empty */
bool nl_list_ASTTupleLiteral_is_empty(List_ASTTupleLiteral *list);

/* Free the list and all its resources */
void nl_list_ASTTupleLiteral_free(List_ASTTupleLiteral *list);

#endif /* LIST_ASTTUPLELITERAL_H */
