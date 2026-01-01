#ifndef LIST_ASTARRAYLITERAL_H
#define LIST_ASTARRAYLITERAL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTArrayLiteral */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTArrayLiteral
#define FORWARD_DEFINED_List_ASTArrayLiteral
typedef struct List_ASTArrayLiteral {
    struct nl_ASTArrayLiteral *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTArrayLiteral;
#endif

/* Create a new empty list */
List_ASTArrayLiteral* nl_list_ASTArrayLiteral_new(void);

/* Create a new list with specified initial capacity */
List_ASTArrayLiteral* nl_list_ASTArrayLiteral_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTArrayLiteral_push(List_ASTArrayLiteral *list, struct nl_ASTArrayLiteral value);

/* Remove and return the last element */
struct nl_ASTArrayLiteral nl_list_ASTArrayLiteral_pop(List_ASTArrayLiteral *list);

/* Insert an element at the specified index */
void nl_list_ASTArrayLiteral_insert(List_ASTArrayLiteral *list, int index, struct nl_ASTArrayLiteral value);

/* Remove and return the element at the specified index */
struct nl_ASTArrayLiteral nl_list_ASTArrayLiteral_remove(List_ASTArrayLiteral *list, int index);

/* Set the value at the specified index */
void nl_list_ASTArrayLiteral_set(List_ASTArrayLiteral *list, int index, struct nl_ASTArrayLiteral value);

/* Get the value at the specified index */
struct nl_ASTArrayLiteral nl_list_ASTArrayLiteral_get(List_ASTArrayLiteral *list, int index);

/* Clear all elements from the list */
void nl_list_ASTArrayLiteral_clear(List_ASTArrayLiteral *list);

/* Get the current length of the list */
int nl_list_ASTArrayLiteral_length(List_ASTArrayLiteral *list);

/* Get the current capacity of the list */
int nl_list_ASTArrayLiteral_capacity(List_ASTArrayLiteral *list);

/* Check if the list is empty */
bool nl_list_ASTArrayLiteral_is_empty(List_ASTArrayLiteral *list);

/* Free the list and all its resources */
void nl_list_ASTArrayLiteral_free(List_ASTArrayLiteral *list);

#endif /* LIST_ASTARRAYLITERAL_H */
