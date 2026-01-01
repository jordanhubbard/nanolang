#ifndef LIST_ASTSTRUCTLITERAL_H
#define LIST_ASTSTRUCTLITERAL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStructLiteral */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStructLiteral
#define DEFINED_List_ASTStructLiteral
typedef struct List_ASTStructLiteral {
    struct nl_ASTStructLiteral *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStructLiteral;
#endif

/* Create a new empty list */
List_ASTStructLiteral* nl_list_ASTStructLiteral_new(void);

/* Create a new list with specified initial capacity */
List_ASTStructLiteral* nl_list_ASTStructLiteral_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStructLiteral_push(List_ASTStructLiteral *list, struct nl_ASTStructLiteral value);

/* Remove and return the last element */
struct nl_ASTStructLiteral nl_list_ASTStructLiteral_pop(List_ASTStructLiteral *list);

/* Insert an element at the specified index */
void nl_list_ASTStructLiteral_insert(List_ASTStructLiteral *list, int index, struct nl_ASTStructLiteral value);

/* Remove and return the element at the specified index */
struct nl_ASTStructLiteral nl_list_ASTStructLiteral_remove(List_ASTStructLiteral *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStructLiteral_set(List_ASTStructLiteral *list, int index, struct nl_ASTStructLiteral value);

/* Get the value at the specified index */
struct nl_ASTStructLiteral nl_list_ASTStructLiteral_get(List_ASTStructLiteral *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStructLiteral_clear(List_ASTStructLiteral *list);

/* Get the current length of the list */
int nl_list_ASTStructLiteral_length(List_ASTStructLiteral *list);

/* Get the current capacity of the list */
int nl_list_ASTStructLiteral_capacity(List_ASTStructLiteral *list);

/* Check if the list is empty */
bool nl_list_ASTStructLiteral_is_empty(List_ASTStructLiteral *list);

/* Free the list and all its resources */
void nl_list_ASTStructLiteral_free(List_ASTStructLiteral *list);

#endif /* LIST_ASTSTRUCTLITERAL_H */
