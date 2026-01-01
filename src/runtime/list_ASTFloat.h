#ifndef LIST_ASTFLOAT_H
#define LIST_ASTFLOAT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFloat */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFloat
#define DEFINED_List_ASTFloat
typedef struct List_ASTFloat {
    struct nl_ASTFloat *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFloat;
#endif

/* Create a new empty list */
List_ASTFloat* nl_list_ASTFloat_new(void);

/* Create a new list with specified initial capacity */
List_ASTFloat* nl_list_ASTFloat_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFloat_push(List_ASTFloat *list, struct nl_ASTFloat value);

/* Remove and return the last element */
struct nl_ASTFloat nl_list_ASTFloat_pop(List_ASTFloat *list);

/* Insert an element at the specified index */
void nl_list_ASTFloat_insert(List_ASTFloat *list, int index, struct nl_ASTFloat value);

/* Remove and return the element at the specified index */
struct nl_ASTFloat nl_list_ASTFloat_remove(List_ASTFloat *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFloat_set(List_ASTFloat *list, int index, struct nl_ASTFloat value);

/* Get the value at the specified index */
struct nl_ASTFloat nl_list_ASTFloat_get(List_ASTFloat *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFloat_clear(List_ASTFloat *list);

/* Get the current length of the list */
int nl_list_ASTFloat_length(List_ASTFloat *list);

/* Get the current capacity of the list */
int nl_list_ASTFloat_capacity(List_ASTFloat *list);

/* Check if the list is empty */
bool nl_list_ASTFloat_is_empty(List_ASTFloat *list);

/* Free the list and all its resources */
void nl_list_ASTFloat_free(List_ASTFloat *list);

#endif /* LIST_ASTFLOAT_H */
