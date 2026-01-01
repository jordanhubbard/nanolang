#ifndef LIST_ASTFUNCTION_H
#define LIST_ASTFUNCTION_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFunction */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFunction
#define DEFINED_List_ASTFunction
typedef struct List_ASTFunction {
    struct nl_ASTFunction *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFunction;
#endif

/* Create a new empty list */
List_ASTFunction* nl_list_ASTFunction_new(void);

/* Create a new list with specified initial capacity */
List_ASTFunction* nl_list_ASTFunction_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFunction_push(List_ASTFunction *list, struct nl_ASTFunction value);

/* Remove and return the last element */
struct nl_ASTFunction nl_list_ASTFunction_pop(List_ASTFunction *list);

/* Insert an element at the specified index */
void nl_list_ASTFunction_insert(List_ASTFunction *list, int index, struct nl_ASTFunction value);

/* Remove and return the element at the specified index */
struct nl_ASTFunction nl_list_ASTFunction_remove(List_ASTFunction *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFunction_set(List_ASTFunction *list, int index, struct nl_ASTFunction value);

/* Get the value at the specified index */
struct nl_ASTFunction nl_list_ASTFunction_get(List_ASTFunction *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFunction_clear(List_ASTFunction *list);

/* Get the current length of the list */
int nl_list_ASTFunction_length(List_ASTFunction *list);

/* Get the current capacity of the list */
int nl_list_ASTFunction_capacity(List_ASTFunction *list);

/* Check if the list is empty */
bool nl_list_ASTFunction_is_empty(List_ASTFunction *list);

/* Free the list and all its resources */
void nl_list_ASTFunction_free(List_ASTFunction *list);

#endif /* LIST_ASTFUNCTION_H */
