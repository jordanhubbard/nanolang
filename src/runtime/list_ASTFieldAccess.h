#ifndef LIST_ASTFIELDACCESS_H
#define LIST_ASTFIELDACCESS_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFieldAccess */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTFieldAccess
#define DEFINED_List_ASTFieldAccess
typedef struct List_ASTFieldAccess {
    struct nl_ASTFieldAccess *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFieldAccess;
#endif

/* Create a new empty list */
List_ASTFieldAccess* nl_list_ASTFieldAccess_new(void);

/* Create a new list with specified initial capacity */
List_ASTFieldAccess* nl_list_ASTFieldAccess_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFieldAccess_push(List_ASTFieldAccess *list, struct nl_ASTFieldAccess value);

/* Remove and return the last element */
struct nl_ASTFieldAccess nl_list_ASTFieldAccess_pop(List_ASTFieldAccess *list);

/* Insert an element at the specified index */
void nl_list_ASTFieldAccess_insert(List_ASTFieldAccess *list, int index, struct nl_ASTFieldAccess value);

/* Remove and return the element at the specified index */
struct nl_ASTFieldAccess nl_list_ASTFieldAccess_remove(List_ASTFieldAccess *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFieldAccess_set(List_ASTFieldAccess *list, int index, struct nl_ASTFieldAccess value);

/* Get the value at the specified index */
struct nl_ASTFieldAccess nl_list_ASTFieldAccess_get(List_ASTFieldAccess *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFieldAccess_clear(List_ASTFieldAccess *list);

/* Get the current length of the list */
int nl_list_ASTFieldAccess_length(List_ASTFieldAccess *list);

/* Get the current capacity of the list */
int nl_list_ASTFieldAccess_capacity(List_ASTFieldAccess *list);

/* Check if the list is empty */
bool nl_list_ASTFieldAccess_is_empty(List_ASTFieldAccess *list);

/* Free the list and all its resources */
void nl_list_ASTFieldAccess_free(List_ASTFieldAccess *list);

#endif /* LIST_ASTFIELDACCESS_H */
