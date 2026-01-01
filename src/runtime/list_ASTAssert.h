#ifndef LIST_ASTASSERT_H
#define LIST_ASTASSERT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTAssert */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTAssert
#define DEFINED_List_ASTAssert
typedef struct List_ASTAssert {
    struct nl_ASTAssert *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTAssert;
#endif

/* Create a new empty list */
List_ASTAssert* nl_list_ASTAssert_new(void);

/* Create a new list with specified initial capacity */
List_ASTAssert* nl_list_ASTAssert_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTAssert_push(List_ASTAssert *list, struct nl_ASTAssert value);

/* Remove and return the last element */
struct nl_ASTAssert nl_list_ASTAssert_pop(List_ASTAssert *list);

/* Insert an element at the specified index */
void nl_list_ASTAssert_insert(List_ASTAssert *list, int index, struct nl_ASTAssert value);

/* Remove and return the element at the specified index */
struct nl_ASTAssert nl_list_ASTAssert_remove(List_ASTAssert *list, int index);

/* Set the value at the specified index */
void nl_list_ASTAssert_set(List_ASTAssert *list, int index, struct nl_ASTAssert value);

/* Get the value at the specified index */
struct nl_ASTAssert nl_list_ASTAssert_get(List_ASTAssert *list, int index);

/* Clear all elements from the list */
void nl_list_ASTAssert_clear(List_ASTAssert *list);

/* Get the current length of the list */
int nl_list_ASTAssert_length(List_ASTAssert *list);

/* Get the current capacity of the list */
int nl_list_ASTAssert_capacity(List_ASTAssert *list);

/* Check if the list is empty */
bool nl_list_ASTAssert_is_empty(List_ASTAssert *list);

/* Free the list and all its resources */
void nl_list_ASTAssert_free(List_ASTAssert *list);

#endif /* LIST_ASTASSERT_H */
