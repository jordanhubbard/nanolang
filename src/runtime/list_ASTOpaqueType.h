#ifndef LIST_ASTOPAQUETYPE_H
#define LIST_ASTOPAQUETYPE_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTOpaqueType */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTOpaqueType
#define FORWARD_DEFINED_List_ASTOpaqueType
typedef struct List_ASTOpaqueType {
    struct nl_ASTOpaqueType *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTOpaqueType;
#endif

/* Create a new empty list */
List_ASTOpaqueType* nl_list_ASTOpaqueType_new(void);

/* Create a new list with specified initial capacity */
List_ASTOpaqueType* nl_list_ASTOpaqueType_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTOpaqueType_push(List_ASTOpaqueType *list, struct nl_ASTOpaqueType value);

/* Remove and return the last element */
struct nl_ASTOpaqueType nl_list_ASTOpaqueType_pop(List_ASTOpaqueType *list);

/* Insert an element at the specified index */
void nl_list_ASTOpaqueType_insert(List_ASTOpaqueType *list, int index, struct nl_ASTOpaqueType value);

/* Remove and return the element at the specified index */
struct nl_ASTOpaqueType nl_list_ASTOpaqueType_remove(List_ASTOpaqueType *list, int index);

/* Set the value at the specified index */
void nl_list_ASTOpaqueType_set(List_ASTOpaqueType *list, int index, struct nl_ASTOpaqueType value);

/* Get the value at the specified index */
struct nl_ASTOpaqueType nl_list_ASTOpaqueType_get(List_ASTOpaqueType *list, int index);

/* Clear all elements from the list */
void nl_list_ASTOpaqueType_clear(List_ASTOpaqueType *list);

/* Get the current length of the list */
int nl_list_ASTOpaqueType_length(List_ASTOpaqueType *list);

/* Get the current capacity of the list */
int nl_list_ASTOpaqueType_capacity(List_ASTOpaqueType *list);

/* Check if the list is empty */
bool nl_list_ASTOpaqueType_is_empty(List_ASTOpaqueType *list);

/* Free the list and all its resources */
void nl_list_ASTOpaqueType_free(List_ASTOpaqueType *list);

#endif /* LIST_ASTOPAQUETYPE_H */
