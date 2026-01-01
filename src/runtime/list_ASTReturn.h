#ifndef LIST_ASTRETURN_H
#define LIST_ASTRETURN_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTReturn */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTReturn
#define DEFINED_List_ASTReturn
typedef struct List_ASTReturn {
    struct nl_ASTReturn *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTReturn;
#endif

/* Create a new empty list */
List_ASTReturn* nl_list_ASTReturn_new(void);

/* Create a new list with specified initial capacity */
List_ASTReturn* nl_list_ASTReturn_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTReturn_push(List_ASTReturn *list, struct nl_ASTReturn value);

/* Remove and return the last element */
struct nl_ASTReturn nl_list_ASTReturn_pop(List_ASTReturn *list);

/* Insert an element at the specified index */
void nl_list_ASTReturn_insert(List_ASTReturn *list, int index, struct nl_ASTReturn value);

/* Remove and return the element at the specified index */
struct nl_ASTReturn nl_list_ASTReturn_remove(List_ASTReturn *list, int index);

/* Set the value at the specified index */
void nl_list_ASTReturn_set(List_ASTReturn *list, int index, struct nl_ASTReturn value);

/* Get the value at the specified index */
struct nl_ASTReturn nl_list_ASTReturn_get(List_ASTReturn *list, int index);

/* Clear all elements from the list */
void nl_list_ASTReturn_clear(List_ASTReturn *list);

/* Get the current length of the list */
int nl_list_ASTReturn_length(List_ASTReturn *list);

/* Get the current capacity of the list */
int nl_list_ASTReturn_capacity(List_ASTReturn *list);

/* Check if the list is empty */
bool nl_list_ASTReturn_is_empty(List_ASTReturn *list);

/* Free the list and all its resources */
void nl_list_ASTReturn_free(List_ASTReturn *list);

#endif /* LIST_ASTRETURN_H */
