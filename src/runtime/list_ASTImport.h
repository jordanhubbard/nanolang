#ifndef LIST_ASTIMPORT_H
#define LIST_ASTIMPORT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTImport */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTImport
#define DEFINED_List_ASTImport
typedef struct List_ASTImport {
    struct nl_ASTImport *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTImport;
#endif

/* Create a new empty list */
List_ASTImport* nl_list_ASTImport_new(void);

/* Create a new list with specified initial capacity */
List_ASTImport* nl_list_ASTImport_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTImport_push(List_ASTImport *list, struct nl_ASTImport value);

/* Remove and return the last element */
struct nl_ASTImport nl_list_ASTImport_pop(List_ASTImport *list);

/* Insert an element at the specified index */
void nl_list_ASTImport_insert(List_ASTImport *list, int index, struct nl_ASTImport value);

/* Remove and return the element at the specified index */
struct nl_ASTImport nl_list_ASTImport_remove(List_ASTImport *list, int index);

/* Set the value at the specified index */
void nl_list_ASTImport_set(List_ASTImport *list, int index, struct nl_ASTImport value);

/* Get the value at the specified index */
struct nl_ASTImport nl_list_ASTImport_get(List_ASTImport *list, int index);

/* Clear all elements from the list */
void nl_list_ASTImport_clear(List_ASTImport *list);

/* Get the current length of the list */
int nl_list_ASTImport_length(List_ASTImport *list);

/* Get the current capacity of the list */
int nl_list_ASTImport_capacity(List_ASTImport *list);

/* Check if the list is empty */
bool nl_list_ASTImport_is_empty(List_ASTImport *list);

/* Free the list and all its resources */
void nl_list_ASTImport_free(List_ASTImport *list);

#endif /* LIST_ASTIMPORT_H */
