#ifndef LIST_ASTMODULEQUALIFIEDCALL_H
#define LIST_ASTMODULEQUALIFIEDCALL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTModuleQualifiedCall */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTModuleQualifiedCall
#define DEFINED_List_ASTModuleQualifiedCall
typedef struct List_ASTModuleQualifiedCall {
    struct nl_ASTModuleQualifiedCall *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTModuleQualifiedCall;
#endif

/* Create a new empty list */
List_ASTModuleQualifiedCall* nl_list_ASTModuleQualifiedCall_new(void);

/* Create a new list with specified initial capacity */
List_ASTModuleQualifiedCall* nl_list_ASTModuleQualifiedCall_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTModuleQualifiedCall_push(List_ASTModuleQualifiedCall *list, struct nl_ASTModuleQualifiedCall value);

/* Remove and return the last element */
struct nl_ASTModuleQualifiedCall nl_list_ASTModuleQualifiedCall_pop(List_ASTModuleQualifiedCall *list);

/* Insert an element at the specified index */
void nl_list_ASTModuleQualifiedCall_insert(List_ASTModuleQualifiedCall *list, int index, struct nl_ASTModuleQualifiedCall value);

/* Remove and return the element at the specified index */
struct nl_ASTModuleQualifiedCall nl_list_ASTModuleQualifiedCall_remove(List_ASTModuleQualifiedCall *list, int index);

/* Set the value at the specified index */
void nl_list_ASTModuleQualifiedCall_set(List_ASTModuleQualifiedCall *list, int index, struct nl_ASTModuleQualifiedCall value);

/* Get the value at the specified index */
struct nl_ASTModuleQualifiedCall nl_list_ASTModuleQualifiedCall_get(List_ASTModuleQualifiedCall *list, int index);

/* Clear all elements from the list */
void nl_list_ASTModuleQualifiedCall_clear(List_ASTModuleQualifiedCall *list);

/* Get the current length of the list */
int nl_list_ASTModuleQualifiedCall_length(List_ASTModuleQualifiedCall *list);

/* Get the current capacity of the list */
int nl_list_ASTModuleQualifiedCall_capacity(List_ASTModuleQualifiedCall *list);

/* Check if the list is empty */
bool nl_list_ASTModuleQualifiedCall_is_empty(List_ASTModuleQualifiedCall *list);

/* Free the list and all its resources */
void nl_list_ASTModuleQualifiedCall_free(List_ASTModuleQualifiedCall *list);

#endif /* LIST_ASTMODULEQUALIFIEDCALL_H */
