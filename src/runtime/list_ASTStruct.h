#ifndef LIST_ASTSTRUCT_H
#define LIST_ASTSTRUCT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTStruct */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTStruct
#define DEFINED_List_ASTStruct
typedef struct List_ASTStruct {
    struct nl_ASTStruct *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTStruct;
#endif

/* Create a new empty list */
List_ASTStruct* nl_list_ASTStruct_new(void);

/* Create a new list with specified initial capacity */
List_ASTStruct* nl_list_ASTStruct_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTStruct_push(List_ASTStruct *list, struct nl_ASTStruct value);

/* Remove and return the last element */
struct nl_ASTStruct nl_list_ASTStruct_pop(List_ASTStruct *list);

/* Insert an element at the specified index */
void nl_list_ASTStruct_insert(List_ASTStruct *list, int index, struct nl_ASTStruct value);

/* Remove and return the element at the specified index */
struct nl_ASTStruct nl_list_ASTStruct_remove(List_ASTStruct *list, int index);

/* Set the value at the specified index */
void nl_list_ASTStruct_set(List_ASTStruct *list, int index, struct nl_ASTStruct value);

/* Get the value at the specified index */
struct nl_ASTStruct nl_list_ASTStruct_get(List_ASTStruct *list, int index);

/* Clear all elements from the list */
void nl_list_ASTStruct_clear(List_ASTStruct *list);

/* Get the current length of the list */
int nl_list_ASTStruct_length(List_ASTStruct *list);

/* Get the current capacity of the list */
int nl_list_ASTStruct_capacity(List_ASTStruct *list);

/* Check if the list is empty */
bool nl_list_ASTStruct_is_empty(List_ASTStruct *list);

/* Free the list and all its resources */
void nl_list_ASTStruct_free(List_ASTStruct *list);

#endif /* LIST_ASTSTRUCT_H */
