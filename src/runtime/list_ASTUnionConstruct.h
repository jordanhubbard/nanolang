#ifndef LIST_ASTUNIONCONSTRUCT_H
#define LIST_ASTUNIONCONSTRUCT_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTUnionConstruct */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTUnionConstruct
#define DEFINED_List_ASTUnionConstruct
typedef struct List_ASTUnionConstruct {
    struct nl_ASTUnionConstruct *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTUnionConstruct;
#endif

/* Create a new empty list */
List_ASTUnionConstruct* nl_list_ASTUnionConstruct_new(void);

/* Create a new list with specified initial capacity */
List_ASTUnionConstruct* nl_list_ASTUnionConstruct_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTUnionConstruct_push(List_ASTUnionConstruct *list, struct nl_ASTUnionConstruct value);

/* Remove and return the last element */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_pop(List_ASTUnionConstruct *list);

/* Insert an element at the specified index */
void nl_list_ASTUnionConstruct_insert(List_ASTUnionConstruct *list, int index, struct nl_ASTUnionConstruct value);

/* Remove and return the element at the specified index */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_remove(List_ASTUnionConstruct *list, int index);

/* Set the value at the specified index */
void nl_list_ASTUnionConstruct_set(List_ASTUnionConstruct *list, int index, struct nl_ASTUnionConstruct value);

/* Get the value at the specified index */
struct nl_ASTUnionConstruct nl_list_ASTUnionConstruct_get(List_ASTUnionConstruct *list, int index);

/* Clear all elements from the list */
void nl_list_ASTUnionConstruct_clear(List_ASTUnionConstruct *list);

/* Get the current length of the list */
int nl_list_ASTUnionConstruct_length(List_ASTUnionConstruct *list);

/* Get the current capacity of the list */
int nl_list_ASTUnionConstruct_capacity(List_ASTUnionConstruct *list);

/* Check if the list is empty */
bool nl_list_ASTUnionConstruct_is_empty(List_ASTUnionConstruct *list);

/* Free the list and all its resources */
void nl_list_ASTUnionConstruct_free(List_ASTUnionConstruct *list);

#endif /* LIST_ASTUNIONCONSTRUCT_H */
