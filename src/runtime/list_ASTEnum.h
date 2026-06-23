#ifndef LIST_ASTENUM_H
#define LIST_ASTENUM_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTEnum */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTEnum
#define FORWARD_DEFINED_List_ASTEnum
typedef struct List_ASTEnum {
    struct nl_ASTEnum *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTEnum;
#endif

/* Create a new empty list */
List_ASTEnum* nl_list_ASTEnum_new(void);

/* Create a new list with specified initial capacity */
List_ASTEnum* nl_list_ASTEnum_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTEnum_push(List_ASTEnum *list, struct nl_ASTEnum value);

/* Remove and return the last element */
struct nl_ASTEnum nl_list_ASTEnum_pop(List_ASTEnum *list);

/* Insert an element at the specified index */
void nl_list_ASTEnum_insert(List_ASTEnum *list, int index, struct nl_ASTEnum value);

/* Remove and return the element at the specified index */
struct nl_ASTEnum nl_list_ASTEnum_remove(List_ASTEnum *list, int index);

/* Set the value at the specified index */
void nl_list_ASTEnum_set(List_ASTEnum *list, int index, struct nl_ASTEnum value);

/* Get the value at the specified index */
struct nl_ASTEnum nl_list_ASTEnum_get(List_ASTEnum *list, int index);

/* Clear all elements from the list */
void nl_list_ASTEnum_clear(List_ASTEnum *list);

/* Get the current length of the list */
int nl_list_ASTEnum_length(List_ASTEnum *list);

/* Get the current capacity of the list */
int nl_list_ASTEnum_capacity(List_ASTEnum *list);

/* Check if the list is empty */
bool nl_list_ASTEnum_is_empty(List_ASTEnum *list);

/* Free the list and all its resources */
void nl_list_ASTEnum_free(List_ASTEnum *list);

#endif /* LIST_ASTENUM_H */
