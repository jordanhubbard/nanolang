#ifndef LIST_ASTLET_H
#define LIST_ASTLET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTLet */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTLet
#define DEFINED_List_ASTLet
typedef struct List_ASTLet {
    struct nl_ASTLet *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTLet;
#endif

/* Create a new empty list */
List_ASTLet* nl_list_ASTLet_new(void);

/* Create a new list with specified initial capacity */
List_ASTLet* nl_list_ASTLet_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTLet_push(List_ASTLet *list, struct nl_ASTLet value);

/* Remove and return the last element */
struct nl_ASTLet nl_list_ASTLet_pop(List_ASTLet *list);

/* Insert an element at the specified index */
void nl_list_ASTLet_insert(List_ASTLet *list, int index, struct nl_ASTLet value);

/* Remove and return the element at the specified index */
struct nl_ASTLet nl_list_ASTLet_remove(List_ASTLet *list, int index);

/* Set the value at the specified index */
void nl_list_ASTLet_set(List_ASTLet *list, int index, struct nl_ASTLet value);

/* Get the value at the specified index */
struct nl_ASTLet nl_list_ASTLet_get(List_ASTLet *list, int index);

/* Clear all elements from the list */
void nl_list_ASTLet_clear(List_ASTLet *list);

/* Get the current length of the list */
int nl_list_ASTLet_length(List_ASTLet *list);

/* Get the current capacity of the list */
int nl_list_ASTLet_capacity(List_ASTLet *list);

/* Check if the list is empty */
bool nl_list_ASTLet_is_empty(List_ASTLet *list);

/* Free the list and all its resources */
void nl_list_ASTLet_free(List_ASTLet *list);

#endif /* LIST_ASTLET_H */
