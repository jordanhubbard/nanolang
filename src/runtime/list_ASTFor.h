#ifndef LIST_ASTFOR_H
#define LIST_ASTFOR_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTFor */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTFor
#define FORWARD_DEFINED_List_ASTFor
typedef struct List_ASTFor {
    struct nl_ASTFor *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTFor;
#endif

/* Create a new empty list */
List_ASTFor* nl_list_ASTFor_new(void);

/* Create a new list with specified initial capacity */
List_ASTFor* nl_list_ASTFor_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTFor_push(List_ASTFor *list, struct nl_ASTFor value);

/* Remove and return the last element */
struct nl_ASTFor nl_list_ASTFor_pop(List_ASTFor *list);

/* Insert an element at the specified index */
void nl_list_ASTFor_insert(List_ASTFor *list, int index, struct nl_ASTFor value);

/* Remove and return the element at the specified index */
struct nl_ASTFor nl_list_ASTFor_remove(List_ASTFor *list, int index);

/* Set the value at the specified index */
void nl_list_ASTFor_set(List_ASTFor *list, int index, struct nl_ASTFor value);

/* Get the value at the specified index */
struct nl_ASTFor nl_list_ASTFor_get(List_ASTFor *list, int index);

/* Clear all elements from the list */
void nl_list_ASTFor_clear(List_ASTFor *list);

/* Get the current length of the list */
int nl_list_ASTFor_length(List_ASTFor *list);

/* Get the current capacity of the list */
int nl_list_ASTFor_capacity(List_ASTFor *list);

/* Check if the list is empty */
bool nl_list_ASTFor_is_empty(List_ASTFor *list);

/* Free the list and all its resources */
void nl_list_ASTFor_free(List_ASTFor *list);

#endif /* LIST_ASTFOR_H */
