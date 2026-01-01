#ifndef LIST_ASTWHILE_H
#define LIST_ASTWHILE_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTWhile */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTWhile
#define DEFINED_List_ASTWhile
typedef struct List_ASTWhile {
    struct nl_ASTWhile *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTWhile;
#endif

/* Create a new empty list */
List_ASTWhile* nl_list_ASTWhile_new(void);

/* Create a new list with specified initial capacity */
List_ASTWhile* nl_list_ASTWhile_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTWhile_push(List_ASTWhile *list, struct nl_ASTWhile value);

/* Remove and return the last element */
struct nl_ASTWhile nl_list_ASTWhile_pop(List_ASTWhile *list);

/* Insert an element at the specified index */
void nl_list_ASTWhile_insert(List_ASTWhile *list, int index, struct nl_ASTWhile value);

/* Remove and return the element at the specified index */
struct nl_ASTWhile nl_list_ASTWhile_remove(List_ASTWhile *list, int index);

/* Set the value at the specified index */
void nl_list_ASTWhile_set(List_ASTWhile *list, int index, struct nl_ASTWhile value);

/* Get the value at the specified index */
struct nl_ASTWhile nl_list_ASTWhile_get(List_ASTWhile *list, int index);

/* Clear all elements from the list */
void nl_list_ASTWhile_clear(List_ASTWhile *list);

/* Get the current length of the list */
int nl_list_ASTWhile_length(List_ASTWhile *list);

/* Get the current capacity of the list */
int nl_list_ASTWhile_capacity(List_ASTWhile *list);

/* Check if the list is empty */
bool nl_list_ASTWhile_is_empty(List_ASTWhile *list);

/* Free the list and all its resources */
void nl_list_ASTWhile_free(List_ASTWhile *list);

#endif /* LIST_ASTWHILE_H */
