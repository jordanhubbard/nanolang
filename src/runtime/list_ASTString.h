#ifndef LIST_ASTSTRING_H
#define LIST_ASTSTRING_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTString */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTString
#define FORWARD_DEFINED_List_ASTString
typedef struct List_ASTString {
    struct nl_ASTString *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTString;
#endif

/* Create a new empty list */
List_ASTString* nl_list_ASTString_new(void);

/* Create a new list with specified initial capacity */
List_ASTString* nl_list_ASTString_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTString_push(List_ASTString *list, struct nl_ASTString value);

/* Remove and return the last element */
struct nl_ASTString nl_list_ASTString_pop(List_ASTString *list);

/* Insert an element at the specified index */
void nl_list_ASTString_insert(List_ASTString *list, int index, struct nl_ASTString value);

/* Remove and return the element at the specified index */
struct nl_ASTString nl_list_ASTString_remove(List_ASTString *list, int index);

/* Set the value at the specified index */
void nl_list_ASTString_set(List_ASTString *list, int index, struct nl_ASTString value);

/* Get the value at the specified index */
struct nl_ASTString nl_list_ASTString_get(List_ASTString *list, int index);

/* Clear all elements from the list */
void nl_list_ASTString_clear(List_ASTString *list);

/* Get the current length of the list */
int nl_list_ASTString_length(List_ASTString *list);

/* Get the current capacity of the list */
int nl_list_ASTString_capacity(List_ASTString *list);

/* Check if the list is empty */
bool nl_list_ASTString_is_empty(List_ASTString *list);

/* Free the list and all its resources */
void nl_list_ASTString_free(List_ASTString *list);

#endif /* LIST_ASTSTRING_H */
