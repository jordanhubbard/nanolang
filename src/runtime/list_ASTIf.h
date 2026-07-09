#ifndef LIST_ASTIF_H
#define LIST_ASTIF_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTIf */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTIf
#define FORWARD_DEFINED_List_ASTIf
typedef struct List_ASTIf {
    struct nl_ASTIf *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTIf;
#endif

/* Create a new empty list */
List_ASTIf* nl_list_ASTIf_new(void);

/* Create a new list with specified initial capacity */
List_ASTIf* nl_list_ASTIf_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTIf_push(List_ASTIf *list, struct nl_ASTIf value);

/* Remove and return the last element */
struct nl_ASTIf nl_list_ASTIf_pop(List_ASTIf *list);

/* Insert an element at the specified index */
void nl_list_ASTIf_insert(List_ASTIf *list, int index, struct nl_ASTIf value);

/* Remove and return the element at the specified index */
struct nl_ASTIf nl_list_ASTIf_remove(List_ASTIf *list, int index);

/* Set the value at the specified index */
void nl_list_ASTIf_set(List_ASTIf *list, int index, struct nl_ASTIf value);

/* Get the value at the specified index */
struct nl_ASTIf nl_list_ASTIf_get(List_ASTIf *list, int index);

/* Clear all elements from the list */
void nl_list_ASTIf_clear(List_ASTIf *list);

/* Get the current length of the list */
int nl_list_ASTIf_length(List_ASTIf *list);

/* Get the current capacity of the list */
int nl_list_ASTIf_capacity(List_ASTIf *list);

/* Check if the list is empty */
bool nl_list_ASTIf_is_empty(List_ASTIf *list);

/* Free the list and all its resources */
void nl_list_ASTIf_free(List_ASTIf *list);

#endif /* LIST_ASTIF_H */
