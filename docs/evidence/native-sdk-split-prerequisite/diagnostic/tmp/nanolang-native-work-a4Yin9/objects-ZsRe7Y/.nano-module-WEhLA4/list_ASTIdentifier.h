#ifndef LIST_ASTIDENTIFIER_H
#define LIST_ASTIDENTIFIER_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTIdentifier */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTIdentifier
#define DEFINED_List_ASTIdentifier
typedef struct List_ASTIdentifier {
    ASTIdentifier *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTIdentifier;
#endif

/* Create a new empty list */
List_ASTIdentifier* nl_list_ASTIdentifier_new(void);

/* Create a new list with specified initial capacity */
List_ASTIdentifier* nl_list_ASTIdentifier_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTIdentifier_push(List_ASTIdentifier *list, ASTIdentifier value);

/* Remove and return the last element */
ASTIdentifier nl_list_ASTIdentifier_pop(List_ASTIdentifier *list);

/* Insert an element at the specified index */
void nl_list_ASTIdentifier_insert(List_ASTIdentifier *list, int index, ASTIdentifier value);

/* Remove and return the element at the specified index */
ASTIdentifier nl_list_ASTIdentifier_remove(List_ASTIdentifier *list, int index);

/* Set the value at the specified index */
void nl_list_ASTIdentifier_set(List_ASTIdentifier *list, int index, ASTIdentifier value);

/* Get the value at the specified index */
ASTIdentifier nl_list_ASTIdentifier_get(List_ASTIdentifier *list, int index);

/* Clear all elements from the list */
void nl_list_ASTIdentifier_clear(List_ASTIdentifier *list);

/* Get the current length of the list */
int nl_list_ASTIdentifier_length(List_ASTIdentifier *list);

/* Get the current capacity of the list */
int nl_list_ASTIdentifier_capacity(List_ASTIdentifier *list);

/* Check if the list is empty */
bool nl_list_ASTIdentifier_is_empty(List_ASTIdentifier *list);

/* Free the list and all its resources */
void nl_list_ASTIdentifier_free(List_ASTIdentifier *list);

#endif /* LIST_ASTIDENTIFIER_H */
