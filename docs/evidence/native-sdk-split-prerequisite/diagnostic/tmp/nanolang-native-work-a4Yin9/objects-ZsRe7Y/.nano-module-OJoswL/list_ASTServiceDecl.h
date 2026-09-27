#ifndef LIST_ASTSERVICEDECL_H
#define LIST_ASTSERVICEDECL_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTServiceDecl */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_ASTServiceDecl
#define DEFINED_List_ASTServiceDecl
typedef struct List_ASTServiceDecl {
    ASTServiceDecl *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTServiceDecl;
#endif

/* Create a new empty list */
List_ASTServiceDecl* nl_list_ASTServiceDecl_new(void);

/* Create a new list with specified initial capacity */
List_ASTServiceDecl* nl_list_ASTServiceDecl_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTServiceDecl_push(List_ASTServiceDecl *list, ASTServiceDecl value);

/* Remove and return the last element */
ASTServiceDecl nl_list_ASTServiceDecl_pop(List_ASTServiceDecl *list);

/* Insert an element at the specified index */
void nl_list_ASTServiceDecl_insert(List_ASTServiceDecl *list, int index, ASTServiceDecl value);

/* Remove and return the element at the specified index */
ASTServiceDecl nl_list_ASTServiceDecl_remove(List_ASTServiceDecl *list, int index);

/* Set the value at the specified index */
void nl_list_ASTServiceDecl_set(List_ASTServiceDecl *list, int index, ASTServiceDecl value);

/* Get the value at the specified index */
ASTServiceDecl nl_list_ASTServiceDecl_get(List_ASTServiceDecl *list, int index);

/* Clear all elements from the list */
void nl_list_ASTServiceDecl_clear(List_ASTServiceDecl *list);

/* Get the current length of the list */
int nl_list_ASTServiceDecl_length(List_ASTServiceDecl *list);

/* Get the current capacity of the list */
int nl_list_ASTServiceDecl_capacity(List_ASTServiceDecl *list);

/* Check if the list is empty */
bool nl_list_ASTServiceDecl_is_empty(List_ASTServiceDecl *list);

/* Free the list and all its resources */
void nl_list_ASTServiceDecl_free(List_ASTServiceDecl *list);

#endif /* LIST_ASTSERVICEDECL_H */
