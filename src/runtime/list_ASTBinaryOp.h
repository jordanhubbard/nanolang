#ifndef LIST_ASTBINARYOP_H
#define LIST_ASTBINARYOP_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of ASTBinaryOp */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_ASTBinaryOp
#define FORWARD_DEFINED_List_ASTBinaryOp
typedef struct List_ASTBinaryOp {
    struct nl_ASTBinaryOp *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_ASTBinaryOp;
#endif

/* Create a new empty list */
List_ASTBinaryOp* nl_list_ASTBinaryOp_new(void);

/* Create a new list with specified initial capacity */
List_ASTBinaryOp* nl_list_ASTBinaryOp_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_ASTBinaryOp_push(List_ASTBinaryOp *list, struct nl_ASTBinaryOp value);

/* Remove and return the last element */
struct nl_ASTBinaryOp nl_list_ASTBinaryOp_pop(List_ASTBinaryOp *list);

/* Insert an element at the specified index */
void nl_list_ASTBinaryOp_insert(List_ASTBinaryOp *list, int index, struct nl_ASTBinaryOp value);

/* Remove and return the element at the specified index */
struct nl_ASTBinaryOp nl_list_ASTBinaryOp_remove(List_ASTBinaryOp *list, int index);

/* Set the value at the specified index */
void nl_list_ASTBinaryOp_set(List_ASTBinaryOp *list, int index, struct nl_ASTBinaryOp value);

/* Get the value at the specified index */
struct nl_ASTBinaryOp nl_list_ASTBinaryOp_get(List_ASTBinaryOp *list, int index);

/* Clear all elements from the list */
void nl_list_ASTBinaryOp_clear(List_ASTBinaryOp *list);

/* Get the current length of the list */
int nl_list_ASTBinaryOp_length(List_ASTBinaryOp *list);

/* Get the current capacity of the list */
int nl_list_ASTBinaryOp_capacity(List_ASTBinaryOp *list);

/* Check if the list is empty */
bool nl_list_ASTBinaryOp_is_empty(List_ASTBinaryOp *list);

/* Free the list and all its resources */
void nl_list_ASTBinaryOp_free(List_ASTBinaryOp *list);

#endif /* LIST_ASTBINARYOP_H */
