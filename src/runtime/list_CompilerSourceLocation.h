#ifndef LIST_COMPILERSOURCELOCATION_H
#define LIST_COMPILERSOURCELOCATION_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerSourceLocation */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_CompilerSourceLocation
#define FORWARD_DEFINED_List_CompilerSourceLocation
typedef struct List_CompilerSourceLocation {
    struct nl_CompilerSourceLocation *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerSourceLocation;
#endif

/* Create a new empty list */
List_CompilerSourceLocation* nl_list_CompilerSourceLocation_new(void);

/* Create a new list with specified initial capacity */
List_CompilerSourceLocation* nl_list_CompilerSourceLocation_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerSourceLocation_push(List_CompilerSourceLocation *list, struct nl_CompilerSourceLocation value);

/* Remove and return the last element */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_pop(List_CompilerSourceLocation *list);

/* Insert an element at the specified index */
void nl_list_CompilerSourceLocation_insert(List_CompilerSourceLocation *list, int index, struct nl_CompilerSourceLocation value);

/* Remove and return the element at the specified index */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_remove(List_CompilerSourceLocation *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerSourceLocation_set(List_CompilerSourceLocation *list, int index, struct nl_CompilerSourceLocation value);

/* Get the value at the specified index */
struct nl_CompilerSourceLocation nl_list_CompilerSourceLocation_get(List_CompilerSourceLocation *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerSourceLocation_clear(List_CompilerSourceLocation *list);

/* Get the current length of the list */
int nl_list_CompilerSourceLocation_length(List_CompilerSourceLocation *list);

/* Get the current capacity of the list */
int nl_list_CompilerSourceLocation_capacity(List_CompilerSourceLocation *list);

/* Check if the list is empty */
bool nl_list_CompilerSourceLocation_is_empty(List_CompilerSourceLocation *list);

/* Free the list and all its resources */
void nl_list_CompilerSourceLocation_free(List_CompilerSourceLocation *list);

#endif /* LIST_COMPILERSOURCELOCATION_H */
