#ifndef LIST_COMPILERDIAGNOSTIC_H
#define LIST_COMPILERDIAGNOSTIC_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerDiagnostic */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_CompilerDiagnostic
#define FORWARD_DEFINED_List_CompilerDiagnostic
typedef struct List_CompilerDiagnostic {
    struct nl_CompilerDiagnostic *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerDiagnostic;
#endif

/* Create a new empty list */
List_CompilerDiagnostic* nl_list_CompilerDiagnostic_new(void);

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic* nl_list_CompilerDiagnostic_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_push(List_CompilerDiagnostic *list, struct nl_CompilerDiagnostic value);

/* Remove and return the last element */
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_pop(List_CompilerDiagnostic *list);

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_insert(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value);

/* Remove and return the element at the specified index */
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_remove(List_CompilerDiagnostic *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_set(List_CompilerDiagnostic *list, int index, struct nl_CompilerDiagnostic value);

/* Get the value at the specified index */
struct nl_CompilerDiagnostic nl_list_CompilerDiagnostic_get(List_CompilerDiagnostic *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_clear(List_CompilerDiagnostic *list);

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_length(List_CompilerDiagnostic *list);

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_capacity(List_CompilerDiagnostic *list);

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_is_empty(List_CompilerDiagnostic *list);

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_free(List_CompilerDiagnostic *list);

#endif /* LIST_COMPILERDIAGNOSTIC_H */
