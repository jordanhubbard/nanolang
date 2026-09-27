#ifndef LIST_COMPILERDIAGNOSTIC_GET_H
#define LIST_COMPILERDIAGNOSTIC_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerDiagnostic_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_CompilerDiagnostic_get
#define DEFINED_List_CompilerDiagnostic_get
typedef struct List_CompilerDiagnostic_get {
    CompilerDiagnostic_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerDiagnostic_get;
#endif

/* Create a new empty list */
List_CompilerDiagnostic_get* nl_list_CompilerDiagnostic_get_new(void);

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic_get* nl_list_CompilerDiagnostic_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_get_push(List_CompilerDiagnostic_get *list, CompilerDiagnostic_get value);

/* Remove and return the last element */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_pop(List_CompilerDiagnostic_get *list);

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_get_insert(List_CompilerDiagnostic_get *list, int index, CompilerDiagnostic_get value);

/* Remove and return the element at the specified index */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_remove(List_CompilerDiagnostic_get *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_get_set(List_CompilerDiagnostic_get *list, int index, CompilerDiagnostic_get value);

/* Get the value at the specified index */
CompilerDiagnostic_get nl_list_CompilerDiagnostic_get_get(List_CompilerDiagnostic_get *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_get_clear(List_CompilerDiagnostic_get *list);

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_get_length(List_CompilerDiagnostic_get *list);

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_get_capacity(List_CompilerDiagnostic_get *list);

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_get_is_empty(List_CompilerDiagnostic_get *list);

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_get_free(List_CompilerDiagnostic_get *list);

#endif /* LIST_COMPILERDIAGNOSTIC_GET_H */
