#ifndef LIST_COMPILERDIAGNOSTIC_LENGTH_H
#define LIST_COMPILERDIAGNOSTIC_LENGTH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerDiagnostic_length */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_CompilerDiagnostic_length
#define DEFINED_List_CompilerDiagnostic_length
typedef struct List_CompilerDiagnostic_length {
    CompilerDiagnostic_length *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerDiagnostic_length;
#endif

/* Create a new empty list */
List_CompilerDiagnostic_length* nl_list_CompilerDiagnostic_length_new(void);

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic_length* nl_list_CompilerDiagnostic_length_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_length_push(List_CompilerDiagnostic_length *list, CompilerDiagnostic_length value);

/* Remove and return the last element */
CompilerDiagnostic_length nl_list_CompilerDiagnostic_length_pop(List_CompilerDiagnostic_length *list);

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_length_insert(List_CompilerDiagnostic_length *list, int index, CompilerDiagnostic_length value);

/* Remove and return the element at the specified index */
CompilerDiagnostic_length nl_list_CompilerDiagnostic_length_remove(List_CompilerDiagnostic_length *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_length_set(List_CompilerDiagnostic_length *list, int index, CompilerDiagnostic_length value);

/* Get the value at the specified index */
CompilerDiagnostic_length nl_list_CompilerDiagnostic_length_get(List_CompilerDiagnostic_length *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_length_clear(List_CompilerDiagnostic_length *list);

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_length_length(List_CompilerDiagnostic_length *list);

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_length_capacity(List_CompilerDiagnostic_length *list);

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_length_is_empty(List_CompilerDiagnostic_length *list);

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_length_free(List_CompilerDiagnostic_length *list);

#endif /* LIST_COMPILERDIAGNOSTIC_LENGTH_H */
