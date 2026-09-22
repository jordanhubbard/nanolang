#ifndef LIST_COMPILERDIAGNOSTIC_NEW_H
#define LIST_COMPILERDIAGNOSTIC_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerDiagnostic_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_CompilerDiagnostic_new
#define DEFINED_List_CompilerDiagnostic_new
typedef struct List_CompilerDiagnostic_new {
    CompilerDiagnostic_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerDiagnostic_new;
#endif

/* Create a new empty list */
List_CompilerDiagnostic_new* nl_list_CompilerDiagnostic_new_new(void);

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic_new* nl_list_CompilerDiagnostic_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_new_push(List_CompilerDiagnostic_new *list, CompilerDiagnostic_new value);

/* Remove and return the last element */
CompilerDiagnostic_new nl_list_CompilerDiagnostic_new_pop(List_CompilerDiagnostic_new *list);

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_new_insert(List_CompilerDiagnostic_new *list, int index, CompilerDiagnostic_new value);

/* Remove and return the element at the specified index */
CompilerDiagnostic_new nl_list_CompilerDiagnostic_new_remove(List_CompilerDiagnostic_new *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_new_set(List_CompilerDiagnostic_new *list, int index, CompilerDiagnostic_new value);

/* Get the value at the specified index */
CompilerDiagnostic_new nl_list_CompilerDiagnostic_new_get(List_CompilerDiagnostic_new *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_new_clear(List_CompilerDiagnostic_new *list);

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_new_length(List_CompilerDiagnostic_new *list);

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_new_capacity(List_CompilerDiagnostic_new *list);

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_new_is_empty(List_CompilerDiagnostic_new *list);

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_new_free(List_CompilerDiagnostic_new *list);

#endif /* LIST_COMPILERDIAGNOSTIC_NEW_H */
