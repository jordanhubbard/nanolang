#ifndef LIST_COMPILERDIAGNOSTIC_PUSH_H
#define LIST_COMPILERDIAGNOSTIC_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of CompilerDiagnostic_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_CompilerDiagnostic_push
#define DEFINED_List_CompilerDiagnostic_push
typedef struct List_CompilerDiagnostic_push {
    CompilerDiagnostic_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_CompilerDiagnostic_push;
#endif

/* Create a new empty list */
List_CompilerDiagnostic_push* nl_list_CompilerDiagnostic_push_new(void);

/* Create a new list with specified initial capacity */
List_CompilerDiagnostic_push* nl_list_CompilerDiagnostic_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_CompilerDiagnostic_push_push(List_CompilerDiagnostic_push *list, CompilerDiagnostic_push value);

/* Remove and return the last element */
CompilerDiagnostic_push nl_list_CompilerDiagnostic_push_pop(List_CompilerDiagnostic_push *list);

/* Insert an element at the specified index */
void nl_list_CompilerDiagnostic_push_insert(List_CompilerDiagnostic_push *list, int index, CompilerDiagnostic_push value);

/* Remove and return the element at the specified index */
CompilerDiagnostic_push nl_list_CompilerDiagnostic_push_remove(List_CompilerDiagnostic_push *list, int index);

/* Set the value at the specified index */
void nl_list_CompilerDiagnostic_push_set(List_CompilerDiagnostic_push *list, int index, CompilerDiagnostic_push value);

/* Get the value at the specified index */
CompilerDiagnostic_push nl_list_CompilerDiagnostic_push_get(List_CompilerDiagnostic_push *list, int index);

/* Clear all elements from the list */
void nl_list_CompilerDiagnostic_push_clear(List_CompilerDiagnostic_push *list);

/* Get the current length of the list */
int nl_list_CompilerDiagnostic_push_length(List_CompilerDiagnostic_push *list);

/* Get the current capacity of the list */
int nl_list_CompilerDiagnostic_push_capacity(List_CompilerDiagnostic_push *list);

/* Check if the list is empty */
bool nl_list_CompilerDiagnostic_push_is_empty(List_CompilerDiagnostic_push *list);

/* Free the list and all its resources */
void nl_list_CompilerDiagnostic_push_free(List_CompilerDiagnostic_push *list);

#endif /* LIST_COMPILERDIAGNOSTIC_PUSH_H */
