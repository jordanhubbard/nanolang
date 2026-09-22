#ifndef LIST_LEXERTOKEN_GET_H
#define LIST_LEXERTOKEN_GET_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of LexerToken_get */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_LexerToken_get
#define DEFINED_List_LexerToken_get
typedef struct List_LexerToken_get {
    LexerToken_get *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_LexerToken_get;
#endif

/* Create a new empty list */
List_LexerToken_get* nl_list_LexerToken_get_new(void);

/* Create a new list with specified initial capacity */
List_LexerToken_get* nl_list_LexerToken_get_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_LexerToken_get_push(List_LexerToken_get *list, LexerToken_get value);

/* Remove and return the last element */
LexerToken_get nl_list_LexerToken_get_pop(List_LexerToken_get *list);

/* Insert an element at the specified index */
void nl_list_LexerToken_get_insert(List_LexerToken_get *list, int index, LexerToken_get value);

/* Remove and return the element at the specified index */
LexerToken_get nl_list_LexerToken_get_remove(List_LexerToken_get *list, int index);

/* Set the value at the specified index */
void nl_list_LexerToken_get_set(List_LexerToken_get *list, int index, LexerToken_get value);

/* Get the value at the specified index */
LexerToken_get nl_list_LexerToken_get_get(List_LexerToken_get *list, int index);

/* Clear all elements from the list */
void nl_list_LexerToken_get_clear(List_LexerToken_get *list);

/* Get the current length of the list */
int nl_list_LexerToken_get_length(List_LexerToken_get *list);

/* Get the current capacity of the list */
int nl_list_LexerToken_get_capacity(List_LexerToken_get *list);

/* Check if the list is empty */
bool nl_list_LexerToken_get_is_empty(List_LexerToken_get *list);

/* Free the list and all its resources */
void nl_list_LexerToken_get_free(List_LexerToken_get *list);

#endif /* LIST_LEXERTOKEN_GET_H */
