#ifndef LIST_LEXERTOKEN_LENGTH_H
#define LIST_LEXERTOKEN_LENGTH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of LexerToken_length */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_LexerToken_length
#define DEFINED_List_LexerToken_length
typedef struct List_LexerToken_length {
    LexerToken_length *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_LexerToken_length;
#endif

/* Create a new empty list */
List_LexerToken_length* nl_list_LexerToken_length_new(void);

/* Create a new list with specified initial capacity */
List_LexerToken_length* nl_list_LexerToken_length_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_LexerToken_length_push(List_LexerToken_length *list, LexerToken_length value);

/* Remove and return the last element */
LexerToken_length nl_list_LexerToken_length_pop(List_LexerToken_length *list);

/* Insert an element at the specified index */
void nl_list_LexerToken_length_insert(List_LexerToken_length *list, int index, LexerToken_length value);

/* Remove and return the element at the specified index */
LexerToken_length nl_list_LexerToken_length_remove(List_LexerToken_length *list, int index);

/* Set the value at the specified index */
void nl_list_LexerToken_length_set(List_LexerToken_length *list, int index, LexerToken_length value);

/* Get the value at the specified index */
LexerToken_length nl_list_LexerToken_length_get(List_LexerToken_length *list, int index);

/* Clear all elements from the list */
void nl_list_LexerToken_length_clear(List_LexerToken_length *list);

/* Get the current length of the list */
int nl_list_LexerToken_length_length(List_LexerToken_length *list);

/* Get the current capacity of the list */
int nl_list_LexerToken_length_capacity(List_LexerToken_length *list);

/* Check if the list is empty */
bool nl_list_LexerToken_length_is_empty(List_LexerToken_length *list);

/* Free the list and all its resources */
void nl_list_LexerToken_length_free(List_LexerToken_length *list);

#endif /* LIST_LEXERTOKEN_LENGTH_H */
