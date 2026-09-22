#ifndef LIST_LEXERTOKEN_NEW_H
#define LIST_LEXERTOKEN_NEW_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of LexerToken_new */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_LexerToken_new
#define DEFINED_List_LexerToken_new
typedef struct List_LexerToken_new {
    LexerToken_new *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_LexerToken_new;
#endif

/* Create a new empty list */
List_LexerToken_new* nl_list_LexerToken_new_new(void);

/* Create a new list with specified initial capacity */
List_LexerToken_new* nl_list_LexerToken_new_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_LexerToken_new_push(List_LexerToken_new *list, LexerToken_new value);

/* Remove and return the last element */
LexerToken_new nl_list_LexerToken_new_pop(List_LexerToken_new *list);

/* Insert an element at the specified index */
void nl_list_LexerToken_new_insert(List_LexerToken_new *list, int index, LexerToken_new value);

/* Remove and return the element at the specified index */
LexerToken_new nl_list_LexerToken_new_remove(List_LexerToken_new *list, int index);

/* Set the value at the specified index */
void nl_list_LexerToken_new_set(List_LexerToken_new *list, int index, LexerToken_new value);

/* Get the value at the specified index */
LexerToken_new nl_list_LexerToken_new_get(List_LexerToken_new *list, int index);

/* Clear all elements from the list */
void nl_list_LexerToken_new_clear(List_LexerToken_new *list);

/* Get the current length of the list */
int nl_list_LexerToken_new_length(List_LexerToken_new *list);

/* Get the current capacity of the list */
int nl_list_LexerToken_new_capacity(List_LexerToken_new *list);

/* Check if the list is empty */
bool nl_list_LexerToken_new_is_empty(List_LexerToken_new *list);

/* Free the list and all its resources */
void nl_list_LexerToken_new_free(List_LexerToken_new *list);

#endif /* LIST_LEXERTOKEN_NEW_H */
