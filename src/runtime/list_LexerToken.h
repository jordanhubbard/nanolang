#ifndef LIST_LEXERTOKEN_H
#define LIST_LEXERTOKEN_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of LexerToken */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_LexerToken
#define FORWARD_DEFINED_List_LexerToken
typedef struct List_LexerToken {
    struct nl_LexerToken *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_LexerToken;
#endif

/* Create a new empty list */
List_LexerToken* nl_list_LexerToken_new(void);

/* Create a new list with specified initial capacity */
List_LexerToken* nl_list_LexerToken_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_LexerToken_push(List_LexerToken *list, struct nl_LexerToken value);

/* Remove and return the last element */
struct nl_LexerToken nl_list_LexerToken_pop(List_LexerToken *list);

/* Insert an element at the specified index */
void nl_list_LexerToken_insert(List_LexerToken *list, int index, struct nl_LexerToken value);

/* Remove and return the element at the specified index */
struct nl_LexerToken nl_list_LexerToken_remove(List_LexerToken *list, int index);

/* Set the value at the specified index */
void nl_list_LexerToken_set(List_LexerToken *list, int index, struct nl_LexerToken value);

/* Get the value at the specified index */
struct nl_LexerToken nl_list_LexerToken_get(List_LexerToken *list, int index);

/* Clear all elements from the list */
void nl_list_LexerToken_clear(List_LexerToken *list);

/* Get the current length of the list */
int nl_list_LexerToken_length(List_LexerToken *list);

/* Get the current capacity of the list */
int nl_list_LexerToken_capacity(List_LexerToken *list);

/* Check if the list is empty */
bool nl_list_LexerToken_is_empty(List_LexerToken *list);

/* Free the list and all its resources */
void nl_list_LexerToken_free(List_LexerToken *list);

#endif /* LIST_LEXERTOKEN_H */
