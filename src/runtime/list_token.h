#ifndef LIST_TOKEN_H
#define LIST_TOKEN_H

#include <stdint.h>
#include <stdbool.h>

/* Forward declaration - actual struct defined elsewhere */
struct Token;

/* Dynamic list of Token */
/* Guard typedef to prevent redefinition warnings */
#ifndef FORWARD_DEFINED_List_Token
#define FORWARD_DEFINED_List_Token
typedef struct List_Token {
    struct nl_LexerToken *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_Token;
#endif

/* Create a new empty list */
List_Token* nl_list_Token_new(void);

/* Create a new list with specified initial capacity */
List_Token* nl_list_Token_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_Token_push(List_Token *list, struct nl_LexerToken value);

/* Remove and return the last element */
struct nl_LexerToken nl_list_Token_pop(List_Token *list);

/* Insert an element at the specified index */
void nl_list_Token_insert(List_Token *list, int index, struct nl_LexerToken value);

/* Remove and return the element at the specified index */
struct nl_LexerToken nl_list_Token_remove(List_Token *list, int index);

/* Set the value at the specified index */
void nl_list_Token_set(List_Token *list, int index, struct nl_LexerToken value);

/* Get the value at the specified index */
struct nl_LexerToken nl_list_Token_get(List_Token *list, int index);

/* Clear all elements from the list */
void nl_list_Token_clear(List_Token *list);

/* Get the current length of the list */
int nl_list_Token_length(List_Token *list);

/* Get the current capacity of the list */
int nl_list_Token_capacity(List_Token *list);

/* Check if the list is empty */
bool nl_list_Token_is_empty(List_Token *list);

/* Free the list and all its resources */
void nl_list_Token_free(List_Token *list);

#endif /* LIST_TOKEN_H */
