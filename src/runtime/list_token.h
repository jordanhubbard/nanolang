#ifndef LIST_TOKEN_H
#define LIST_TOKEN_H

#include "nanolang.h"
#include <stdbool.h>

/* Dynamic list of Token structs */
typedef struct {
    Token *data;         /* Array of Token structs */
    int length;          /* Current number of elements */
    int capacity;        /* Allocated capacity */
} List_token;

/* Create a new empty list */
List_token* list_token_new(void);

/* Create a new list with specified initial capacity */
List_token* list_token_with_capacity(int capacity);

/* Append an element to the end of the list (copies the token) */
void list_token_push(List_token *list, Token token);

/* Remove and return the last element (caller must free token.value) */
Token list_token_pop(List_token *list);

/* Insert an element at the specified index (copies the token) */
void list_token_insert(List_token *list, int index, Token token);

/* Remove and return the element at the specified index (caller must free token.value) */
Token list_token_remove(List_token *list, int index);

/* Set the value at the specified index (copies the token, frees old token.value) */
void list_token_set(List_token *list, int index, Token token);

/* Get the value at the specified index (returns pointer to internal token) */
Token* list_token_get(List_token *list, int index);

/* Clear all elements from the list (frees all token.value strings) */
void list_token_clear(List_token *list);

/* Get the current length of the list */
int list_token_length(List_token *list);

/* Get the current capacity of the list */
int list_token_capacity(List_token *list);

/* Check if the list is empty */
bool list_token_is_empty(List_token *list);

/* Free the list and all its resources (frees all token.value strings) */
void list_token_free(List_token *list);

#endif /* LIST_TOKEN_H */

