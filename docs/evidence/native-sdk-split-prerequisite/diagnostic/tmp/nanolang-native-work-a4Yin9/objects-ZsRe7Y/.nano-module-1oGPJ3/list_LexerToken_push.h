#ifndef LIST_LEXERTOKEN_PUSH_H
#define LIST_LEXERTOKEN_PUSH_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of LexerToken_push */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_LexerToken_push
#define DEFINED_List_LexerToken_push
typedef struct List_LexerToken_push {
    LexerToken_push *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_LexerToken_push;
#endif

/* Create a new empty list */
List_LexerToken_push* nl_list_LexerToken_push_new(void);

/* Create a new list with specified initial capacity */
List_LexerToken_push* nl_list_LexerToken_push_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_LexerToken_push_push(List_LexerToken_push *list, LexerToken_push value);

/* Remove and return the last element */
LexerToken_push nl_list_LexerToken_push_pop(List_LexerToken_push *list);

/* Insert an element at the specified index */
void nl_list_LexerToken_push_insert(List_LexerToken_push *list, int index, LexerToken_push value);

/* Remove and return the element at the specified index */
LexerToken_push nl_list_LexerToken_push_remove(List_LexerToken_push *list, int index);

/* Set the value at the specified index */
void nl_list_LexerToken_push_set(List_LexerToken_push *list, int index, LexerToken_push value);

/* Get the value at the specified index */
LexerToken_push nl_list_LexerToken_push_get(List_LexerToken_push *list, int index);

/* Clear all elements from the list */
void nl_list_LexerToken_push_clear(List_LexerToken_push *list);

/* Get the current length of the list */
int nl_list_LexerToken_push_length(List_LexerToken_push *list);

/* Get the current capacity of the list */
int nl_list_LexerToken_push_capacity(List_LexerToken_push *list);

/* Check if the list is empty */
bool nl_list_LexerToken_push_is_empty(List_LexerToken_push *list);

/* Free the list and all its resources */
void nl_list_LexerToken_push_free(List_LexerToken_push *list);

#endif /* LIST_LEXERTOKEN_PUSH_H */
