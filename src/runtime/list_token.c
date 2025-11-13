#include "list_token.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Copy a token (deep copy of value string) */
static Token token_copy(const Token *src) {
    Token dst;
    dst.type = src->type;
    dst.value = src->value ? strdup(src->value) : NULL;
    dst.line = src->line;
    dst.column = src->column;
    return dst;
}

/* Helper: Free a token's value string */
static void token_free_value(Token *token) {
    if (token->value) {
        free(token->value);
        token->value = NULL;
    }
}

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity(List_token *list, int min_capacity) {
    if (list->capacity >= min_capacity) {
        return;
    }
    
    int new_capacity = list->capacity;
    if (new_capacity == 0) {
        new_capacity = INITIAL_CAPACITY;
    }
    
    while (new_capacity < min_capacity) {
        new_capacity *= GROWTH_FACTOR;
    }
    
    Token *new_data = realloc(list->data, sizeof(Token) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list_token\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_token* list_token_new(void) {
    return list_token_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_token* list_token_with_capacity(int capacity) {
    List_token *list = malloc(sizeof(List_token));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list_token\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(Token) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list_token data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list (copies the token) */
void list_token_push(List_token *list, Token token) {
    ensure_capacity(list, list->length + 1);
    list->data[list->length] = token_copy(&token);
    list->length++;
}

/* Remove and return the last element (caller must free token.value) */
Token list_token_pop(List_token *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list_token\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index (copies the token) */
void list_token_insert(List_token *list, int index, Token token) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list_token of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(Token) * (list->length - index));
    
    list->data[index] = token_copy(&token);
    list->length++;
}

/* Remove and return the element at the specified index (caller must free token.value) */
Token list_token_remove(List_token *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list_token of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    Token removed = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(Token) * (list->length - index - 1));
    
    list->length--;
    return removed;
}

/* Set the value at the specified index (copies the token, frees old token.value) */
void list_token_set(List_token *list, int index, Token token) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list_token of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    /* Free old token's value */
    token_free_value(&list->data[index]);
    
    /* Copy new token */
    list->data[index] = token_copy(&token);
}

/* Get the value at the specified index (returns pointer to internal token) */
Token* list_token_get(List_token *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list_token of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return &list->data[index];
}

/* Clear all elements from the list (frees all token.value strings) */
void list_token_clear(List_token *list) {
    for (int i = 0; i < list->length; i++) {
        token_free_value(&list->data[i]);
    }
    list->length = 0;
}

/* Get the current length of the list */
int list_token_length(List_token *list) {
    return list->length;
}

/* Get the current capacity of the list */
int list_token_capacity(List_token *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool list_token_is_empty(List_token *list) {
    return list->length == 0;
}

/* Free the list and all its resources (frees all token.value strings) */
void list_token_free(List_token *list) {
    if (!list) {
        return;
    }
    
    /* Free all token values */
    for (int i = 0; i < list->length; i++) {
        token_free_value(&list->data[i]);
    }
    
    /* Free data array */
    if (list->data) {
        free(list->data);
    }
    
    /* Free list structure */
    free(list);
}

