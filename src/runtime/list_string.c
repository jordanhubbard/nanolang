#include "list_string.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity(List_string *list, int min_capacity) {
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
    
    char **new_data = realloc(list->data, sizeof(char*) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_string* list_string_new(void) {
    return list_string_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_string* list_string_with_capacity(int capacity) {
    List_string *list = malloc(sizeof(List_string));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof(char*) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void list_string_push(List_string *list, const char *value) {
    ensure_capacity(list, list->length + 1);
    list->data[list->length] = strdup(value);  /* Copy the string */
    list->length++;
}

/* Remove and return the last element */
char* list_string_pop(List_string *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void list_string_insert(List_string *list, int index, const char *value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof(char*) * (list->length - index));
    
    list->data[index] = strdup(value);  /* Copy the string */
    list->length++;
}

/* Remove and return the element at the specified index */
char* list_string_remove(List_string *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    char *value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof(char*) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void list_string_set(List_string *list, int index, const char *value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    free(list->data[index]);  /* Free old string */
    list->data[index] = strdup(value);  /* Copy new string */
}

/* Get the value at the specified index */
char* list_string_get(List_string *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void list_string_clear(List_string *list) {
    for (int i = 0; i < list->length; i++) {
        free(list->data[i]);
    }
    list->length = 0;
}

/* Get the current length of the list */
int list_string_length(List_string *list) {
    return list->length;
}

/* Get the current capacity of the list */
int list_string_capacity(List_string *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool list_string_is_empty(List_string *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void list_string_free(List_string *list) {
    if (list) {
        for (int i = 0; i < list->length; i++) {
            free(list->data[i]);
        }
        free(list->data);
        free(list);
    }
}

