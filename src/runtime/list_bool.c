#include "list_bool.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "list_capacity.h"

#define INITIAL_CAPACITY 8

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity(List_bool *list, int min_capacity) {
    int new_capacity = nl_list_grown_capacity(list->capacity, min_capacity, sizeof(*list->data));
    if (new_capacity == list->capacity) return;

    bool *new_data = realloc(list->data, sizeof(bool) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }

    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_bool* list_bool_new(void) {
    return list_bool_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_bool* list_bool_with_capacity(int capacity) {
    nl_list_validate_capacity(capacity, sizeof(*((List_bool *)0)->data));
    List_bool *list = malloc(sizeof(List_bool));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }

    list->data = capacity ? malloc(sizeof(*list->data) * (size_t)capacity) : NULL;
    if (capacity && !list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }

    list->length = 0;
    list->capacity = capacity;

    return list;
}

/* Append an element to the end of the list */
void list_bool_push(List_bool *list, bool value) {
    ensure_capacity(list, nl_list_next_length(list->length));
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
bool list_bool_pop(List_bool *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }

    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void list_bool_insert(List_bool *list, int index, bool value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n",
                index, list->length);
        exit(1);
    }

    ensure_capacity(list, nl_list_next_length(list->length));

    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index],
            sizeof(bool) * (list->length - index));

    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
bool list_bool_remove(List_bool *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n",
                index, list->length);
        exit(1);
    }

    bool value = list->data[index];

    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1],
            sizeof(bool) * (list->length - index - 1));

    list->length--;
    return value;
}

/* Set the value at the specified index */
void list_bool_set(List_bool *list, int index, bool value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n",
                index, list->length);
        exit(1);
    }

    list->data[index] = value;
}

/* Get the value at the specified index */
bool list_bool_get(List_bool *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n",
                index, list->length);
        exit(1);
    }

    return list->data[index];
}

/* Clear all elements from the list */
void list_bool_clear(List_bool *list) {
    list->length = 0;
}

/* Get the current length of the list */
int list_bool_length(List_bool *list) {
    return list->length;
}

/* Get the current capacity of the list */
int list_bool_capacity(List_bool *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool list_bool_is_empty(List_bool *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void list_bool_free(List_bool *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
