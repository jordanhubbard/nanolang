#!/bin/bash
# Generic List Generator for nanolang
# Usage: generate_list.sh TypeName OutputDir
#
# Generates list_TypeName.h and list_TypeName.c for any struct type

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 TypeName OutputDir [TypeDef]"
    echo "Example: $0 Point /tmp 'struct Point'"
    exit 1
fi

TYPE_NAME="$1"
OUTPUT_DIR="$2"
TYPE_DEF="${3:-$TYPE_NAME}"  # Default to TypeName if not provided
TYPE_NAME_UPPER=$(echo "$TYPE_NAME" | tr '[:lower:]' '[:upper:]')

HEADER_FILE="$OUTPUT_DIR/list_$TYPE_NAME.h"
SOURCE_FILE="$OUTPUT_DIR/list_$TYPE_NAME.c"

# Generate header file
cat > "$HEADER_FILE" << EOF
#ifndef LIST_${TYPE_NAME_UPPER}_H
#define LIST_${TYPE_NAME_UPPER}_H

#include <stdint.h>
#include <stdbool.h>

/* Dynamic list of $TYPE_NAME */
/* Guard typedef to prevent redefinition warnings */
#ifndef DEFINED_List_$TYPE_NAME
#define DEFINED_List_$TYPE_NAME
typedef struct List_$TYPE_NAME {
    $TYPE_DEF *data;      /* Array of elements */
    int length;                       /* Current number of elements */
    int capacity;                     /* Allocated capacity */
} List_$TYPE_NAME;
#endif

/* Create a new empty list */
List_$TYPE_NAME* nl_list_${TYPE_NAME}_new(void);

/* Create a new list with specified initial capacity */
List_$TYPE_NAME* nl_list_${TYPE_NAME}_with_capacity(int capacity);

/* Append an element to the end of the list */
void nl_list_${TYPE_NAME}_push(List_$TYPE_NAME *list, $TYPE_DEF value);

/* Remove and return the last element */
$TYPE_DEF nl_list_${TYPE_NAME}_pop(List_$TYPE_NAME *list);

/* Insert an element at the specified index */
void nl_list_${TYPE_NAME}_insert(List_$TYPE_NAME *list, int index, $TYPE_DEF value);

/* Remove and return the element at the specified index */
$TYPE_DEF nl_list_${TYPE_NAME}_remove(List_$TYPE_NAME *list, int index);

/* Set the value at the specified index */
void nl_list_${TYPE_NAME}_set(List_$TYPE_NAME *list, int index, $TYPE_DEF value);

/* Get the value at the specified index */
$TYPE_DEF nl_list_${TYPE_NAME}_get(List_$TYPE_NAME *list, int index);

/* Clear all elements from the list */
void nl_list_${TYPE_NAME}_clear(List_$TYPE_NAME *list);

/* Get the current length of the list */
int nl_list_${TYPE_NAME}_length(List_$TYPE_NAME *list);

/* Get the current capacity of the list */
int nl_list_${TYPE_NAME}_capacity(List_$TYPE_NAME *list);

/* Check if the list is empty */
bool nl_list_${TYPE_NAME}_is_empty(List_$TYPE_NAME *list);

/* Free the list and all its resources */
void nl_list_${TYPE_NAME}_free(List_$TYPE_NAME *list);

#endif /* LIST_${TYPE_NAME_UPPER}_H */
EOF

# Generate source file
cat > "$SOURCE_FILE" << 'EOF'
#include "list_TYPENAME.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Note: The actual struct nl_TYPENAME definition must be included */
/* before this file in the compilation */

#define INITIAL_CAPACITY 8
#define GROWTH_FACTOR 2

/* Helper: Ensure the list has enough capacity */
static void ensure_capacity_TYPENAME(List_TYPENAME *list, int min_capacity) {
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
    
    $TYPE_DEF *new_data = realloc(list->data, sizeof($TYPE_DEF) * new_capacity);
    if (!new_data) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = new_data;
    list->capacity = new_capacity;
}

/* Create a new empty list */
List_TYPENAME* nl_list_TYPENAME_new(void) {
    return nl_list_TYPENAME_with_capacity(INITIAL_CAPACITY);
}

/* Create a new list with specified initial capacity */
List_TYPENAME* nl_list_TYPENAME_with_capacity(int capacity) {
    List_TYPENAME *list = malloc(sizeof(List_TYPENAME));
    if (!list) {
        fprintf(stderr, "Error: Failed to allocate memory for list\n");
        exit(1);
    }
    
    list->data = malloc(sizeof($TYPE_DEF) * capacity);
    if (!list->data) {
        fprintf(stderr, "Error: Failed to allocate memory for list data\n");
        exit(1);
    }
    
    list->length = 0;
    list->capacity = capacity;
    
    return list;
}

/* Append an element to the end of the list */
void nl_list_TYPENAME_push(List_TYPENAME *list, $TYPE_DEF value) {
    ensure_capacity_TYPENAME(list, list->length + 1);
    list->data[list->length] = value;
    list->length++;
}

/* Remove and return the last element */
$TYPE_DEF nl_list_TYPENAME_pop(List_TYPENAME *list) {
    if (list->length == 0) {
        fprintf(stderr, "Error: Cannot pop from empty list\n");
        exit(1);
    }
    
    list->length--;
    return list->data[list->length];
}

/* Insert an element at the specified index */
void nl_list_TYPENAME_insert(List_TYPENAME *list, int index, $TYPE_DEF value) {
    if (index < 0 || index > list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    ensure_capacity_TYPENAME(list, list->length + 1);
    
    /* Shift elements to the right */
    memmove(&list->data[index + 1], &list->data[index], 
            sizeof($TYPE_DEF) * (list->length - index));
    
    list->data[index] = value;
    list->length++;
}

/* Remove and return the element at the specified index */
$TYPE_DEF nl_list_TYPENAME_remove(List_TYPENAME *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    $TYPE_DEF value = list->data[index];
    
    /* Shift elements to the left */
    memmove(&list->data[index], &list->data[index + 1], 
            sizeof($TYPE_DEF) * (list->length - index - 1));
    
    list->length--;
    return value;
}

/* Set the value at the specified index */
void nl_list_TYPENAME_set(List_TYPENAME *list, int index, $TYPE_DEF value) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    list->data[index] = value;
}

/* Get the value at the specified index */
$TYPE_DEF nl_list_TYPENAME_get(List_TYPENAME *list, int index) {
    if (index < 0 || index >= list->length) {
        fprintf(stderr, "Error: Index %d out of bounds for list of length %d\n", 
                index, list->length);
        exit(1);
    }
    
    return list->data[index];
}

/* Clear all elements from the list */
void nl_list_TYPENAME_clear(List_TYPENAME *list) {
    list->length = 0;
}

/* Get the current length of the list */
int nl_list_TYPENAME_length(List_TYPENAME *list) {
    return list->length;
}

/* Get the current capacity of the list */
int nl_list_TYPENAME_capacity(List_TYPENAME *list) {
    return list->capacity;
}

/* Check if the list is empty */
bool nl_list_TYPENAME_is_empty(List_TYPENAME *list) {
    return list->length == 0;
}

/* Free the list and all its resources */
void nl_list_TYPENAME_free(List_TYPENAME *list) {
    if (list) {
        free(list->data);
        free(list);
    }
}
EOF

# Replace TYPENAME and TYPE_DEF with actual values
sed -i.bak "s|TYPENAME|$TYPE_NAME|g" "$SOURCE_FILE"
sed -i.bak "s|\$TYPE_DEF|$TYPE_DEF|g" "$SOURCE_FILE"
sed -i.bak "s|\$TYPE_NAME|$TYPE_NAME|g" "$HEADER_FILE"
sed -i.bak "s|\$TYPE_DEF|$TYPE_DEF|g" "$HEADER_FILE"
rm -f "$SOURCE_FILE.bak" "$HEADER_FILE.bak"

echo "Generated list_$TYPE_NAME.h and list_$TYPE_NAME.c in $OUTPUT_DIR"
