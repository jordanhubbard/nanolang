#ifndef LIST_STRING_H
#define LIST_STRING_H

#include <stdbool.h>

/* Dynamic list of strings (char*) */
typedef struct {
    char **data;         /* Array of string pointers */
    int length;          /* Current number of elements */
    int capacity;        /* Allocated capacity */
} List_string;

/* Stub functions - to be fully implemented later */
List_string* list_string_new(void);
List_string* list_string_with_capacity(int capacity);
void list_string_push(List_string *list, const char *value);
char* list_string_pop(List_string *list);
void list_string_insert(List_string *list, int index, const char *value);
char* list_string_remove(List_string *list, int index);
void list_string_set(List_string *list, int index, const char *value);
char* list_string_get(List_string *list, int index);
void list_string_clear(List_string *list);
int list_string_length(List_string *list);
int list_string_capacity(List_string *list);
bool list_string_is_empty(List_string *list);
void list_string_free(List_string *list);

#endif /* LIST_STRING_H */

