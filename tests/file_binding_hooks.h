#ifndef FILE_BINDING_TEST_HOOKS_H
#define FILE_BINDING_TEST_HOOKS_H
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
void *binding_test_malloc(size_t);
void *binding_test_calloc(size_t, size_t);
void *binding_test_realloc(void *, size_t);
void binding_test_free(void *);
char *binding_test_strdup(const char *);
#ifndef BINDING_TEST_IMPLEMENTATION
#define malloc binding_test_malloc
#define calloc binding_test_calloc
#define realloc binding_test_realloc
#define free binding_test_free
#define strdup binding_test_strdup
#endif
#endif
