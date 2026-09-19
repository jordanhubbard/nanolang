#ifndef FILE_HOSTED_TEST_ALLOC_H
#define FILE_HOSTED_TEST_ALLOC_H
#include <stddef.h>
void *file_test_malloc(size_t);
void *file_test_calloc(size_t,size_t);
void *file_test_realloc(void *,size_t);
void file_test_free(void *);
#endif
