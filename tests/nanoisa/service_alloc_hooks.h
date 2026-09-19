#ifndef SERVICE_ALLOC_HOOKS_H
#define SERVICE_ALLOC_HOOKS_H
#include <stddef.h>
void *service_test_malloc(size_t);
void *service_test_calloc(size_t,size_t);
void *service_test_realloc(void *,size_t);
#endif
