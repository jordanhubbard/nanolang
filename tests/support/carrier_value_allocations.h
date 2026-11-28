#ifndef NANO_TEST_CARRIER_VALUE_ALLOCATIONS_H
#define NANO_TEST_CARRIER_VALUE_ALLOCATIONS_H
#include <stdlib.h>
void *nano_carrier_test_malloc(size_t size);
void *nano_carrier_test_calloc(size_t count, size_t size);
void *nano_carrier_test_realloc(void *pointer, size_t size);
void nano_carrier_test_free(void *pointer);
#define malloc nano_carrier_test_malloc
#define calloc nano_carrier_test_calloc
#define realloc nano_carrier_test_realloc
#define free nano_carrier_test_free
#endif
