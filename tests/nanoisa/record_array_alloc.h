#ifndef RECORD_ARRAY_TEST_ALLOC_H
#define RECORD_ARRAY_TEST_ALLOC_H
#include <stdlib.h>
#include <stddef.h>
void *ra_test_malloc(size_t);
void *ra_test_calloc(size_t,size_t);
void *ra_test_realloc(void *,size_t);
void ra_test_free(void *);
extern size_t ra_live,ra_bytes,ra_peak,ra_calls,ra_fail;
extern int ra_persistent;
#ifdef RA_ALLOC_WRAP
#define malloc ra_test_malloc
#define calloc ra_test_calloc
#define realloc ra_test_realloc
#define free ra_test_free
#endif
#endif
