#ifndef PUBLICATION_ALLOC_H
#define PUBLICATION_ALLOC_H
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
void *pa_malloc(size_t);
void *pa_calloc(size_t,size_t);
void *pa_realloc(void *,size_t);
void *pa_cleanup_realloc(void *,size_t);
void pa_free(void *);
extern size_t pa_live,pa_calls,pa_fail,pa_denied;
extern bool pa_active,pa_persistent;
#ifdef PUBLICATION_ALLOC_WRAP
#define malloc pa_malloc
#define calloc pa_calloc
#define realloc pa_realloc
#define free pa_free
#endif
#endif
