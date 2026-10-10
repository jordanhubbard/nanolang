#ifndef MIXED_WEBSOCKET_RUNTIME_ALLOC_H
#define MIXED_WEBSOCKET_RUNTIME_ALLOC_H
#include <stddef.h>
void *mixed_runtime_malloc(size_t);
void *mixed_runtime_calloc(size_t,size_t);
#endif
