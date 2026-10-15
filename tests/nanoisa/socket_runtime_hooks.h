#ifndef NANO_SOCKET_RUNTIME_TEST_HOOKS_H
#define NANO_SOCKET_RUNTIME_TEST_HOOKS_H
#include <stddef.h>
void *socket_runtime_malloc(size_t);
void *socket_runtime_calloc(size_t,size_t);
void socket_runtime_free(void *);
int socket_runtime_socket(int,int,int);
int socket_runtime_close(int);
#endif
