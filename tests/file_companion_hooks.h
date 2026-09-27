#ifndef FILE_COMPANION_HOOKS_H
#define FILE_COMPANION_HOOKS_H
#include <stddef.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
void *fc_test_malloc(size_t);
void *fc_test_calloc(size_t,size_t);
void *fc_test_realloc(void *,size_t);
void fc_test_free(void *);
char *fc_test_strdup(const char *);
int fc_test_open(const char *,int,...);
int fc_test_openat(int,const char *,int,...);
int fc_test_fstat(int,struct stat *);
ssize_t fc_test_read(int,void *,size_t);
int fc_test_close(int);
#define malloc fc_test_malloc
#define calloc fc_test_calloc
#define realloc fc_test_realloc
#define free fc_test_free
#define strdup fc_test_strdup
#define open fc_test_open
#define openat fc_test_openat
#define fstat fc_test_fstat
#define read fc_test_read
#define close fc_test_close
#endif
