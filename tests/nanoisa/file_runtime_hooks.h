#ifndef FILE_RUNTIME_TEST_HOOKS_H
#define FILE_RUNTIME_TEST_HOOKS_H
#include <stdio.h>
#include <stdbool.h>
#include <sys/types.h>
#include "file_hosted_alloc.h"
FILE *file_runtime_tmpfile(void);
int file_runtime_fclose(FILE *);
size_t file_runtime_fread(void *,size_t,size_t,FILE *);
size_t file_runtime_fwrite(const void *,size_t,size_t,FILE *);
int file_runtime_fseek(FILE *,long,int);
bool file_runtime_loader_init(bool);
bool file_runtime_loader_open(const char *,const char *);
pid_t file_runtime_fork(void);
/* SDK stdio macros are read before I install the fixture hooks. */
#ifdef FILE_RUNTIME_HOST_HOOKS
#undef tmpfile
#undef fclose
#undef fread
#undef fwrite
#undef fseek
#define tmpfile file_runtime_tmpfile
#define fclose file_runtime_fclose
#define fread file_runtime_fread
#define fwrite file_runtime_fwrite
#define fseek file_runtime_fseek
#endif
#endif
