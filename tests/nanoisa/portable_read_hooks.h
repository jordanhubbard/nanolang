#ifndef TEST_PORTABLE_READ_HOOKS_H
#define TEST_PORTABLE_READ_HOOKS_H
#include <stdio.h>
#include <stdlib.h>
void *pr_test_malloc(size_t);
void *pr_test_calloc(size_t,size_t);
void pr_test_free(void *);
FILE *pr_test_fopen(const char *,const char *);
size_t pr_test_fread(void *,size_t,size_t,FILE *);
int pr_test_ferror(FILE *);
int pr_test_fclose(FILE *);
#ifdef READ_ALLOC_HOOKS
#define malloc pr_test_malloc
#define calloc pr_test_calloc
#define free pr_test_free
#endif
#ifdef READ_IO_HOOKS
#define fopen pr_test_fopen
#define fread pr_test_fread
#define ferror pr_test_ferror
#define fclose pr_test_fclose
#endif
#endif
