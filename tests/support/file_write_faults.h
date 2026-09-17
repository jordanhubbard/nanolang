#ifndef NANO_TEST_FILE_WRITE_FAULTS_H
#define NANO_TEST_FILE_WRITE_FAULTS_H

/* Force-include only in dedicated production objects, never test drivers. */
#include <stdio.h>
size_t nano_test_fwrite(const void *ptr, size_t size, size_t count, FILE *stream);
int nano_test_fclose(FILE *stream);
#define fwrite nano_test_fwrite
#define fclose nano_test_fclose

#endif
