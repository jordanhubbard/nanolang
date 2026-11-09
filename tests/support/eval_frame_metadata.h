/* I intercept allocation/free only in the interpreter unit's private objects. */
#ifndef NANO_TEST_EVAL_FRAME_METADATA_H
#define NANO_TEST_EVAL_FRAME_METADATA_H
#include "nanolang.h"

char *nano_test_metadata_strdup(const char *text);
void nano_test_metadata_free(void *pointer);

#ifdef NANO_TEST_BINDING_ALLOCATIONS
#define strdup nano_test_metadata_strdup
#endif
#define free nano_test_metadata_free
#endif
