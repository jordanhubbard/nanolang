#ifndef NANOLANG_EVALUATOR_COLLECTION_ALLOC_H
#define NANOLANG_EVALUATOR_COLLECTION_ALLOC_H
#include <stdlib.h>
#include <string.h>
#ifdef NANO_TEST_COLLECTION_ALLOC
extern void *nano_test_collection_calloc(size_t, size_t);
#define NANO_COLLECTION_CALLOC nano_test_collection_calloc
#else
#define NANO_COLLECTION_CALLOC calloc
#endif
static inline char *eval_collection_copy_string(const char *source) {
    size_t bytes = strlen(source) + 1;
    char *copy = NANO_COLLECTION_CALLOC(bytes, 1);
    if (copy) memcpy(copy, source, bytes);
    return copy;
}
#endif
