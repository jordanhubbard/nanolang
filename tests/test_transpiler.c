/*
 * StringBuilder is the one transpiler implementation detail I test directly.
 * Its growth arithmetic is memory-safety critical and easier to diagnose in
 * isolation. Language lowering is covered by compile-and-run CLI tests.
 */

#include "../src/nanolang.h"
#include "../src/stdlib_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int g_argc = 0;
char **g_argv = NULL;
char g_project_root[4096] = ".";
const char *get_project_root(void) { return g_project_root; }

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAILED: %s at line %d\n", #cond, __LINE__); \
        exit(1); \
    } \
} while (0)

static void test_append_sequence_across_growth_boundaries(void) {
    static const char *chunks[] = {
        "", "nano", "lang", "-", "0123456789", "xy", "a", "bcdef"
    };
    enum { REPEATS = 3000, ORACLE_SIZE = 65536 };
    char oracle[ORACLE_SIZE] = "";
    size_t oracle_length = 0;
    StringBuilder *builder = sb_create();

    ASSERT(builder != NULL);
    int initial_capacity = builder->capacity;
    for (int i = 0; i < REPEATS; i++) {
        const char *chunk = chunks[i % (int)(sizeof(chunks) / sizeof(chunks[0]))];
        size_t length = strlen(chunk);
        ASSERT(oracle_length + length + 1 < sizeof(oracle));
        memcpy(oracle + oracle_length, chunk, length + 1);
        oracle_length += length;
        sb_append(builder, chunk);

        ASSERT((size_t)builder->length == oracle_length);
        ASSERT(builder->buffer[builder->length] == '\0');
        ASSERT(strcmp(builder->buffer, oracle) == 0);
    }

    ASSERT(builder->capacity > initial_capacity);
    free(builder->buffer);
    free(builder);
}

static void test_single_append_larger_than_initial_capacity(void) {
    char input[4097];
    memset(input, 'z', sizeof(input) - 1);
    input[sizeof(input) - 1] = '\0';

    StringBuilder *builder = sb_create();
    ASSERT(builder != NULL);
    sb_append(builder, input);

    ASSERT(builder->length == sizeof(input) - 1);
    ASSERT(builder->capacity > builder->length);
    ASSERT(memcmp(builder->buffer, input, sizeof(input)) == 0);

    free(builder->buffer);
    free(builder);
}

int main(void) {
    test_append_sequence_across_growth_boundaries();
    test_single_append_larger_than_initial_capacity();
    printf("StringBuilder boundary tests passed\n");
    return 0;
}
