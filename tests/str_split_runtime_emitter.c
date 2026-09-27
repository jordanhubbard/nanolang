/* I retain the complete actual C runtime output before selecting its helper. */
#include "stdlib_runtime.h"
#include <stdio.h>
#include <stdlib.h>
int g_argc;
char **g_argv;
int main(void) {
    StringBuilder *builder = sb_create();
    if (!builder) return 1;
    generate_string_operations(builder);
    int failed = fwrite(builder->buffer, 1, (size_t)builder->length, stdout) != (size_t)builder->length;
    free(builder->buffer); free(builder);
    return failed;
}
