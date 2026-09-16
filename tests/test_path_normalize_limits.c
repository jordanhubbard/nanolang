#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fs.h"

static char *repeated_path(const char *part, size_t count) {
    size_t part_len = strlen(part);
    char *path = malloc(count * (part_len + 1) + 1);
    assert(path != NULL);

    size_t pos = 0;
    for (size_t i = 0; i < count; i++) {
        if (i > 0) path[pos++] = '/';
        memcpy(path + pos, part, part_len);
        pos += part_len;
    }
    path[pos] = '\0';
    return path;
}

int main(void) {
    char *long_path = repeated_path("component", 600);
    const char *long_normalized = path_normalize(long_path);
    assert(long_normalized != NULL);
    assert(strlen(long_normalized) == strlen(long_path));
    assert(strcmp(long_normalized, long_path) == 0);

    char *parents = repeated_path("..", 600);
    const char *parents_normalized = path_normalize(parents);
    assert(parents_normalized != NULL);
    assert(strcmp(parents_normalized, parents) == 0);

    free((void *)parents_normalized);
    free(parents);
    free((void *)long_normalized);
    free(long_path);
    puts("path normalization limit tests passed");
    return 0;
}
