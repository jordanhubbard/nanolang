#ifndef NANOLANG_FILE_WRITE_H
#define NANOLANG_FILE_WRITE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* I report buffered write and close failures. Failure can leave partial output;
 * I do not provide atomic replacement or a durability guarantee. */
static inline int64_t nl_write_file_text(const char *path, const char *text,
                                         const char *mode) {
    if (!path || !text) return -1;
    FILE *file = fopen(path, mode);
    if (!file) return -1;
    size_t length = strlen(text);
    size_t written = fwrite(text, 1, length, file);
    int closed = fclose(file);
    return written == length && closed == 0 ? 0 : -1;
}

#endif
