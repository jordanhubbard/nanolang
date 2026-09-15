#ifndef NANOLANG_FILE_BYTES_H
#define NANOLANG_FILE_BYTES_H

#include "dyn_array.h"
#include <stdio.h>

/* I consume and close the stream. Read/close failures discard partial bytes. */
static inline DynArray *nl_read_byte_stream(FILE *file) {
    DynArray *bytes = dyn_array_new(ELEM_U8);
    if (!bytes || !file) {
        if (file) fclose(file);
        return bytes;
    }
    unsigned char buffer[4096];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file)) != 0) {
        for (size_t i = 0; i < count; i++) dyn_array_push_u8(bytes, buffer[i]);
    }
    int failed = ferror(file);
    if (fclose(file) != 0) failed = 1;
    if (failed) bytes->length = 0;
    return bytes;
}

static inline DynArray *nl_read_file_bytes(const char *path) {
    return nl_read_byte_stream(path ? fopen(path, "rb") : NULL);
}

#endif
