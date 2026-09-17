#ifndef NANOLANG_FILE_TEXT_H
#define NANOLANG_FILE_TEXT_H

#include "file_bytes.h"
#include "gc.h"
#include <stdlib.h>
#include <string.h>

/* I return owned, NUL-terminated text. NULL means allocation failed.
 * I consume/close the stream, and reject embedded NUL rather than truncate. */
static inline char *nl_read_text_stream(FILE *file) {
    DynArray *bytes = nl_read_byte_stream(file);
    if (!bytes) return NULL;
    size_t length = (size_t)bytes->length;
    if (length && memchr(bytes->data, 0, length)) length = 0;
    char *text = length < SIZE_MAX ? malloc(length + 1) : NULL;
    if (text) {
        if (length) memcpy(text, bytes->data, length);
        text[length] = '\0';
    }
    gc_release(bytes);
    return text;
}

static inline char *nl_read_file_text(const char *path) {
    return nl_read_text_stream(path ? fopen(path, "rb") : NULL);
}

#endif
