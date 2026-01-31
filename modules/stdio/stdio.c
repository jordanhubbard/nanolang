#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

static FILE* nl_stdio_cast_file(int64_t handle) {
    return (FILE*)(intptr_t)handle;
}

int64_t nl_stdio_fopen(const char* filename, const char* mode) {
    FILE* f = fopen(filename, mode);
    return (int64_t)(intptr_t)f;
}

int64_t nl_stdio_fclose(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)fclose(f);
}

int64_t nl_stdio_fread(int64_t ptr, int64_t size, int64_t count, int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return 0;
    return (int64_t)fread((void*)(intptr_t)ptr, (size_t)size, (size_t)count, f);
}

int64_t nl_stdio_fwrite(int64_t ptr, int64_t size, int64_t count, int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return 0;
    return (int64_t)fwrite((const void*)(intptr_t)ptr, (size_t)size, (size_t)count, f);
}

int64_t nl_stdio_fseek(int64_t file, int64_t offset, int64_t whence) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)fseek(f, (long)offset, (int)whence);
}

int64_t nl_stdio_ftell(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)ftell(f);
}

int64_t nl_stdio_rewind(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    rewind(f);
    return 0;
}

int64_t nl_stdio_feof(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return 1;
    return (int64_t)feof(f);
}

int64_t nl_stdio_ferror(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return 1;
    return (int64_t)ferror(f);
}

int64_t nl_stdio_clearerr(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return 1;
    clearerr(f);
    return 0;
}

int64_t nl_stdio_fflush(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)fflush(f);
}

int64_t nl_stdio_fgetc(int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)fgetc(f);
}

int64_t nl_stdio_fputc(int64_t c, int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)fputc((int)c, f);
}

int64_t nl_stdio_ungetc(int64_t c, int64_t file) {
    FILE* f = nl_stdio_cast_file(file);
    if (!f) return -1;
    return (int64_t)ungetc((int)c, f);
}
