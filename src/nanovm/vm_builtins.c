/*
 * VM Built-in Functions
 *
 * C-callable implementations of nanolang built-in functions
 * for the NanoVM FFI bridge. These mirror the transpiler's
 * nl_os_* functions but are actual callable symbols.
 */

#define _POSIX_C_SOURCE 200809L  /* For strdup(), mkdtemp() */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <dirent.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include "runtime/dyn_array.h"
#include "runtime/process_capture.h"
#include "runtime/file_bytes.h"
#include "runtime/file_text.h"
#include "utf8.h"

/* mkdtemp declaration (not exposed on macOS with -std=c99) */
#ifndef _DARWIN_C_SOURCE
char *mkdtemp(char *);
#endif

/* ── OS / File System ─────────────────────────────────────────────── */

const char *nl_exec_capture(const char *command) {
    static __thread char output[65536];
    output[0] = '\0';
    FILE *pipe = popen(command, "r");
    if (!pipe) return output;
    size_t used = fread(output, 1, sizeof(output) - 1, pipe);
    output[used] = '\0';
    /* I drain excess output before waiting so the child cannot block on a
     * full pipe after my retained-output bound has been reached. */
    char discard[4096];
    while (fread(discard, 1, sizeof(discard), pipe)) {}
    pclose(pipe);
    return output;
}

int64_t nl_exec_shell(const char *command) {
    return (int64_t)system(command);
}

int64_t nl_timing_get_microseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now)) return -1;
    return (int64_t)now.tv_sec * 1000000 + now.tv_nsec / 1000;
}

int64_t nl_timing_get_nanoseconds(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now)) return -1;
    return (int64_t)now.tv_sec * 1000000000 + now.tv_nsec;
}

int64_t nl_get_time_ms(void) {
    int64_t us = nl_timing_get_microseconds();
    return us < 0 ? -1 : us / 1000;
}

char *vm_getcwd(void) {
    char buf[1024];
    if (getcwd(buf, sizeof(buf)) == NULL) return strdup("");
    return strdup(buf);
}

int64_t vm_chdir(const char *path) {
    return chdir(path) == 0 ? 0 : -1;
}

char *vm_file_read(const char *path) {
    return nl_read_file_text(path);
}

DynArray *vm_file_read_bytes(const char *path) {
    return nl_read_file_bytes(path);
}

int64_t vm_file_write(const char *path, const char *content) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    size_t len = strlen(content);
    size_t written = fwrite(content, 1, len, f);
    fclose(f);
    return (int64_t)written == (int64_t)len ? 0 : -1;
}

int64_t vm_file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 ? 1 : 0;
}

int64_t vm_dir_exists(const char *path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) ? 1 : 0;
}

int64_t vm_dir_create(const char *path) {
    return mkdir(path, 0755) == 0 ? 0 : -1;
}

DynArray *vm_dir_list(const char *path) {
    DynArray *arr = dyn_array_new(ELEM_STRING);
    DIR *d = opendir(path);
    if (!d) return arr;
    struct dirent *entry;
    while ((entry = readdir(d)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        /* Copy: entry->d_name is owned by the DIR stream and is invalidated by
         * the next readdir()/closedir(), so storing the raw pointer would leave
         * every element dangling (they read back empty). */
        dyn_array_push_string_copy(arr, entry->d_name);
    }
    closedir(d);
    return arr;
}

/* The directory temporary files belong in. Mirrors nl_os_tmp_dir() in the
 * native runtime so a program sees the same path under either backend. */
char *vm_tmp_dir(void) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    return strdup(tmp);
}

char *vm_mktemp_dir(const char *prefix) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    char tmpl[1024];
    snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmp, prefix ? prefix : "nano_");
    char *result = mkdtemp(tmpl);
    return result ? strdup(result) : strdup("");
}

/* Create a temporary *file* and return its path. mkstemp() creates the file
 * and hands back a descriptor; the caller only wants the name, so close it. */
char *vm_mktemp(const char *prefix) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    char tmpl[1024];
    snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmp,
             (prefix && prefix[0]) ? prefix : "nanolang_");
    int fd = mkstemp(tmpl);
    if (fd < 0) return strdup("");
    close(fd);
    return strdup(tmpl);
}

char *vm_getenv(const char *name) {
    const char *val = getenv(name);
    return val ? strdup(val) : strdup("");
}

int64_t vm_setenv(const char *name, const char *value) {
    return setenv(name, value, 1) == 0 ? 0 : -1;
}

/* ── String operations (delegate to nl_cstr_* shared primitives) ── */
#include "runtime/nl_string.h"

int64_t vm_str_index_of(const char *haystack, const char *needle) {
    return nl_cstr_index_of(haystack, needle);
}

int64_t vm_str_last_index_of(const char *haystack, const char *needle) {
    return nl_cstr_last_index_of(haystack, needle);
}

/* ── String building ──────────────────────────────────────────────── */

char *vm_string_from_char(int64_t code) {
    return nl_cstr_from_char(code);
}

/* ── Character classification ────────────────────────────────────── */

int64_t vm_is_digit(int64_t c) { return nl_ascii_isdigit((int)c) ? 1 : 0; }
int64_t vm_is_alpha(int64_t c) { return nl_ascii_isalpha((int)c) ? 1 : 0; }
int64_t vm_is_alnum(int64_t c) { return nl_ascii_isalnum((int)c) ? 1 : 0; }
int64_t vm_is_space(int64_t c) { return nl_ascii_isspace((int)c) ? 1 : 0; }
int64_t vm_is_upper(int64_t c) { return nl_ascii_isupper((int)c) ? 1 : 0; }
int64_t vm_is_lower(int64_t c) { return nl_ascii_islower((int)c) ? 1 : 0; }
int64_t vm_is_whitespace(int64_t c) {
    return (c == ' ' || c == '\t' || c == '\n' || c == '\r') ? 1 : 0;
}

int64_t vm_digit_value(int64_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    return -1;
}

int64_t vm_char_to_lower(int64_t c) {
    if (c >= 'A' && c <= 'Z') return c + 32;
    return c;
}

int64_t vm_char_to_upper(int64_t c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    return c;
}

int64_t vm_bstr_utf8_length(const char *str) {
    if (!str) return 0;
    int64_t count = 0;
    const unsigned char *s = (const unsigned char *)str;
    while (*s) {
        if ((*s & 0x80) == 0) s += 1;
        else if ((*s & 0xE0) == 0xC0) s += 2;
        else if ((*s & 0xF0) == 0xE0) s += 3;
        else if ((*s & 0xF8) == 0xF0) s += 4;
        else s += 1;
        count++;
    }
    return count;
}

int64_t vm_bstr_utf8_char_at(const char *str, int64_t char_index) {
    if (!str || char_index < 0) return -1;
    const unsigned char *s = (const unsigned char *)str;
    int64_t idx = 0;
    while (*s && idx < char_index) {
        if ((*s & 0x80) == 0) s += 1;
        else if ((*s & 0xE0) == 0xC0) s += 2;
        else if ((*s & 0xF0) == 0xE0) s += 3;
        else if ((*s & 0xF8) == 0xF0) s += 4;
        else s += 1;
        idx++;
    }
    if (!*s) return -1;
    /* Decode UTF-8 codepoint */
    if ((*s & 0x80) == 0) return *s;
    if ((*s & 0xE0) == 0xC0) return ((s[0] & 0x1F) << 6) | (s[1] & 0x3F);
    if ((*s & 0xF0) == 0xE0) return ((s[0] & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F);
    if ((*s & 0xF8) == 0xF0) return ((s[0] & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F);
    return -1;
}

int64_t vm_bstr_validate_utf8(const char *str) {
    if (!str) return 0;
    return nl_utf8_validate(str, strlen(str), NULL) ? 1 : 0;
}

/* ── Binary string operations ────────────────────────────────────── */

static bool vm_trim_space(unsigned char byte) {
    return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r';
}

char *vm_str_trim_left(const char *str) {
    if (!str) return strdup("");
    while (vm_trim_space((unsigned char)*str)) str++;
    return strdup(str);
}

char *vm_str_trim_right(const char *str) {
    if (!str) return strdup("");
    size_t end = strlen(str);
    while (end && vm_trim_space((unsigned char)str[end - 1])) end--;
    char *result = malloc(end + 1);
    if (!result) return NULL;
    memcpy(result, str, end);
    result[end] = '\0';
    return result;
}

char *vm_format(const char *template, DynArray *arguments) {
    if (!template || !arguments || arguments->elem_type != ELEM_STRING ||
        arguments->elem_size != sizeof(char *) || arguments->length < 0 ||
        arguments->capacity < arguments->length ||
        (uint64_t)arguments->capacity > SIZE_MAX / sizeof(char *) ||
        (arguments->length && !arguments->data)) return NULL;
    for (int64_t i = 0; i < arguments->length; i++)
        if (!dyn_array_get_string(arguments, i)) return NULL;

    size_t length = 0;
    char *result = NULL;
    for (int pass = 0; pass < 2; pass++) {
        size_t offset = 0;
        int64_t used = 0;
        const char *cursor = template;
        while (*cursor) {
            const char *part = cursor;
            size_t size = 1;
            if (*cursor == '%' && used < arguments->length &&
                (cursor[1] == 's' || cursor[1] == 'd' ||
                 cursor[1] == 'f' || cursor[1] == 'g')) {
                part = dyn_array_get_string(arguments, used++);
                size = strlen(part);
                cursor += 2;
            } else {
                cursor++;
            }
            if (size > SIZE_MAX - 1 - offset) { free(result); return NULL; }
            if (pass) memcpy(result + offset, part, size);
            offset += size;
        }
        if (!pass) {
            length = offset;
            result = malloc(length + 1);
            if (!result) return NULL;
        }
    }
    result[length] = '\0';
    return result;
}

char *vm_str_join(DynArray *parts, const char *separator) {
    if (!parts || !separator || parts->elem_type != ELEM_STRING ||
        parts->elem_size != sizeof(char *) || parts->length < 0 ||
        parts->capacity < parts->length ||
        (uint64_t)parts->capacity > SIZE_MAX / sizeof(char *) ||
        (parts->length && !parts->data)) return NULL;
    size_t length = 0, separator_length = strlen(separator);
    for (int64_t i = 0; i < parts->length; i++) {
        const char *part = dyn_array_get_string(parts, i);
        if (!part) return NULL;
        size_t size = strlen(part);
        if (i) {
            if (separator_length > SIZE_MAX - 1 - length) return NULL;
            length += separator_length;
        }
        if (size > SIZE_MAX - 1 - length) return NULL;
        length += size;
    }
    char *result = malloc(length + 1);
    if (!result) return NULL;
    size_t offset = 0;
    for (int64_t i = 0; i < parts->length; i++) {
        if (i) {
            memcpy(result + offset, separator, separator_length);
            offset += separator_length;
        }
        const char *part = dyn_array_get_string(parts, i);
        size_t size = strlen(part);
        memcpy(result + offset, part, size);
        offset += size;
    }
    result[offset] = '\0';
    return result;
}

DynArray *vm_array_sort(DynArray *array) {
    return dyn_array_sorted(array);
}

DynArray *vm_bytes_from_string(const char *str) {
    DynArray *arr = dyn_array_new(ELEM_INT);
    if (!str) return arr;
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        dyn_array_push_int(arr, (int64_t)(unsigned char)str[i]);
    }
    return arr;
}

char *vm_string_from_bytes(DynArray *arr) {
    if (!arr) return strdup("");
    size_t len = (size_t)arr->length;
    char *buf = malloc(len + 1);
    if (!buf) return strdup("");
    for (size_t i = 0; i < len; i++) {
        buf[i] = arr->elem_type == ELEM_U8
            ? (char)dyn_array_get_u8(arr, (int64_t)i)
            : (char)dyn_array_get_int(arr, (int64_t)i);
    }
    buf[len] = '\0';
    return buf;
}

/* ── Process ──────────────────────────────────────────────────────── */

DynArray *vm_process_run(const char *cmd) {
    return nl_process_run_capture(cmd);
}
