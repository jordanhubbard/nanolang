/*
 * VM Built-in Functions
 *
 * C-callable implementations of nanolang built-in functions
 * for the NanoVM FFI bridge. These mirror the transpiler's
 * nl_os_* functions but are actual callable symbols.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ctype.h>
#include <math.h>
#include "runtime/dyn_array.h"

/* ── OS / File System ─────────────────────────────────────────────── */

char *vm_getcwd(void) {
    char buf[1024];
    if (getcwd(buf, sizeof(buf)) == NULL) return strdup("");
    return strdup(buf);
}

int64_t vm_chdir(const char *path) {
    return chdir(path) == 0 ? 0 : -1;
}

char *vm_file_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return strdup("");
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len < 0) { fclose(f); return strdup(""); }
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf) { fclose(f); return strdup(""); }
    size_t n = fread(buf, 1, (size_t)len, f);
    buf[n] = '\0';
    fclose(f);
    return buf;
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
        dyn_array_push_string(arr, entry->d_name);
    }
    closedir(d);
    return arr;
}

char *vm_mktemp_dir(const char *prefix) {
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    char tmpl[1024];
    snprintf(tmpl, sizeof(tmpl), "%s/%sXXXXXX", tmp, prefix ? prefix : "nano_");
    char *result = mkdtemp(tmpl);
    return result ? strdup(result) : strdup("");
}

char *vm_getenv(const char *name) {
    const char *val = getenv(name);
    return val ? strdup(val) : strdup("");
}

int64_t vm_setenv(const char *name, const char *value) {
    return setenv(name, value, 1) == 0 ? 0 : -1;
}

/* ── String operations ────────────────────────────────────────────── */

int64_t vm_str_index_of(const char *haystack, const char *needle) {
    if (!haystack || !needle) return -1;
    const char *p = strstr(haystack, needle);
    return p ? (int64_t)(p - haystack) : -1;
}

/* ── String building ──────────────────────────────────────────────── */

char *vm_string_from_char(int64_t code) {
    char buf[2];
    buf[0] = (char)code;
    buf[1] = '\0';
    return strdup(buf);
}

/* ── Character classification ────────────────────────────────────── */

int64_t vm_is_digit(int64_t c) { return (c >= '0' && c <= '9') ? 1 : 0; }
int64_t vm_is_alpha(int64_t c) { return isalpha((int)c) ? 1 : 0; }
int64_t vm_is_alnum(int64_t c) { return isalnum((int)c) ? 1 : 0; }
int64_t vm_is_space(int64_t c) { return isspace((int)c) ? 1 : 0; }
int64_t vm_is_upper(int64_t c) { return isupper((int)c) ? 1 : 0; }
int64_t vm_is_lower(int64_t c) { return islower((int)c) ? 1 : 0; }
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
    const unsigned char *s = (const unsigned char *)str;
    while (*s) {
        if (*s < 0x80) { s++; }
        else if ((*s & 0xE0) == 0xC0) {
            if ((s[1] & 0xC0) != 0x80) return 0;
            s += 2;
        } else if ((*s & 0xF0) == 0xE0) {
            if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80) return 0;
            s += 3;
        } else if ((*s & 0xF8) == 0xF0) {
            if ((s[1] & 0xC0) != 0x80 || (s[2] & 0xC0) != 0x80 || (s[3] & 0xC0) != 0x80) return 0;
            s += 4;
        } else {
            return 0;
        }
    }
    return 1;
}

/* ── Binary string operations ────────────────────────────────────── */

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
        buf[i] = (char)dyn_array_get_int(arr, (int64_t)i);
    }
    buf[len] = '\0';
    return buf;
}

/* ── Process ──────────────────────────────────────────────────────── */

int64_t vm_process_run(const char *cmd) {
    return (int64_t)system(cmd);
}
