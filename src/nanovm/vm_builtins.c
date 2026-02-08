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

/* ── Process ──────────────────────────────────────────────────────── */

int64_t vm_process_run(const char *cmd) {
    return (int64_t)system(cmd);
}
