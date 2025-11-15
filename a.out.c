#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>

/* nanolang runtime */
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>

/* ========== OS Standard Library ========== */

static char* nl_os_file_read(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

static int64_t nl_os_file_write(const char* path, const char* content) {
    FILE* f = fopen(path, "w");
    if (!f) return -1;
    fputs(content, f);
    fclose(f);
    return 0;
}

static int64_t nl_os_file_append(const char* path, const char* content) {
    FILE* f = fopen(path, "a");
    if (!f) return -1;
    fputs(content, f);
    fclose(f);
    return 0;
}

static int64_t nl_os_file_remove(const char* path) {
    return remove(path) == 0 ? 0 : -1;
}

static int64_t nl_os_file_rename(const char* old_path, const char* new_path) {
    return rename(old_path, new_path) == 0 ? 0 : -1;
}

static bool nl_os_file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

static int64_t nl_os_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size;
}

static int64_t nl_os_dir_create(const char* path) {
    return mkdir(path, 0755) == 0 ? 0 : -1;
}

static int64_t nl_os_dir_remove(const char* path) {
    return rmdir(path) == 0 ? 0 : -1;
}

static char* nl_os_dir_list(const char* path) {
    DIR* dir = opendir(path);
    if (!dir) return "";
    char* buffer = malloc(4096);
    buffer[0] = '\0';
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        strcat(buffer, entry->d_name);
        strcat(buffer, "\n");
    }
    closedir(dir);
    return buffer;
}

static bool nl_os_dir_exists(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static char* nl_os_getcwd(void) {
    char* buffer = malloc(1024);
    if (getcwd(buffer, 1024) == NULL) {
        buffer[0] = '\0';
    }
    return buffer;
}

static int64_t nl_os_chdir(const char* path) {
    return chdir(path) == 0 ? 0 : -1;
}

static bool nl_os_path_isfile(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISREG(st.st_mode);
}

static bool nl_os_path_isdir(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return false;
    return S_ISDIR(st.st_mode);
}

static char* nl_os_path_join(const char* a, const char* b) {
    char* buffer = malloc(2048);
    if (strlen(a) == 0) {
        snprintf(buffer, 2048, "%s", b);
    } else if (a[strlen(a) - 1] == '/') {
        snprintf(buffer, 2048, "%s%s", a, b);
    } else {
        snprintf(buffer, 2048, "%s/%s", a, b);
    }
    return buffer;
}

static char* nl_os_path_basename(const char* path) {
    char* path_copy = strdup(path);
    char* base = basename(path_copy);
    char* result = strdup(base);
    free(path_copy);
    return result;
}

static char* nl_os_path_dirname(const char* path) {
    char* path_copy = strdup(path);
    char* dir = dirname(path_copy);
    char* result = strdup(dir);
    free(path_copy);
    return result;
}

static int64_t nl_os_system(const char* command) {
    return system(command);
}

static void nl_os_exit(int64_t code) {
    exit((int)code);
}

static char* nl_os_getenv(const char* name) {
    const char* value = getenv(name);
    return value ? (char*)value : "";
}

/* ========== End OS Standard Library ========== */

/* ========== Advanced String Operations ========== */

static int64_t char_at(const char* s, int64_t index) {
    int len = strlen(s);
    if (index < 0 || index >= len) {
        fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", index, len);
        return 0;
    }
    return (unsigned char)s[index];
}

static char* string_from_char(int64_t c) {
    char* buffer = malloc(2);
    buffer[0] = (char)c;
    buffer[1] = '\0';
    return buffer;
}

static bool is_digit(int64_t c) {
    return c >= '0' && c <= '9';
}

static bool is_alpha(int64_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool is_alnum(int64_t c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

static bool is_whitespace(int64_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static bool is_upper(int64_t c) {
    return c >= 'A' && c <= 'Z';
}

static bool is_lower(int64_t c) {
    return c >= 'a' && c <= 'z';
}

static char* int_to_string(int64_t n) {
    char* buffer = malloc(32);
    snprintf(buffer, 32, "%lld", n);
    return buffer;
}

static int64_t string_to_int(const char* s) {
    return strtoll(s, NULL, 10);
}

static int64_t digit_value(int64_t c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    return -1;
}

static int64_t char_to_lower(int64_t c) {
    if (c >= 'A' && c <= 'Z') {
        return c + 32;
    }
    return c;
}

static int64_t char_to_upper(int64_t c) {
    if (c >= 'a' && c <= 'z') {
        return c - 32;
    }
    return c;
}

/* ========== End Advanced String Operations ========== */

/* ========== Math and Utility Built-in Functions ========== */

#define nl_abs(x) _Generic((x), \
    int64_t: (int64_t)((x) < 0 ? -(x) : (x)), \
    double: (double)((x) < 0.0 ? -(x) : (x)))

#define nl_min(a, b) _Generic((a), \
    int64_t: (int64_t)((a) < (b) ? (a) : (b)), \
    double: (double)((a) < (b) ? (a) : (b)))

#define nl_max(a, b) _Generic((a), \
    int64_t: (int64_t)((a) > (b) ? (a) : (b)), \
    double: (double)((a) > (b) ? (a) : (b)))

static void nl_println(void* value_ptr) {
    /* This is a placeholder - actual implementation uses type info from checker */
}

static void nl_print_int(int64_t value) {
    printf("%lld", value);
}

static void nl_print_float(double value) {
    printf("%g", value);
}

static void nl_print_string(const char* value) {
    printf("%s", value);
}

static void nl_print_bool(bool value) {
    printf(value ? "true" : "false");
}

static void nl_println_int(int64_t value) {
    printf("%lld\n", value);
}

static void nl_println_float(double value) {
    printf("%g\n", value);
}

static void nl_println_string(const char* value) {
    printf("%s\n", value);
}

/* String concatenation */
static const char* nl_str_concat(const char* s1, const char* s2) {
    size_t len1 = strlen(s1);
    size_t len2 = strlen(s2);
    char* result = malloc(len1 + len2 + 1);
    if (!result) return "";
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

/* String substring */
static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {
    int64_t str_len = strlen(str);
    if (start < 0 || start >= str_len || length < 0) return "";
    if (start + length > str_len) length = str_len - start;
    char* result = malloc(length + 1);
    if (!result) return "";
    strncpy(result, str + start, length);
    result[length] = '\0';
    return result;
}

/* String contains */
static bool nl_str_contains(const char* str, const char* substr) {
    return strstr(str, substr) != NULL;
}

/* String equals */
static bool nl_str_equals(const char* s1, const char* s2) {
    return strcmp(s1, s2) == 0;
}

static void nl_println_bool(bool value) {
    printf("%s\n", value ? "true" : "false");
}

/* ========== Array Operations (With Bounds Checking!) ========== */

/* Array struct */
typedef struct {
    int64_t length;
    void* data;
    size_t element_size;
} nl_array;

/* Array access - BOUNDS CHECKED! */
static int64_t nl_array_at_int(nl_array* arr, int64_t index) {
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n", index, arr->length);
        exit(1);
    }
    return ((int64_t*)arr->data)[index];
}

static double nl_array_at_float(nl_array* arr, int64_t index) {
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n", index, arr->length);
        exit(1);
    }
    return ((double*)arr->data)[index];
}

static const char* nl_array_at_string(nl_array* arr, int64_t index) {
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n", index, arr->length);
        exit(1);
    }
    return ((const char**)arr->data)[index];
}

/* Array length */
static int64_t nl_array_length(nl_array* arr) {
    return arr->length;
}

/* Array creation */
static nl_array* nl_array_new_int(int64_t size, int64_t default_val) {
    nl_array* arr = malloc(sizeof(nl_array));
    arr->length = size;
    arr->element_size = sizeof(int64_t);
    arr->data = malloc(size * sizeof(int64_t));
    for (int64_t i = 0; i < size; i++) {
        ((int64_t*)arr->data)[i] = default_val;
    }
    return arr;
}

/* Array set - BOUNDS CHECKED! */
static void nl_array_set_int(nl_array* arr, int64_t index, int64_t value) {
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n", index, arr->length);
        exit(1);
    }
    ((int64_t*)arr->data)[index] = value;
}

/* Array literal creation helper */
static nl_array* nl_array_literal_int(int64_t count, ...) {
    nl_array* arr = malloc(sizeof(nl_array));
    arr->length = count;
    arr->element_size = sizeof(int64_t);
    arr->data = malloc(count * sizeof(int64_t));
    va_list args;
    va_start(args, count);
    for (int64_t i = 0; i < count; i++) {
        ((int64_t*)arr->data)[i] = va_arg(args, int64_t);
    }
    va_end(args);
    return arr;
}

/* ========== End Array Operations ========== */

/* ========== End Math and Utility Built-in Functions ========== */

/* ========== Struct Definitions ========== */

/* ========== End Struct Definitions ========== */

/* ========== Enum Definitions ========== */

/* ========== End Enum Definitions ========== */

/* ========== Generic List Specializations ========== */

/* ========== End Generic List Specializations ========== */

/* ========== Union Definitions ========== */

/* ========== End Union Definitions ========== */

int64_t nl_add(int64_t a, int64_t b);
int64_t nl_main();

int64_t nl_add(int64_t a, int64_t b) {
    return (a + b);
}

int64_t nl_main() {
    nl_println_string("Simple function variable test");
    BinaryOp_0 my_func = nl_add;
    int64_t result = my_func(10LL, 20LL);
    nl_print_string("Result: ");
    nl_println_int(result);
    return 0LL;
}


/* C main() entry point - calls nanolang main (nl_main) */
int main() {
    return (int)nl_main();
}
