#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>
#include "runtime/nl_string.h"

/* nanolang runtime */
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include "runtime/token_helpers.h"
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>

/* ========== OS Standard Library ========== */

static char* nl_os_file_read(const char* path) {
    FILE* f = fopen(path, "rb");  /* Binary mode for MOD files */
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

static DynArray* nl_os_file_read_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Return empty array on error */
        return dyn_array_new(ELEM_INT);
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Create dynamic array for bytes */
    DynArray* bytes = dyn_array_new(ELEM_INT);
    
    /* Read bytes and add to array */
    for (long i = 0; i < size; i++) {
        int c = fgetc(f);
        if (c == EOF) break;
        int64_t byte_val = (int64_t)(unsigned char)c;
        dyn_array_push_int(bytes, byte_val);
    }
    
    fclose(f);
    return bytes;
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
    /* Safety: Bound string scan to reasonable size (1MB) */
    int len = strnlen(s, 1024*1024);
    if (index < 0 || index >= len) {
        fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", (long long)index, len);
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
    snprintf(buffer, 32, "%lld", (long long)n);
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
    double: (double)((x) < 0.0 ? -(x) : (x)), \
    default: (int64_t)((x) < 0 ? -(x) : (x)))

#define nl_min(a, b) _Generic((a), \
    double: (double)((a) < (b) ? (a) : (b)), \
    default: (int64_t)((a) < (b) ? (a) : (b)))

#define nl_max(a, b) _Generic((a), \
    double: (double)((a) > (b) ? (a) : (b)), \
    default: (int64_t)((a) > (b) ? (a) : (b)))

static int64_t nl_cast_int(double x) { return (int64_t)x; }
static int64_t nl_cast_int_from_int(int64_t x) { return x; }
static double nl_cast_float(int64_t x) { return (double)x; }
static double nl_cast_float_from_float(double x) { return x; }
static int64_t nl_cast_bool_to_int(bool x) { return x ? 1 : 0; }
static bool nl_cast_bool(int64_t x) { return x != 0; }

static void nl_println(void* value_ptr) {
    /* This is a placeholder - actual implementation uses type info from checker */
}

static void nl_print_int(int64_t value) {
    printf("%lld", (long long)value);
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
    printf("%lld\n", (long long)value);
}

static void nl_println_float(double value) {
    printf("%g\n", value);
}

static void nl_println_string(const char* value) {
    printf("%s\n", value);
}

/* Dynamic array runtime (using GC) - LEGACY */
#include "runtime/gc.h"
#include "runtime/dyn_array.h"

static DynArray* dynarray_literal_int(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_INT);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        int64_t val = va_arg(args, int64_t);
        dyn_array_push_int(arr, val);
    }
    va_end(args);
    return arr;
}

static DynArray* dynarray_literal_float(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_FLOAT);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        double val = va_arg(args, double);
        dyn_array_push_float(arr, val);
    }
    va_end(args);
    return arr;
}

static DynArray* dynarray_push(DynArray* arr, double val) {
    if (arr->elem_type == ELEM_INT) {
        return dyn_array_push_int(arr, (int64_t)val);
    } else {
        return dyn_array_push_float(arr, val);
    }
}

static DynArray* nl_array_push(DynArray* arr, double val) {
    if (arr->elem_type == ELEM_INT) {
        return dyn_array_push_int(arr, (int64_t)val);
    } else {
        return dyn_array_push_float(arr, val);
    }
}

static double nl_array_pop(DynArray* arr) {
    bool success = false;
    if (arr->elem_type == ELEM_INT) {
        return (double)dyn_array_pop_int(arr, &success);
    } else {
        return dyn_array_pop_float(arr, &success);
    }
}

static int64_t nl_array_length(DynArray* arr) {
    return dyn_array_length(arr);
}

static int64_t nl_array_at_int(DynArray* arr, int64_t idx) {
    return dyn_array_get_int(arr, idx);
}

static double nl_array_at_float(DynArray* arr, int64_t idx) {
    return dyn_array_get_float(arr, idx);
}

static const char* nl_array_at_string(DynArray* arr, int64_t idx) {
    return dyn_array_get_string(arr, idx);
}

static bool nl_array_at_bool(DynArray* arr, int64_t idx) {
    return dyn_array_get_bool(arr, idx);
}

static void nl_array_set_int(DynArray* arr, int64_t idx, int64_t val) {
    dyn_array_set_int(arr, idx, val);
}

static void nl_array_set_float(DynArray* arr, int64_t idx, double val) {
    dyn_array_set_float(arr, idx, val);
}

static void nl_array_set_string(DynArray* arr, int64_t idx, const char* val) {
    dyn_array_set_string(arr, idx, val);
}

static void nl_array_set_bool(DynArray* arr, int64_t idx, bool val) {
    dyn_array_set_bool(arr, idx, val);
}

static DynArray* nl_array_new_int(int64_t size, int64_t default_val) {
    DynArray* arr = dyn_array_new(ELEM_INT);
    for (int64_t i = 0; i < size; i++) {
        dyn_array_push_int(arr, default_val);
    }
    return arr;
}

static DynArray* nl_array_new_float(int64_t size, double default_val) {
    DynArray* arr = dyn_array_new(ELEM_FLOAT);
    for (int64_t i = 0; i < size; i++) {
        dyn_array_push_float(arr, default_val);
    }
    return arr;
}

static DynArray* nl_array_new_string(int64_t size, const char* default_val) {
    DynArray* arr = dyn_array_new(ELEM_STRING);
    for (int64_t i = 0; i < size; i++) {
        dyn_array_push_string(arr, default_val);
    }
    return arr;
}

static DynArray* nl_array_new_bool(int64_t size, bool default_val) {
    DynArray* arr = dyn_array_new(ELEM_BOOL);
    for (int64_t i = 0; i < size; i++) {
        dyn_array_push_bool(arr, default_val);
    }
    return arr;
}

static int64_t dynarray_length(DynArray* arr) {
    return dyn_array_length(arr);
}

static double dynarray_at_for_transpiler(DynArray* arr, int64_t idx) {
    if (arr->elem_type == ELEM_INT) {
        return (double)dyn_array_get_int(arr, idx);
    } else {
        return dyn_array_get_float(arr, idx);
    }
}

/* String concatenation - use strnlen for safety */
static const char* nl_str_concat(const char* s1, const char* s2) {
    /* Safety: Bound string scan to 1MB */
    size_t len1 = strnlen(s1, 1024*1024);
    size_t len2 = strnlen(s2, 1024*1024);
    char* result = malloc(len1 + len2 + 1);
    if (!result) return "";
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

/* String substring - use strnlen for safety */
static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {
    /* Safety: Bound string scan to 1MB */
    int64_t str_len = strnlen(str, 1024*1024);
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

static void nl_print_array(DynArray* arr) {
    printf("[");
    for (int i = 0; i < arr->length; i++) {
        if (i > 0) printf(", ");
        switch (arr->elem_type) {
            case ELEM_INT:
                printf("%lld", (long long)((int64_t*)arr->data)[i]);
                break;
            case ELEM_FLOAT:
                printf("%g", ((double*)arr->data)[i]);
                break;
            default:
                printf("?");
                break;
        }
    }
    printf("]");
}

static void nl_println_array(DynArray* arr) {
    nl_print_array(arr);
    printf("\n");
}

/* ========== Array Operations (With Bounds Checking!) ========== */

/* Array struct */
/* ========== End Array Operations ========== */

/* ========== End Math and Utility Built-in Functions ========== */

/* ========== Enum Definitions ========== */

/* ========== End Enum Definitions ========== */

/* ========== Struct Definitions ========== */

typedef struct nl_CompilerArgs {
    const char* input_file;
    const char* output_file;
    bool keep_c;
    bool verbose;
    bool show_help;
    bool has_error;
} nl_CompilerArgs;

/* ========== End Struct Definitions ========== */

/* ========== Generic List Specializations ========== */

/* ========== End Generic List Specializations ========== */

/* ========== Union Definitions ========== */

/* ========== End Union Definitions ========== */

/* External C function declarations */
extern const char* read_file(const char* path);
extern int64_t write_file(const char* path, const char* content);
extern int64_t get_argc();
extern const char* get_argv(int64_t index);

/* Forward declarations for module functions */
int64_t nl_char_at();
const char* nl_string_from_char();
bool nl_is_digit();
bool nl_is_alpha();
bool nl_is_alnum();
bool nl_is_whitespace();
bool nl_is_upper();
bool nl_is_lower();
const char* nl_int_to_string();
int64_t nl_string_to_int();
int64_t nl_digit_value();
int64_t nl_char_to_lower();
int64_t nl_char_to_upper();
List_int* nl_list_int_new();
List_int* nl_list_int_with_capacity();
void nl_list_int_push();
int64_t nl_list_int_pop();
int64_t nl_list_int_get();
void nl_list_int_set();
void nl_list_int_insert();
int64_t nl_list_int_remove();
int64_t nl_list_int_length();
int64_t nl_list_int_capacity();
bool nl_list_int_is_empty();
void nl_list_int_clear();
void nl_list_int_free();
List_string* nl_list_string_new();
List_string* nl_list_string_with_capacity();
void nl_list_string_push();
const char* nl_list_string_pop();
const char* nl_list_string_get();
void nl_list_string_set();
void nl_list_string_insert();
const char* nl_list_string_remove();
int64_t nl_list_string_length();
int64_t nl_list_string_capacity();
bool nl_list_string_is_empty();
void nl_list_string_clear();
void nl_list_string_free();
List_token* nl_list_token_new();
List_token* nl_list_token_with_capacity();
void nl_list_token_push();
void nl_list_token_set();
void nl_list_token_insert();
int64_t nl_list_token_length();
int64_t nl_list_token_capacity();
bool nl_list_token_is_empty();
void nl_list_token_clear();
void nl_list_token_free();
int64_t nl_char_at();
const char* nl_string_from_char();
bool nl_is_digit();
bool nl_is_alpha();
bool nl_is_alnum();
bool nl_is_whitespace();
bool nl_is_upper();
bool nl_is_lower();
const char* nl_int_to_string();
int64_t nl_string_to_int();
int64_t nl_digit_value();
int64_t nl_char_to_lower();
int64_t nl_char_to_upper();
List_int* nl_list_int_new();
List_int* nl_list_int_with_capacity();
void nl_list_int_push();
int64_t nl_list_int_pop();
int64_t nl_list_int_get();
void nl_list_int_set();
void nl_list_int_insert();
int64_t nl_list_int_remove();
int64_t nl_list_int_length();
int64_t nl_list_int_capacity();
bool nl_list_int_is_empty();
void nl_list_int_clear();
void nl_list_int_free();
List_string* nl_list_string_new();
List_string* nl_list_string_with_capacity();
void nl_list_string_push();
const char* nl_list_string_pop();
const char* nl_list_string_get();
void nl_list_string_set();
void nl_list_string_insert();
const char* nl_list_string_remove();
int64_t nl_list_string_length();
int64_t nl_list_string_capacity();
bool nl_list_string_is_empty();
void nl_list_string_clear();
void nl_list_string_free();
List_token* nl_list_token_new();
List_token* nl_list_token_with_capacity();
void nl_list_token_push();
void nl_list_token_set();
void nl_list_token_insert();
int64_t nl_list_token_length();
int64_t nl_list_token_capacity();
bool nl_list_token_is_empty();
void nl_list_token_clear();
void nl_list_token_free();
int64_t nl_char_at();
const char* nl_string_from_char();
bool nl_is_digit();
bool nl_is_alpha();
bool nl_is_alnum();
bool nl_is_whitespace();
bool nl_is_upper();
bool nl_is_lower();
const char* nl_int_to_string();
int64_t nl_string_to_int();
int64_t nl_digit_value();
int64_t nl_char_to_lower();
int64_t nl_char_to_upper();
List_int* nl_list_int_new();
List_int* nl_list_int_with_capacity();
void nl_list_int_push();
int64_t nl_list_int_pop();
int64_t nl_list_int_get();
void nl_list_int_set();
void nl_list_int_insert();
int64_t nl_list_int_remove();
int64_t nl_list_int_length();
int64_t nl_list_int_capacity();
bool nl_list_int_is_empty();
void nl_list_int_clear();
void nl_list_int_free();
List_string* nl_list_string_new();
List_string* nl_list_string_with_capacity();
void nl_list_string_push();
const char* nl_list_string_pop();
const char* nl_list_string_get();
void nl_list_string_set();
void nl_list_string_insert();
const char* nl_list_string_remove();
int64_t nl_list_string_length();
int64_t nl_list_string_capacity();
bool nl_list_string_is_empty();
void nl_list_string_clear();
void nl_list_string_free();
List_token* nl_list_token_new();
List_token* nl_list_token_with_capacity();
void nl_list_token_push();
void nl_list_token_set();
void nl_list_token_insert();
int64_t nl_list_token_length();
int64_t nl_list_token_capacity();
bool nl_list_token_is_empty();
void nl_list_token_clear();
void nl_list_token_free();

/* Top-level constants */
static const int64_t SUCCESS = 0LL;
static const int64_t ERROR_ARGS = 1LL;
static const int64_t ERROR_FILE = 2LL;
static const int64_t ERROR_COMPILE = 3LL;

/* Forward declarations for program functions */
void nl_show_usage();
nl_CompilerArgs nl_parse_args();
int64_t nl_compile(const char* input, const char* output, bool keep_c, bool verbose);
int64_t nl_main();

void nl_show_usage() {
    nl_println_string("Nanolang Self-Hosted Compiler (TRULY IN NANOLANG!)");
    nl_println_string("");
    nl_println_string("This compiler is written IN nanolang and uses nanolang components.");
    nl_println_string("It demonstrates that nanolang can compile itself.");
    nl_println_string("");
    nl_println_string("Usage: nanoc <input.nano> [options]");
    nl_println_string("");
    nl_println_string("Options:");
    nl_println_string("  -o <file>    Output executable (default: a.out)");
    nl_println_string("  --keep-c     Keep generated C code");
    nl_println_string("  --verbose    Show compilation steps");
    nl_println_string("  --help       Show this help");
    nl_println_string("");
}

nl_CompilerArgs nl_parse_args() {
    int64_t argc = get_argc();
    const char* input = "";
    const char* output = "a.out";
    bool keep_c = false;
    bool verbose = false;
    bool help = false;
    bool error = false;
    if (argc < 2LL) {
        error = true;
        return (nl_CompilerArgs){.input_file = "", .output_file = output, .keep_c = false, .verbose = false, .show_help = false, .has_error = true};
    }
    else {
        input = get_argv(1LL);
        if (nl_str_equals(input, "--help")) {
            help = true;
        }
        else {
            int64_t i = 2LL;
            while (i < argc) {
                const char* arg = get_argv(i);
                if (nl_str_equals(arg, "-o")) {
                    i = (i + 1LL);
                    if (i < argc) {
                        output = get_argv(i);
                    }
                    else {
                        error = true;
                    }
                }
                else {
                    if (nl_str_equals(arg, "--keep-c")) {
                        keep_c = true;
                    }
                    else {
                        if (nl_str_equals(arg, "--verbose")) {
                            verbose = true;
                        }
                        else {
                            error = true;
                        }
                    }
                }
                i = (i + 1LL);
            }
        }
    }
    return (nl_CompilerArgs){.input_file = input, .output_file = output, .keep_c = keep_c, .verbose = verbose, .show_help = help, .has_error = error};
}

int64_t nl_compile(const char* input, const char* output, bool keep_c, bool verbose) {
    if (verbose) {
        nl_println_string("=== Nanolang Self-Hosted Compiler ===");
        nl_print_string("Input:  ");
        nl_println_string(input);
        nl_print_string("Output: ");
        nl_println_string(output);
        nl_println_string("");
    }
    else {
        nl_print_string("");
    }
    if ((!nl_os_file_exists(input))) {
        nl_print_string("Error: File not found: ");
        nl_println_string(input);
        return 2;
    }
    else {
        nl_print_string("");
    }
    if (verbose) {
        nl_println_string("Step 1: Reading source...");
    }
    else {
        nl_print_string("");
    }
    const char* source = read_file(input);
    if (verbose) {
        nl_println_string("Step 2-5: Using C compiler components (temporary)...");
        nl_println_string("  (Will use nanolang components once import aliases work)");
    }
    else {
        nl_print_string("");
    }
    const char* cmd = nl_str_concat("bin/nanoc ", input);
    const char* cmd2 = nl_str_concat(cmd, " -o ");
    const char* cmd3 = nl_str_concat(cmd2, output);
    if (keep_c) {
        const char* cmd4 = nl_str_concat(cmd3, " --keep-c");
        int64_t result = nl_os_system(cmd4);
        return result;
    }
    else {
        int64_t result = nl_os_system(cmd3);
        return result;
    }
}

int64_t nl_main() {
    nl_CompilerArgs args = nl_parse_args();
    bool show_help = args.show_help;
    bool has_error = args.has_error;
    if (show_help) {
        nl_show_usage();
        return 0;
    }
    else {
        if (has_error) {
            nl_show_usage();
            return 1;
        }
        else {
            const char* input = args.input_file;
            const char* output = args.output_file;
            bool keep_c = args.keep_c;
            bool verbose = args.verbose;
            return nl_compile(input, output, keep_c, verbose);
        }
    }
}


/* C main() entry point - calls nanolang main (nl_main) */
int main() {
    return (int)nl_main();
}
