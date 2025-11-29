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

/* Headers from imported modules (sorted by priority) */
#include <SDL.h>  /* priority: 0 */
#include <sdl_helpers.h>  /* priority: 0 */
#include <SDL_ttf.h>  /* priority: 0 */

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

static void nl_array_set_int(DynArray* arr, int64_t idx, int64_t val) {
    dyn_array_set_int(arr, idx, val);
}

static void nl_array_set_float(DynArray* arr, int64_t idx, double val) {
    dyn_array_set_float(arr, idx, val);
}

static DynArray* nl_array_new_int(int64_t size, int64_t default_val) {
    DynArray* arr = dyn_array_new(ELEM_INT);
    for (int64_t i = 0; i < size; i++) {
        dyn_array_push_int(arr, default_val);
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

/* ========== Struct Definitions ========== */

typedef struct nl_Vec3 {
    double x;
    double y;
    double z;
} nl_Vec3;

typedef struct nl_Ray {
    nl_Vec3 origin;
    nl_Vec3 direction;
} nl_Ray;

typedef struct nl_Sphere {
    nl_Vec3 center;
    double radius;
    nl_Vec3 color;
} nl_Sphere;

typedef struct nl_HitRecord {
    bool hit;
    double t;
    nl_Vec3 point;
    nl_Vec3 normal;
    nl_Vec3 color;
} nl_HitRecord;

/* ========== End Struct Definitions ========== */

/* ========== Enum Definitions ========== */

/* ========== End Enum Definitions ========== */

/* ========== Generic List Specializations ========== */

/* ========== End Generic List Specializations ========== */

/* ========== Union Definitions ========== */

/* ========== End Union Definitions ========== */

/* External C function declarations */

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
int64_t nl_draw_text_blended(int64_t renderer, int64_t font, const char* text, int64_t x, int64_t y, int64_t r, int64_t g, int64_t b, int64_t a);
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
static const int64_t WINDOW_WIDTH = 800LL;
static const int64_t WINDOW_HEIGHT = 600LL;

/* Forward declarations for program functions */
nl_Vec3 nl_vec3_new(double x, double y, double z);
nl_Vec3 nl_vec3_add(nl_Vec3 a, nl_Vec3 b);
nl_Vec3 nl_vec3_sub(nl_Vec3 a, nl_Vec3 b);
nl_Vec3 nl_vec3_mul_scalar(nl_Vec3 v, double s);
double nl_vec3_dot(nl_Vec3 a, nl_Vec3 b);
double nl_vec3_length_squared(nl_Vec3 v);
double nl_vec3_length(nl_Vec3 v);
nl_Vec3 nl_vec3_normalize(nl_Vec3 v);
nl_Vec3 nl_vec3_clamp(nl_Vec3 v, double min_val, double max_val);
nl_Vec3 nl_ray_at(nl_Ray r, double t);
nl_HitRecord nl_sphere_hit(nl_Sphere sphere, nl_Ray r, double t_min, double t_max);
nl_Vec3 nl_calculate_lighting(nl_HitRecord hit, nl_Vec3 light_pos, nl_Vec3 view_dir);
nl_Vec3 nl_scene_hit(nl_Ray r, nl_Vec3 light_pos);
nl_Ray nl_get_ray(double u, double v, nl_Vec3 camera_origin, double viewport_width, double viewport_height);
void nl_render_scene(int64_t renderer, nl_Vec3 light_pos, int64_t width, int64_t height);
void nl_draw_ui(int64_t renderer, int64_t font, int64_t light_x, int64_t light_y);
int64_t nl_main();

nl_Vec3 nl_vec3_new(double x, double y, double z) {
    return (nl_Vec3){.x = x, .y = y, .z = z};
}

nl_Vec3 nl_vec3_add(nl_Vec3 a, nl_Vec3 b) {
    return (nl_Vec3){.x = (a.x + b.x), .y = (a.y + b.y), .z = (a.z + b.z)};
}

nl_Vec3 nl_vec3_sub(nl_Vec3 a, nl_Vec3 b) {
    return (nl_Vec3){.x = (a.x - b.x), .y = (a.y - b.y), .z = (a.z - b.z)};
}

nl_Vec3 nl_vec3_mul_scalar(nl_Vec3 v, double s) {
    return (nl_Vec3){.x = (v.x * s), .y = (v.y * s), .z = (v.z * s)};
}

double nl_vec3_dot(nl_Vec3 a, nl_Vec3 b) {
    return (((a.x * b.x) + (a.y * b.y)) + (a.z * b.z));
}

double nl_vec3_length_squared(nl_Vec3 v) {
    return (((v.x * v.x) + (v.y * v.y)) + (v.z * v.z));
}

double nl_vec3_length(nl_Vec3 v) {
    return sqrt(nl_vec3_length_squared(v));
}

nl_Vec3 nl_vec3_normalize(nl_Vec3 v) {
    double len = nl_vec3_length(v);
    if (len > 0.0001) {
        return nl_vec3_mul_scalar(v, (1.0 / len));
    }
    else {
        return (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
    }
}

nl_Vec3 nl_vec3_clamp(nl_Vec3 v, double min_val, double max_val) {
    return (nl_Vec3){.x = nl_max(min_val, nl_min(v.x, max_val)), .y = nl_max(min_val, nl_min(v.y, max_val)), .z = nl_max(min_val, nl_min(v.z, max_val))};
}

nl_Vec3 nl_ray_at(nl_Ray r, double t) {
    nl_Vec3 origin = r.origin;
    nl_Vec3 direction = r.direction;
    nl_Vec3 scaled_dir = nl_vec3_mul_scalar(direction, t);
    return nl_vec3_add(origin, scaled_dir);
}

nl_HitRecord nl_sphere_hit(nl_Sphere sphere, nl_Ray r, double t_min, double t_max) {
    nl_Vec3 ray_origin = r.origin;
    nl_Vec3 ray_direction = r.direction;
    nl_Vec3 sphere_center = sphere.center;
    nl_Vec3 oc = nl_vec3_sub(ray_origin, sphere_center);
    double a = nl_vec3_length_squared(ray_direction);
    double half_b = nl_vec3_dot(oc, ray_direction);
    double c = (nl_vec3_length_squared(oc) - (sphere.radius * sphere.radius));
    double discriminant = ((half_b * half_b) - (a * c));
    nl_HitRecord no_hit = (nl_HitRecord){.hit = false, .t = 0.0, .point = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0}, .normal = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0}, .color = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0}};
    if (discriminant < 0.0) {
        return no_hit;
    }
    else {
        double sqrtd = sqrt(discriminant);
        double root = (((0.0 - half_b) - sqrtd) / a);
        double t_val = root;
        bool out_of_range1 = t_val < t_min;
        bool out_of_range2 = t_val > t_max;
        if (out_of_range1 || out_of_range2) {
            t_val = (((0.0 - half_b) + sqrtd) / a);
            bool out_of_range3 = t_val < t_min;
            bool out_of_range4 = t_val > t_max;
            if (out_of_range3 || out_of_range4) {
                return no_hit;
            }
            else {
                nl_print_string("");
            }
        }
        else {
            nl_print_string("");
        }
        nl_Vec3 hit_point = nl_ray_at(r, t_val);
        nl_Vec3 sphere_center2 = sphere.center;
        nl_Vec3 diff_from_center = nl_vec3_sub(hit_point, sphere_center2);
        nl_Vec3 normal = nl_vec3_normalize(diff_from_center);
        nl_Vec3 sphere_color = sphere.color;
        return (nl_HitRecord){.hit = true, .t = t_val, .point = hit_point, .normal = normal, .color = sphere_color};
    }
}

nl_Vec3 nl_calculate_lighting(nl_HitRecord hit, nl_Vec3 light_pos, nl_Vec3 view_dir) {
    nl_Vec3 hit_color = hit.color;
    nl_Vec3 hit_point = hit.point;
    nl_Vec3 hit_normal = hit.normal;
    nl_Vec3 ambient = nl_vec3_mul_scalar(hit_color, 0.2);
    nl_Vec3 light_to_hit = nl_vec3_sub(light_pos, hit_point);
    nl_Vec3 light_dir = nl_vec3_normalize(light_to_hit);
    double diff = nl_max(0.0, nl_vec3_dot(hit_normal, light_dir));
    nl_Vec3 diffuse = nl_vec3_mul_scalar(hit_color, diff);
    nl_Vec3 half_vec = nl_vec3_add(light_dir, view_dir);
    nl_Vec3 half_dir = nl_vec3_normalize(half_vec);
    double spec = pow(nl_max(0.0, nl_vec3_dot(hit_normal, half_dir)), 32.0);
    nl_Vec3 specular = (nl_Vec3){.x = spec, .y = spec, .z = spec};
    nl_Vec3 result = nl_vec3_add(nl_vec3_add(ambient, diffuse), specular);
    return nl_vec3_clamp(result, 0.0, 1.0);
}

nl_Vec3 nl_scene_hit(nl_Ray r, nl_Vec3 light_pos) {
    nl_Sphere ground = (nl_Sphere){.center = (nl_Vec3){.x = 0.0, .y = -100.5, .z = -1.0}, .radius = 100.0, .color = (nl_Vec3){.x = 0.5, .y = 0.5, .z = 0.5}};
    nl_Sphere center_sphere = (nl_Sphere){.center = (nl_Vec3){.x = 0.0, .y = 0.0, .z = -1.0}, .radius = 0.5, .color = (nl_Vec3){.x = 0.8, .y = 0.3, .z = 0.3}};
    nl_Sphere left_sphere = (nl_Sphere){.center = (nl_Vec3){.x = -1.0, .y = 0.0, .z = -1.0}, .radius = 0.5, .color = (nl_Vec3){.x = 0.3, .y = 0.8, .z = 0.3}};
    nl_Sphere right_sphere = (nl_Sphere){.center = (nl_Vec3){.x = 1.0, .y = 0.0, .z = -1.0}, .radius = 0.5, .color = (nl_Vec3){.x = 0.3, .y = 0.3, .z = 0.8}};
    double closest_t = 1000000.0;
    bool did_hit = false;
    nl_Vec3 hit_point = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
    nl_Vec3 hit_normal = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
    nl_Vec3 hit_color = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
    nl_HitRecord hit_ground = nl_sphere_hit(ground, r, 0.001, closest_t);
    bool ground_hit = hit_ground.hit;
    if (ground_hit) {
        double ground_t = hit_ground.t;
        nl_Vec3 ground_point = hit_ground.point;
        nl_Vec3 ground_normal = hit_ground.normal;
        nl_Vec3 ground_color = hit_ground.color;
        closest_t = ground_t;
        did_hit = true;
        hit_point = ground_point;
        hit_normal = ground_normal;
        hit_color = ground_color;
    }
    else {
        nl_print_string("");
    }
    nl_HitRecord hit_center = nl_sphere_hit(center_sphere, r, 0.001, closest_t);
    bool center_hit = hit_center.hit;
    if (center_hit) {
        double center_t = hit_center.t;
        nl_Vec3 center_point = hit_center.point;
        nl_Vec3 center_normal = hit_center.normal;
        nl_Vec3 center_color = hit_center.color;
        closest_t = center_t;
        did_hit = true;
        hit_point = center_point;
        hit_normal = center_normal;
        hit_color = center_color;
    }
    else {
        nl_print_string("");
    }
    nl_HitRecord hit_left = nl_sphere_hit(left_sphere, r, 0.001, closest_t);
    bool left_hit = hit_left.hit;
    if (left_hit) {
        double left_t = hit_left.t;
        nl_Vec3 left_point = hit_left.point;
        nl_Vec3 left_normal = hit_left.normal;
        nl_Vec3 left_color = hit_left.color;
        closest_t = left_t;
        did_hit = true;
        hit_point = left_point;
        hit_normal = left_normal;
        hit_color = left_color;
    }
    else {
        nl_print_string("");
    }
    nl_HitRecord hit_right = nl_sphere_hit(right_sphere, r, 0.001, closest_t);
    bool right_hit = hit_right.hit;
    if (right_hit) {
        double right_t = hit_right.t;
        nl_Vec3 right_point = hit_right.point;
        nl_Vec3 right_normal = hit_right.normal;
        nl_Vec3 right_color = hit_right.color;
        closest_t = right_t;
        did_hit = true;
        hit_point = right_point;
        hit_normal = right_normal;
        hit_color = right_color;
    }
    else {
        nl_print_string("");
    }
    if (did_hit) {
        nl_HitRecord closest_hit = (nl_HitRecord){.hit = true, .t = closest_t, .point = hit_point, .normal = hit_normal, .color = hit_color};
        nl_Vec3 ray_dir = r.direction;
        nl_Vec3 neg_ray_dir = nl_vec3_mul_scalar(ray_dir, -1.0);
        nl_Vec3 view_dir = nl_vec3_normalize(neg_ray_dir);
        return nl_calculate_lighting(closest_hit, light_pos, view_dir);
    }
    else {
        nl_Vec3 ray_dir2 = r.direction;
        nl_Vec3 unit_dir = nl_vec3_normalize(ray_dir2);
        double t = ((unit_dir.y + 1.0) * 0.5);
        nl_Vec3 white = (nl_Vec3){.x = 1.0, .y = 1.0, .z = 1.0};
        nl_Vec3 blue = (nl_Vec3){.x = 0.5, .y = 0.7, .z = 1.0};
        nl_Vec3 sky_white = nl_vec3_mul_scalar(white, (1.0 - t));
        nl_Vec3 sky_blue = nl_vec3_mul_scalar(blue, t);
        return nl_vec3_add(sky_white, sky_blue);
    }
}

nl_Ray nl_get_ray(double u, double v, nl_Vec3 camera_origin, double viewport_width, double viewport_height) {
    double focal_length = 1.0;
    nl_Vec3 horizontal = (nl_Vec3){.x = viewport_width, .y = 0.0, .z = 0.0};
    nl_Vec3 vertical = (nl_Vec3){.x = 0.0, .y = viewport_height, .z = 0.0};
    nl_Vec3 lower_left = (nl_Vec3){.x = ((camera_origin.x - (viewport_width / 2.0)) - 0.0), .y = ((camera_origin.y - (viewport_height / 2.0)) - 0.0), .z = (camera_origin.z - focal_length)};
    nl_Vec3 h_offset = nl_vec3_mul_scalar(horizontal, u);
    nl_Vec3 v_offset = nl_vec3_mul_scalar(vertical, v);
    nl_Vec3 target = nl_vec3_add(nl_vec3_add(lower_left, h_offset), v_offset);
    nl_Vec3 direction = nl_vec3_sub(target, camera_origin);
    nl_Vec3 dir_normalized = nl_vec3_normalize(direction);
    return (nl_Ray){.origin = camera_origin, .direction = dir_normalized};
}

void nl_render_scene(int64_t renderer, nl_Vec3 light_pos, int64_t width, int64_t height) {
    nl_println_string("Rendering scene...");
    double aspect_ratio = (nl_cast_float(width) / nl_cast_float(height));
    double viewport_height = 2.0;
    double viewport_width = (aspect_ratio * viewport_height);
    nl_Vec3 camera_origin = (nl_Vec3){.x = 0.0, .y = 0.0, .z = 0.0};
    SDL_SetRenderDrawColor((SDL_Renderer*)renderer, (Uint8)0LL, (Uint8)0LL, (Uint8)0LL, (Uint8)255LL);
    SDL_RenderClear((SDL_Renderer*)renderer);
    int64_t y = 0LL;
    while (y < height) {
        if ((y % 50LL) == 0LL) {
            nl_print_string("Rendering row: ");
            nl_println_int(y);
        }
        else {
            nl_print_string("");
        }
        int64_t x = 0LL;
        while (x < width) {
            int64_t width_minus_1 = (width - 1LL);
            int64_t height_minus_1 = (height - 1LL);
            int64_t height_minus_y_minus_1 = ((height - y) - 1LL);
            double u = (nl_cast_float(x) / nl_cast_float(width_minus_1));
            double v = (nl_cast_float(height_minus_y_minus_1) / nl_cast_float(height_minus_1));
            nl_Ray r = nl_get_ray(u, v, camera_origin, viewport_width, viewport_height);
            nl_Vec3 color = nl_scene_hit(r, light_pos);
            int64_t ir = nl_cast_int((color.x * 255.0));
            int64_t ig = nl_cast_int((color.y * 255.0));
            int64_t ib = nl_cast_int((color.z * 255.0));
            SDL_SetRenderDrawColor((SDL_Renderer*)renderer, (Uint8)ir, (Uint8)ig, (Uint8)ib, (Uint8)255LL);
            SDL_RenderDrawPoint((SDL_Renderer*)renderer, x, y);
            x = (x + 1LL);
        }
        y = (y + 1LL);
    }
    SDL_RenderPresent((SDL_Renderer*)renderer);
    nl_println_string("Rendering complete!");
}

void nl_draw_ui(int64_t renderer, int64_t font, int64_t light_x, int64_t light_y) {
    SDL_SetRenderDrawColor((SDL_Renderer*)renderer, (Uint8)0LL, (Uint8)0LL, (Uint8)0LL, (Uint8)200LL);
    nl_sdl_render_fill_rect(renderer, 0LL, (600 - 60LL), 800, 60LL);
    nl_draw_text_blended(renderer, font, "SPACE = Render    ESC = Quit", 10LL, (600 - 50LL), 200LL, 200LL, 255LL, 255LL);
    const char* light_text = "Light: (";
    light_text = nl_str_concat(light_text, int_to_string(light_x));
    light_text = nl_str_concat(light_text, ", ");
    light_text = nl_str_concat(light_text, int_to_string(light_y));
    light_text = nl_str_concat(light_text, ")");
    nl_draw_text_blended(renderer, font, light_text, 10LL, (600 - 30LL), 255LL, 200LL, 100LL, 255LL);
    nl_draw_text_blended(renderer, font, "Click to set light position", ((800 / 2LL) - 100LL), (600 - 30LL), 150LL, 150LL, 150LL, 255LL);
}

int64_t nl_main() {
    SDL_Init((Uint32)32LL);
    int64_t window = (int64_t)SDL_CreateWindow("Ray Tracer Demo - Nanolang", 100LL, 100LL, 800, 600, (Uint32)4LL);
    int64_t renderer = (int64_t)SDL_CreateRenderer((SDL_Window*)window, -1LL, (Uint32)2LL);
    if (renderer == 0LL) {
        nl_println_string("Failed to create renderer");
        return 1LL;
    }
    else {
        nl_println_string("✓ SDL initialized");
    }
    TTF_Init();
    int64_t font = (int64_t)TTF_OpenFont("/System/Library/Fonts/Supplemental/Arial.ttf", 16LL);
    if (font == 0LL) {
        nl_println_string("Failed to load font");
        SDL_DestroyRenderer((SDL_Renderer*)renderer);
        SDL_DestroyWindow((SDL_Window*)window);
        SDL_Quit();
        return 1LL;
    }
    else {
        nl_println_string("✓ Font loaded");
    }
    SDL_EventState(1024LL, 0LL);
    nl_println_string("✓ Scene created with 4 spheres");
    int64_t light_x = 400LL;
    int64_t light_y = 200LL;
    double light_z = 2.0;
    bool running = true;
    bool needs_render = true;
    nl_println_string("");
    nl_println_string("=== RAY TRACER DEMO ===");
    nl_println_string("Click to set light position");
    nl_println_string("Press SPACE to render");
    nl_println_string("Press ESC to quit");
    nl_println_string("");
    SDL_SetRenderDrawColor((SDL_Renderer*)renderer, (Uint8)20LL, (Uint8)20LL, (Uint8)30LL, (Uint8)255LL);
    SDL_RenderClear((SDL_Renderer*)renderer);
    nl_draw_ui(renderer, font, light_x, light_y);
    SDL_RenderPresent((SDL_Renderer*)renderer);
    while (running) {
        int64_t key = nl_sdl_poll_keypress();
        if (key > -1LL) {
            if (key == 41LL) {
                running = false;
            }
            else {
                if (key == 44LL) {
                    needs_render = true;
                }
                else {
                    nl_print_string("");
                }
            }
        }
        else {
            nl_print_string("");
        }
        int64_t quit = nl_sdl_poll_event_quit();
        if (quit == 1LL) {
            running = false;
        }
        else {
            nl_print_string("");
        }
        int64_t mouse = nl_sdl_poll_mouse_click();
        if (mouse > -1LL) {
            light_x = (mouse / 10000LL);
            light_y = (mouse % 10000LL);
            needs_render = true;
            nl_print_string("Light position set to: (");
            nl_print_int(light_x);
            nl_print_string(", ");
            nl_print_int(light_y);
            nl_println_string(")");
        }
        else {
            nl_print_string("");
        }
        if (needs_render) {
            needs_render = false;
            double light_world_x = (((nl_cast_float(light_x) - (nl_cast_float(800) / 2.0)) * 4.0) / nl_cast_float(800));
            double light_world_y = ((((nl_cast_float(600) / 2.0) - nl_cast_float(light_y)) * 4.0) / nl_cast_float(600));
            nl_Vec3 light_pos = (nl_Vec3){.x = light_world_x, .y = light_world_y, .z = light_z};
            nl_render_scene(renderer, light_pos, 800, 600);
            nl_draw_ui(renderer, font, light_x, light_y);
            SDL_RenderPresent((SDL_Renderer*)renderer);
        }
        else {
            nl_draw_ui(renderer, font, light_x, light_y);
            SDL_RenderPresent((SDL_Renderer*)renderer);
        }
        SDL_Delay((Uint32)16LL);
    }
    nl_println_string("");
    nl_println_string("✨ Exiting...");
    nl_println_string("");
    TTF_CloseFont((TTF_Font*)font);
    TTF_Quit();
    SDL_DestroyRenderer((SDL_Renderer*)renderer);
    SDL_DestroyWindow((SDL_Window*)window);
    SDL_Quit();
    return 0LL;
}


/* C main() entry point - calls nanolang main (nl_main) */
int main() {
    return (int)nl_main();
}
