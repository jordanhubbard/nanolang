#define _GNU_SOURCE 1
#define _DARWIN_C_SOURCE 1
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
#ifdef __wasm__
#  include "runtime/refcount_gc.h"
#  define gc_alloc_string(len)   nl_rc_str_new(NULL, (len))
#  define gc_retain(p)           nl_rc_retain(p)
#  define gc_release(p)          nl_rc_release(p)
#else
#  include "runtime/gc.h"
#endif
#include "runtime/dyn_array.h"
#include "runtime/native_array_abi.h"
#ifndef __wasm__
#  include "nanolang.h"
#endif

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
#include <sys/wait.h>
#include <spawn.h>
#include <fcntl.h>
/* fs.h forward declarations */
DynArray* fs_walkdir(const char* root);
const char* path_normalize(const char* path);
const char* path_join(const char* a, const char* b);
const char* path_basename(const char* path);
const char* path_dirname(const char* path);
int64_t  fs_mkdir_p(const char* path);
const char* path_relpath(const char* target, const char* base);
const char* file_read(const char* path);
int64_t  file_write(const char* path, const char* content);
int64_t  file_append(const char* path, const char* content);
bool     file_exists(const char* path);
int64_t  file_delete(const char* path);
int64_t  file_copy(const char* src, const char* dst);
int64_t  dir_copy(const char* src, const char* dst);
int64_t  nl_os_process_spawn(const char* command);
int64_t  nl_os_process_is_running(int64_t pid);
int64_t  nl_os_process_wait(int64_t pid);
DynArray* nl_os_process_spawn_with_pipes(const char* command);
const char* nl_os_fd_read_available(int64_t fd);
int64_t  nl_os_fd_close(int64_t fd);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-const-variable"

/* ========== OS Standard Library ========== */

#include "runtime/file_text.h"
static const char* nl_os_file_read(const char* path) {
    char* text = nl_read_file_text(path);
    if (!text) return gc_alloc_string(0);
    size_t length = strlen(text);
    char* result = gc_alloc_string(length);
    if (result) memcpy(result, text, length + 1);
    free(text);
    return result;
}

#include "runtime/file_bytes.h"
static DynArray* nl_os_file_read_bytes(const char* path) {
    return nl_read_file_bytes(path);
}

#include "runtime/file_write.h"
static int64_t nl_os_file_write(const char* path, const char* content) {
    return nl_write_file_text(path, content, "w");
}

static int64_t nl_os_file_append(const char* path, const char* content) {
    return nl_write_file_text(path, content, "a");
}

static int64_t nl_os_file_remove(const char* path) {
    return remove(path) == 0 ? 0 : -1;
}

static int64_t nl_os_file_delete(const char* path) {
    return nl_os_file_remove(path);
}

static int64_t nl_os_file_rename(const char* old_path, const char* new_path) {
    return rename(old_path, new_path) == 0 ? 0 : -1;
}

static int64_t nl_os_file_size(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return (int64_t)st.st_size;
}

static bool nl_os_file_exists(const char* path) {
    struct stat st;
    return stat(path, &st) == 0;
}

static char* nl_os_tmp_dir(void) {
    const char* tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    size_t len = strlen(tmp);
    char* out = gc_alloc_string(len);
    if (!out) return gc_alloc_string(0);
    memcpy(out, tmp, len);
    out[len] = '\0';
    return out;
}

static char* nl_os_mktemp(const char* prefix) {
    const char* tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    const char* p = (prefix && prefix[0]) ? prefix : "nanolang_";
    char templ[1024];
    snprintf(templ, sizeof(templ), "%s/%sXXXXXX", tmp, p);
    int fd = mkstemp(templ);
    if (fd < 0) return gc_alloc_string(0);
    close(fd);
    size_t len = strlen(templ);
    char* out = gc_alloc_string(len);
    if (!out) return gc_alloc_string(0);
    memcpy(out, templ, len);
    out[len] = '\0';
    return out;
}

static char* nl_os_mktemp_dir(const char* prefix) {
    const char* tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    const char* p = (prefix && prefix[0]) ? prefix : "nanolang_dir_";
    char path[1024];
    for (int i = 0; i < 100; i++) {
        snprintf(path, sizeof(path), "%s/%s%lld_%d", tmp, p, (long long)time(NULL), i);
        if (mkdir(path, 0700) == 0) {
            size_t len = strlen(path);
            char* out = gc_alloc_string(len);
            if (!out) return gc_alloc_string(0);
            memcpy(out, path, len);
            out[len] = '\0';
            return out;
        }
    }
    return gc_alloc_string(0);
}

static int64_t nl_os_dir_create(const char* path) {
    return mkdir(path, 0755) == 0 ? 0 : -1;
}

static int64_t nl_os_dir_remove(const char* path) {
    return rmdir(path) == 0 ? 0 : -1;
}

static char* nl_os_dir_list(const char* path) {
    DIR* dir = opendir(path);
    if (!dir) return gc_alloc_string(0);
    size_t capacity = 4096;
    size_t used = 0;
    char* buffer = gc_alloc_string(capacity);
    if (!buffer) { closedir(dir); return gc_alloc_string(0); }
    buffer[0] = '\0';
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        size_t name_len = strlen(entry->d_name);
        size_t needed = used + name_len + 2; /* +1 for newline, +1 for null */
        if (needed > capacity) {
            capacity = needed * 2;
            char* new_buffer = gc_alloc_string(capacity);
            if (!new_buffer) { gc_release(buffer); closedir(dir); return gc_alloc_string(0); }
            memcpy(new_buffer, buffer, used);
            gc_release(buffer);
            buffer = new_buffer;
        }
        memcpy(buffer + used, entry->d_name, name_len);
        used += name_len;
        buffer[used++] = '\n';
        buffer[used] = '\0';
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
    char temp_buffer[1024];
    if (getcwd(temp_buffer, sizeof(temp_buffer)) == NULL) {
        return gc_alloc_string(0);
    }
    size_t len = strlen(temp_buffer);
    char* buffer = gc_alloc_string(len);
    if (buffer) memcpy(buffer, temp_buffer, len + 1);
    return buffer ? buffer : gc_alloc_string(0);
}

static int64_t nl_os_chdir(const char* path) {
    return chdir(path) == 0 ? 0 : -1;
}

#include "runtime/directory_walk.h"
static DynArray* nl_os_walkdir(const char* root) {
    return nl_fs_walkdir(root);
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
    size_t len_a = strlen(a);
    size_t len_b = strlen(b);
    size_t total_len = len_a + len_b + 2; /* +1 for '/', +1 for null */
    char* buffer = gc_alloc_string(total_len);
    if (!buffer) return gc_alloc_string(0);
    if (len_a == 0) {
        snprintf(buffer, total_len, "%s", b);
    } else if (a[len_a - 1] == '/') {
        snprintf(buffer, total_len, "%s%s", a, b);
    } else {
        snprintf(buffer, total_len, "%s/%s", a, b);
    }
    return buffer;
}

static char* nl_os_path_basename(const char* path) {
    char* path_copy = strdup(path);
    char* base = basename(path_copy);
    size_t len = strlen(base);
    char* result = gc_alloc_string(len);
    if (result) memcpy(result, base, len + 1);
    free(path_copy);
    return result ? result : gc_alloc_string(0);
}

static char* nl_os_path_dirname(const char* path) {
    char* path_copy = strdup(path);
    char* dir = dirname(path_copy);
    size_t len = strlen(dir);
    char* result = gc_alloc_string(len);
    if (result) memcpy(result, dir, len + 1);
    free(path_copy);
    return result ? result : gc_alloc_string(0);
}

#include "runtime/path_normalize.h"
static char* nl_os_path_normalize(const char* path) {
    if (!path) return gc_alloc_string(0);
    char* normalized = nl_normalize_path(path);
    if (!normalized) return gc_alloc_string(0);
    size_t length = strlen(normalized);
    char* out = gc_alloc_string(length);
    if (out) memcpy(out, normalized, length + 1);
    free(normalized);
    return out ? out : gc_alloc_string(0);
}

static int64_t nl_os_system(const char* command) {
    return system(command);
}

static void nl_os_exit(int64_t code) {
    exit((int)code);
}

static const char* nl_os_getenv(const char* name) {
    const char* value = getenv(name);
    return value ? value : "";
}

/* system() wrapper - stdlib system() available via stdlib.h */
static inline int64_t nl_exec_shell(const char* cmd) {
    return (int64_t)system(cmd);
}

/* isatty() wrapper - avoids int64_t type clash with unistd.h */
static inline int64_t nl_isatty(int64_t fd) {
    return (int64_t)isatty((int)fd);
}

/* Capture stdout from a shell command */
static inline const char* nl_exec_capture(const char* cmd) {
    FILE* pipe = popen(cmd, "r");
    if (!pipe) return "";
    char* out = (char*)malloc(65536);
    if (!out) { pclose(pipe); return ""; }
    size_t total = 0;
    while (total < 65535) {
        size_t n = fread(out + total, 1, 65535 - total, pipe);
        if (n == 0) break;
        total += n;
    }
    out[total] = '\0';
    pclose(pipe);
    return out;
}

#include "runtime/process_capture.h"
#ifndef NANOLANG_STD_PROCESS_H
static DynArray* nl_os_process_run(const char* command) {
    return nl_process_run_capture(command);
}
#endif /* NANOLANG_STD_PROCESS_H */

/* ========== End OS Standard Library ========== */

/* I canonicalize only scalar binary arithmetic results, never transported bits. */
#ifndef NANOLANG_BINARY64_ARITHMETIC_H
#define NANOLANG_BINARY64_ARITHMETIC_H
#include <float.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I require ordinary IEEE arithmetic without fast-math."
#endif
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 double arithmetic."
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "I require operations evaluated in their binary64 type."
#endif
/* I retain a compile-time storage check in both C99 and C11 output. */
typedef char nano_rt_binary64_storage_guard[
    sizeof(double) == 8 && sizeof(uint64_t) == 8 ? 1 : -1];

/* I inspect a rounded result with integer operations, not another FP operation. */
static inline double nano_rt_f64_arithmetic_result(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0) {
        bits = UINT64_C(0x7ff8000000000000);
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

/* Each volatile store/load is a binary64 rounding and noncontraction boundary. */
static inline double nano_rt_f64_add(double a, double b) {
    volatile double rounded = a + b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_sub(double a, double b) {
    volatile double rounded = a - b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_mul(double a, double b) {
    volatile double rounded = a * b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_div(double a, double b) {
    uint64_t divisor;
    memcpy(&divisor, &b, sizeof(divisor));
    /* Either signed zero takes precedence even over a signaling NaN numerator. */
    if ((divisor & UINT64_C(0x7fffffffffffffff)) == 0) return 0.0;
    volatile double rounded = a / b;
    return nano_rt_f64_arithmetic_result(rounded);
}
#endif
#ifndef NANOLANG_BINARY64_FORMAT_H
#define NANOLANG_BINARY64_FORMAT_H
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 formatting storage."
#endif
typedef char nano_rt_f64_format_storage[(sizeof(double) == 8 && sizeof(uint64_t) == 8) ? 1 : -1]; static inline const char *nano_rt_f64_nonfinite(double value) { uint64_t bits; memcpy(&bits, &value, sizeof bits); if ((bits & 0x7ff0000000000000UL) != 0x7ff0000000000000UL) return ((void *)0); if (bits & 0x000fffffffffffffUL) return bits >> 63 ? "-nan" : "nan"; return bits >> 63 ? "-inf" : "inf"; } static inline int nano_rt_f64_format(char *out, size_t size, double value) { const char *special = nano_rt_f64_nonfinite(value); return special ? snprintf(out, size, "%s", special) : snprintf(out, size, "%g", value); } static inline int nano_rt_f64_print(FILE *out, double value) { const char *special = nano_rt_f64_nonfinite(value); return special ? fprintf(out, "%s", special) : fprintf(out, "%g", value); }
#endif
/* ========== Advanced String Operations ========== */

static int64_t char_at(const char* s, int64_t index) {
    /* Safety: Bound string scan to 64MB */
    int len = strnlen(s, 64*1024*1024);
    if (index < 0 || index >= len) {
        fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", (long long)index, len);
        return 0;
    }
    return (unsigned char)s[index];
}

static char* string_from_char(int64_t c) {
    char* buffer = gc_alloc_string(1);
    if (!buffer) return "";
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
    char* buffer = gc_alloc_string(31);
    if (!buffer) return "";
    snprintf(buffer, 32, "%lld", (long long)n);
    return buffer;
}

static char* float_to_string(double x) {
    char* buffer = gc_alloc_string(63);
    if (!buffer) return "";
    nano_rt_f64_format(buffer, 64, x);
    /* Ensure at least one decimal place for whole-number floats
     * so 0.0 -> "0.0" rather than "0" */
    if (!strchr(buffer, '.') && !strchr(buffer, 'e')
            && !strchr(buffer, 'n') && !strchr(buffer, 'i')) {
        size_t len = strlen(buffer);
        if (len + 2 < 64) { buffer[len] = '.'; buffer[len+1] = '0'; buffer[len+2] = '\0'; }
    }
    return buffer;
}

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} nl_fmt_sb_t;

static void nl_fmt_sb_ensure(nl_fmt_sb_t *sb, size_t extra) {
    if (!sb) return;
    size_t needed = sb->len + extra + 1;
    if (needed <= sb->cap) return;
    size_t new_cap = sb->cap ? sb->cap : 128;
    while (new_cap < needed) new_cap *= 2;
    char *new_buf = realloc(sb->buf, new_cap);
    if (!new_buf) return;
    sb->buf = new_buf;
    sb->cap = new_cap;
}

static nl_fmt_sb_t nl_fmt_sb_new(size_t initial_cap) {
    nl_fmt_sb_t sb = {0};
    sb.cap = initial_cap ? initial_cap : 128;
    sb.buf = (char*)malloc(sb.cap);
    sb.len = 0;
    if (sb.buf) sb.buf[0] = '\0';
    return sb;
}

static void nl_fmt_sb_append_cstr(nl_fmt_sb_t *sb, const char *s) {
    if (!sb || !s) return;
    size_t n = strlen(s);
    nl_fmt_sb_ensure(sb, n);
    if (!sb->buf) return;
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}

static void nl_fmt_sb_append_char(nl_fmt_sb_t *sb, char c) {
    if (!sb) return;
    nl_fmt_sb_ensure(sb, 1);
    if (!sb->buf) return;
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = '\0';
}

static char* nl_fmt_sb_build(nl_fmt_sb_t *sb) {
    if (!sb || !sb->buf) return "";
    return sb->buf;
}

static const char* nl_to_string_int(int64_t v) { return int_to_string(v); }
static const char* nl_to_string_float(double v) { return float_to_string(v); }
static const char* nl_to_string_bool(bool v) { return v ? "true" : "false"; }
static const char* nl_to_string_string(const char* v) { return v ? v : ""; }

static const char* nl_to_string_array(DynArray* arr) {
    if (!arr) return "[]";
    nl_fmt_sb_t sb = nl_fmt_sb_new(256);
    nl_fmt_sb_append_char(&sb, '[');
    int64_t len = dyn_array_length(arr);
    ElementType t = dyn_array_get_elem_type(arr);
    for (int64_t i = 0; i < len; i++) {
        if (i > 0) nl_fmt_sb_append_cstr(&sb, ", ");
        switch (t) {
            case ELEM_INT: {
                const char* s = nl_to_string_int(dyn_array_get_int(arr, i));
                nl_fmt_sb_append_cstr(&sb, s);
                break;
            }
            case ELEM_U8: {
                const char* s = nl_to_string_int((int64_t)dyn_array_get_u8(arr, i));
                nl_fmt_sb_append_cstr(&sb, s);
                break;
            }
            case ELEM_FLOAT: {
                const char* s = nl_to_string_float(dyn_array_get_float(arr, i));
                nl_fmt_sb_append_cstr(&sb, s);
                break;
            }
            case ELEM_BOOL: {
                nl_fmt_sb_append_cstr(&sb, nl_to_string_bool(dyn_array_get_bool(arr, i)));
                break;
            }
            case ELEM_STRING: {
                nl_fmt_sb_append_char(&sb, '"');
                nl_fmt_sb_append_cstr(&sb, nl_to_string_string(dyn_array_get_string(arr, i)));
                nl_fmt_sb_append_char(&sb, '"');
                break;
            }
            case ELEM_ARRAY: {
                const char* s = nl_to_string_array(dyn_array_get_array(arr, i));
                nl_fmt_sb_append_cstr(&sb, s);
                break;
            }
            case ELEM_STRUCT: {
                nl_fmt_sb_append_cstr(&sb, "<struct>");
                break;
            }
            default: {
                nl_fmt_sb_append_cstr(&sb, "?");
                break;
            }
        }
    }
    nl_fmt_sb_append_char(&sb, ']');
    return nl_fmt_sb_build(&sb);
}

static const char* nl_str_concat(const char* s1, const char* s2);
static DynArray* nl_array_add(DynArray* a, DynArray* b);
static DynArray* nl_array_sub(DynArray* a, DynArray* b);
static DynArray* nl_array_mul(DynArray* a, DynArray* b);
static DynArray* nl_array_div(DynArray* a, DynArray* b);
static DynArray* nl_array_mod(DynArray* a, DynArray* b);

static void nl_array_assert_compatible(DynArray* a, DynArray* b) {
    assert(a && b);
    assert(dyn_array_length(a) == dyn_array_length(b));
    assert(dyn_array_get_elem_type(a) == dyn_array_get_elem_type(b));
}

static DynArray* nl_array_add(DynArray* a, DynArray* b) {
    nl_array_assert_compatible(a, b);
    ElementType t = dyn_array_get_elem_type(a);
    int64_t len = dyn_array_length(a);
    DynArray* out = dyn_array_new(t);
    switch (t) {
        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)+dyn_array_get_int(b,i)); break;
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_add(dyn_array_get_float(a,i), dyn_array_get_float(b,i))); break;
        case ELEM_STRING: for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(dyn_array_get_string(a,i), dyn_array_get_string(b,i))); break;
        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_add(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;
        default: assert(false && "nl_array_add: unsupported element type");
    }
    return out;
}

static DynArray* nl_array_sub(DynArray* a, DynArray* b) {
    nl_array_assert_compatible(a, b);
    ElementType t = dyn_array_get_elem_type(a);
    int64_t len = dyn_array_length(a);
    DynArray* out = dyn_array_new(t);
    switch (t) {
        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)-dyn_array_get_int(b,i)); break;
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_sub(dyn_array_get_float(a,i), dyn_array_get_float(b,i))); break;
        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_sub(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;
        default: assert(false && "nl_array_sub: unsupported element type");
    }
    return out;
}

static DynArray* nl_array_mul(DynArray* a, DynArray* b) {
    nl_array_assert_compatible(a, b);
    ElementType t = dyn_array_get_elem_type(a);
    int64_t len = dyn_array_length(a);
    DynArray* out = dyn_array_new(t);
    switch (t) {
        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)*dyn_array_get_int(b,i)); break;
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_mul(dyn_array_get_float(a,i), dyn_array_get_float(b,i))); break;
        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_mul(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;
        default: assert(false && "nl_array_mul: unsupported element type");
    }
    return out;
}

static DynArray* nl_array_div(DynArray* a, DynArray* b) {
    nl_array_assert_compatible(a, b);
    ElementType t = dyn_array_get_elem_type(a);
    int64_t len = dyn_array_length(a);
    DynArray* out = dyn_array_new(t);
    switch (t) {
        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)/dyn_array_get_int(b,i)); break;
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_div(dyn_array_get_float(a,i), dyn_array_get_float(b,i))); break;
        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_div(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;
        default: assert(false && "nl_array_div: unsupported element type");
    }
    return out;
}

static DynArray* nl_array_mod(DynArray* a, DynArray* b) {
    nl_array_assert_compatible(a, b);
    ElementType t = dyn_array_get_elem_type(a);
    int64_t len = dyn_array_length(a);
    DynArray* out = dyn_array_new(t);
    switch (t) {
        case ELEM_INT: for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i)%dyn_array_get_int(b,i)); break;
        case ELEM_ARRAY: for (int64_t i=0;i<len;i++) dyn_array_push_array(out, nl_array_mod(dyn_array_get_array(a,i), dyn_array_get_array(b,i))); break;
        default: assert(false && "nl_array_mod: unsupported element type");
    }
    return out;
}

static DynArray* nl_array_add_scalar_int(DynArray* a, int64_t s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) + s);
    return out;
}

static DynArray* nl_array_radd_scalar_int(int64_t s, DynArray* a) { return nl_array_add_scalar_int(a, s); }

static DynArray* nl_array_sub_scalar_int(DynArray* a, int64_t s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) - s);
    return out;
}

static DynArray* nl_array_rsub_scalar_int(int64_t s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s - dyn_array_get_int(a,i));
    return out;
}

static DynArray* nl_array_mul_scalar_int(DynArray* a, int64_t s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) * s);
    return out;
}

static DynArray* nl_array_rmul_scalar_int(int64_t s, DynArray* a) { return nl_array_mul_scalar_int(a, s); }

static DynArray* nl_array_div_scalar_int(DynArray* a, int64_t s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) / s);
    return out;
}

static DynArray* nl_array_rdiv_scalar_int(int64_t s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s / dyn_array_get_int(a,i));
    return out;
}

static DynArray* nl_array_mod_scalar_int(DynArray* a, int64_t s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, dyn_array_get_int(a,i) % s);
    return out;
}

static DynArray* nl_array_rmod_scalar_int(int64_t s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_INT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_INT);
    for (int64_t i=0;i<len;i++) dyn_array_push_int(out, s % dyn_array_get_int(a,i));
    return out;
}

static DynArray* nl_array_add_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_add(dyn_array_get_float(a,i), s));
    return out;
}

static DynArray* nl_array_radd_scalar_float(double s, DynArray* a) { return nl_array_add_scalar_float(a, s); }

static DynArray* nl_array_sub_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_sub(dyn_array_get_float(a,i), s));
    return out;
}

static DynArray* nl_array_rsub_scalar_float(double s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_sub(s, dyn_array_get_float(a,i)));
    return out;
}

static DynArray* nl_array_mul_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_mul(dyn_array_get_float(a,i), s));
    return out;
}

static DynArray* nl_array_rmul_scalar_float(double s, DynArray* a) { return nl_array_mul_scalar_float(a, s); }

static DynArray* nl_array_div_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_div(dyn_array_get_float(a,i), s));
    return out;
}

static DynArray* nl_array_rdiv_scalar_float(double s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, nano_rt_f64_div(s, dyn_array_get_float(a,i)));
    return out;
}

static DynArray* nl_array_add_scalar_string(DynArray* a, const char* s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_STRING);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_STRING);
    for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(dyn_array_get_string(a,i), s));
    return out;
}

static DynArray* nl_array_radd_scalar_string(const char* s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_STRING);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_STRING);
    for (int64_t i=0;i<len;i++) dyn_array_push_string(out, nl_str_concat(s, dyn_array_get_string(a,i)));
    return out;
}

static int64_t string_to_int(const char* s) {
    return strtoll(s, NULL, 10);
}

#include "runtime/binary64_parse.h"
static double string_to_float(const char* s) { return nl_binary64_prefix(s); }

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

#include "runtime/string_edges.h"
static const char* nl_str_trim_left(const char* s) {
    if (!s) return "";
    size_t len = strnlen(s, 64*1024*1024);
    size_t start = 0;
    while (start < len && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' || s[start] == '\r')) start++;
    size_t new_len = len - start;
    char* result = gc_alloc_string(new_len);
    if (!result) return "";
    memcpy(result, s + start, new_len);
    result[new_len] = '\0';
    return result;
}

static const char* nl_str_trim_right(const char* s) {
    if (!s) return "";
    size_t len = strnlen(s, 64*1024*1024);
    size_t end = len;
    while (end > 0 && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n' || s[end-1] == '\r')) end--;
    char* result = gc_alloc_string(end);
    if (!result) return "";
    memcpy(result, s, end);
    result[end] = '\0';
    return result;
}

static const char* nl_str_to_lower(const char* s) {
    if (!s) return "";
    size_t len = strnlen(s, 64*1024*1024);
    char* result = gc_alloc_string(len);
    if (!result) return "";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        result[i] = (c >= 'A' && c <= 'Z') ? (char)(c + 32) : (char)c;
    }
    result[len] = '\0';
    return result;
}

static const char* nl_str_to_upper(const char* s) {
    if (!s) return "";
    size_t len = strnlen(s, 64*1024*1024);
    char* result = gc_alloc_string(len);
    if (!result) return "";
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)s[i];
        result[i] = (c >= 'a' && c <= 'z') ? (char)(c - 32) : (char)c;
    }
    result[len] = '\0';
    return result;
}

static const char* nl_str_replace(const char* s, const char* old_str, const char* new_str) {
    if (!s || !old_str || !new_str) return s ? s : "";
    size_t old_len = strlen(old_str);
    size_t s_len = strlen(s);
    if (old_len == 0) {
        char* copy = gc_alloc_string(s_len);
        if (!copy) return s;
        memcpy(copy, s, s_len + 1);
        return copy;
    }
    size_t new_len = strlen(new_str);
    int64_t count = 0;
    const char* p = s;
    const char* found;
    while ((found = strstr(p, old_str)) != NULL) { count++; p = found + old_len; }
    if (count == 0) {
        char* copy = gc_alloc_string(s_len);
        if (!copy) return s;
        memcpy(copy, s, s_len + 1);
        return copy;
    }
    int64_t result_len = (int64_t)s_len + count * ((int64_t)new_len - (int64_t)old_len);
    if (result_len < 0) return "";
    char* result = gc_alloc_string((size_t)result_len);
    if (!result) return "";
    const char* src = s;
    char* dst = result;
    while ((found = strstr(src, old_str)) != NULL) {
        size_t seg_len = (size_t)(found - src);
        memcpy(dst, src, seg_len);
        dst += seg_len;
        memcpy(dst, new_str, new_len);
        dst += new_len;
        src = found + old_len;
    }
    size_t rest = strlen(src);
    memcpy(dst, src, rest);
    dst[rest] = '\0';
    return result;
}

static DynArray* nl_str_split(const char* str, const char* delim) {
    DynArray* result = dyn_array_new(ELEM_STRING);
    if (!result) return NULL;
    if (!str) return result;
    size_t delim_len = strlen(delim);
    if (delim_len == 0) {
        size_t str_len = strnlen(str, 64*1024*1024);
        for (size_t i = 0; i < str_len; i++) {
            char* ch = gc_alloc_string(1);
            if (!ch) break;
            ch[0] = str[i]; ch[1] = '\0';
            dyn_array_push_string(result, ch);
        }
        return result;
    }
    const char* start = str;
    const char* found;
    while ((found = strstr(start, delim)) != NULL) {
        size_t seg_len = (size_t)(found - start);
        char* seg = gc_alloc_string(seg_len);
        if (!seg) break;
        memcpy(seg, start, seg_len);
        seg[seg_len] = '\0';
        dyn_array_push_string(result, seg);
        start = found + delim_len;
    }
    size_t rest_len = strlen(start);
    char* seg = gc_alloc_string(rest_len);
    if (seg) {
        memcpy(seg, start, rest_len);
        seg[rest_len] = '\0';
        dyn_array_push_string(result, seg);
    }
    return result;
}

static const char* nl_str_join(DynArray* arr, const char* delim) {
    if (!arr) return "";
    int64_t count = dyn_array_length(arr);
    if (count == 0) return "";
    size_t delim_len = strlen(delim);
    size_t total = 0;
    for (int64_t i = 0; i < count; i++) {
        const char* s = dyn_array_get_string(arr, i);
        if (s) total += strlen(s);
        if (i < count - 1) total += delim_len;
    }
    char* result = gc_alloc_string(total);
    if (!result) return "";
    size_t pos = 0;
    for (int64_t i = 0; i < count; i++) {
        const char* s = dyn_array_get_string(arr, i);
        if (s) { size_t slen = strlen(s); memcpy(result + pos, s, slen); pos += slen; }
        if (i < count - 1) { memcpy(result + pos, delim, delim_len); pos += delim_len; }
    }
    result[pos] = '\0';
    return result;
}

static const char* nl_format(const char *fmt, int n_args, ...) {
    if (!fmt) return "";
    va_list ap;
    va_start(ap, n_args);
    nl_fmt_sb_t out = nl_fmt_sb_new(128);
    const char *p = fmt;
    int used = 0;
    while (*p) {
        if (*p == '%' && (p[1] == 's' || p[1] == 'd' || p[1] == 'f' || p[1] == 'g') && used < n_args) {
            const char *arg = va_arg(ap, const char *);
            if (arg) nl_fmt_sb_append_cstr(&out, arg);
            used++;
            p += 2;
        } else {
            nl_fmt_sb_append_char(&out, *p++);
        }
    }
    va_end(ap);
    char *result = gc_alloc_string(out.len);
    if (result && out.buf) { memcpy(result, out.buf, out.len + 1); }
    if (out.buf) free(out.buf);
    return result ? result : "";
}

/* ========== End Advanced String Operations ========== */

/* ========== Timing Utilities ========== */

#include <sys/time.h>
#ifdef __MACH__
#include <mach/mach_time.h>
#endif

/* Get current time in microseconds since epoch */
static int64_t nl_timing_get_microseconds(void) {
#ifdef CLOCK_REALTIME
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((int64_t)ts.tv_sec * 1000000LL) + (int64_t)(ts.tv_nsec / 1000);
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((int64_t)tv.tv_sec * 1000000LL) + (int64_t)tv.tv_usec;
#endif
}

/* Get high-resolution time in nanoseconds */
static int64_t nl_timing_get_nanoseconds(void) {
#ifdef __MACH__
    static mach_timebase_info_data_t timebase;
    static int initialized = 0;
    if (!initialized) {
        mach_timebase_info(&timebase);
        initialized = 1;
    }
    uint64_t mach_time = mach_absolute_time();
    return (int64_t)((mach_time * timebase.numer) / timebase.denom);
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((int64_t)ts.tv_sec * 1000000000LL) + (int64_t)ts.tv_nsec;
#else
    return nl_timing_get_microseconds() * 1000LL;
#endif
}

/* Convenience: current time in milliseconds */
static int64_t nl_get_time_ms(void) { return nl_timing_get_microseconds() / 1000LL; }

/* ========== End Timing Utilities ========== */

/* ========== Console I/O Utilities ========== */

/* Read a line from stdin, returns heap-allocated string */
/* Static to avoid duplicate symbols when linking multiple modules */
static const char* nl_read_line(void) {
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
        char* empty = malloc(1);
        if (empty) empty[0] = '\0';
        return empty ? empty : "";
    }
    /* Remove trailing newline if present */
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len-1] == '\n') {
        buffer[len-1] = '\0';
        len--;
    }
    char* result = malloc(len + 1);
    if (!result) return "";
    memcpy(result, buffer, len + 1);
    return result;
}

/* ========== End Console I/O Utilities ========== */

#include <string.h>
static inline double nl_float_from_bits(int64_t value) { uint64_t bits = (uint64_t)value; double result; (void)sizeof(char[sizeof(result) == sizeof(bits) ? 1 : -1]); memcpy(&result, &bits, sizeof(result)); return result; }
static inline int64_t nl_float_to_bits(double value) { uint64_t bits; (void)sizeof(char[sizeof(value) == sizeof(bits) ? 1 : -1]); memcpy(&bits, &value, sizeof(bits)); return bits <= (9223372036854775807L) ? (int64_t)bits : -1 - (int64_t)((18446744073709551615UL) - bits); }
/* I canonicalize only scalar binary arithmetic results, never transported bits. */
#ifndef NANOLANG_BINARY64_ARITHMETIC_H
#define NANOLANG_BINARY64_ARITHMETIC_H
#include <float.h>
#include <stdint.h>
#include <string.h>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I require ordinary IEEE arithmetic without fast-math."
#endif
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 double arithmetic."
#endif
#if !defined(FLT_EVAL_METHOD) || FLT_EVAL_METHOD != 0
#error "I require operations evaluated in their binary64 type."
#endif
/* I retain a compile-time storage check in both C99 and C11 output. */
typedef char nano_rt_binary64_storage_guard[
    sizeof(double) == 8 && sizeof(uint64_t) == 8 ? 1 : -1];

/* I inspect a rounded result with integer operations, not another FP operation. */
static inline double nano_rt_f64_arithmetic_result(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000) &&
        (bits & UINT64_C(0x000fffffffffffff)) != 0) {
        bits = UINT64_C(0x7ff8000000000000);
        memcpy(&value, &bits, sizeof(value));
    }
    return value;
}

/* Each volatile store/load is a binary64 rounding and noncontraction boundary. */
static inline double nano_rt_f64_add(double a, double b) {
    volatile double rounded = a + b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_sub(double a, double b) {
    volatile double rounded = a - b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_mul(double a, double b) {
    volatile double rounded = a * b;
    return nano_rt_f64_arithmetic_result(rounded);
}
static inline double nano_rt_f64_div(double a, double b) {
    uint64_t divisor;
    memcpy(&divisor, &b, sizeof(divisor));
    /* Either signed zero takes precedence even over a signaling NaN numerator. */
    if ((divisor & UINT64_C(0x7fffffffffffffff)) == 0) return 0.0;
    volatile double rounded = a / b;
    return nano_rt_f64_arithmetic_result(rounded);
}
#endif
#ifndef NANOLANG_BINARY64_FORMAT_H
#define NANOLANG_BINARY64_FORMAT_H
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#if FLT_RADIX != 2 || DBL_MANT_DIG != 53 || DBL_MAX_EXP != 1024 || DBL_MIN_EXP != -1021
#error "I require binary64 formatting storage."
#endif
typedef char nano_rt_f64_format_storage[(sizeof(double) == 8 && sizeof(uint64_t) == 8) ? 1 : -1]; static inline const char *nano_rt_f64_nonfinite(double value) { uint64_t bits; memcpy(&bits, &value, sizeof bits); if ((bits & 0x7ff0000000000000UL) != 0x7ff0000000000000UL) return ((void *)0); if (bits & 0x000fffffffffffffUL) return bits >> 63 ? "-nan" : "nan"; return bits >> 63 ? "-inf" : "inf"; } static inline int nano_rt_f64_format(char *out, size_t size, double value) { const char *special = nano_rt_f64_nonfinite(value); return special ? snprintf(out, size, "%s", special) : snprintf(out, size, "%g", value); } static inline int nano_rt_f64_print(FILE *out, double value) { const char *special = nano_rt_f64_nonfinite(value); return special ? fprintf(out, "%s", special) : fprintf(out, "%g", value); }
#endif
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

/* Trigonometric functions */
static double nl_sin(double x) { return sin(x); }
static double nl_cos(double x) { return cos(x); }
static double nl_tan(double x) { return tan(x); }
static double nl_atan2(double y, double x) { return atan2(y, x); }

/* Power and root functions */
static double nl_sqrt(double x) { return sqrt(x); }
static double nl_pow(double base, double exp) { return pow(base, exp); }

/* Rounding functions */
static double nl_floor(double x) { return floor(x); }
static double nl_ceil(double x) { return ceil(x); }
static double nl_round(double x) { return round(x); }

static int64_t nl_cast_int(double x) { if (!(x >= -0x1p63 && x < 0x1p63)) { fputs("I cannot convert this float to int: I require a finite value in [-2^63, 2^63).\n", stderr); exit(EXIT_FAILURE); } return (int64_t)x; }
static int64_t nl_cast_int_from_int(int64_t x) { return x; }
static double nl_cast_float(int64_t x) { return (double)x; }
static double nl_cast_float_from_float(double x) { return x; }
static void* nl_null_opaque() { return NULL; }
static int64_t nl_cast_bool_to_int(bool x) { return x ? 1 : 0; }
static bool nl_cast_bool(int64_t x) { return x != 0; }
static int64_t nano_rt_idiv(int64_t a, int64_t b) { if (b == 0) return 0; if (a == INT64_MIN && b == -1) return INT64_MIN; return a / b; }
static int64_t nano_rt_imod(int64_t a, int64_t b) { if (b == 0) return 0; if (a == INT64_MIN && b == -1) return 0; return a % b; }

static void nl_println(void* value_ptr) {
    (void)value_ptr; /* Unused - actual implementation uses type info from checker */
}

static void nl_print_int(int64_t value) {
    printf("%lld", (long long)value);
}

static void nl_print_float(double value) {
    nano_rt_f64_print(stdout, value);
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
    nano_rt_f64_print(stdout, value); fputc('\n', stdout);
}

static void nl_println_string(const char* value) {
    printf("%s\n", value);
}

/* Dynamic array runtime (using GC) - LEGACY */
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include "runtime/nl_string.h"

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

static DynArray* dynarray_literal_u8(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_U8);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        int val = va_arg(args, int); /* default promotion */
        dyn_array_push_u8(arr, (uint8_t)val);
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

static DynArray* dynarray_literal_string(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_STRING);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        const char* val = va_arg(args, const char*);
        dyn_array_push_string(arr, val);
    }
    va_end(args);
    return arr;
}

static DynArray* dynarray_literal_bool(int count, ...) {
    DynArray* arr = dyn_array_new(ELEM_BOOL);
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        int val = va_arg(args, int); /* bool promotes to int */
        dyn_array_push_bool(arr, val);
    }
    va_end(args);
    return arr;
}

static DynArray* dynarray_push(DynArray* arr, double val) {
    if (arr->elem_type == ELEM_U8) {
        return dyn_array_push_u8(arr, (uint8_t)val);
    } else if (arr->elem_type == ELEM_INT) {
        return dyn_array_push_int(arr, (int64_t)val);
    } else {
        return dyn_array_push_float(arr, val);
    }
}

static DynArray* nl_array_push(DynArray* arr, double val) {
    if (arr->elem_type == ELEM_U8) {
        return dyn_array_push_u8(arr, (uint8_t)val);
    } else if (arr->elem_type == ELEM_INT) {
        return dyn_array_push_int(arr, (int64_t)val);
    } else {
        return dyn_array_push_float(arr, val);
    }
}

static double nl_array_pop(DynArray* arr) {
    bool success = false;
    if (arr->elem_type == ELEM_U8) {
        return (double)dyn_array_pop_u8(arr, &success);
    } else if (arr->elem_type == ELEM_INT) {
        return (double)dyn_array_pop_int(arr, &success);
    } else {
        return dyn_array_pop_float(arr, &success);
    }
}

static int64_t nl_array_length(DynArray* arr) {
    return dyn_array_length(arr);
}

static DynArray* nl_array_remove_at(DynArray* arr, int64_t index) {
    return dyn_array_remove_at(arr, index);
}

static int64_t nl_array_at_int(DynArray* arr, int64_t idx) {
    return dyn_array_get_int(arr, idx);
}

static uint8_t nl_array_at_u8(DynArray* arr, int64_t idx) {
    return dyn_array_get_u8(arr, idx);
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

static void nl_array_set_u8(DynArray* arr, int64_t idx, uint8_t val) {
    dyn_array_set_u8(arr, idx, val);
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

static DynArray* nl_array_at_array(DynArray* arr, int64_t idx) {
    return dyn_array_get_array(arr, idx);
}

static void nl_array_set_array(DynArray* arr, int64_t idx, DynArray* val) {
    dyn_array_set_array(arr, idx, val);
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
    if (arr->elem_type == ELEM_U8) {
        return (double)dyn_array_get_u8(arr, idx);
    } else if (arr->elem_type == ELEM_INT) {
        return (double)dyn_array_get_int(arr, idx);
    } else {
        return dyn_array_get_float(arr, idx);
    }
}

/* bstring helpers (nl_string_t wrappers) */
static nl_string_t* bstr_new(const char* cstr) {
    if (!cstr) cstr = "";
    return nl_string_new(cstr);
}

static nl_string_t* bstr_new_binary(DynArray* bytes) {
    if (!bytes || dyn_array_get_elem_type(bytes) != ELEM_U8) {
        return nl_string_new_binary("", 0);
    }
    int64_t len = dyn_array_length(bytes);
    if (len <= 0) {
        return nl_string_new_binary("", 0);
    }
    uint8_t* buffer = malloc((size_t)len);
    if (!buffer) {
        return nl_string_new_binary("", 0);
    }
    for (int64_t i = 0; i < len; i++) {
        buffer[i] = dyn_array_get_u8(bytes, i);
    }
    nl_string_t* result = nl_string_new_binary(buffer, (size_t)len);
    free(buffer);
    return result;
}

static size_t bstr_length(nl_string_t* str) {
    if (!str) return 0;
    return nl_string_length(str);
}

static int64_t bstr_byte_at(nl_string_t* str, int64_t index) {
    if (!str || index < 0 || (size_t)index >= nl_string_length(str)) {
        return 0;
    }
    return (unsigned char)nl_string_byte_at(str, (size_t)index);
}

static nl_string_t* bstr_concat(nl_string_t* a, nl_string_t* b) {
    if (!a && !b) return nl_string_new("");
    if (!a) return nl_string_clone(b);
    if (!b) return nl_string_clone(a);
    return nl_string_concat(a, b);
}

static nl_string_t* bstr_substring(nl_string_t* str, int64_t start, int64_t length) {
    if (!str || start < 0 || length < 0) {
        return nl_string_new("");
    }
    size_t len = nl_string_length(str);
    if ((size_t)start > len) {
        start = (int64_t)len;
    }
    if ((size_t)(start + length) > len) {
        length = (int64_t)len - start;
    }
    return nl_string_substring(str, (size_t)start, (size_t)length);
}

static bool bstr_equals(nl_string_t* a, nl_string_t* b) {
    if (!a || !b) return a == b;
    return nl_string_equals(a, b);
}

static bool bstr_validate_utf8(nl_string_t* str) {
    if (!str) return false;
    return nl_string_validate_utf8(str);
}

static int64_t bstr_utf8_length(nl_string_t* str) {
    if (!str) return 0;
    return nl_string_utf8_length(str);
}

static int64_t bstr_utf8_char_at(nl_string_t* str, int64_t char_index) {
    if (!str || char_index < 0) return -1;
    return nl_string_utf8_char_at(str, (size_t)char_index);
}

static const char* bstr_to_cstr(nl_string_t* str) {
    if (!str) return "";
    return nl_string_to_cstr(str);
}

static void bstr_free(nl_string_t* str) {
    if (str) {
        nl_string_free(str);
    }
}

/* String concatenation - use strnlen for safety */
static const char* nl_str_concat(const char* s1, const char* s2) {
    /* Safety: Bound string scan to 64MB */
    size_t len1 = strnlen(s1, 64*1024*1024);
    size_t len2 = strnlen(s2, 64*1024*1024);
    char* result = gc_alloc_string(len1 + len2);
    if (!result) return "";
    memcpy(result, s1, len1);
    memcpy(result + len1, s2, len2);
    result[len1 + len2] = '\0';
    return result;
}

/* String substring - use strnlen for safety */
static const char* nl_str_substring(const char* str, int64_t start, int64_t length) {
    /* Safety: Bound string scan to 64MB */
    int64_t str_len = strnlen(str, 64*1024*1024);
    if (start < 0 || start > str_len || length < 0) return "";
    if (start == str_len) return "";
    if (start + length > str_len) length = str_len - start;
    char* result = gc_alloc_string(length);
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

/* String starts_with */
static bool nl_str_starts_with(const char* s, const char* prefix) {
    if (!s || !prefix) return false;
    size_t slen = strnlen(s, 64*1024*1024);
    size_t plen = strnlen(prefix, 64*1024*1024);
    if (plen > slen) return false;
    return strncmp(s, prefix, plen) == 0;
}

/* String ends_with */
#include "runtime/string_edges.h"
#include "runtime/string_search.h"
static DynArray* nl_bytes_from_string(const char* s) {
    DynArray* out = dyn_array_new(ELEM_U8);
    if (!out) return NULL;
    if (!s) return out;
    size_t len = strnlen(s, 64*1024*1024);
    for (size_t i = 0; i < len; i++) {
        dyn_array_push_u8(out, (uint8_t)(unsigned char)s[i]);
    }
    return out;
}

static const char* nl_string_from_bytes(DynArray* bytes) {
    if (!bytes) return "";
    if (dyn_array_get_elem_type(bytes) != ELEM_U8) return "";
    int64_t len = dyn_array_length(bytes);
    if (len < 0) return "";
    char* out = gc_alloc_string((size_t)len);
    if (!out) return "";
    for (int64_t i = 0; i < len; i++) {
        out[i] = (char)dyn_array_get_u8(bytes, i);
    }
    out[len] = '\0';
    return out;
}

static DynArray* nl_array_slice(DynArray* arr, int64_t start, int64_t length) {
    if (!arr) return dyn_array_new(ELEM_INT);
    if (start < 0) start = 0;
    if (length < 0) length = 0;
    int64_t len = dyn_array_length(arr);
    if (start > len) start = len;
    if (length > len - start) length = len - start;
    int64_t end = start + length;
    if (end > len) end = len;
    ElementType t = dyn_array_get_elem_type(arr);
    DynArray* out = dyn_array_new(t);
    if (!out) return NULL;
    for (int64_t i = start; i < end; i++) {
        switch (t) {
            case ELEM_U8: dyn_array_push_u8(out, dyn_array_get_u8(arr, i)); break;
            case ELEM_INT: dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;
            case ELEM_FLOAT: dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;
            case ELEM_BOOL: dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;
            case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;
            case ELEM_ARRAY: dyn_array_push_array(out, dyn_array_get_array(arr, i)); break;
            case ELEM_STRUCT: dyn_array_push_struct(out, dyn_array_get_struct(arr, i), (size_t)arr->elem_size); break;
            default: assert(false && "nl_array_slice: unsupported element type");
        }
    }
    return out;
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
            case ELEM_U8:
                printf("%u", (unsigned)((uint8_t*)arr->data)[i]);
                break;
            case ELEM_FLOAT:
                nano_rt_f64_print(stdout, ((double*)arr->data)[i]);
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

static DynArray* nl_array_sort(DynArray* arr) {
    return dyn_array_sorted(arr);
}

static DynArray* nl_array_reverse(DynArray* arr) {
    if (!arr) return dyn_array_new(ELEM_INT);
    int64_t len = dyn_array_length(arr);
    ElementType t = dyn_array_get_elem_type(arr);
    DynArray* out = dyn_array_new(t);
    if (!out) return NULL;
    for (int64_t i = len - 1; i >= 0; i--) {
        switch (t) {
            case ELEM_INT:    dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;
            case ELEM_FLOAT:  dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;
            case ELEM_BOOL:   dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;
            case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;
            default: dyn_array_push_int(out, 0); break;
        }
    }
    return out;
}

static bool nl_array_contains(DynArray* arr, int64_t elem) {
    if (!arr) return false;
    int64_t len = dyn_array_length(arr);
    for (int64_t i = 0; i < len; i++) {
        if (dyn_array_get_int(arr, i) == elem) return true;
    }
    return false;
}

static int64_t nl_array_index_of(DynArray* arr, int64_t elem) {
    if (!arr) return -1;
    int64_t len = dyn_array_length(arr);
    for (int64_t i = 0; i < len; i++) {
        if (dyn_array_get_int(arr, i) == elem) return i;
    }
    return -1;
}

/* ========== End Array Operations ========== */

/* ========== End Math and Utility Built-in Functions ========== */

/* ========== Coroutine Runtime Builtins ========== */
#include "coroutine.h"

typedef struct { void *fn; int64_t args[8]; int nargs; } NlSpawnCtx;
static Value nl_coro_spawn_trampoline(void *raw, int coro_id) {
    (void)coro_id;
    NlSpawnCtx *ctx = (NlSpawnCtx *)raw;
    Value v; memset(&v, 0, sizeof(v)); v.type = VAL_INT;
    switch (ctx->nargs) {
        case 0: v.as.int_val = ((int64_t(*)(void))ctx->fn)(); break;
        case 1: v.as.int_val = ((int64_t(*)(int64_t))ctx->fn)(ctx->args[0]); break;
        case 2: v.as.int_val = ((int64_t(*)(int64_t,int64_t))ctx->fn)(ctx->args[0],ctx->args[1]); break;
        case 3: v.as.int_val = ((int64_t(*)(int64_t,int64_t,int64_t))ctx->fn)(ctx->args[0],ctx->args[1],ctx->args[2]); break;
        default: break;
    }
    return v;
}

static int64_t nl_coro_spawn_n(void *fn, int nargs, int64_t a0, int64_t a1, int64_t a2) {
    nano_scheduler_init();
    NlSpawnCtx *ctx = (NlSpawnCtx *)malloc(sizeof(NlSpawnCtx));
    if (!ctx) return -1;
    ctx->fn = fn; ctx->nargs = nargs;
    ctx->args[0] = a0; ctx->args[1] = a1; ctx->args[2] = a2;
    return (int64_t)nano_coro_spawn(nl_coro_spawn_trampoline, ctx);
}
/* nl_coro_spawn: select correct overload by argument count */
#define _nl_cs0(fn)           nl_coro_spawn_n((void*)(fn),0,0LL,0LL,0LL)
#define _nl_cs1(fn,a)         nl_coro_spawn_n((void*)(fn),1,(int64_t)(a),0LL,0LL)
#define _nl_cs2(fn,a,b)       nl_coro_spawn_n((void*)(fn),2,(int64_t)(a),(int64_t)(b),0LL)
#define _nl_cs3(fn,a,b,c)     nl_coro_spawn_n((void*)(fn),3,(int64_t)(a),(int64_t)(b),(int64_t)(c))
#define _nl_cs_pick(_1,_2,_3,_4,X,...) X
#define nl_coro_spawn(fn,...) _nl_cs_pick(fn,##__VA_ARGS__,_nl_cs3,_nl_cs2,_nl_cs1,_nl_cs0)(fn,##__VA_ARGS__)

/* nl_scheduler_run(): drain all pending coroutines */
static void nl_scheduler_run(void) {
    nano_scheduler_init();
    nano_scheduler_run_until_done();
}

/* nl_scheduler_step(): run one step */
static int64_t nl_scheduler_step(void) {
    nano_scheduler_init();
    return nano_scheduler_step() ? 1 : 0;
}

/* nl_coro_result(id): get result value of completed coroutine */
static int64_t nl_coro_result(int64_t id) {
    Value v = nano_coro_result((int)id);
    return v.as.int_val;
}

/* nl_coro_done(id): 1 if coroutine completed */
static int64_t nl_coro_done(int64_t id) {
    return nano_coro_is_done((int)id) ? 1 : 0;
}

/* nl_coro_yield(): cooperative yield hint */
static void nl_coro_yield(void) { nano_coro_yield(); }
/* ========== End Coroutine Runtime Builtins ========== */

/* ========== Enum Definitions ========== */

/* ========== End Enum Definitions ========== */

/* ========== Struct and Union Definitions ========== */

/* ========== End Struct and Union Definitions ========== */

/* ========== Auto-Generated Struct Metadata ========== */

/* ========== End Struct Metadata ========== */

/* ========== Auto-Generated Module Metadata ========== */

/* ========== End Module Metadata ========== */

/* ========== HashMap Runtime (Generated) ========== */

static uint64_t nl_hashmap_hash_string(const char *s) {
    if (!s) return 0;
    uint64_t hash = 1469598103934665603ULL;
    while (*s) { hash ^= (uint8_t)(*s++); hash *= 1099511628211ULL; }
    return hash;
}

static uint64_t nl_hashmap_hash_int(int64_t x) {
    uint64_t z = (uint64_t)x;
    z ^= z >> 33;
    z *= 0xff51afd7ed558ccdULL;
    z ^= z >> 33;
    z *= 0xc4ceb9fe1a85ec53ULL;
    z ^= z >> 33;
    return z;
}

static bool nl_hashmap_key_eq_string(const char *a, const char *b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

/* (no HashMap instantiations) */

/* ========== End HashMap Runtime (Generated) ========== */

/* ========== To-String Helpers ========== */

/* To-String forward declarations */

/* ========== End To-String Helpers ========== */

/* External C function declarations */

/* Forward declarations for imported module functions */

/* Forward declarations for program functions */
static bool nl_agrees(int64_t i, int64_t j);
static int64_t nl_main();

/* Top-level globals */

static bool nl_agrees(int64_t i, int64_t j) {
#line 2 "/tmp/nanolang-comparison-original/comparison.nano"
    return i == j == j == i;
}

static int64_t nl_main() {
#line 6 "/tmp/nanolang-comparison-original/comparison.nano"
    int64_t i = 0LL;
#line 7 "/tmp/nanolang-comparison-original/comparison.nano"
    while (i < 4LL)     {
#line 8 "/tmp/nanolang-comparison-original/comparison.nano"
        int64_t j = 0LL;
#line 9 "/tmp/nanolang-comparison-original/comparison.nano"
        while (j < 4LL)         {
#line 10 "/tmp/nanolang-comparison-original/comparison.nano"
            if (!(({ __auto_type __nl_arg_0_0 = i; __auto_type __nl_arg_0_1 = j; nl_agrees(__nl_arg_0_0, __nl_arg_0_1); }))) { fputs("Contract violation at line 10: (agrees i j)\n", stderr); exit(1); }
#line 11 "/tmp/nanolang-comparison-original/comparison.nano"
            if (!(i < j == j > i)) { fputs("Contract violation at line 11: (== (< i j) (> j i))\n", stderr); exit(1); }
#line 12 "/tmp/nanolang-comparison-original/comparison.nano"
            if (!((i + 1LL) > j == j < (i + 1LL))) { fputs("Contract violation at line 12: (== (> (+ i 1) j) (< j (+ i 1)))\n", stderr); exit(1); }
#line 13 "/tmp/nanolang-comparison-original/comparison.nano"
            j = (j + 1LL);
        }
#line 15 "/tmp/nanolang-comparison-original/comparison.nano"
        i = (i + 1LL);
    }
#line 17 "/tmp/nanolang-comparison-original/comparison.nano"
    return 0LL;
}


/* C main() entry point - calls nanolang main */
/* Global argc/argv for CLI runtime support */
int g_argc = 0;
char **g_argv = NULL;

int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    setvbuf(stdout, NULL, _IOLBF, 0);
    return (int)nl_main();
}

#pragma GCC diagnostic pop
