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
#include "runtime/gc.h"
#include "runtime/dyn_array.h"

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

/* ========== OS Standard Library ========== */

static char* nl_os_file_read(const char* path) {
    FILE* f = fopen(path, "rb");  /* Binary mode for MOD files */
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(size + 1);
    if (!buffer) { fclose(f); return ""; }
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);
    return buffer;
}

static DynArray* nl_os_file_read_bytes(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        /* Return empty array on error */
        return dyn_array_new(ELEM_U8);
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    /* Create dynamic array for bytes */
    DynArray* bytes = dyn_array_new(ELEM_U8);
    
    /* Read bytes and add to array */
    for (long i = 0; i < size; i++) {
        int c = fgetc(f);
        if (c == EOF) break;
        dyn_array_push_u8(bytes, (uint8_t)(unsigned char)c);
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
    char* out = malloc(len + 1);
    if (!out) return "";
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
    if (fd < 0) return "";
    close(fd);
    size_t len = strlen(templ);
    char* out = malloc(len + 1);
    if (!out) return "";
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
            char* out = malloc(len + 1);
            if (!out) return "";
            memcpy(out, path, len);
            out[len] = '\0';
            return out;
        }
    }
    return "";
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
    size_t capacity = 4096;
    size_t used = 0;
    char* buffer = malloc(capacity);
    if (!buffer) { closedir(dir); return ""; }
    buffer[0] = '\0';
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        size_t name_len = strlen(entry->d_name);
        size_t needed = used + name_len + 2; /* +1 for newline, +1 for null */
        if (needed > capacity) {
            capacity = needed * 2;
            char* new_buffer = realloc(buffer, capacity);
            if (!new_buffer) { free(buffer); closedir(dir); return ""; }
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
    char* buffer = malloc(1024);
    if (getcwd(buffer, 1024) == NULL) {
        buffer[0] = '\0';
    }
    return buffer;
}

static int64_t nl_os_chdir(const char* path) {
    return chdir(path) == 0 ? 0 : -1;
}

static void nl_os_walkdir_rec(const char* root, DynArray* out) {
    DIR* dir = opendir(root);
    if (!dir) return;
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) continue;
        size_t root_len = strlen(root);
        size_t name_len = strlen(entry->d_name);
        bool needs_slash = (root_len > 0 && root[root_len - 1] != '/');
        size_t cap = root_len + (needs_slash ? 1 : 0) + name_len + 1;
        char* path = malloc(cap);
        if (!path) continue;
        if (needs_slash) snprintf(path, cap, "%s/%s", root, entry->d_name);
        else snprintf(path, cap, "%s%s", root, entry->d_name);

        struct stat st;
        if (stat(path, &st) != 0) { free(path); continue; }
        if (S_ISDIR(st.st_mode)) {
            nl_os_walkdir_rec(path, out);
            free(path);
        } else if (S_ISREG(st.st_mode)) {
            dyn_array_push_string(out, path);
        } else {
            free(path);
        }
    }
    closedir(dir);
}

static DynArray* nl_os_walkdir(const char* root) {
    DynArray* out = dyn_array_new(ELEM_STRING);
    if (!root || root[0] == '\0') return out;
    nl_os_walkdir_rec(root, out);
    return out;
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
    if (!buffer) return "";
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

static char* nl_os_path_normalize(const char* path) {
    if (!path) return "";
    bool abs = (path[0] == '/');
    char* copy = strdup(path);
    if (!copy) return "";

    const char* parts[512];
    int count = 0;
    char* save = NULL;
    char* tok = strtok_r(copy, "/", &save);
    while (tok) {
        if (strcmp(tok, "") == 0 || strcmp(tok, ".") == 0) {
            /* skip */
        } else if (strcmp(tok, "..") == 0) {
            if (count > 0 && strcmp(parts[count - 1], "..") != 0) {
                count--;
            } else if (!abs) {
                parts[count++] = tok;
            }
        } else {
            if (count < 512) parts[count++] = tok;
        }
        tok = strtok_r(NULL, "/", &save);
    }

    size_t cap = strlen(path) + 3;
    char* out = malloc(cap);
    if (!out) { free(copy); return ""; }
    size_t pos = 0;
    if (abs) out[pos++] = '/';

    for (int i = 0; i < count; i++) {
        size_t len = strlen(parts[i]);
        if (pos + len + 2 > cap) {
            cap = (pos + len + 2) * 2;
            char* n = realloc(out, cap);
            if (!n) { free(out); free(copy); return ""; }
            out = n;
        }
        if (pos > 0 && out[pos - 1] != '/') out[pos++] = '/';
        memcpy(out + pos, parts[i], len);
        pos += len;
    }

    if (pos == 0) {
        if (abs) { out[pos++] = '/'; } else { out[pos++] = '.'; }
    }
    out[pos] = '\0';

    free(copy);
    return out;
}

static int64_t nl_os_system(const char* command) {
    return system(command);
}

static void nl_os_exit(int64_t code) {
    exit((int)code);
}

static char* nl_os_getenv(const char* name) {
    const char* value = getenv(name);
    return value ? (char*)value : (char*)"";
}

/* system() wrapper - stdlib system() available via stdlib.h */
static inline int64_t nl_exec_shell(const char* cmd) {
    return (int64_t)system(cmd);
}

static char* nl_os_read_all_fd(int fd) {
    size_t cap = 4096;
    size_t len = 0;
    char* buf = malloc(cap);
    if (!buf) return strdup("");
    while (1) {
        if (len + 1 >= cap) {
            cap *= 2;
            char* n = realloc(buf, cap);
            if (!n) { free(buf); return strdup(""); }
            buf = n;
        }
        ssize_t r = read(fd, buf + len, cap - len - 1);
        if (r <= 0) break;
        len += (size_t)r;
    }
    buf[len] = '\0';
    return buf;
}

static DynArray* nl_os_process_run(const char* command) {
    DynArray* out = dyn_array_new(ELEM_STRING);
    if (!command) {
        dyn_array_push_string(out, strdup("-1"));
        dyn_array_push_string(out, strdup(""));
        dyn_array_push_string(out, strdup(""));
        return out;
    }

    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        dyn_array_push_string(out, strdup("-1"));
        dyn_array_push_string(out, strdup(""));
        dyn_array_push_string(out, strdup(""));
        return out;
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[0]);

    pid_t pid = 0;
    char* argv[] = { "sh", "-c", (char*)command, NULL };
    extern char **environ;
    int rc = posix_spawn(&pid, "/bin/sh", &actions, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&actions);

    close(out_pipe[1]);
    close(err_pipe[1]);

    char* out_s = nl_os_read_all_fd(out_pipe[0]);
    char* err_s = nl_os_read_all_fd(err_pipe[0]);
    close(out_pipe[0]);
    close(err_pipe[0]);

    int code = -1;
    if (rc != 0) {
        code = rc;
    } else {
        int status = 0;
        (void)waitpid(pid, &status, 0);
        if (WIFEXITED(status)) code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) code = 128 + WTERMSIG(status);
        else code = -1;
    }

    char code_buf[64];
    snprintf(code_buf, sizeof(code_buf), "%d", code);
    dyn_array_push_string(out, strdup(code_buf));
    dyn_array_push_string(out, out_s);
    dyn_array_push_string(out, err_s);
    return out;
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
    char* buffer = malloc(32);
    if (!buffer) return "";
    snprintf(buffer, 32, "%lld", (long long)n);
    return buffer;
}

static char* float_to_string(double x) {
    char* buffer = malloc(64);
    if (!buffer) return "";
    snprintf(buffer, 64, "%g", x);
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
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)+dyn_array_get_float(b,i)); break;
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
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)-dyn_array_get_float(b,i)); break;
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
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)*dyn_array_get_float(b,i)); break;
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
        case ELEM_FLOAT: for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i)/dyn_array_get_float(b,i)); break;
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
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) + s);
    return out;
}

static DynArray* nl_array_radd_scalar_float(double s, DynArray* a) { return nl_array_add_scalar_float(a, s); }

static DynArray* nl_array_sub_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) - s);
    return out;
}

static DynArray* nl_array_rsub_scalar_float(double s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, s - dyn_array_get_float(a,i));
    return out;
}

static DynArray* nl_array_mul_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) * s);
    return out;
}

static DynArray* nl_array_rmul_scalar_float(double s, DynArray* a) { return nl_array_mul_scalar_float(a, s); }

static DynArray* nl_array_div_scalar_float(DynArray* a, double s) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, dyn_array_get_float(a,i) / s);
    return out;
}

static DynArray* nl_array_rdiv_scalar_float(double s, DynArray* a) {
    assert(a); assert(dyn_array_get_elem_type(a) == ELEM_FLOAT);
    int64_t len = dyn_array_length(a); DynArray* out = dyn_array_new(ELEM_FLOAT);
    for (int64_t i=0;i<len;i++) dyn_array_push_float(out, s / dyn_array_get_float(a,i));
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

/* String concatenation - use strnlen for safety */
static const char* nl_str_concat(const char* s1, const char* s2) {
    /* Safety: Bound string scan to 1MB */
    size_t len1 = strnlen(s1, 1024*1024);
    size_t len2 = strnlen(s2, 1024*1024);
    char* result = malloc(len1 + len2 + 1);
    if (!result) return "";
    memcpy(result, s1, len1);
    memcpy(result + len1, s2, len2);
    result[len1 + len2] = '\0';
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

static DynArray* nl_bytes_from_string(const char* s) {
    DynArray* out = dyn_array_new(ELEM_U8);
    if (!out) return NULL;
    if (!s) return out;
    size_t len = strnlen(s, 1024*1024);
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
    char* out = malloc((size_t)len + 1);
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

/* ========== Struct and Union Definitions ========== */

/* ========== End Struct and Union Definitions ========== */

/* ========== Generic List Specializations ========== */

/* ========== End Generic List Specializations ========== */

/* ========== To-String Helpers ========== */

/* To-String forward declarations */

/* ========== End To-String Helpers ========== */

/* External C function declarations */
extern const char* path_normalize(const char* path);
extern const char* path_join(const char* a, const char* b);
extern const char* path_basename(const char* path);
extern const char* path_dirname(const char* path);

/* Forward declarations for imported module functions */

/* Top-level constants */

/* Forward declarations for program functions */
static int64_t nl_main();

static int64_t nl_main() {
    nl_println_string("Testing path operations");
    const char* p1 = "";
    /* unsafe */ {
        p1 = nl_os_path_normalize("/foo/bar");
    }
    nl_println_string(p1);
    const char* p2 = "";
    /* unsafe */ {
        p2 = nl_os_path_join("foo", "bar");
    }
    nl_println_string(p2);
    const char* p3 = "";
    /* unsafe */ {
        p3 = nl_os_path_basename("/foo/bar/baz.txt");
    }
    nl_println_string(p3);
    const char* p4 = "";
    /* unsafe */ {
        p4 = nl_os_path_dirname("/foo/bar/baz.txt");
    }
    nl_println_string(p4);
    return 0LL;
}


/* C main() entry point - calls nanolang main */
/* Global argc/argv for CLI runtime support */
int g_argc = 0;
char **g_argv = NULL;

int main(int argc, char **argv) {
    g_argc = argc;
    g_argv = argv;
    return (int)nl_main();
}
