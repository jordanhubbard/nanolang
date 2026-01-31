#define _POSIX_C_SOURCE 200809L  /* For mkstemp/mkdtemp */

#include "nanolang.h"
#include "runtime/list_int.h"
#include "runtime/list_string.h"
#include "runtime/list_token.h"
#include "runtime/gc.h"
#include "runtime/dyn_array.h"
#include "tracing.h"
#include "interpreter_ffi.h"
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/wait.h>
#include <spawn.h>
#include <math.h>


/* Forward declarations */
static Value eval_expression(ASTNode *expr, Environment *env);
static Value eval_statement(ASTNode *stmt, Environment *env);
static Value create_dyn_array(DynArray *arr);

static DynArray* eval_dyn_array_binop(DynArray *a, DynArray *b, TokenType op);
static DynArray* eval_dyn_array_scalar_right(DynArray *a, Value scalar, TokenType op);
static DynArray* eval_dyn_array_scalar_left(Value scalar, DynArray *a, TokenType op);

static DynArray* eval_dyn_array_binop(DynArray *a, DynArray *b, TokenType op) {
    if (!a || !b) return NULL;
    int64_t len = dyn_array_length(a);
    if (len != dyn_array_length(b)) return NULL;
    ElementType t = dyn_array_get_elem_type(a);
    if (t != dyn_array_get_elem_type(b)) return NULL;

    DynArray *out = dyn_array_new(t);
    for (int64_t i = 0; i < len; i++) {
        switch (t) {
            case ELEM_INT: {
                int64_t x = dyn_array_get_int(a, i);
                int64_t y = dyn_array_get_int(b, i);
                int64_t r = 0;
                switch (op) {
                    case TOKEN_PLUS: r = x + y; break;
                    case TOKEN_MINUS: r = x - y; break;
                    case TOKEN_STAR: r = x * y; break;
                    case TOKEN_SLASH: r = x / y; break;
                    case TOKEN_PERCENT: r = x % y; break;
                    default: break;
                }
                dyn_array_push_int(out, r);
                break;
            }
            case ELEM_FLOAT: {
                double x = dyn_array_get_float(a, i);
                double y = dyn_array_get_float(b, i);
                double r = 0.0;
                switch (op) {
                    case TOKEN_PLUS: r = x + y; break;
                    case TOKEN_MINUS: r = x - y; break;
                    case TOKEN_STAR: r = x * y; break;
                    case TOKEN_SLASH: r = x / y; break;
                    default: break;
                }
                dyn_array_push_float(out, r);
                break;
            }
            case ELEM_STRING: {
                if (op != TOKEN_PLUS) return NULL;
                const char *x = dyn_array_get_string(a, i);
                const char *y = dyn_array_get_string(b, i);
                size_t lx = strlen(x);
                size_t ly = strlen(y);
                char *buf = malloc(lx + ly + 1);
                memcpy(buf, x, lx);
                memcpy(buf + lx, y, ly);
                buf[lx + ly] = '\0';
                dyn_array_push_string(out, buf);
                break;
            }
            case ELEM_ARRAY: {
                DynArray *x = dyn_array_get_array(a, i);
                DynArray *y = dyn_array_get_array(b, i);
                DynArray *r = eval_dyn_array_binop(x, y, op);
                if (!r) return NULL;
                dyn_array_push_array(out, r);
                break;
            }
            default:
                return NULL;
        }
    }
    return out;
}

static DynArray* eval_dyn_array_scalar_right(DynArray *a, Value scalar, TokenType op) {
    if (!a) return NULL;
    int64_t len = dyn_array_length(a);
    ElementType t = dyn_array_get_elem_type(a);
    DynArray *out = dyn_array_new(t);

    for (int64_t i = 0; i < len; i++) {
        if (t == ELEM_ARRAY) {
            DynArray *inner = dyn_array_get_array(a, i);
            DynArray *r = eval_dyn_array_scalar_right(inner, scalar, op);
            if (!r) return NULL;
            dyn_array_push_array(out, r);
            continue;
        }

        if (t == ELEM_INT && scalar.type == VAL_INT) {
            int64_t x = dyn_array_get_int(a, i);
            int64_t s = scalar.as.int_val;
            int64_t r = 0;
            switch (op) {
                case TOKEN_PLUS: r = x + s; break;
                case TOKEN_MINUS: r = x - s; break;
                case TOKEN_STAR: r = x * s; break;
                case TOKEN_SLASH: r = x / s; break;
                case TOKEN_PERCENT: r = x % s; break;
                default: break;
            }
            dyn_array_push_int(out, r);
        } else if (t == ELEM_FLOAT && scalar.type == VAL_FLOAT) {
            double x = dyn_array_get_float(a, i);
            double s = scalar.as.float_val;
            double r = 0.0;
            switch (op) {
                case TOKEN_PLUS: r = x + s; break;
                case TOKEN_MINUS: r = x - s; break;
                case TOKEN_STAR: r = x * s; break;
                case TOKEN_SLASH: r = x / s; break;
                default: break;
            }
            dyn_array_push_float(out, r);
        } else if (t == ELEM_STRING && scalar.type == VAL_STRING) {
            if (op != TOKEN_PLUS) return NULL;
            const char *x = dyn_array_get_string(a, i);
            const char *s = scalar.as.string_val;
            size_t lx = strlen(x);
            size_t ls = strlen(s);
            char *buf = malloc(lx + ls + 1);
            memcpy(buf, x, lx);
            memcpy(buf + lx, s, ls);
            buf[lx + ls] = '\0';
            dyn_array_push_string(out, buf);
        } else {
            return NULL;
        }
    }
    return out;
}

static DynArray* eval_dyn_array_scalar_left(Value scalar, DynArray *a, TokenType op) {
    if (!a) return NULL;
    int64_t len = dyn_array_length(a);
    ElementType t = dyn_array_get_elem_type(a);
    DynArray *out = dyn_array_new(t);

    for (int64_t i = 0; i < len; i++) {
        if (t == ELEM_ARRAY) {
            DynArray *inner = dyn_array_get_array(a, i);
            DynArray *r = eval_dyn_array_scalar_left(scalar, inner, op);
            if (!r) return NULL;
            dyn_array_push_array(out, r);
            continue;
        }

        if (t == ELEM_INT && scalar.type == VAL_INT) {
            int64_t s = scalar.as.int_val;
            int64_t y = dyn_array_get_int(a, i);
            int64_t r = 0;
            switch (op) {
                case TOKEN_PLUS: r = s + y; break;
                case TOKEN_MINUS: r = s - y; break;
                case TOKEN_STAR: r = s * y; break;
                case TOKEN_SLASH: r = s / y; break;
                case TOKEN_PERCENT: r = s % y; break;
                default: break;
            }
            dyn_array_push_int(out, r);
        } else if (t == ELEM_FLOAT && scalar.type == VAL_FLOAT) {
            double s = scalar.as.float_val;
            double y = dyn_array_get_float(a, i);
            double r = 0.0;
            switch (op) {
                case TOKEN_PLUS: r = s + y; break;
                case TOKEN_MINUS: r = s - y; break;
                case TOKEN_STAR: r = s * y; break;
                case TOKEN_SLASH: r = s / y; break;
                default: break;
            }
            dyn_array_push_float(out, r);
        } else if (t == ELEM_STRING && scalar.type == VAL_STRING) {
            if (op != TOKEN_PLUS) return NULL;
            const char *s = scalar.as.string_val;
            const char *y = dyn_array_get_string(a, i);
            size_t ls = strlen(s);
            size_t ly = strlen(y);
            char *buf = malloc(ls + ly + 1);
            memcpy(buf, s, ls);
            memcpy(buf + ls, y, ly);
            buf[ls + ly] = '\0';
            dyn_array_push_string(out, buf);
        } else {
            return NULL;
        }
    }
    return out;
}

/* ==========================================================================
 * Built-in OS Functions Implementation
 * ========================================================================== */

/* File Operations */
static Value builtin_file_read(Value *args) {
    const char *path = args[0].as.string_val;
    FILE *f = fopen(path, "rb");  /* Binary mode for MOD files and other binary data */
    if (!f) return create_string("");

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    fclose(f);

    Value result = create_string(buffer);
    free(buffer);
    return result;
}

static Value builtin_file_read_bytes(Value *args) {
    const char *path = args[0].as.string_val;
    FILE *f = fopen(path, "rb");
    if (!f) {
        return create_dyn_array(dyn_array_new(ELEM_INT));
    }

    DynArray *bytes = dyn_array_new(ELEM_INT);
    int c;
    while ((c = fgetc(f)) != EOF) {
        dyn_array_push_int(bytes, (int64_t)(unsigned char)c);
    }
    fclose(f);
    return create_dyn_array(bytes);
}

static Value builtin_bytes_from_string(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: bytes_from_string requires a string argument\n");
        return create_void();
    }

    const char *s = args[0].as.string_val;
    size_t len = strlen(s);
    DynArray *out = dyn_array_new(ELEM_INT);
    for (size_t i = 0; i < len; i++) {
        dyn_array_push_int(out, (int64_t)(unsigned char)s[i]);
    }
    return create_dyn_array(out);
}

static Value builtin_string_from_bytes(Value *args) {
    if (args[0].type != VAL_ARRAY && args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: string_from_bytes requires an array argument\n");
        return create_void();
    }

    int64_t len = 0;
    char *buf = NULL;

    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        if (arr->element_type != VAL_INT) {
            fprintf(stderr, "Error: string_from_bytes requires array<int>\n");
            return create_void();
        }
        len = arr->length;
        buf = malloc((size_t)len + 1);
        for (int64_t i = 0; i < len; i++) {
            long long v = ((long long*)arr->data)[i];
            buf[i] = (char)(unsigned char)v;
        }
    } else {
        DynArray *arr = args[0].as.dyn_array_val;
        ElementType t = dyn_array_get_elem_type(arr);
        if (t != ELEM_INT && t != ELEM_U8) {
            fprintf(stderr, "Error: string_from_bytes requires array<u8> (represented as int/u8)\n");
            return create_void();
        }
        len = dyn_array_length(arr);
        buf = malloc((size_t)len + 1);
        for (int64_t i = 0; i < len; i++) {
            int64_t v = (t == ELEM_U8) ? (int64_t)dyn_array_get_u8(arr, i) : dyn_array_get_int(arr, i);
            buf[i] = (char)(unsigned char)v;
        }
    }

    if (!buf) return create_string("");
    buf[len] = '\0';
    Value result = create_string(buf);
    free(buf);
    return result;
}

static Value builtin_file_write(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "w");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_append(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "a");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

static Value builtin_file_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(remove(path) == 0 ? 0 : -1);
}

static Value builtin_file_rename(Value *args) {
    const char *old_path = args[0].as.string_val;
    const char *new_path = args[1].as.string_val;
    return create_int(rename(old_path, new_path) == 0 ? 0 : -1);
}

static Value builtin_file_exists(Value *args) {
    const char *path = args[0].as.string_val;
    return create_bool(access(path, F_OK) == 0);
}

static Value builtin_file_size(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_int(-1);
    return create_int(st.st_size);
}

static Value builtin_tmp_dir(Value *args) {
    (void)args;
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    return create_string(tmp);
}

static Value builtin_mktemp(Value *args) {
    const char *prefix = args[0].as.string_val;
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";

    char templ[1024];
    const char *p = (prefix && prefix[0] != '\0') ? prefix : "nanolang_";
    snprintf(templ, sizeof(templ), "%s/%sXXXXXX", tmp, p);

    int fd = mkstemp(templ);
    if (fd < 0) return create_string("");
    close(fd);
    return create_string(templ);
}

static Value builtin_mktemp_dir(Value *args) {
    const char *prefix = args[0].as.string_val;
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";

    char path[1024];
    const char *p = (prefix && prefix[0] != '\0') ? prefix : "nanolang_dir_";

    for (int i = 0; i < 100; i++) {
        /* Best-effort: unique-ish name; mkdir is atomic */
        snprintf(path, sizeof(path), "%s/%s%lld_%d", tmp, p, (long long)time(NULL), i);
        if (mkdir(path, 0700) == 0) {
            return create_string(path);
        }
    }

    return create_string("");
}

/* Directory Operations */
static Value builtin_dir_create(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(mkdir(path, 0755) == 0 ? 0 : -1);
}

static Value builtin_dir_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(rmdir(path) == 0 ? 0 : -1);
}

static Value builtin_dir_list(Value *args) {
    const char *path = args[0].as.string_val;
    DIR *dir = opendir(path);
    if (!dir) return create_string("");

    /* Build newline-separated list */
    char buffer[4096] = "";
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        /* Skip . and .. */
        assert(entry->d_name != NULL);
        if (safe_strcmp(entry->d_name, ".") == 0 || safe_strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        safe_strncat(buffer, entry->d_name, sizeof(buffer));
        safe_strncat(buffer, "\n", sizeof(buffer));
    }
    closedir(dir);

    return create_string(buffer);
}

static Value builtin_dir_exists(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_getcwd(Value *args) {
    (void)args;  /* Unused */
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        return create_string("");
    }
    return create_string(buffer);
}

static Value builtin_chdir(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(chdir(path) == 0 ? 0 : -1);
}

/* Path Operations */
static Value builtin_path_isfile(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISREG(st.st_mode));
}

static Value builtin_path_isdir(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

static Value builtin_path_join(Value *args) {
    const char *a = args[0].as.string_val;
    const char *b = args[1].as.string_val;
    char buffer[2048];

    /* Handle various cases */
    assert(a != NULL);
    assert(b != NULL);
    if (safe_strlen(a) == 0) {
        snprintf(buffer, sizeof(buffer), "%s", b);
    } else if (a[safe_strlen(a) - 1] == '/') {
        snprintf(buffer, sizeof(buffer), "%s%s", a, b);
    } else {
        snprintf(buffer, sizeof(buffer), "%s/%s", a, b);
    }

    return create_string(buffer);
}

static Value builtin_path_basename(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *base = basename(path_copy);
    Value result = create_string(base);
    free(path_copy);
    return result;
}

static Value builtin_path_dirname(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    Value result = create_string(dir);
    free(path_copy);
    return result;
}

static char* nl_path_normalize(const char* path) {
    if (!path) return strdup("");
    bool abs = (path[0] == '/');
    char* copy = strdup(path);
    if (!copy) return strdup("");

    const char* parts[512];
    int count = 0;
    char* save = NULL;
    char* tok = strtok_r(copy, "/", &save);
    while (tok) {
        if (tok[0] == '\0' || strcmp(tok, ".") == 0) {
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
    if (!out) { free(copy); return strdup(""); }
    size_t pos = 0;
    if (abs) out[pos++] = '/';
    for (int i = 0; i < count; i++) {
        size_t len = strlen(parts[i]);
        if (pos + len + 2 > cap) {
            cap = (pos + len + 2) * 2;
            char* n = realloc(out, cap);
            if (!n) { free(out); free(copy); return strdup(""); }
            out = n;
        }
        if (pos > 0 && out[pos - 1] != '/') out[pos++] = '/';
        memcpy(out + pos, parts[i], len);
        pos += len;
    }
    if (pos == 0) {
        if (abs) out[pos++] = '/';
        else out[pos++] = '.';
    }
    out[pos] = '\0';

    free(copy);
    return out;
}

static Value builtin_path_normalize(Value *args) {
    char* norm = nl_path_normalize(args[0].as.string_val);
    Value v = create_string(norm);
    free(norm);
    return v;
}

static void nl_walkdir_rec(const char* root, DynArray* out) {
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
            nl_walkdir_rec(path, out);
            free(path);
        } else if (S_ISREG(st.st_mode)) {
            dyn_array_push_string(out, path);
        } else {
            free(path);
        }
    }
    closedir(dir);
}

static Value builtin_fs_walkdir(Value *args) {
    const char* root = args[0].as.string_val;
    DynArray* out = dyn_array_new(ELEM_STRING);
    if (root && root[0] != '\0') {
        nl_walkdir_rec(root, out);
    }
    return create_dyn_array(out);
}

/* Process Operations */
static Value builtin_system(Value *args) {
    const char *command = args[0].as.string_val;
    return create_int(system(command));
}

static Value builtin_exit(Value *args) {
    int code = (int)args[0].as.int_val;
    exit(code);
    return create_void();  /* Never reached */
}

static Value builtin_getenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = getenv(name);
    return create_string(value ? value : "");
}

static Value builtin_setenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = args[1].as.string_val;
    int overwrite = (int)args[2].as.int_val;
    return create_int(setenv(name, value, overwrite) == 0 ? 0 : -1);
}

static Value builtin_unsetenv(Value *args) {
    const char *name = args[0].as.string_val;
    return create_int(unsetenv(name) == 0 ? 0 : -1);
}

static char* nl_read_all_fd(int fd) {
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

static Value builtin_process_run(Value *args) {
    const char* command = args[0].as.string_val;
    DynArray* out = dyn_array_new(ELEM_STRING);

    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        dyn_array_push_string(out, strdup("-1"));
        dyn_array_push_string(out, strdup(""));
        dyn_array_push_string(out, strdup(""));
        return create_dyn_array(out);
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

    char* out_s = nl_read_all_fd(out_pipe[0]);
    char* err_s = nl_read_all_fd(err_pipe[0]);
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
    return create_dyn_array(out);
}

static Value builtin_result_is_ok(Value *args) {
    if (args[0].type != VAL_UNION) return create_bool(false);
    UnionValue *uv = args[0].as.union_val;
    return create_bool(uv && strcmp(uv->variant_name, "Ok") == 0);
}

static Value builtin_result_is_err(Value *args) {
    if (args[0].type != VAL_UNION) return create_bool(false);
    UnionValue *uv = args[0].as.union_val;
    return create_bool(uv && strcmp(uv->variant_name, "Err") == 0);
}

static Value builtin_result_unwrap(Value *args) {
    if (args[0].type != VAL_UNION) {
        fprintf(stderr, "panic: result_unwrap called on non-union\n");
        exit(1);
    }
    UnionValue *uv = args[0].as.union_val;
    if (!uv || strcmp(uv->variant_name, "Ok") != 0 || uv->field_count < 1) {
        fprintf(stderr, "panic: result_unwrap on non-Ok\n");
        exit(1);
    }
    return uv->field_values[0];
}

static Value builtin_result_unwrap_err(Value *args) {
    if (args[0].type != VAL_UNION) {
        fprintf(stderr, "panic: result_unwrap_err called on non-union\n");
        exit(1);
    }
    UnionValue *uv = args[0].as.union_val;
    if (!uv || strcmp(uv->variant_name, "Err") != 0 || uv->field_count < 1) {
        fprintf(stderr, "panic: result_unwrap_err on non-Err\n");
        exit(1);
    }
    return uv->field_values[0];
}

static Value builtin_result_unwrap_or(Value *args) {
    if (args[0].type != VAL_UNION) return args[1];
    UnionValue *uv = args[0].as.union_val;
    if (uv && strcmp(uv->variant_name, "Ok") == 0 && uv->field_count >= 1) {
        return uv->field_values[0];
    }
    return args[1];
}

static Value builtin_result_map(Value *args, Environment *env) {
    if (args[0].type != VAL_UNION) return args[0];
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "panic: result_map requires a function\n");
        exit(1);
    }

    UnionValue *uv = args[0].as.union_val;
    if (!uv || strcmp(uv->variant_name, "Ok") != 0 || uv->field_count < 1) {
        return args[0];
    }

    Value call_args[1];
    call_args[0] = uv->field_values[0];
    const char *fn_name = args[1].as.function_val.function_name;
    Value mapped = call_function(fn_name, call_args, 1, env);
    mapped.is_return = false;

    char *field_names[1] = { "value" };
    Value field_values[1] = { mapped };
    return create_union(uv->union_name, 0, "Ok", field_names, field_values, 1);
}

static Value builtin_result_and_then(Value *args, Environment *env) {
    if (args[0].type != VAL_UNION) return args[0];
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "panic: result_and_then requires a function\n");
        exit(1);
    }

    UnionValue *uv = args[0].as.union_val;
    if (!uv || strcmp(uv->variant_name, "Ok") != 0 || uv->field_count < 1) {
        return args[0];
    }

    Value call_args[1];
    call_args[0] = uv->field_values[0];
    const char *fn_name = args[1].as.function_val.function_name;
    Value next = call_function(fn_name, call_args, 1, env);
    next.is_return = false;
    if (next.type != VAL_UNION) {
        fprintf(stderr, "panic: result_and_then callback did not return a Result\n");
        exit(1);
    }
    return next;
}

/* ==========================================================================
 * End of Built-in OS Functions
 * ========================================================================== */

/* Print a value (used by println and eval) */
static void print_value(Value val) {
    switch (val.type) {
        case VAL_INT:
            printf("%lld", (long long)val.as.int_val);
            break;
        case VAL_FLOAT:
            printf("%g", val.as.float_val);
            break;
        case VAL_BOOL:
            printf("%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            printf("%s", val.as.string_val);
            break;
        case VAL_ARRAY: {
            /* Print array as [elem1, elem2, ...] */
            Array *arr = val.as.array_val;
            printf("[");
            for (int i = 0; i < arr->length; i++) {
                if (i > 0) printf(", ");
                switch (arr->element_type) {
                    case VAL_INT:
                        printf("%lld", ((long long*)arr->data)[i]);
                        break;
                    case VAL_FLOAT:
                        printf("%g", ((double*)arr->data)[i]);
                        break;
                    case VAL_BOOL:
                        printf("%s", ((bool*)arr->data)[i] ? "true" : "false");
                        break;
                    case VAL_STRING:
                        printf("\"%s\"", ((char**)arr->data)[i]);
                        break;
                    default:
                        break;
                }
            }
            printf("]");
            break;
        }
        case VAL_DYN_ARRAY: {
            /* Print dynamic array as [elem1, elem2, ...] */
            DynArray *arr = val.as.dyn_array_val;
            printf("[");
            int64_t len = dyn_array_length(arr);
            ElementType elem_type = dyn_array_get_elem_type(arr);
            for (int64_t i = 0; i < len; i++) {
                if (i > 0) printf(", ");
                switch (elem_type) {
                    case ELEM_INT:
                        printf("%lld", (long long)dyn_array_get_int(arr, i));
                        break;
                    case ELEM_FLOAT:
                        printf("%g", dyn_array_get_float(arr, i));
                        break;
                    case ELEM_BOOL:
                        printf("%s", dyn_array_get_bool(arr, i) ? "true" : "false");
                        break;
                    case ELEM_STRING:
                        printf("\"%s\"", dyn_array_get_string(arr, i));
                        break;
                    default:
                        printf("?");
                        break;
                }
            }
            printf("]");
            break;
        }
        case VAL_STRUCT: {
            /* Print struct as StructName { field1: value1, field2: value2 } */
            StructValue *sv = val.as.struct_val;
            printf("%s { ", sv->struct_name);
            for (int i = 0; i < sv->field_count; i++) {
                if (i > 0) printf(", ");
                printf("%s: ", sv->field_names[i]);
                print_value(sv->field_values[i]);
            }
            printf(" }");
            break;
        }
        case VAL_FUNCTION: {
            /* Print function value */
            printf("<function %s>", val.as.function_val.function_name);
            break;
        }
        case VAL_GC_STRUCT: {
            /* Print GC struct */
            printf("<gc_struct>");
            break;
        }
        case VAL_UNION: {
            /* Print union value */
            UnionValue *uv = val.as.union_val;
            printf("%s.%s { ", uv->union_name, uv->variant_name);
            for (int i = 0; i < uv->field_count; i++) {
                if (i > 0) printf(", ");
                printf("%s: ", uv->field_names[i]);
                print_value(uv->field_values[i]);
            }
            printf(" }");
            break;
        }
        case VAL_TUPLE: {
            /* Print tuple value */
            TupleValue *tv = val.as.tuple_val;
            printf("(");
            for (int i = 0; i < tv->element_count; i++) {
                if (i > 0) printf(", ");
                print_value(tv->elements[i]);
            }
            printf(")");
            break;
        }
        case VAL_VOID:
            printf("void");
            break;
    }
}

/* ==========================================================================
 * Math and Utility Built-in Functions
 * ========================================================================== */

static Value builtin_abs(Value *args) {
    if (args[0].type == VAL_INT) {
        long long val = args[0].as.int_val;
        return create_int(val < 0 ? -val : val);
    } else if (args[0].type == VAL_FLOAT) {
        double val = args[0].as.float_val;
        return create_float(val < 0 ? -val : val);
    }
    fprintf(stderr, "Error: abs requires int or float argument\n");
    return create_void();
}

static Value builtin_min(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a < b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a < b ? a : b);
    }
    fprintf(stderr, "Error: min requires two arguments of same type (int or float)\n");
    return create_void();
}

static Value builtin_max(Value *args) {
    if (args[0].type == VAL_INT && args[1].type == VAL_INT) {
        long long a = args[0].as.int_val;
        long long b = args[1].as.int_val;
        return create_int(a > b ? a : b);
    } else if (args[0].type == VAL_FLOAT && args[1].type == VAL_FLOAT) {
        double a = args[0].as.float_val;
        double b = args[1].as.float_val;
        return create_float(a > b ? a : b);
    }
    fprintf(stderr, "Error: max requires two arguments of same type (int or float)\n");
    return create_void();
}

/* Advanced Math Functions */
static Value builtin_sqrt(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sqrt(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sqrt((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sqrt requires numeric argument\n");
    return create_void();
}

static Value builtin_pow(Value *args) {
    double base, exponent;
    if (args[0].type == VAL_FLOAT) {
        base = args[0].as.float_val;
    } else if (args[0].type == VAL_INT) {
        base = (double)args[0].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }
    
    if (args[1].type == VAL_FLOAT) {
        exponent = args[1].as.float_val;
    } else if (args[1].type == VAL_INT) {
        exponent = (double)args[1].as.int_val;
    } else {
        fprintf(stderr, "Error: pow requires numeric arguments\n");
        return create_void();
    }
    
    return create_float(pow(base, exponent));
}

static Value builtin_floor(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(floor(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: floor requires numeric argument\n");
    return create_void();
}

static Value builtin_ceil(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(ceil(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: ceil requires numeric argument\n");
    return create_void();
}

static Value builtin_round(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(round(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_int(args[0].as.int_val);  /* Already an integer */
    }
    fprintf(stderr, "Error: round requires numeric argument\n");
    return create_void();
}

/* Trigonometric Functions */
static Value builtin_sin(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(sin(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(sin((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: sin requires numeric argument\n");
    return create_void();
}

static Value builtin_cos(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(cos(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(cos((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: cos requires numeric argument\n");
    return create_void();
}

static Value builtin_tan(Value *args) {
    if (args[0].type == VAL_FLOAT) {
        return create_float(tan(args[0].as.float_val));
    } else if (args[0].type == VAL_INT) {
        return create_float(tan((double)args[0].as.int_val));
    }
    fprintf(stderr, "Error: tan requires numeric argument\n");
    return create_void();
}

static Value builtin_atan2(Value *args) {
    double y = 0.0, x = 0.0;
    
    /* Get y value */
    if (args[0].type == VAL_FLOAT) {
        y = args[0].as.float_val;
    } else if (args[0].type == VAL_INT) {
        y = (double)args[0].as.int_val;
    } else {
        fprintf(stderr, "Error: atan2 requires numeric arguments\n");
        return create_void();
    }
    
    /* Get x value */
    if (args[1].type == VAL_FLOAT) {
        x = args[1].as.float_val;
    } else if (args[1].type == VAL_INT) {
        x = (double)args[1].as.int_val;
    } else {
        fprintf(stderr, "Error: atan2 requires numeric arguments\n");
        return create_void();
    }
    
    return create_float(atan2(y, x));
}

/* ============================================================================
 * Type Casting Functions
 * ========================================================================== */

static Value builtin_cast_int(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_INT) {
        return arg;  /* Already an int */
    } else if (arg.type == VAL_FLOAT) {
        return create_int((long long)arg.as.float_val);  /* Truncate */
    } else if (arg.type == VAL_BOOL) {
        return create_int(arg.as.bool_val ? 1 : 0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to int */
        char *endptr;
        long long val = strtoll(arg.as.string_val, &endptr, 10);
        if (endptr == arg.as.string_val || *endptr != '\0') {
            fprintf(stderr, "Error: cast_int cannot parse '%s' as integer\n", arg.as.string_val);
            return create_int(0);
        }
        return create_int(val);
    } else {
        fprintf(stderr, "Error: cast_int cannot convert type to int\n");
        return create_void();
    }
}

static Value builtin_cast_float(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_FLOAT) {
        return arg;  /* Already a float */
    } else if (arg.type == VAL_INT) {
        return create_float((double)arg.as.int_val);
    } else if (arg.type == VAL_BOOL) {
        return create_float(arg.as.bool_val ? 1.0 : 0.0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to float */
        char *endptr;
        double val = strtod(arg.as.string_val, &endptr);
        if (endptr == arg.as.string_val || *endptr != '\0') {
            fprintf(stderr, "Error: cast_float cannot parse '%s' as float\n", arg.as.string_val);
            return create_float(0.0);
        }
        return create_float(val);
    } else {
        fprintf(stderr, "Error: cast_float cannot convert type to float\n");
        return create_void();
    }
}

static Value builtin_cast_bool(Value *args) {
    Value arg = args[0];
    
    if (arg.type == VAL_BOOL) {
        return arg;  /* Already a bool */
    } else if (arg.type == VAL_INT) {
        return create_bool(arg.as.int_val != 0);
    } else if (arg.type == VAL_FLOAT) {
        return create_bool(arg.as.float_val != 0.0);
    } else if (arg.type == VAL_STRING) {
        /* Parse string to bool */
        if (strcmp(arg.as.string_val, "true") == 0 || strcmp(arg.as.string_val, "1") == 0) {
            return create_bool(true);
        } else {
            return create_bool(false);
        }
    } else {
        fprintf(stderr, "Error: cast_bool cannot convert type to bool\n");
        return create_void();
    }
}

typedef struct {
    char *buf;
    size_t len;
    size_t cap;
} EvalSB;

static void eval_sb_ensure(EvalSB *sb, size_t extra) {
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

static EvalSB eval_sb_new(size_t initial_cap) {
    EvalSB sb = {0};
    sb.cap = initial_cap ? initial_cap : 128;
    sb.buf = malloc(sb.cap);
    sb.len = 0;
    if (sb.buf) sb.buf[0] = '\0';
    return sb;
}

static void eval_sb_append_cstr(EvalSB *sb, const char *s) {
    if (!sb || !s) return;
    size_t n = strlen(s);
    eval_sb_ensure(sb, n);
    if (!sb->buf) return;
    memcpy(sb->buf + sb->len, s, n);
    sb->len += n;
    sb->buf[sb->len] = '\0';
}

static void eval_sb_append_char(EvalSB *sb, char c) {
    if (!sb) return;
    eval_sb_ensure(sb, 1);
    if (!sb->buf) return;
    sb->buf[sb->len++] = c;
    sb->buf[sb->len] = '\0';
}

static void eval_sb_append_value(EvalSB *sb, Value val);

static void eval_sb_append_dyn_array(EvalSB *sb, DynArray *arr) {
    eval_sb_append_char(sb, '[');
    int64_t len = dyn_array_length(arr);
    ElementType elem_type = dyn_array_get_elem_type(arr);
    for (int64_t i = 0; i < len; i++) {
        if (i > 0) eval_sb_append_cstr(sb, ", ");
        switch (elem_type) {
            case ELEM_INT: {
                char tmp[64];
                snprintf(tmp, sizeof(tmp), "%lld", (long long)dyn_array_get_int(arr, i));
                eval_sb_append_cstr(sb, tmp);
                break;
            }
            case ELEM_FLOAT: {
                char tmp[64];
                snprintf(tmp, sizeof(tmp), "%g", dyn_array_get_float(arr, i));
                eval_sb_append_cstr(sb, tmp);
                break;
            }
            case ELEM_BOOL:
                eval_sb_append_cstr(sb, dyn_array_get_bool(arr, i) ? "true" : "false");
                break;
            case ELEM_STRING:
                eval_sb_append_char(sb, '"');
                eval_sb_append_cstr(sb, dyn_array_get_string(arr, i));
                eval_sb_append_char(sb, '"');
                break;
            case ELEM_ARRAY:
                eval_sb_append_dyn_array(sb, dyn_array_get_array(arr, i));
                break;
            default:
                eval_sb_append_cstr(sb, "?");
                break;
        }
    }
    eval_sb_append_char(sb, ']');
}

static void eval_sb_append_value(EvalSB *sb, Value val) {
    switch (val.type) {
        case VAL_INT: {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%lld", (long long)val.as.int_val);
            eval_sb_append_cstr(sb, tmp);
            break;
        }
        case VAL_FLOAT: {
            char tmp[64];
            snprintf(tmp, sizeof(tmp), "%g", val.as.float_val);
            eval_sb_append_cstr(sb, tmp);
            break;
        }
        case VAL_BOOL:
            eval_sb_append_cstr(sb, val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            eval_sb_append_cstr(sb, val.as.string_val ? val.as.string_val : "");
            break;
        case VAL_ARRAY: {
            Array *arr = val.as.array_val;
            eval_sb_append_char(sb, '[');
            for (int i = 0; i < arr->length; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                switch (arr->element_type) {
                    case VAL_INT: {
                        char tmp[64];
                        snprintf(tmp, sizeof(tmp), "%lld", (long long)((long long*)arr->data)[i]);
                        eval_sb_append_cstr(sb, tmp);
                        break;
                    }
                    case VAL_FLOAT: {
                        char tmp[64];
                        snprintf(tmp, sizeof(tmp), "%g", ((double*)arr->data)[i]);
                        eval_sb_append_cstr(sb, tmp);
                        break;
                    }
                    case VAL_BOOL:
                        eval_sb_append_cstr(sb, ((bool*)arr->data)[i] ? "true" : "false");
                        break;
                    case VAL_STRING:
                        eval_sb_append_char(sb, '"');
                        eval_sb_append_cstr(sb, ((char**)arr->data)[i]);
                        eval_sb_append_char(sb, '"');
                        break;
                    default:
                        eval_sb_append_cstr(sb, "?");
                        break;
                }
            }
            eval_sb_append_char(sb, ']');
            break;
        }
        case VAL_DYN_ARRAY: {
            DynArray *arr = val.as.dyn_array_val;
            eval_sb_append_dyn_array(sb, arr);
            break;
        }
        case VAL_STRUCT: {
            StructValue *sv = val.as.struct_val;
            eval_sb_append_cstr(sb, sv->struct_name);
            eval_sb_append_cstr(sb, " { ");
            for (int i = 0; i < sv->field_count; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                eval_sb_append_cstr(sb, sv->field_names[i]);
                eval_sb_append_cstr(sb, ": ");
                eval_sb_append_value(sb, sv->field_values[i]);
            }
            eval_sb_append_cstr(sb, " }");
            break;
        }
        case VAL_UNION: {
            UnionValue *uv = val.as.union_val;
            eval_sb_append_cstr(sb, uv->union_name);
            eval_sb_append_char(sb, '.');
            eval_sb_append_cstr(sb, uv->variant_name);
            if (uv->field_count > 0) {
                eval_sb_append_cstr(sb, " { ");
                for (int i = 0; i < uv->field_count; i++) {
                    if (i > 0) eval_sb_append_cstr(sb, ", ");
                    eval_sb_append_cstr(sb, uv->field_names[i]);
                    eval_sb_append_cstr(sb, ": ");
                    eval_sb_append_value(sb, uv->field_values[i]);
                }
                eval_sb_append_cstr(sb, " }");
            }
            break;
        }
        case VAL_TUPLE: {
            TupleValue *tv = val.as.tuple_val;
            eval_sb_append_char(sb, '(');
            for (int i = 0; i < tv->element_count; i++) {
                if (i > 0) eval_sb_append_cstr(sb, ", ");
                eval_sb_append_value(sb, tv->elements[i]);
            }
            eval_sb_append_char(sb, ')');
            break;
        }
        case VAL_FUNCTION:
            eval_sb_append_cstr(sb, "<function>");
            break;
        case VAL_GC_STRUCT:
            eval_sb_append_cstr(sb, "<gc_struct>");
            break;
        case VAL_VOID:
            eval_sb_append_cstr(sb, "void");
            break;
    }
}

static Value builtin_to_string(Value *args) {
    Value arg = args[0];
    if (arg.type == VAL_STRING) return arg;

    EvalSB sb = eval_sb_new(256);
    eval_sb_append_value(&sb, arg);
    const char *out = sb.buf ? sb.buf : "";
    Value v = create_string(out);
    free(sb.buf);
    return v;
}

static Value builtin_cast_string(Value *args) {
    return builtin_to_string(args);
}

static Value builtin_print(Value *args) {
    print_value(args[0]);
    return create_void();
}

static Value builtin_println(Value *args) {
    print_value(args[0]);
    printf("\n");
    return create_void();
}

/* ============================================================================
 * String Operations
 * ========================================================================== */

static Value builtin_str_length(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_length requires string argument\n");
        return create_void();
    }
    assert(args[0].as.string_val != NULL);
    return create_int(safe_strlen(args[0].as.string_val));
}

static Value builtin_str_concat(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_concat requires two string arguments\n");
        return create_void();
    }
    
    assert(args[0].as.string_val != NULL);
    assert(args[1].as.string_val != NULL);
    size_t len1 = safe_strlen(args[0].as.string_val);
    size_t len2 = safe_strlen(args[1].as.string_val);
    char *result = malloc(len1 + len2 + 1);
    if (!result) {
        safe_fprintf(stderr, "Error: Memory allocation failed in str_concat\n");
        return create_void();
    }
    
    safe_strncpy(result, args[0].as.string_val, len1 + len2 + 1);
    safe_strncat(result, args[1].as.string_val, len1 + len2 + 1);
    
    return create_string(result);
}

static Value builtin_str_substring(Value *args) {
    if (args[0].type != VAL_STRING) {
        fprintf(stderr, "Error: str_substring requires string as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT || args[2].type != VAL_INT) {
        fprintf(stderr, "Error: str_substring requires integer start and length\n");
        return create_void();
    }
    
    const char *str = args[0].as.string_val;
    assert(str != NULL);
    long long start = args[1].as.int_val;
    long long length = args[2].as.int_val;
    long long str_len = safe_strlen(str);
    
    if (start < 0 || start >= str_len) {
        fprintf(stderr, "Error: str_substring start index out of bounds\n");
        return create_void();
    }
    
    if (length < 0) {
        fprintf(stderr, "Error: str_substring length cannot be negative\n");
        return create_void();
    }
    
    /* Adjust length if it exceeds string bounds */
    if (start + length > str_len) {
        length = str_len - start;
    }
    
    char *result = malloc(length + 1);
    if (!result) {
        fprintf(stderr, "Error: Memory allocation failed in str_substring\n");
        return create_void();
    }
    
    strncpy(result, str + start, length);
    result[length] = '\0';
    
    return create_string(result);
}

static Value builtin_str_contains(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_contains requires two string arguments\n");
        return create_void();
    }
    
    const char *str = args[0].as.string_val;
    const char *substr = args[1].as.string_val;
    
    return create_bool(strstr(str, substr) != NULL);
}

static Value builtin_str_equals(Value *args) {
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
        fprintf(stderr, "Error: str_equals requires two string arguments\n");
        return create_void();
    }
    
    return create_bool(strcmp(args[0].as.string_val, args[1].as.string_val) == 0);
}

/* ==========================================================================
 * Array Built-in Functions (With Bounds Checking!)
 * ========================================================================== */

static Value builtin_at(Value *args) {
    /* at(array, index) -> element */
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: at() requires an integer index\n");
        return create_void();
    }
    
    long long index = args[1].as.int_val;
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        
        /* BOUNDS CHECKING - This is the safety guarantee! */
        if (index < 0 || index >= arr->length) {
            fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                    (long long)index, arr->length);
            exit(1);  /* Fail fast - no undefined behavior! */
        }
        
        /* Return element based on type */
        switch (arr->element_type) {
            case VAL_INT:
                return create_int(((long long*)arr->data)[index]);
            case VAL_FLOAT:
                return create_float(((double*)arr->data)[index]);
            case VAL_BOOL:
                return create_bool(((bool*)arr->data)[index]);
            case VAL_STRING:
                return create_string(((char**)arr->data)[index]);
            default:
                fprintf(stderr, "Error: Unsupported array element type\n");
                return create_void();
        }
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        
        /* BOUNDS CHECKING */
        if (index < 0 || index >= len) {
            fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%lld)\n",
                    (long long)index, (long long)len);
            exit(1);
        }
        
        /* Return element based on type */
        ElementType elem_type = dyn_array_get_elem_type(arr);
        switch (elem_type) {
            case ELEM_INT:
                return create_int(dyn_array_get_int(arr, index));
            case ELEM_FLOAT:
                return create_float(dyn_array_get_float(arr, index));
            case ELEM_BOOL:
                return create_bool(dyn_array_get_bool(arr, index));
            case ELEM_STRING:
                return create_string(dyn_array_get_string(arr, index));
            case ELEM_ARRAY:
                return create_dyn_array(dyn_array_get_array(arr, index));
            default:
                fprintf(stderr, "Error: Unsupported array element type\n");
                return create_void();
        }
    }
    
    fprintf(stderr, "Error: at() requires an array as first argument\n");
    return create_void();
}

static Value builtin_array_length(Value *args) {
    /* array_length(array) -> int */
    if (args[0].type == VAL_ARRAY) {
        return create_int(args[0].as.array_val->length);
    }
    if (args[0].type == VAL_DYN_ARRAY) {
        return create_int(dyn_array_length(args[0].as.dyn_array_val));
    }
    
    fprintf(stderr, "Error: array_length() requires an array argument\n");
    return create_void();
}

static Value builtin_array_new(Value *args) {
    /* array_new(size, default_value) -> array */
    if (args[0].type != VAL_INT) {
        fprintf(stderr, "Error: array_new() requires an integer size\n");
        return create_void();
    }
    
    long long size = args[0].as.int_val;
    if (size < 0) {
        fprintf(stderr, "Error: array_new() size must be non-negative\n");
        return create_void();
    }
    
    ValueType elem_type = args[1].type;
    Value arr = create_array(elem_type, size, size);
    
    /* Initialize all elements with default value */
    for (long long i = 0; i < size; i++) {
        switch (elem_type) {
            case VAL_INT:
                ((long long*)arr.as.array_val->data)[i] = args[1].as.int_val;
                break;
            case VAL_FLOAT:
                ((double*)arr.as.array_val->data)[i] = args[1].as.float_val;
                break;
            case VAL_BOOL:
                ((bool*)arr.as.array_val->data)[i] = args[1].as.bool_val;
                break;
            case VAL_STRING:
                ((char**)arr.as.array_val->data)[i] = strdup(args[1].as.string_val);
                break;
            default:
                break;
        }
    }
    
    return arr;
}

static Value builtin_array_set(Value *args) {
    /* array_set(array, index, value) -> void */
    if (args[0].type != VAL_ARRAY) {
        fprintf(stderr, "Error: array_set() requires an array as first argument\n");
        return create_void();
    }
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: array_set() requires an integer index\n");
        return create_void();
    }
    
    Array *arr = args[0].as.array_val;
    long long index = args[1].as.int_val;
    
    /* BOUNDS CHECKING */
    if (index < 0 || index >= arr->length) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds [0..%d)\n",
                (long long)index, arr->length);
        exit(1);  /* Fail fast! */
    }
    
    /* Set element based on type */
    switch (arr->element_type) {
        case VAL_INT:
            if (args[2].type != VAL_INT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((long long*)arr->data)[index] = args[2].as.int_val;
            break;
        case VAL_FLOAT:
            if (args[2].type != VAL_FLOAT) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((double*)arr->data)[index] = args[2].as.float_val;
            break;
        case VAL_BOOL:
            if (args[2].type != VAL_BOOL) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            ((bool*)arr->data)[index] = args[2].as.bool_val;
            break;
        case VAL_STRING:
            if (args[2].type != VAL_STRING) {
                fprintf(stderr, "Error: Type mismatch in array_set\n");
                return create_void();
            }
            /* Free old string if exists */
            if (((char**)arr->data)[index]) {
                free(((char**)arr->data)[index]);
            }
            ((char**)arr->data)[index] = strdup(args[2].as.string_val);
            break;
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            break;
    }
    
    return create_void();
}

static Value builtin_array_slice(Value *args) {
    /* array_slice(array, start, length) -> array */
    if (args[1].type != VAL_INT || args[2].type != VAL_INT) {
        fprintf(stderr, "Error: array_slice() requires integer start and length\n");
        return create_void();
    }

    int64_t start = args[1].as.int_val;
    int64_t length = args[2].as.int_val;
    if (start < 0) start = 0;
    if (length < 0) length = 0;

    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        int64_t len = arr->length;
        if (start > len) start = len;
        int64_t end = start + length;
        if (end > len) end = len;
        int64_t out_len = end - start;

        Value out = create_array(arr->element_type, out_len, out_len);
        switch (arr->element_type) {
            case VAL_INT:
                for (int64_t i = 0; i < out_len; i++) {
                    ((long long*)out.as.array_val->data)[i] = ((long long*)arr->data)[start + i];
                }
                break;
            case VAL_FLOAT:
                for (int64_t i = 0; i < out_len; i++) {
                    ((double*)out.as.array_val->data)[i] = ((double*)arr->data)[start + i];
                }
                break;
            case VAL_BOOL:
                for (int64_t i = 0; i < out_len; i++) {
                    ((bool*)out.as.array_val->data)[i] = ((bool*)arr->data)[start + i];
                }
                break;
            case VAL_STRING:
                for (int64_t i = 0; i < out_len; i++) {
                    ((char**)out.as.array_val->data)[i] = strdup(((char**)arr->data)[start + i]);
                }
                break;
            default:
                break;
        }
        return out;
    }

    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        if (start > len) start = len;
        int64_t end = start + length;
        if (end > len) end = len;

        ElementType t = dyn_array_get_elem_type(arr);
        DynArray *out = dyn_array_new(t);
        for (int64_t i = start; i < end; i++) {
            switch (t) {
                case ELEM_INT: dyn_array_push_int(out, dyn_array_get_int(arr, i)); break;
                case ELEM_U8: dyn_array_push_int(out, (int64_t)dyn_array_get_u8(arr, i)); break;
                case ELEM_FLOAT: dyn_array_push_float(out, dyn_array_get_float(arr, i)); break;
                case ELEM_BOOL: dyn_array_push_bool(out, dyn_array_get_bool(arr, i)); break;
                case ELEM_STRING: dyn_array_push_string(out, dyn_array_get_string(arr, i)); break;
                case ELEM_ARRAY: dyn_array_push_array(out, dyn_array_get_array(arr, i)); break;
                default:
                    fprintf(stderr, "Error: array_slice unsupported element type\n");
                    return create_void();
            }
        }
        return create_dyn_array(out);
    }

    fprintf(stderr, "Error: array_slice() requires an array argument\n");
    return create_void();
}

/* ==========================================================================
 * Dynamic Array Operations (GC-Managed)
 * ========================================================================== */

/* Helper to create a dynamic array Value */
static Value create_dyn_array(DynArray *arr) {
    Value val;
    val.type = VAL_DYN_ARRAY;
    val.is_return = false;
    val.as.dyn_array_val = arr;
    return val;
}

/* Helper to map ValueType to ElementType */
static ElementType value_type_to_elem_type(ValueType vtype) {
    switch (vtype) {
        case VAL_INT: return ELEM_INT;
        case VAL_FLOAT: return ELEM_FLOAT;
        case VAL_BOOL: return ELEM_BOOL;
        case VAL_STRING: return ELEM_STRING;
        case VAL_DYN_ARRAY: return ELEM_ARRAY;  /* Nested arrays */
        case VAL_STRUCT:
        case VAL_GC_STRUCT: return ELEM_STRUCT;  /* Structs */
        default: return ELEM_INT; /* Default */
    }
}

static Value builtin_array_push(Value *args) {
    /* array_push(array, value) -> array
     * For empty array literal [], infers type from first push
     * For dynamic arrays, appends element
     */
    
    /* If arg[0] is an empty static array, convert to dynamic */
    if (args[0].type == VAL_ARRAY && args[0].as.array_val->length == 0) {
        /* Create new dynamic array with element type from value */
        ElementType elem_type = value_type_to_elem_type(args[1].type);
        DynArray *arr = dyn_array_new(elem_type);
        
        /* Push the first element */
        switch (args[1].type) {
            case VAL_INT:
                dyn_array_push_int(arr, args[1].as.int_val);
                break;
            case VAL_FLOAT:
                dyn_array_push_float(arr, args[1].as.float_val);
                break;
            case VAL_BOOL:
                dyn_array_push_bool(arr, args[1].as.bool_val);
                break;
            case VAL_STRING:
                dyn_array_push_string(arr, args[1].as.string_val);
                break;
            case VAL_DYN_ARRAY:
                dyn_array_push_array(arr, args[1].as.dyn_array_val);
                break;
            default:
                fprintf(stderr, "Error: Unsupported array element type\n");
                gc_release(arr);
                return create_void();
        }
        
        return create_dyn_array(arr);
    }
    
    /* Must be a dynamic array */
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_push() requires a dynamic array (use [] to create one)\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    
    /* Type check */
    ElementType expected_type = dyn_array_get_elem_type(arr);
    ValueType value_type = args[1].type;
    
    if (value_type_to_elem_type(value_type) != expected_type) {
        fprintf(stderr, "Error: Type mismatch in array_push\n");
        return create_void();
    }
    
    /* Push element */
    switch (value_type) {
        case VAL_INT:
            dyn_array_push_int(arr, args[1].as.int_val);
            break;
        case VAL_FLOAT:
            dyn_array_push_float(arr, args[1].as.float_val);
            break;
        case VAL_BOOL:
            dyn_array_push_bool(arr, args[1].as.bool_val);
            break;
        case VAL_STRING:
            dyn_array_push_string(arr, args[1].as.string_val);
            break;
        case VAL_DYN_ARRAY:
            dyn_array_push_array(arr, args[1].as.dyn_array_val);
            break;
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            return create_void();
    }
    
    /* Return the same array (it's mutated in-place) */
    return args[0];
}

static Value builtin_array_pop(Value *args) {
    /* array_pop(array) -> value */
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_pop() requires a dynamic array\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    
    if (dyn_array_length(arr) == 0) {
        fprintf(stderr, "Error: array_pop() on empty array\n");
        return create_void();
    }
    
    /* Pop element based on type */
    bool success = false;
    ElementType elem_type = dyn_array_get_elem_type(arr);
    switch (elem_type) {
        case ELEM_INT: {
            int64_t val = dyn_array_pop_int(arr, &success);
            return success ? create_int(val) : create_void();
        }
        case ELEM_FLOAT: {
            double val = dyn_array_pop_float(arr, &success);
            return success ? create_float(val) : create_void();
        }
        case ELEM_BOOL: {
            bool val = dyn_array_pop_bool(arr, &success);
            return success ? create_bool(val) : create_void();
        }
        case ELEM_STRING: {
            const char *val = dyn_array_pop_string(arr, &success);
            return success ? create_string(val) : create_void();
        }
        case ELEM_ARRAY: {
            DynArray *val = dyn_array_pop_array(arr, &success);
            return success ? create_dyn_array(val) : create_void();
        }
        default:
            fprintf(stderr, "Error: Unsupported array element type\n");
            return create_void();
    }
}

static Value builtin_array_remove_at(Value *args) {
    /* array_remove_at(array, index) -> array */
    if (args[0].type != VAL_DYN_ARRAY) {
        fprintf(stderr, "Error: array_remove_at() requires a dynamic array\n");
        return create_void();
    }
    if (args[1].type != VAL_INT) {
        fprintf(stderr, "Error: array_remove_at() requires an integer index\n");
        return create_void();
    }
    
    DynArray *arr = args[0].as.dyn_array_val;
    int64_t index = args[1].as.int_val;
    
    if (index < 0 || index >= dyn_array_length(arr)) {
        fprintf(stderr, "Runtime Error: Array index %lld out of bounds\n", (long long)index);
        exit(1);
    }
    
    dyn_array_remove_at(arr, index);
    
    /* Return the modified array */
    return args[0];
}

/* ==========================================================================
 * Higher-Order Array Functions (map, filter, reduce)
 * ========================================================================== */

static Value builtin_map(Value *args, Environment *env) {
    /* map(array, transform_fn) -> array
     * Applies transform_fn to each element and returns a new array
     */
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: map() requires a function as second argument\n");
        return create_void();
    }
    
    const char *transform_fn_name = args[1].as.function_val.function_name;
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *input_arr = args[0].as.array_val;
        int64_t len = input_arr->length;
        
        /* Create new array of same type and size */
        Value result = create_array(input_arr->element_type, len, len);
        Array *output_arr = result.as.array_val;
        
        /* Apply transform to each element */
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = input_arr->element_type;
            elem.is_return = false;
            
            /* Get element from input array */
            switch (input_arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)input_arr->data)[i];
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in map\n");
                    return create_void();
            }
            
            /* Call transform function with this element */
            Value call_args[1];
            call_args[0] = elem;
            Value transformed = call_function(transform_fn_name, call_args, 1, env);
            
            /* Store transformed value in output array */
            switch (output_arr->element_type) {
                case VAL_INT:
                    if (transformed.type != VAL_INT) {
                        fprintf(stderr, "Error: Transform function must return same type as array elements\n");
                        return create_void();
                    }
                    ((long long*)output_arr->data)[i] = transformed.as.int_val;
                    break;
                case VAL_FLOAT:
                    if (transformed.type != VAL_FLOAT) {
                        fprintf(stderr, "Error: Transform function must return same type as array elements\n");
                        return create_void();
                    }
                    ((double*)output_arr->data)[i] = transformed.as.float_val;
                    break;
                case VAL_BOOL:
                    if (transformed.type != VAL_BOOL) {
                        fprintf(stderr, "Error: Transform function must return same type as array elements\n");
                        return create_void();
                    }
                    ((bool*)output_arr->data)[i] = transformed.as.bool_val;
                    break;
                case VAL_STRING:
                    if (transformed.type != VAL_STRING) {
                        fprintf(stderr, "Error: Transform function must return same type as array elements\n");
                        return create_void();
                    }
                    ((char**)output_arr->data)[i] = strdup(transformed.as.string_val);
                    break;
                default:
                    break;
            }
        }
        
        return result;
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *input_arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(input_arr);
        ElementType elem_type = dyn_array_get_elem_type(input_arr);
        
        /* Create new dynamic array of same type */
        DynArray *output_arr = dyn_array_new(elem_type);
        
        /* Apply transform to each element */
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;
            
            /* Get element from input array */
            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(input_arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(input_arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(input_arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(input_arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(input_arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in map\n");
                    return create_void();
            }
            
            /* Call transform function */
            Value call_args[1];
            call_args[0] = elem;
            Value transformed = call_function(transform_fn_name, call_args, 1, env);
            
            /* Push transformed value to output array */
            switch (elem_type) {
                case ELEM_INT:
                    if (transformed.type != VAL_INT) {
                        fprintf(stderr, "Error: Transform function must return same type\n");
                        return create_void();
                    }
                    dyn_array_push_int(output_arr, transformed.as.int_val);
                    break;
                case ELEM_FLOAT:
                    if (transformed.type != VAL_FLOAT) {
                        fprintf(stderr, "Error: Transform function must return same type\n");
                        return create_void();
                    }
                    dyn_array_push_float(output_arr, transformed.as.float_val);
                    break;
                case ELEM_BOOL:
                    if (transformed.type != VAL_BOOL) {
                        fprintf(stderr, "Error: Transform function must return same type\n");
                        return create_void();
                    }
                    dyn_array_push_bool(output_arr, transformed.as.bool_val);
                    break;
                case ELEM_STRING:
                    if (transformed.type != VAL_STRING) {
                        fprintf(stderr, "Error: Transform function must return same type\n");
                        return create_void();
                    }
                    dyn_array_push_string(output_arr, transformed.as.string_val);
                    break;
                case ELEM_ARRAY:
                    if (transformed.type != VAL_DYN_ARRAY) {
                        fprintf(stderr, "Error: Transform function must return same type\n");
                        return create_void();
                    }
                    dyn_array_push_array(output_arr, transformed.as.dyn_array_val);
                    break;
                default:
                    break;
            }
        }
        
        return create_dyn_array(output_arr);
    }
    
    fprintf(stderr, "Error: map() requires an array as first argument\n");
    return create_void();
}

static Value builtin_filter(Value *args, Environment *env) {
    /* filter(array, predicate_fn) -> array
     * Returns a new array containing only elements where predicate_fn(elem) is true.
     */
    if (args[1].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: filter() requires a function as second argument\n");
        return create_void();
    }

    const char *pred_fn_name = args[1].as.function_val.function_name;

    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *input_arr = args[0].as.array_val;
        int64_t len = input_arr->length;

        bool *keep = (bool*)calloc((size_t)len, sizeof(bool));
        if (!keep) {
            fprintf(stderr, "Error: Out of memory in filter()\n");
            return create_void();
        }

        int64_t out_len = 0;
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = input_arr->element_type;
            elem.is_return = false;

            switch (input_arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)input_arr->data)[i];
                    break;
                default:
                    free(keep);
                    fprintf(stderr, "Error: Unsupported array element type in filter\n");
                    return create_void();
            }

            Value call_args[1];
            call_args[0] = elem;
            Value pred = call_function(pred_fn_name, call_args, 1, env);
            if (pred.type != VAL_BOOL) {
                free(keep);
                fprintf(stderr, "Error: filter predicate must return bool\n");
                return create_void();
            }
            keep[i] = pred.as.bool_val;
            if (keep[i]) out_len++;
        }

        Value result = create_array(input_arr->element_type, out_len, out_len);
        Array *output_arr = result.as.array_val;

        int64_t out_i = 0;
        for (int64_t i = 0; i < len; i++) {
            if (!keep[i]) continue;

            switch (input_arr->element_type) {
                case VAL_INT:
                    ((long long*)output_arr->data)[out_i] = ((long long*)input_arr->data)[i];
                    break;
                case VAL_FLOAT:
                    ((double*)output_arr->data)[out_i] = ((double*)input_arr->data)[i];
                    break;
                case VAL_BOOL:
                    ((bool*)output_arr->data)[out_i] = ((bool*)input_arr->data)[i];
                    break;
                case VAL_STRING:
                    ((char**)output_arr->data)[out_i] = strdup(((char**)input_arr->data)[i]);
                    break;
                default:
                    break;
            }
            out_i++;
        }

        free(keep);
        return result;
    }

    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *input_arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(input_arr);
        ElementType elem_type = dyn_array_get_elem_type(input_arr);

        DynArray *output_arr = dyn_array_new(elem_type);
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;

            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(input_arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(input_arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(input_arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(input_arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(input_arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in filter\n");
                    return create_void();
            }

            Value call_args[1];
            call_args[0] = elem;
            Value pred = call_function(pred_fn_name, call_args, 1, env);
            if (pred.type != VAL_BOOL) {
                fprintf(stderr, "Error: filter predicate must return bool\n");
                return create_void();
            }

            if (!pred.as.bool_val) continue;

            switch (elem_type) {
                case ELEM_INT:
                    dyn_array_push_int(output_arr, elem.as.int_val);
                    break;
                case ELEM_FLOAT:
                    dyn_array_push_float(output_arr, elem.as.float_val);
                    break;
                case ELEM_BOOL:
                    dyn_array_push_bool(output_arr, elem.as.bool_val);
                    break;
                case ELEM_STRING:
                    dyn_array_push_string(output_arr, elem.as.string_val);
                    break;
                case ELEM_ARRAY:
                    dyn_array_push_array(output_arr, elem.as.dyn_array_val);
                    break;
                default:
                    break;
            }
        }

        return create_dyn_array(output_arr);
    }

    fprintf(stderr, "Error: filter() requires an array as first argument\n");
    return create_void();
}

static Value builtin_reduce(Value *args, Environment *env) {
    /* reduce(array, initial_value, combine_fn) -> value
     * Combines all elements using combine_fn, starting with initial_value
     */
    if (args[2].type != VAL_FUNCTION) {
        fprintf(stderr, "Error: reduce() requires a function as third argument\n");
        return create_void();
    }
    
    const char *combine_fn_name = args[2].as.function_val.function_name;
    Value accumulator = args[1];  /* Initial value */
    
    /* Handle static arrays */
    if (args[0].type == VAL_ARRAY) {
        Array *arr = args[0].as.array_val;
        int64_t len = arr->length;
        
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.type = arr->element_type;
            elem.is_return = false;
            
            /* Get element */
            switch (arr->element_type) {
                case VAL_INT:
                    elem.as.int_val = ((long long*)arr->data)[i];
                    break;
                case VAL_FLOAT:
                    elem.as.float_val = ((double*)arr->data)[i];
                    break;
                case VAL_BOOL:
                    elem.as.bool_val = ((bool*)arr->data)[i];
                    break;
                case VAL_STRING:
                    elem.as.string_val = ((char**)arr->data)[i];
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in reduce\n");
                    return create_void();
            }
            
            /* Call combine function with accumulator and element */
            Value call_args[2];
            call_args[0] = accumulator;
            call_args[1] = elem;
            accumulator = call_function(combine_fn_name, call_args, 2, env);
        }
        
        return accumulator;
    }
    
    /* Handle dynamic arrays */
    if (args[0].type == VAL_DYN_ARRAY) {
        DynArray *arr = args[0].as.dyn_array_val;
        int64_t len = dyn_array_length(arr);
        ElementType elem_type = dyn_array_get_elem_type(arr);
        
        for (int64_t i = 0; i < len; i++) {
            Value elem;
            elem.is_return = false;
            
            /* Get element */
            switch (elem_type) {
                case ELEM_INT:
                    elem.type = VAL_INT;
                    elem.as.int_val = dyn_array_get_int(arr, i);
                    break;
                case ELEM_FLOAT:
                    elem.type = VAL_FLOAT;
                    elem.as.float_val = dyn_array_get_float(arr, i);
                    break;
                case ELEM_BOOL:
                    elem.type = VAL_BOOL;
                    elem.as.bool_val = dyn_array_get_bool(arr, i);
                    break;
                case ELEM_STRING:
                    elem.type = VAL_STRING;
                    elem.as.string_val = (char*)dyn_array_get_string(arr, i);
                    break;
                case ELEM_ARRAY:
                    elem.type = VAL_DYN_ARRAY;
                    elem.as.dyn_array_val = dyn_array_get_array(arr, i);
                    break;
                default:
                    fprintf(stderr, "Error: Unsupported array element type in reduce\n");
                    return create_void();
            }
            
            /* Call combine function */
            Value call_args[2];
            call_args[0] = accumulator;
            call_args[1] = elem;
            accumulator = call_function(combine_fn_name, call_args, 2, env);
        }
        
        return accumulator;
    }
    
    fprintf(stderr, "Error: reduce() requires an array as first argument\n");
    return create_void();
}

/* ==========================================================================
 * End of Math and Utility Built-in Functions
 * ========================================================================== */

/* Helper to convert value to boolean */
static bool is_truthy(Value val) {
    switch (val.type) {
        case VAL_BOOL:
            return val.as.bool_val;
        case VAL_INT:
            return val.as.int_val != 0;
        case VAL_FLOAT:
            return val.as.float_val != 0.0;
        case VAL_VOID:
            return false;
        default:
            return true; /* Strings are truthy if non-null */
    }
}

/* Evaluate prefix operation */
static Value eval_prefix_op(ASTNode *node, Environment *env) {
    TokenType op = node->as.prefix_op.op;
    int arg_count = node->as.prefix_op.arg_count;


    /* Arithmetic operators */
    if (op == TOKEN_PLUS || op == TOKEN_MINUS || op == TOKEN_STAR ||
        op == TOKEN_SLASH || op == TOKEN_PERCENT) {
        
        /* Handle unary minus: (- x) */
        if (op == TOKEN_MINUS && arg_count == 1) {
            Value arg = eval_expression(node->as.prefix_op.args[0], env);
            if (arg.type == VAL_INT) {
                return create_int(-arg.as.int_val);
            } else if (arg.type == VAL_FLOAT) {
                return create_float(-arg.as.float_val);
            } else if (arg.type == VAL_DYN_ARRAY) {
                DynArray *a = arg.as.dyn_array_val;
                assert(a);
                ElementType t = dyn_array_get_elem_type(a);
                int64_t len = dyn_array_length(a);
                if (t == ELEM_INT) {
                    DynArray *out = dyn_array_new(ELEM_INT);
                    for (int64_t i = 0; i < len; i++) dyn_array_push_int(out, -dyn_array_get_int(a, i));
                    return create_dyn_array(out);
                } else if (t == ELEM_FLOAT) {
                    DynArray *out = dyn_array_new(ELEM_FLOAT);
                    for (int64_t i = 0; i < len; i++) dyn_array_push_float(out, -dyn_array_get_float(a, i));
                    return create_dyn_array(out);
                }
                fprintf(stderr, "Error: Unary minus requires array<int> or array<float>\n");
                return create_void();
            } else if (arg.type == VAL_ARRAY) {
                Array *a = arg.as.array_val;
                if (!a) return create_void();
                if (a->element_type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) ((long long*)out.as.array_val->data)[i] = -((long long*)a->data)[i];
                    return out;
                } else if (a->element_type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) ((double*)out.as.array_val->data)[i] = -((double*)a->data)[i];
                    return out;
                }
                fprintf(stderr, "Error: Unary minus requires array<int> or array<float>\n");
                return create_void();
            } else {
                fprintf(stderr, "Error: Unary minus requires numeric argument\n");
                return create_void();
            }
        }
        
        /* Binary arithmetic operations */
        if (arg_count != 2) {
            fprintf(stderr, "Error: Binary arithmetic operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        /* Array arithmetic (elementwise) */
        if (left.type == VAL_DYN_ARRAY || right.type == VAL_DYN_ARRAY || left.type == VAL_ARRAY || right.type == VAL_ARRAY) {
            /* DynArray path */
            if (left.type == VAL_DYN_ARRAY || right.type == VAL_DYN_ARRAY) {
                if (left.type == VAL_DYN_ARRAY && right.type == VAL_DYN_ARRAY) {
                    DynArray *a = left.as.dyn_array_val;
                    DynArray *b = right.as.dyn_array_val;
                    assert(a && b);
                    int64_t len = dyn_array_length(a);
                    if (len != dyn_array_length(b)) {
                        fprintf(stderr, "Error: Array length mismatch in operator\n");
                        return create_void();
                    }
                    ElementType t = dyn_array_get_elem_type(a);
                    if (t != dyn_array_get_elem_type(b)) {
                        fprintf(stderr, "Error: Array element type mismatch in operator\n");
                        return create_void();
                    }
                    DynArray *out = eval_dyn_array_binop(a, b, op);
                    if (!out) {
                        fprintf(stderr, "Error: Array mismatch in operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                }

                /* Broadcast scalar over array */
                if (left.type == VAL_DYN_ARRAY) {
                    DynArray *a = left.as.dyn_array_val;
                    DynArray *out = eval_dyn_array_scalar_right(a, right, op);
                    if (!out) {
                        fprintf(stderr, "Error: Type mismatch in array-scalar operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                } else if (right.type == VAL_DYN_ARRAY) {
                    DynArray *a = right.as.dyn_array_val;
                    DynArray *out = eval_dyn_array_scalar_left(left, a, op);
                    if (!out) {
                        fprintf(stderr, "Error: Type mismatch in scalar-array operator\n");
                        return create_void();
                    }
                    return create_dyn_array(out);
                }
            }

            /* Static Array path */
            if (left.type == VAL_ARRAY && right.type == VAL_ARRAY) {
                Array *a = left.as.array_val;
                Array *b = right.as.array_val;
                if (!a || !b || a->length != b->length || a->element_type != b->element_type) {
                    fprintf(stderr, "Error: Array mismatch in operator\n");
                    return create_void();
                }
                Value out = create_array(a->element_type, a->length, a->length);
                for (int i = 0; i < a->length; i++) {
                    switch (a->element_type) {
                        case VAL_INT: {
                            long long x = ((long long*)a->data)[i];
                            long long y = ((long long*)b->data)[i];
                            long long r = 0;
                            switch (op) {
                                case TOKEN_PLUS: r = x + y; break;
                                case TOKEN_MINUS: r = x - y; break;
                                case TOKEN_STAR: r = x * y; break;
                                case TOKEN_SLASH: r = x / y; break;
                                case TOKEN_PERCENT: r = x % y; break;
                                default: break;
                            }
                            ((long long*)out.as.array_val->data)[i] = r;
                            break;
                        }
                        case VAL_FLOAT: {
                            double x = ((double*)a->data)[i];
                            double y = ((double*)b->data)[i];
                            double r = 0.0;
                            switch (op) {
                                case TOKEN_PLUS: r = x + y; break;
                                case TOKEN_MINUS: r = x - y; break;
                                case TOKEN_STAR: r = x * y; break;
                                case TOKEN_SLASH: r = x / y; break;
                                default: break;
                            }
                            ((double*)out.as.array_val->data)[i] = r;
                            break;
                        }
                        case VAL_STRING: {
                            if (op != TOKEN_PLUS) {
                                fprintf(stderr, "Error: string arrays only support +\n");
                                return create_void();
                            }
                            const char *x = ((char**)a->data)[i];
                            const char *y = ((char**)b->data)[i];
                            size_t lx = strlen(x);
                            size_t ly = strlen(y);
                            char *buf = malloc(lx + ly + 1);
                            memcpy(buf, x, lx);
                            memcpy(buf + lx, y, ly);
                            buf[lx + ly] = '\0';
                            ((char**)out.as.array_val->data)[i] = buf;
                            break;
                        }
                        default:
                            fprintf(stderr, "Error: unsupported array element type in operator\n");
                            return create_void();
                    }
                }
                return out;
            }

            /* Static array + scalar broadcast */
            if (left.type == VAL_ARRAY && (right.type == VAL_INT || right.type == VAL_FLOAT || right.type == VAL_STRING)) {
                Array *a = left.as.array_val;
                if (!a) return create_void();

                if (a->element_type == VAL_INT && right.type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        long long x = ((long long*)a->data)[i];
                        long long s = right.as.int_val;
                        long long r = 0;
                        switch (op) {
                            case TOKEN_PLUS: r = x + s; break;
                            case TOKEN_MINUS: r = x - s; break;
                            case TOKEN_STAR: r = x * s; break;
                            case TOKEN_SLASH: r = x / s; break;
                            case TOKEN_PERCENT: r = x % s; break;
                            default: break;
                        }
                        ((long long*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_FLOAT && right.type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        double x = ((double*)a->data)[i];
                        double s = right.as.float_val;
                        double r = 0.0;
                        switch (op) {
                            case TOKEN_PLUS: r = x + s; break;
                            case TOKEN_MINUS: r = x - s; break;
                            case TOKEN_STAR: r = x * s; break;
                            case TOKEN_SLASH: r = x / s; break;
                            default: break;
                        }
                        ((double*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_STRING && right.type == VAL_STRING) {
                    if (op != TOKEN_PLUS) {
                        fprintf(stderr, "Error: string arrays only support +\n");
                        return create_void();
                    }
                    Value out = create_array(VAL_STRING, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        const char *x = ((char**)a->data)[i];
                        const char *s = right.as.string_val;
                        size_t lx = strlen(x);
                        size_t ls = strlen(s);
                        char *buf = malloc(lx + ls + 1);
                        memcpy(buf, x, lx);
                        memcpy(buf + lx, s, ls);
                        buf[lx + ls] = '\0';
                        ((char**)out.as.array_val->data)[i] = buf;
                    }
                    return out;
                }
            }

            /* Scalar + static array broadcast */
            if (right.type == VAL_ARRAY && (left.type == VAL_INT || left.type == VAL_FLOAT || left.type == VAL_STRING)) {
                Array *a = right.as.array_val;
                if (!a) return create_void();

                if (a->element_type == VAL_INT && left.type == VAL_INT) {
                    Value out = create_array(VAL_INT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        long long s = left.as.int_val;
                        long long y = ((long long*)a->data)[i];
                        long long r = 0;
                        switch (op) {
                            case TOKEN_PLUS: r = s + y; break;
                            case TOKEN_MINUS: r = s - y; break;
                            case TOKEN_STAR: r = s * y; break;
                            case TOKEN_SLASH: r = s / y; break;
                            case TOKEN_PERCENT: r = s % y; break;
                            default: break;
                        }
                        ((long long*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_FLOAT && left.type == VAL_FLOAT) {
                    Value out = create_array(VAL_FLOAT, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        double s = left.as.float_val;
                        double y = ((double*)a->data)[i];
                        double r = 0.0;
                        switch (op) {
                            case TOKEN_PLUS: r = s + y; break;
                            case TOKEN_MINUS: r = s - y; break;
                            case TOKEN_STAR: r = s * y; break;
                            case TOKEN_SLASH: r = s / y; break;
                            default: break;
                        }
                        ((double*)out.as.array_val->data)[i] = r;
                    }
                    return out;
                }

                if (a->element_type == VAL_STRING && left.type == VAL_STRING) {
                    if (op != TOKEN_PLUS) {
                        fprintf(stderr, "Error: string arrays only support +\n");
                        return create_void();
                    }
                    Value out = create_array(VAL_STRING, a->length, a->length);
                    for (int i = 0; i < a->length; i++) {
                        const char *s = left.as.string_val;
                        const char *y = ((char**)a->data)[i];
                        size_t ls = strlen(s);
                        size_t ly = strlen(y);
                        char *buf = malloc(ls + ly + 1);
                        memcpy(buf, s, ls);
                        memcpy(buf + ls, y, ly);
                        buf[ls + ly] = '\0';
                        ((char**)out.as.array_val->data)[i] = buf;
                    }
                    return out;
                }
            }
        }

        if (left.type == VAL_INT && right.type == VAL_INT) {
            long long result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.int_val + right.as.int_val; break;
                case TOKEN_MINUS: result = left.as.int_val - right.as.int_val; break;
                case TOKEN_STAR: result = left.as.int_val * right.as.int_val; break;
                case TOKEN_SLASH:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val / right.as.int_val;
                    break;
                case TOKEN_PERCENT:
                    if (right.as.int_val == 0) {
                        fprintf(stderr, "Error: Modulo by zero\n");
                        return create_void();
                    }
                    result = left.as.int_val % right.as.int_val;
                    break;
                default: result = 0;
            }
            return create_int(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            double result;
            switch (op) {
                case TOKEN_PLUS: result = left.as.float_val + right.as.float_val; break;
                case TOKEN_MINUS: result = left.as.float_val - right.as.float_val; break;
                case TOKEN_STAR: result = left.as.float_val * right.as.float_val; break;
                case TOKEN_SLASH:
                    if (right.as.float_val == 0.0) {
                        fprintf(stderr, "Error: Division by zero\n");
                        return create_void();
                    }
                    result = left.as.float_val / right.as.float_val;
                    break;
                default: result = 0.0;
            }
            return create_float(result);
        }
    }

    /* Comparison operators */
    if (op == TOKEN_LT || op == TOKEN_LE || op == TOKEN_GT || op == TOKEN_GE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Comparison operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        if (left.type == VAL_INT && right.type == VAL_INT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.int_val < right.as.int_val; break;
                case TOKEN_LE: result = left.as.int_val <= right.as.int_val; break;
                case TOKEN_GT: result = left.as.int_val > right.as.int_val; break;
                case TOKEN_GE: result = left.as.int_val >= right.as.int_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_FLOAT) {
            bool result;
            switch (op) {
                case TOKEN_LT: result = left.as.float_val < right.as.float_val; break;
                case TOKEN_LE: result = left.as.float_val <= right.as.float_val; break;
                case TOKEN_GT: result = left.as.float_val > right.as.float_val; break;
                case TOKEN_GE: result = left.as.float_val >= right.as.float_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_INT && right.type == VAL_FLOAT) {
            /* Mixed int/float comparison: convert int to float */
            bool result;
            double left_f = (double)left.as.int_val;
            switch (op) {
                case TOKEN_LT: result = left_f < right.as.float_val; break;
                case TOKEN_LE: result = left_f <= right.as.float_val; break;
                case TOKEN_GT: result = left_f > right.as.float_val; break;
                case TOKEN_GE: result = left_f >= right.as.float_val; break;
                default: result = false;
            }
            return create_bool(result);
        } else if (left.type == VAL_FLOAT && right.type == VAL_INT) {
            /* Mixed float/int comparison: convert int to float */
            bool result;
            double right_f = (double)right.as.int_val;
            switch (op) {
                case TOKEN_LT: result = left.as.float_val < right_f; break;
                case TOKEN_LE: result = left.as.float_val <= right_f; break;
                case TOKEN_GT: result = left.as.float_val > right_f; break;
                case TOKEN_GE: result = left.as.float_val >= right_f; break;
                default: result = false;
            }
            return create_bool(result);
        }
    }

    /* Equality operators */
    if (op == TOKEN_EQ || op == TOKEN_NE) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Equality operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);
        Value right = eval_expression(node->as.prefix_op.args[1], env);

        bool equal = false;
        if (left.type == right.type) {
            switch (left.type) {
                case VAL_INT: equal = left.as.int_val == right.as.int_val; break;
                case VAL_FLOAT: equal = left.as.float_val == right.as.float_val; break;
                case VAL_BOOL: equal = left.as.bool_val == right.as.bool_val; break;
                case VAL_STRING: equal = strcmp(left.as.string_val, right.as.string_val) == 0; break;
                case VAL_STRUCT: {
                    /* Structs are equal if they're the same type and all fields are equal */
                    StructValue *left_sv = left.as.struct_val;
                    StructValue *right_sv = right.as.struct_val;
                    if (strcmp(left_sv->struct_name, right_sv->struct_name) != 0 ||
                        left_sv->field_count != right_sv->field_count) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_sv->field_count && equal; i++) {
                            Value left_field = left_sv->field_values[i];
                            Value right_field = right_sv->field_values[i];
                            /* Recursively compare field values (simplified - only int/float/bool/string) */
                            if (left_field.type != right_field.type) {
                                equal = false;
                            } else if (left_field.type == VAL_INT) {
                                equal = left_field.as.int_val == right_field.as.int_val;
                            } else if (left_field.type == VAL_FLOAT) {
                                equal = left_field.as.float_val == right_field.as.float_val;
                            } else if (left_field.type == VAL_BOOL) {
                                equal = left_field.as.bool_val == right_field.as.bool_val;
                            } else if (left_field.type == VAL_STRING) {
                                equal = strcmp(left_field.as.string_val, right_field.as.string_val) == 0;
                            }
                        }
                    }
                    break;
                }
                case VAL_VOID: equal = true; break;  /* void == void */
                case VAL_ARRAY: {
                    /* Arrays are equal if they have same length and all elements equal */
                    Array *left_arr = left.as.array_val;
                    Array *right_arr = right.as.array_val;
                    if (left_arr->length != right_arr->length) {
                        equal = false;
                    } else {
                        equal = true;
                        for (int i = 0; i < left_arr->length && equal; i++) {
                            switch (left_arr->element_type) {
                                case VAL_INT:
                                    equal = ((long long*)left_arr->data)[i] == ((long long*)right_arr->data)[i];
                                    break;
                                case VAL_FLOAT:
                                    equal = ((double*)left_arr->data)[i] == ((double*)right_arr->data)[i];
                                    break;
                                case VAL_BOOL:
                                    equal = ((bool*)left_arr->data)[i] == ((bool*)right_arr->data)[i];
                                    break;
                                case VAL_STRING:
                                    equal = strcmp(((char**)left_arr->data)[i], ((char**)right_arr->data)[i]) == 0;
                                    break;
                                default:
                                    equal = false;
                                    break;
                            }
                        }
                    }
                    break;
                }
                case VAL_DYN_ARRAY:
                case VAL_GC_STRUCT:
                case VAL_UNION:
                case VAL_TUPLE:
                case VAL_FUNCTION:
                    /* These types don't support equality comparison yet */
                    equal = false;
                    break;
            }
        }

        return create_bool(op == TOKEN_EQ ? equal : !equal);
    }

    /* Logical operators */
    if (op == TOKEN_AND || op == TOKEN_OR) {
        if (arg_count != 2) {
            fprintf(stderr, "Error: Logical operators require 2 arguments\n");
            return create_void();
        }
        Value left = eval_expression(node->as.prefix_op.args[0], env);

        if (op == TOKEN_AND) {
            if (!is_truthy(left)) return create_bool(false);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        } else { /* OR */
            if (is_truthy(left)) return create_bool(true);
            Value right = eval_expression(node->as.prefix_op.args[1], env);
            return create_bool(is_truthy(right));
        }
    }

    if (op == TOKEN_NOT) {
        if (arg_count != 1) {
            fprintf(stderr, "Error: 'not' requires 1 argument\n");
            return create_void();
        }
        Value arg = eval_expression(node->as.prefix_op.args[0], env);
        return create_bool(!is_truthy(arg));
    }

    return create_void();
}

/* Evaluate function call */
static Value eval_call(ASTNode *node, Environment *env) {
    /* Check if this is a function call returning a function: ((func_call) arg1 arg2) */
    if (node->as.call.func_expr) {
        /* Evaluate the inner function call to get the function */
        Value func_val = eval_expression(node->as.call.func_expr, env);
        if (func_val.type != VAL_FUNCTION) {
            fprintf(stderr, "Error: Expression does not return a function\n");
            return create_void();
        }
        
        /* Get the function name from the function value */
        const char *func_name = func_val.as.function_val.function_name;
        if (!func_name) {
            fprintf(stderr, "Error: Cannot get function name from function value\n");
            return create_void();
        }
        
        /* Evaluate arguments */
        Value *args = malloc(sizeof(Value) * node->as.call.arg_count);
        for (int i = 0; i < node->as.call.arg_count; i++) {
            args[i] = eval_expression(node->as.call.args[i], env);
        }
        
        /* Call the function */
        Function *func = env_get_function(env, func_name);
        if (!func) {
            fprintf(stderr, "Error: Function '%s' not found\n", func_name);
            free(args);
            return create_void();
        }
        
        Value result = call_function(func_name, args, node->as.call.arg_count, env);
        free(args);
        return result;
    }
    
    const char *name = node->as.call.name;

    /* Check if the name refers to a function variable (for first-class functions) */
    Symbol *func_var = env_get_var(env, name);
    if (func_var && func_var->value.type == VAL_FUNCTION) {
        /* This is a function value stored in a variable - use its actual function name */
        name = func_var->value.as.function_val.function_name;
    }

    /* Special built-in: range (used in for loops only) */
    if (strcmp(name, "range") == 0) {
        /* This should not be called directly */
        return create_void();
    }

    /* Check for built-in OS functions */
    /* Evaluate arguments first */
    Value args[16];  /* Max args for function calls */
    for (int i = 0; i < node->as.call.arg_count; i++) {
        args[i] = eval_expression(node->as.call.args[i], env);
    }

    /* File operations */
    if (strcmp(name, "file_read") == 0) return builtin_file_read(args);
    if (strcmp(name, "file_read_bytes") == 0) return builtin_file_read_bytes(args);
    if (strcmp(name, "file_write") == 0) return builtin_file_write(args);
    if (strcmp(name, "file_append") == 0) return builtin_file_append(args);
    if (strcmp(name, "file_remove") == 0) return builtin_file_remove(args);
    if (strcmp(name, "file_rename") == 0) return builtin_file_rename(args);
    if (strcmp(name, "file_exists") == 0) return builtin_file_exists(args);
    if (strcmp(name, "file_size") == 0) return builtin_file_size(args);

    /* Temp helpers */
    if (strcmp(name, "tmp_dir") == 0) return builtin_tmp_dir(args);
    if (strcmp(name, "mktemp") == 0) return builtin_mktemp(args);
    if (strcmp(name, "mktemp_dir") == 0) return builtin_mktemp_dir(args);

    /* Directory operations */
    if (strcmp(name, "dir_create") == 0) return builtin_dir_create(args);
    if (strcmp(name, "dir_remove") == 0) return builtin_dir_remove(args);
    if (strcmp(name, "dir_list") == 0) return builtin_dir_list(args);
    if (strcmp(name, "dir_exists") == 0) return builtin_dir_exists(args);
    if (strcmp(name, "getcwd") == 0) return builtin_getcwd(args);
    if (strcmp(name, "chdir") == 0) return builtin_chdir(args);
    if (strcmp(name, "fs_walkdir") == 0) return builtin_fs_walkdir(args);

    /* Path operations */
    if (strcmp(name, "path_isfile") == 0) return builtin_path_isfile(args);
    if (strcmp(name, "path_isdir") == 0) return builtin_path_isdir(args);
    if (strcmp(name, "path_join") == 0) return builtin_path_join(args);
    if (strcmp(name, "path_basename") == 0) return builtin_path_basename(args);
    if (strcmp(name, "path_dirname") == 0) return builtin_path_dirname(args);
    if (strcmp(name, "path_normalize") == 0) return builtin_path_normalize(args);

    /* Process operations */
    if (strcmp(name, "system") == 0) return builtin_system(args);
    if (strcmp(name, "exit") == 0) return builtin_exit(args);
    if (strcmp(name, "getenv") == 0) return builtin_getenv(args);
    if (strcmp(name, "setenv") == 0) return builtin_setenv(args);
    if (strcmp(name, "unsetenv") == 0) return builtin_unsetenv(args);
    if (strcmp(name, "process_run") == 0) return builtin_process_run(args);

    /* Result helpers */
    if (strcmp(name, "result_is_ok") == 0) return builtin_result_is_ok(args);
    if (strcmp(name, "result_is_err") == 0) return builtin_result_is_err(args);
    if (strcmp(name, "result_unwrap") == 0) return builtin_result_unwrap(args);
    if (strcmp(name, "result_unwrap_err") == 0) return builtin_result_unwrap_err(args);
    if (strcmp(name, "result_unwrap_or") == 0) return builtin_result_unwrap_or(args);
    if (strcmp(name, "result_map") == 0) return builtin_result_map(args, env);
    if (strcmp(name, "result_and_then") == 0) return builtin_result_and_then(args, env);

    /* Math and utility functions */
    if (strcmp(name, "abs") == 0) return builtin_abs(args);
    if (strcmp(name, "min") == 0) return builtin_min(args);
    if (strcmp(name, "max") == 0) return builtin_max(args);
    if (strcmp(name, "print") == 0) return builtin_print(args);
    if (strcmp(name, "println") == 0) return builtin_println(args);
    
    /* Advanced math functions */
    if (strcmp(name, "sqrt") == 0) return builtin_sqrt(args);
    if (strcmp(name, "pow") == 0) return builtin_pow(args);
    if (strcmp(name, "floor") == 0) return builtin_floor(args);
    if (strcmp(name, "ceil") == 0) return builtin_ceil(args);
    if (strcmp(name, "round") == 0) return builtin_round(args);
    
    /* Trigonometric functions */
    if (strcmp(name, "sin") == 0) return builtin_sin(args);
    if (strcmp(name, "cos") == 0) return builtin_cos(args);
    if (strcmp(name, "tan") == 0) return builtin_tan(args);
    if (strcmp(name, "atan2") == 0) return builtin_atan2(args);
    
    /* Type casting functions */
    if (strcmp(name, "cast_int") == 0) return builtin_cast_int(args);
    if (strcmp(name, "cast_float") == 0) return builtin_cast_float(args);
    if (strcmp(name, "cast_bool") == 0) return builtin_cast_bool(args);
    if (strcmp(name, "cast_string") == 0) return builtin_cast_string(args);
    if (strcmp(name, "to_string") == 0) return builtin_to_string(args);
    
    /* String operations */
    if (strcmp(name, "str_length") == 0) return builtin_str_length(args);
    if (strcmp(name, "str_concat") == 0) return builtin_str_concat(args);
    if (strcmp(name, "str_substring") == 0) return builtin_str_substring(args);
    if (strcmp(name, "str_contains") == 0) return builtin_str_contains(args);
    if (strcmp(name, "str_equals") == 0) return builtin_str_equals(args);

    /* Bytes helpers */
    if (strcmp(name, "bytes_from_string") == 0) return builtin_bytes_from_string(args);
    if (strcmp(name, "string_from_bytes") == 0) return builtin_string_from_bytes(args);
    
    /* Advanced string operations */
    if (strcmp(name, "char_at") == 0) {
        if (args[0].type != VAL_STRING || args[1].type != VAL_INT) {
            fprintf(stderr, "Error: char_at requires string and int\n");
            return create_void();
        }
        const char *str = args[0].as.string_val;
        long long index = args[1].as.int_val;
        /* Safety: Bound string scan to 1MB */
        int len = strnlen(str, 1024*1024);
        if (index < 0 || index >= len) {
            fprintf(stderr, "Error: Index %lld out of bounds (string length %d)\n", (long long)index, len);
            return create_void();
        }
        return create_int((unsigned char)str[index]);
    }
    
    if (strcmp(name, "string_from_char") == 0) {
        if (args[0].type != VAL_INT) {
            fprintf(stderr, "Error: string_from_char requires int\n");
            return create_void();
        }
        char buffer[2];
        buffer[0] = (char)args[0].as.int_val;
        buffer[1] = '\0';
        return create_string(buffer);
    }
    
    /* Character classification */
    if (strcmp(name, "is_digit") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= '0' && c <= '9');
    }
    
    if (strcmp(name, "is_alpha") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_alnum") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool((c >= '0' && c <= '9') || 
                           (c >= 'a' && c <= 'z') || 
                           (c >= 'A' && c <= 'Z'));
    }
    
    if (strcmp(name, "is_whitespace") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c == ' ' || c == '\t' || c == '\n' || c == '\r');
    }
    
    if (strcmp(name, "is_upper") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'A' && c <= 'Z');
    }
    
    if (strcmp(name, "is_lower") == 0) {
        if (args[0].type != VAL_INT) return create_bool(false);
        int c = (int)args[0].as.int_val;
        return create_bool(c >= 'a' && c <= 'z');
    }
    
    /* Type conversions */
    if (strcmp(name, "int_to_string") == 0) {
        if (args[0].type != VAL_INT) {
            return create_string("0");
        }
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%lld", (long long)args[0].as.int_val);
        return create_string(buffer);
    }
    
    if (strcmp(name, "string_to_int") == 0) {
        if (args[0].type != VAL_STRING) {
            return create_int(0);
        }
        long long result = strtoll(args[0].as.string_val, NULL, 10);
        return create_int(result);
    }
    
    if (strcmp(name, "digit_value") == 0) {
        if (args[0].type != VAL_INT) return create_int(-1);
        int c = (int)args[0].as.int_val;
        if (c >= '0' && c <= '9') {
            return create_int(c - '0');
        }
        return create_int(-1);
    }
    
    if (strcmp(name, "char_to_lower") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'A' && c <= 'Z') {
            return create_int(c + 32);
        }
        return create_int(c);
    }
    
    if (strcmp(name, "char_to_upper") == 0) {
        if (args[0].type != VAL_INT) return create_int(args[0].as.int_val);
        int c = (int)args[0].as.int_val;
        if (c >= 'a' && c <= 'z') {
            return create_int(c - 32);
        }
        return create_int(c);
    }
    
    /* Array operations */
    if (strcmp(name, "at") == 0) return builtin_at(args);
    if (strcmp(name, "array_length") == 0) return builtin_array_length(args);
    if (strcmp(name, "array_new") == 0) return builtin_array_new(args);
    if (strcmp(name, "array_set") == 0) return builtin_array_set(args);
    if (strcmp(name, "array_slice") == 0) return builtin_array_slice(args);
    
    /* Higher-order array functions */
    if (strcmp(name, "map") == 0) return builtin_map(args, env);
    if (strcmp(name, "filter") == 0) return builtin_filter(args, env);
    if (strcmp(name, "reduce") == 0) return builtin_reduce(args, env);
    
    /* Dynamic array operations (GC-managed) */
    if (strcmp(name, "array_push") == 0) return builtin_array_push(args);
    if (strcmp(name, "array_pop") == 0) return builtin_array_pop(args);
    if (strcmp(name, "array_remove_at") == 0) return builtin_array_remove_at(args);
    
    /* list_int operations - delegate to C runtime */
    if (strcmp(name, "list_int_new") == 0) {
        List_int *list = list_int_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_int_with_capacity") == 0) {
        List_int *list = list_int_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_int_push") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_push(list, args[1].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_pop") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_pop(list));
    }
    if (strcmp(name, "list_int_get") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_get(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_set") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_set(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_insert") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_insert(list, args[1].as.int_val, args[2].as.int_val);
        return create_void();
    }
    if (strcmp(name, "list_int_remove") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_remove(list, args[1].as.int_val));
    }
    if (strcmp(name, "list_int_length") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_length(list));
    }
    if (strcmp(name, "list_int_capacity") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_int(list_int_capacity(list));
    }
    if (strcmp(name, "list_int_is_empty") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        return create_bool(list_int_is_empty(list));
    }
    if (strcmp(name, "list_int_clear") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_int_free") == 0) {
        List_int *list = (List_int*)args[0].as.int_val;
        list_int_free(list);
        return create_void();
    }

    /* list_string operations - delegate to C runtime */
    if (strcmp(name, "list_string_new") == 0) {
        List_string *list = list_string_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_string_with_capacity") == 0) {
        List_string *list = list_string_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_string_push") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_push(list, args[1].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_pop") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_pop(list);
        Value result = create_string(str);
        free(str);  /* list_string_pop returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_get") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_get(list, args[1].as.int_val);
        return create_string(str);
    }
    if (strcmp(name, "list_string_set") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_set(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_insert") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_insert(list, args[1].as.int_val, args[2].as.string_val);
        return create_void();
    }
    if (strcmp(name, "list_string_remove") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        char *str = list_string_remove(list, args[1].as.int_val);
        Value result = create_string(str);
        free(str);  /* list_string_remove returns strdup'd string */
        return result;
    }
    if (strcmp(name, "list_string_length") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_length(list));
    }
    if (strcmp(name, "list_string_capacity") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_int(list_string_capacity(list));
    }
    if (strcmp(name, "list_string_is_empty") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        return create_bool(list_string_is_empty(list));
    }
    if (strcmp(name, "list_string_clear") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_string_free") == 0) {
        List_string *list = (List_string*)args[0].as.int_val;
        list_string_free(list);
        return create_void();
    }

    /* list_token operations - delegate to C runtime */
    /* Note: Token structs are stored as pointers for now */
    /* When we rewrite lexer in nanolang, we'll use proper Token struct values */
    if (strcmp(name, "list_token_new") == 0) {
        List_token *list = list_token_new();
        Value result = create_int((long long)list);
        return result;
    }
    if (strcmp(name, "list_token_with_capacity") == 0) {
        List_token *list = list_token_with_capacity(args[0].as.int_val);
        return create_int((long long)list);
    }
    if (strcmp(name, "list_token_push") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        /* For now, args[1] should be a Token struct pointer */
        /* When we have proper Token struct support, this will change */
        Token *token = (Token*)args[1].as.int_val;
        if (token) {
            list_token_push(list, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_pop") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token token = list_token_pop(list);
        /* Return token as struct value - for now return pointer */
        /* TODO: Convert Token to proper struct value when we have Token struct support */
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "list_token_get") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = list_token_get(list, args[1].as.int_val);
        /* Return token pointer for now */
        return create_int((long long)token);
    }
    if (strcmp(name, "list_token_set") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            list_token_set(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_insert") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token *token = (Token*)args[2].as.int_val;
        if (token) {
            list_token_insert(list, args[1].as.int_val, *token);
        }
        return create_void();
    }
    if (strcmp(name, "list_token_remove") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        Token token = list_token_remove(list, args[1].as.int_val);
        Token *token_ptr = malloc(sizeof(Token));
        *token_ptr = token;
        return create_int((long long)token_ptr);
    }
    if (strcmp(name, "list_token_length") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_int(list_token_length(list));
    }
    if (strcmp(name, "list_token_capacity") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_int(list_token_capacity(list));
    }
    if (strcmp(name, "list_token_is_empty") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        return create_bool(list_token_is_empty(list));
    }
    if (strcmp(name, "list_token_clear") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        list_token_clear(list);
        return create_void();
    }
    if (strcmp(name, "list_token_free") == 0) {
        List_token *list = (List_token*)args[0].as.int_val;
        list_token_free(list);
        return create_void();
    }

    /* Generic list functions: list_TypeName_operation for user-defined types */
    /* Pattern: list_ASTNumber_new, list_Point_push, etc. */
    /* For interpreter/shadow tests, we use a simple generic list that stores pointers */
    if (strncmp(name, "list_", 5) == 0) {
        /* Extract the operation: list_TypeName_op -> op */
        const char *last_underscore = strrchr(name, '_');
        if (last_underscore) {
            const char *operation = last_underscore + 1;
            
            /* Use list_int as the underlying implementation (stores pointers as int64) */
            if (strcmp(operation, "new") == 0) {
                List_int *list = list_int_new();  /* Generic list stores pointers */
                return create_int((long long)list);
            }
            if (strcmp(operation, "with_capacity") == 0) {
                List_int *list = list_int_with_capacity(args[0].as.int_val);
                return create_int((long long)list);
            }
            if (strcmp(operation, "push") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                /* Handle different value types */
                if (args[1].type == VAL_STRUCT) {
                    /* Allocate struct on heap and store pointer */
                    StructValue *sv_copy = malloc(sizeof(StructValue));
                    sv_copy->struct_name = strdup(args[1].as.struct_val->struct_name);
                    sv_copy->field_count = args[1].as.struct_val->field_count;
                    sv_copy->field_names = malloc(sizeof(char*) * sv_copy->field_count);
                    sv_copy->field_values = malloc(sizeof(Value) * sv_copy->field_count);
                    for (int i = 0; i < sv_copy->field_count; i++) {
                        sv_copy->field_names[i] = strdup(args[1].as.struct_val->field_names[i]);
                        sv_copy->field_values[i] = args[1].as.struct_val->field_values[i];
                        /* Deep-copy strings so they outlive the caller's stack frame */
                        if (sv_copy->field_values[i].type == VAL_STRING && sv_copy->field_values[i].as.string_val) {
                            sv_copy->field_values[i].as.string_val = strdup(sv_copy->field_values[i].as.string_val);
                        }
                    }
                    list_int_push(list, (int64_t)sv_copy);
                } else {
                    /* For non-struct types, store as int */
                    list_int_push(list, args[1].as.int_val);
                }
                return create_void();
            }
            if (strcmp(operation, "pop") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                int64_t stored_val = list_int_pop(list);
                
                /* Extract type name to determine if this is a struct list */
                const char *type_start = name + 5;  /* Skip "list_" */
                int type_name_len = (int)(last_underscore - type_start);
                char *type_name = malloc(type_name_len + 1);
                strncpy(type_name, type_start, type_name_len);
                type_name[type_name_len] = '\0';
                
                /* Check if this is a struct type */
                if (strcmp(type_name, "int") != 0 && 
                    strcmp(type_name, "string") != 0 && 
                    strcmp(type_name, "token") != 0 &&
                    strcmp(type_name, "Token") != 0) {
                    /* Struct type */
                    StructValue *sv = (StructValue*)stored_val;
                    Value result;
                    result.type = VAL_STRUCT;
                    result.is_return = false;
                    result.as.struct_val = sv;
                    free(type_name);
                    return result;
                } else {
                    free(type_name);
                    return create_int(stored_val);
                }
            }
            if (strcmp(operation, "get") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                int64_t stored_val = list_int_get(list, args[1].as.int_val);
                
                /* Determine if this list stores structs by checking the type name in the function */
                /* Extract type name: list_TypeName_get -> TypeName */
                const char *type_start = name + 5;  /* Skip "list_" */
                int type_name_len = (int)(last_underscore - type_start);
                char *type_name = malloc(type_name_len + 1);
                strncpy(type_name, type_start, type_name_len);
                type_name[type_name_len] = '\0';
                
                /* Check if this type name is a struct (not int/string/token) */
                if (strcmp(type_name, "int") != 0 && 
                    strcmp(type_name, "string") != 0 && 
                    strcmp(type_name, "token") != 0 &&
                    strcmp(type_name, "Token") != 0) {
                    /* This is a struct type - stored value is a pointer to StructValue */
                    StructValue *sv = (StructValue*)stored_val;
                    Value result;
                    result.type = VAL_STRUCT;
                    result.is_return = false;
                    result.as.struct_val = sv;
                    free(type_name);
                    return result;
                } else {
                    /* This is a primitive type */
                    free(type_name);
                    return create_int(stored_val);
                }
            }
            if (strcmp(operation, "set") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                list_int_set(list, args[1].as.int_val, args[2].as.int_val);
                return create_void();
            }
            if (strcmp(operation, "insert") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                list_int_insert(list, args[1].as.int_val, args[2].as.int_val);
                return create_void();
            }
            if (strcmp(operation, "remove") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                return create_int(list_int_remove(list, args[1].as.int_val));
            }
            if (strcmp(operation, "length") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                return create_int(list_int_length(list));
            }
            if (strcmp(operation, "capacity") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                return create_int(list_int_capacity(list));
            }
            if (strcmp(operation, "is_empty") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                return create_bool(list_int_is_empty(list));
            }
            if (strcmp(operation, "clear") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                list_int_clear(list);
                return create_void();
            }
            if (strcmp(operation, "free") == 0) {
                List_int *list = (List_int*)args[0].as.int_val;
                list_int_free(list);
                return create_void();
            }
        }
    }

    /* External C library functions - provide interpreter implementations */
    if (strcmp(name, "rand") == 0) {
        return create_int(rand());
    }
    if (strcmp(name, "srand") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: srand expects 1 int argument\n");
            return create_void();
        }
        srand((unsigned int)args[0].as.int_val);
        return create_void();
    }
    if (strcmp(name, "time") == 0) {
        /* Simplified: ignore the argument, just return current time */
        return create_int((long long)time(NULL));
    }
    
    /* C string functions - map to interpreter built-ins */
    if (strcmp(name, "strlen") == 0) {
        return builtin_str_length(args);
    }
    if (strcmp(name, "strcmp") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_STRING || args[1].type != VAL_STRING) {
            fprintf(stderr, "Error: strcmp requires 2 string arguments\n");
            return create_void();
        }
        int result = strcmp(args[0].as.string_val, args[1].as.string_val);
        return create_int(result);
    }
    if (strcmp(name, "strncmp") == 0) {
        if (node->as.call.arg_count < 3 || args[0].type != VAL_STRING || args[1].type != VAL_STRING || args[2].type != VAL_INT) {
            fprintf(stderr, "Error: strncmp requires 2 string arguments and 1 int argument\n");
            return create_void();
        }
        int n = (int)args[2].as.int_val;
        if (n < 0) n = 0;
        int result = strncmp(args[0].as.string_val, args[1].as.string_val, (size_t)n);
        return create_int(result);
    }
    
    /* C math functions - provide interpreter implementations */
    if (strcmp(name, "asin") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: asin requires 1 float argument\n");
            return create_void();
        }
        return create_float(asin(args[0].as.float_val));
    }
    if (strcmp(name, "acos") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: acos requires 1 float argument\n");
            return create_void();
        }
        return create_float(acos(args[0].as.float_val));
    }
    if (strcmp(name, "atan") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: atan requires 1 float argument\n");
            return create_void();
        }
        return create_float(atan(args[0].as.float_val));
    }
    if (strcmp(name, "exp") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: exp requires 1 float argument\n");
            return create_void();
        }
        return create_float(exp(args[0].as.float_val));
    }
    if (strcmp(name, "exp2") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: exp2 requires 1 float argument\n");
            return create_void();
        }
        return create_float(exp2(args[0].as.float_val));
    }
    if (strcmp(name, "log") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log requires 1 float argument\n");
            return create_void();
        }
        return create_float(log(args[0].as.float_val));
    }
    if (strcmp(name, "log10") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log10 requires 1 float argument\n");
            return create_void();
        }
        return create_float(log10(args[0].as.float_val));
    }
    if (strcmp(name, "log2") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log2 requires 1 float argument\n");
            return create_void();
        }
        return create_float(log2(args[0].as.float_val));
    }
    if (strcmp(name, "log1p") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: log1p requires 1 float argument\n");
            return create_void();
        }
        return create_float(log1p(args[0].as.float_val));
    }
    if (strcmp(name, "expm1") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: expm1 requires 1 float argument\n");
            return create_void();
        }
        return create_float(expm1(args[0].as.float_val));
    }
    if (strcmp(name, "cbrt") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: cbrt requires 1 float argument\n");
            return create_void();
        }
        return create_float(cbrt(args[0].as.float_val));
    }
    if (strcmp(name, "hypot") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: hypot requires 2 float arguments\n");
            return create_void();
        }
        return create_float(hypot(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "sinh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: sinh requires 1 float argument\n");
            return create_void();
        }
        return create_float(sinh(args[0].as.float_val));
    }
    if (strcmp(name, "cosh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: cosh requires 1 float argument\n");
            return create_void();
        }
        return create_float(cosh(args[0].as.float_val));
    }
    if (strcmp(name, "tanh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: tanh requires 1 float argument\n");
            return create_void();
        }
        return create_float(tanh(args[0].as.float_val));
    }
    if (strcmp(name, "asinh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: asinh requires 1 float argument\n");
            return create_void();
        }
        return create_float(asinh(args[0].as.float_val));
    }
    if (strcmp(name, "acosh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: acosh requires 1 float argument\n");
            return create_void();
        }
        return create_float(acosh(args[0].as.float_val));
    }
    if (strcmp(name, "atanh") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: atanh requires 1 float argument\n");
            return create_void();
        }
        return create_float(atanh(args[0].as.float_val));
    }
    if (strcmp(name, "fmod") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmod requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmod(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "trunc") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: trunc requires 1 float argument\n");
            return create_void();
        }
        return create_float(trunc(args[0].as.float_val));
    }
    if (strcmp(name, "rint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: rint requires 1 float argument\n");
            return create_void();
        }
        return create_float(rint(args[0].as.float_val));
    }
    if (strcmp(name, "nearbyint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: nearbyint requires 1 float argument\n");
            return create_void();
        }
        return create_float(nearbyint(args[0].as.float_val));
    }
    if (strcmp(name, "remainder") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: remainder requires 2 float arguments\n");
            return create_void();
        }
        return create_float(remainder(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fmin") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmin requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmin(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fmax") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fmax requires 2 float arguments\n");
            return create_void();
        }
        return create_float(fmax(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "copysign") == 0) {
        if (node->as.call.arg_count < 2 || args[0].type != VAL_FLOAT || args[1].type != VAL_FLOAT) {
            fprintf(stderr, "Error: copysign requires 2 float arguments\n");
            return create_void();
        }
        return create_float(copysign(args[0].as.float_val, args[1].as.float_val));
    }
    if (strcmp(name, "fabs") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_FLOAT) {
            fprintf(stderr, "Error: fabs requires 1 float argument\n");
            return create_void();
        }
        return create_float(fabs(args[0].as.float_val));
    }
    
    /* C character functions */
    if (strcmp(name, "getchar") == 0) {
        int c = getchar();
        return create_int(c);
    }
    if (strcmp(name, "putchar") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: putchar requires 1 int argument\n");
            return create_void();
        }
        int c = putchar((int)args[0].as.int_val);
        return create_int(c);
    }
    if (strcmp(name, "isalpha") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isalpha requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isalpha((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isdigit") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isdigit requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isdigit((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isalnum") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isalnum requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isalnum((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "islower") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: islower requires 1 int argument\n");
            return create_void();
        }
        return create_bool(islower((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isupper") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isupper requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isupper((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "tolower") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: tolower requires 1 int argument\n");
            return create_void();
        }
        return create_int(tolower((int)args[0].as.int_val));
    }
    if (strcmp(name, "toupper") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: toupper requires 1 int argument\n");
            return create_void();
        }
        return create_int(toupper((int)args[0].as.int_val));
    }
    if (strcmp(name, "isspace") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isspace requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isspace((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "isprint") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: isprint requires 1 int argument\n");
            return create_void();
        }
        return create_bool(isprint((int)args[0].as.int_val) != 0);
    }
    if (strcmp(name, "ispunct") == 0) {
        if (node->as.call.arg_count < 1 || args[0].type != VAL_INT) {
            fprintf(stderr, "Error: ispunct requires 1 int argument\n");
            return create_void();
        }
        return create_bool(ispunct((int)args[0].as.int_val) != 0);
    }

    /* Get user-defined function */
    Function *func = env_get_function(env, name);
    
    /* Check if this is a generic list function (List_TypeName_new, List_TypeName_push, etc.) */
    /* This check happens before checking func->body because generic list functions are registered as extern */
    if (!func || (func->is_extern && func->body == NULL && strncmp(name, "List_", 5) == 0)) {
        if (strncmp(name, "List_", 5) == 0) {
            /* Extract element type name and operation from function name */
            /* Format: List_TypeName_new, List_TypeName_push, List_TypeName_get, List_TypeName_length */
            const char *type_start = name + 5;  /* Skip "List_" */
            const char *func_suffix = strrchr(name, '_');
            if (func_suffix && func_suffix > type_start) {
                func_suffix++;  /* Skip '_' */
                int type_name_len = (int)(func_suffix - type_start - 1);
                char *type_name = malloc(type_name_len + 1);
                strncpy(type_name, type_start, type_name_len);
                type_name[type_name_len] = '\0';
                
                /* Check which operation */
                if (strcmp(func_suffix, "new") == 0) {
                    /* List_TypeName_new() -> List<TypeName> */
                    /* Use DynArray with ELEM_INT to store struct pointers as int64_t */
                    DynArray *list = dyn_array_new(ELEM_INT);
                    free(type_name);
                    return create_int((long long)list);
                } else if (strcmp(func_suffix, "push") == 0) {
                    /* List_TypeName_push(list, value) -> void */
                    if (node->as.call.arg_count < 2) {
                        fprintf(stderr, "Error: %s requires 2 arguments\n", name);
                        free(type_name);
                        return create_void();
                    }
                    DynArray *list = (DynArray*)args[0].as.int_val;
                    /* Store struct value as pointer (int64_t) */
                    /* args[1] should be a struct value */
                    if (args[1].type != VAL_STRUCT) {
                        fprintf(stderr, "Error: %s_push expects struct value\n", name);
                        free(type_name);
                        return create_void();
                    }
                    int64_t value_ptr = (int64_t)args[1].as.struct_val;  /* Store StructValue* as int64_t */
                    dyn_array_push_int(list, value_ptr);
                    free(type_name);
                    return create_void();
                } else if (strcmp(func_suffix, "get") == 0) {
                    /* List_TypeName_get(list, index) -> TypeName */
                    if (node->as.call.arg_count < 2) {
                        fprintf(stderr, "Error: %s requires 2 arguments\n", name);
                        free(type_name);
                        return create_void();
                    }
                    DynArray *list = (DynArray*)args[0].as.int_val;
                    int index = (int)args[1].as.int_val;
                    if (index < 0 || index >= list->length) {
                        fprintf(stderr, "Error: Index %d out of bounds\n", index);
                        free(type_name);
                        return create_void();
                    }
                    int64_t value_ptr = dyn_array_get_int(list, index);
                    /* Cast back to StructValue* and return as struct value */
                    StructValue *sv = (StructValue*)value_ptr;
                    Value result;
                    result.type = VAL_STRUCT;
                    result.is_return = false;
                    result.as.struct_val = sv;
                    free(type_name);
                    return result;
                } else if (strcmp(func_suffix, "length") == 0) {
                    /* List_TypeName_length(list) -> int */
                    if (node->as.call.arg_count < 1) {
                        fprintf(stderr, "Error: %s requires 1 argument\n", name);
                        free(type_name);
                        return create_void();
                    }
                    DynArray *list = (DynArray*)args[0].as.int_val;
                    free(type_name);
                    return create_int(list->length);
                }
                free(type_name);
            }
        }
    }
    
    if (!func) {
        fprintf(stderr, "Error: Undefined function '%s'\n", name);
        return create_void();
    }

    /* If built-in with no body, already handled above */
    if (func->body == NULL && !(func->is_extern && strncmp(name, "List_", 5) == 0)) {
        /* Try FFI for extern functions */
        if (func->is_extern && ffi_is_available()) {
            return ffi_call_extern(name, args, node->as.call.arg_count, func, env);
        }
        
        fprintf(stderr, "Error: Built-in function '%s' not implemented in interpreter\n", name);
        return create_void();
    }

    /* Trace function call */
    const char **param_names = NULL;
    if (func->params) {
        param_names = malloc(sizeof(char*) * func->param_count);
        for (int i = 0; i < func->param_count; i++) {
            param_names[i] = func->params[i].name;
        }
    }
    tracing_push_call(name);
    trace_function_call(name, args, node->as.call.arg_count, param_names, 
                        node->line, node->column);
    if (param_names) free(param_names);

    /* Create new environment for function */
    int old_symbol_count = env->symbol_count;

    /* Bind parameters with copies of string values */
    for (int i = 0; i < func->param_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute function body */
    Value result = create_void();
    for (int i = 0; i < func->body->as.block.count; i++) {
        ASTNode *stmt = func->body->as.block.statements[i];
        if (stmt->type == AST_RETURN) {
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            }
            break;
        }
        result = eval_statement(stmt, env);
        /* If statement returned a value (e.g., from if block with return), propagate it */
        if (result.is_return) {
            break;
        }
    }

    /* Pop call stack */
    tracing_pop_call();

    /*
     * Make a copy of the result BEFORE cleaning up parameters.
     *
     * Function-local variables (including string temporaries) are freed when we
     * unwind the call frame, so any returned value that references them must
     * deep-copy those strings.
     */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    } else if (result.type == VAL_STRUCT && result.as.struct_val) {
        StructValue *src = result.as.struct_val;
        StructValue *dst = malloc(sizeof(StructValue));
        if (!dst) {
            fprintf(stderr, "Error: Out of memory copying struct return value\n");
            exit(1);
        }
        dst->struct_name = strdup(src->struct_name);
        dst->field_count = src->field_count;
        dst->field_names = malloc(sizeof(char*) * dst->field_count);
        dst->field_values = malloc(sizeof(Value) * dst->field_count);
        for (int i = 0; i < dst->field_count; i++) {
            dst->field_names[i] = strdup(src->field_names[i]);
            dst->field_values[i] = src->field_values[i];
            if (dst->field_values[i].type == VAL_STRING && dst->field_values[i].as.string_val) {
                dst->field_values[i].as.string_val = strdup(dst->field_values[i].as.string_val);
            }
        }

        return_value.type = VAL_STRUCT;
        return_value.is_return = false;
        return_value.as.struct_val = dst;
    }

    /* Clean up parameter strings and restore environment */
    for (int i = old_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
        if (env->symbols[i].value.type == VAL_FUNCTION) {
            /* Free function value - both function_name and signature */
            if (env->symbols[i].value.as.function_val.function_name) {
                free((char*)env->symbols[i].value.as.function_val.function_name);
            }
            if (env->symbols[i].value.as.function_val.signature) {
                free_function_signature(env->symbols[i].value.as.function_val.signature);
            }
        }
    }
    env->symbol_count = old_symbol_count;

    return return_value;
}

/* Evaluate expression */
static Value eval_expression(ASTNode *expr, Environment *env) {
    if (!expr) return create_void();


    switch (expr->type) {
        case AST_NUMBER:
            return create_int(expr->as.number);

        case AST_FLOAT:
            return create_float(expr->as.float_val);

        case AST_STRING:
            return create_string(expr->as.string_val);

        case AST_BOOL:
            return create_bool(expr->as.bool_val);

        case AST_IDENTIFIER: {
            /* First check if it's a variable */
            Symbol *sym = env_get_var(env, expr->as.identifier);
            if (sym) {
                /* Trace variable read */
#ifdef TRACING_ENABLED
                const char *scope = (g_tracing_config.call_stack_size > 0) ?
                    g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
                trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, scope);
#else
                trace_var_read(expr->as.identifier, sym->value, expr->line, expr->column, NULL);
#endif
                return sym->value;
            }
            
            /* If not a variable, check if it's a function (for first-class function support) */
            Function *func = env_get_function(env, expr->as.identifier);
            if (func) {
                /* Return a function value - create signature from function's parameters */
                Type *param_types = NULL;
                int actual_param_count = 0;
                if (func->param_count > 0 && func->params) {
                    param_types = malloc(sizeof(Type) * func->param_count);
                    for (int i = 0; i < func->param_count; i++) {
                        param_types[i] = func->params[i].type;
                    }
                    actual_param_count = func->param_count;
                }
                FunctionSignature *sig = create_function_signature(
                    param_types,
                    actual_param_count,
                    func->return_type
                );
                return create_function(expr->as.identifier, sig);
            }
            
            /* Neither variable nor function */
            fprintf(stderr, "Error: Undefined variable or function '%s'\n", expr->as.identifier);
            return create_void();
        }

        case AST_PREFIX_OP:
            return eval_prefix_op(expr, env);

        case AST_CALL:
            return eval_call(expr, env);

        case AST_ARRAY_LITERAL: {
            /* Evaluate array literal: [1, 2, 3] */
            int count = expr->as.array_literal.element_count;
            
            /* Empty array */
            if (count == 0) {
                /* Create empty array - type will be determined by context */
                return create_array(VAL_INT, 0, 0);  /* Default to int for now */
            }
            
            /* Evaluate first element to determine type */
            Value first = eval_expression(expr->as.array_literal.elements[0], env);
            ValueType elem_type = first.type;
            
            /* Create array */
            Value arr = create_array(elem_type, count, count);
            
            /* Set elements */
            for (int i = 0; i < count; i++) {
                Value elem = eval_expression(expr->as.array_literal.elements[i], env);
                
                /* Store element in array data */
                switch (elem_type) {
                    case VAL_INT:
                        ((long long*)arr.as.array_val->data)[i] = elem.as.int_val;
                        break;
                    case VAL_FLOAT:
                        ((double*)arr.as.array_val->data)[i] = elem.as.float_val;
                        break;
                    case VAL_BOOL:
                        ((bool*)arr.as.array_val->data)[i] = elem.as.bool_val;
                        break;
                    case VAL_STRING:
                        ((char**)arr.as.array_val->data)[i] = strdup(elem.as.string_val);
                        break;
                    default:
                        fprintf(stderr, "Error: Unsupported array element type\n");
                        break;
                }
            }
            
            return arr;
        }

        case AST_IF: {
            Value cond = eval_expression(expr->as.if_stmt.condition, env);
            if (is_truthy(cond)) {
                return eval_statement(expr->as.if_stmt.then_branch, env);
            } else {
                return eval_statement(expr->as.if_stmt.else_branch, env);
            }
        }

        case AST_STRUCT_LITERAL: {
            /* Evaluate struct literal: Point { x: 10, y: 20 } */
            
            const char *struct_name = expr->as.struct_literal.struct_name;
            int field_count = expr->as.struct_literal.field_count;
            
            
            /* Get struct definition to verify field order */
            StructDef *struct_def = env_get_struct(env, struct_name);
            if (!struct_def) {
                fprintf(stderr, "Error: Undefined struct '%s'\n", struct_name);
                return create_void();
            }
            
            
            /* Allocate arrays for field names and values */
            char **field_names = malloc(sizeof(char*) * field_count);
            Value *field_values = malloc(sizeof(Value) * field_count);
            
            
            /* Evaluate each field value */
            for (int i = 0; i < field_count; i++) {
                field_names[i] = expr->as.struct_literal.field_names[i];
                field_values[i] = eval_expression(expr->as.struct_literal.field_values[i], env);
            }
            
            
            /* Create struct value */
            Value result = create_struct(struct_name, field_names, field_values, field_count);
            
            
            /* Free temporary arrays (create_struct makes copies) */
            free(field_names);
            free(field_values);
            
            
            return result;
        }

        case AST_FIELD_ACCESS: {
            /* Check object is not NULL */
            if (!expr->as.field_access.object) {
                fprintf(stderr, "Error: NULL object in field access\n");
                return create_void();
            }
            
            /* Special case: Check if this is an enum variant access */
            if (expr->as.field_access.object->type == AST_IDENTIFIER) {
                const char *enum_name = expr->as.field_access.object->as.identifier;
                assert(enum_name != NULL);
                if (!enum_name) {
                    safe_fprintf(stderr, "Error: NULL enum name in field access\n");
                    return create_void();
                }
                EnumDef *enum_def = env_get_enum(env, enum_name);
                
                if (enum_def && enum_def->variant_names) {
                    /* This is an enum variant access (e.g., Color.Red) */
                    const char *variant_name = expr->as.field_access.field_name;
                    
                    assert(variant_name != NULL);
                    if (!variant_name) {
                        safe_fprintf(stderr, "Error: NULL variant name in enum access\n");
                        return create_void();
                    }
                    
                    /* Lookup variant value */
                    for (int i = 0; i < enum_def->variant_count; i++) {
                        if (safe_strcmp(enum_def->variant_names[i], variant_name) == 0) {
                            return create_int(enum_def->variant_values ? enum_def->variant_values[i] : i);
                        }
                    }
                    
                    safe_fprintf(stderr, "Error: Enum '%s' has no variant '%s'\n",
                            safe_format_string(enum_name), safe_format_string(variant_name));
                    return create_void();
                }
            }
            
            /* Regular struct field access */
            /* Evaluate field access: point.x */
            Value obj = eval_expression(expr->as.field_access.object, env);
            
            if (obj.type != VAL_STRUCT) {
                fprintf(stderr, "Error: Cannot access field on non-struct value\n");
                return create_void();
            }
            
            const char *field_name = expr->as.field_access.field_name;
            StructValue *sv = obj.as.struct_val;
            
            /* Find field in struct */
            for (int i = 0; i < sv->field_count; i++) {
                if (strcmp(sv->field_names[i], field_name) == 0) {
                    return sv->field_values[i];
                }
            }
            
            fprintf(stderr, "Error: Struct '%s' has no field '%s'\n", 
                    sv->struct_name, field_name);
            return create_void();
        }

        case AST_UNION_CONSTRUCT: {
            /* Evaluate union construction: Status.Ok {} or Result.Error { code: 404 } */
            const char *union_name = expr->as.union_construct.union_name;
            const char *variant_name = expr->as.union_construct.variant_name;
            
            /* Get variant index */
            int variant_idx = env_get_union_variant_index(env, union_name, variant_name);
            if (variant_idx < 0) {
                fprintf(stderr, "Error: Unknown variant '%s' for union '%s'\n", variant_name, union_name);
                return create_void();
            }
            
            /* Evaluate field values */
            int field_count = expr->as.union_construct.field_count;
            char **field_names = NULL;
            Value *field_values = NULL;
            
            if (field_count > 0) {
                field_names = malloc(sizeof(char*) * field_count);
                field_values = malloc(sizeof(Value) * field_count);
                
                for (int i = 0; i < field_count; i++) {
                    field_names[i] = expr->as.union_construct.field_names[i];
                    field_values[i] = eval_expression(expr->as.union_construct.field_values[i], env);
                }
            }
            
            Value result = create_union(union_name, variant_idx, variant_name, 
                                       field_names, field_values, field_count);
            
            /* Free temporary arrays (create_union makes copies) */
            if (field_count > 0) {
                free(field_names);
                free(field_values);
            }
            
            return result;
        }

        case AST_MATCH: {
            /* Evaluate match expression: match status { Ok(x) => 1, Error(e) => 0 } */
            Value match_val = eval_expression(expr->as.match_expr.expr, env);
            
            if (match_val.type != VAL_UNION) {
                fprintf(stderr, "Error: Match expression requires a union value (got type %d)\n", match_val.type);
                return create_void();
            }
            
            UnionValue *uval = match_val.as.union_val;
            
            /* Find matching arm by comparing variant names */
            for (int i = 0; i < expr->as.match_expr.arm_count; i++) {
                const char *pattern_variant = expr->as.match_expr.pattern_variants[i];
                
                if (strcmp(uval->variant_name, pattern_variant) == 0) {
                    /* This arm matches! */
                    const char *binding = expr->as.match_expr.pattern_bindings[i];
                    
                    /* Save environment state for scope */
                    int saved_symbol_count = env->symbol_count;
                    
                    /* Bind the pattern variable to a struct value representing the variant's fields
                     * This allows field access like binding.field_name in the match arm body
                     */
                    Value binding_val;
                    if (uval->field_count > 0) {
                        /* Create a struct-like value with the variant's fields
                         * We need to duplicate the field names and values for the struct
                         */
                        char **field_names_copy = malloc(sizeof(char*) * uval->field_count);
                        Value *field_values_copy = malloc(sizeof(Value) * uval->field_count);
                        
                        for (int j = 0; j < uval->field_count; j++) {
                            field_names_copy[j] = uval->field_names[j];  /* Share string pointers */
                            field_values_copy[j] = uval->field_values[j];  /* Copy values */
                        }
                        
                        binding_val = create_struct(uval->union_name, 
                                                   field_names_copy, 
                                                   field_values_copy, 
                                                   uval->field_count);
                    } else {
                        /* Variant has no fields - create a placeholder */
                        binding_val = create_void();
                    }
                    env_define_var(env, binding, TYPE_STRUCT, false, binding_val);
                    
                    /* Evaluate arm body */
                    Value result = eval_expression(expr->as.match_expr.arm_bodies[i], env);
                    
                    /* Restore environment */
                    /* Note: Symbols added here will be leaked, but interpreter is short-lived */
                    env->symbol_count = saved_symbol_count;
                    
                    return result;
                }
            }
            
            /* No matching arm found - this should be caught by typechecker */
            fprintf(stderr, "Error: No matching arm for variant '%s'\n", uval->variant_name);
            return create_void();
        }

        case AST_BLOCK: {
            /* Blocks can be used as expressions in match arms
             * Execute statements and return the last return value
             */
            Value result = create_void();
            for (int i = 0; i < expr->as.block.count; i++) {
                result = eval_statement(expr->as.block.statements[i], env);
                /* If statement returned a value, propagate it immediately */
                if (result.is_return) {
                    /* Clear the return flag since we're handling it */
                    result.is_return = false;
                    return result;
                }
            }
            return result;
        }

        case AST_RETURN: {
            /* Return statements can appear in blocks that are used as expressions */
            Value result;
            if (expr->as.return_stmt.value) {
                result = eval_expression(expr->as.return_stmt.value, env);
            } else {
                result = create_void();
            }
            /* Don't set is_return flag here - let the block handler deal with it */
            return result;
        }

        case AST_TUPLE_LITERAL: {
            /* Evaluate tuple literal: (1, "hello", true) */
            int element_count = expr->as.tuple_literal.element_count;
            
            /* Empty tuple */
            if (element_count == 0) {
                return create_tuple(NULL, 0);
            }
            
            /* Evaluate each element */
            Value *elements = malloc(sizeof(Value) * element_count);
            for (int i = 0; i < element_count; i++) {
                elements[i] = eval_expression(expr->as.tuple_literal.elements[i], env);
            }
            
            /* Create tuple value */
            Value result = create_tuple(elements, element_count);
            free(elements);  /* create_tuple makes a copy */
            
            return result;
        }

        case AST_TUPLE_INDEX: {
            /* Evaluate tuple index access: tuple.0, tuple.1 */
            Value tuple = eval_expression(expr->as.tuple_index.tuple, env);
            
            if (tuple.type != VAL_TUPLE) {
                fprintf(stderr, "Error: Tuple index access on non-tuple value (type %d)\n", tuple.type);
                return create_void();
            }
            
            int index = expr->as.tuple_index.index;
            TupleValue *tv = tuple.as.tuple_val;
            
            if (index < 0 || index >= tv->element_count) {
                fprintf(stderr, "Error: Tuple index %d out of bounds (tuple has %d elements)\n",
                        index, tv->element_count);
                return create_void();
            }
            
            return tv->elements[index];
        }

        default:
            return create_void();
    }
}

/* Evaluate statement */
static Value eval_statement(ASTNode *stmt, Environment *env) {
    if (!stmt) return create_void();


    switch (stmt->type) {
        case AST_LET: {
            Value value = eval_expression(stmt->as.let.value, env);
            env_define_var(env, stmt->as.let.name, stmt->as.let.var_type, stmt->as.let.is_mut, value);
            
            /* Trace variable declaration */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, scope);
#else
            trace_var_decl(stmt->as.let.name, stmt->as.let.var_type, value, 
                          stmt->as.let.is_mut, stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_SET: {
            Value value = eval_expression(stmt->as.set.value, env);
            env_set_var(env, stmt->as.set.name, value);
            
            /* Trace variable assignment */
#ifdef TRACING_ENABLED
            const char *scope = (g_tracing_config.call_stack_size > 0) ?
                g_tracing_config.call_stack[g_tracing_config.call_stack_size - 1] : NULL;
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, scope);
#else
            trace_var_set(stmt->as.set.name, old_value, value, 
                         stmt->line, stmt->column, NULL);
#endif
            
            return create_void();
        }

        case AST_WHILE: {
            Value result = create_void();
            while (is_truthy(eval_expression(stmt->as.while_stmt.condition, env))) {
                result = eval_statement(stmt->as.while_stmt.body, env);
                /* If body returned a value, propagate it immediately */
                if (result.is_return) {
                    return result;
                }
            }
            return result;
        }

        case AST_FOR: {
            /* Evaluate range */
            ASTNode *range_expr = stmt->as.for_stmt.range_expr;
            if (range_expr->type != AST_CALL || strcmp(range_expr->as.call.name, "range") != 0) {
                fprintf(stderr, "Error: for loop requires range expression\n");
                return create_void();
            }

            if (range_expr->as.call.arg_count != 2) {
                fprintf(stderr, "Error: range requires 2 arguments\n");
                return create_void();
            }

            Value start_val = eval_expression(range_expr->as.call.args[0], env);
            Value end_val = eval_expression(range_expr->as.call.args[1], env);

            if (start_val.type != VAL_INT || end_val.type != VAL_INT) {
                fprintf(stderr, "Error: range requires int arguments\n");
                return create_void();
            }

            long long start = start_val.as.int_val;
            long long end = end_val.as.int_val;

            /* Define loop variable before the loop */
            int loop_var_index = env->symbol_count;
            env_define_var(env, stmt->as.for_stmt.var_name, TYPE_INT, false, create_int(start));

            Value result = create_void();
            for (long long i = start; i < end; i++) {
                /* Update loop variable value */
                env->symbols[loop_var_index].value = create_int(i);

                /* Execute loop body */
                result = eval_statement(stmt->as.for_stmt.body, env);
                
                /* If body returned a value, propagate it immediately */
                if (result.is_return) {
                    env->symbol_count = loop_var_index;  /* Clean up before return */
                    return result;
                }
            }

            /* Remove loop variable from scope */
            env->symbol_count = loop_var_index;

            return result;
        }

        case AST_RETURN: {
            Value result;
            if (stmt->as.return_stmt.value) {
                result = eval_expression(stmt->as.return_stmt.value, env);
            } else {
                result = create_void();
            }
            result.is_return = true;  /* Mark as return value */
            return result;
        }

        case AST_BLOCK: {
            Value result = create_void();
            for (int i = 0; i < stmt->as.block.count; i++) {
                result = eval_statement(stmt->as.block.statements[i], env);
                /* If statement returned a value, propagate it immediately */
                if (result.is_return) {
                    return result;
                }
            }
            return result;
        }

        case AST_PRINT: {
            Value value = eval_expression(stmt->as.print.expr, env);
            print_value(value);
            printf("\n");
            return create_void();
        }

        case AST_ASSERT: {
            Value cond = eval_expression(stmt->as.assert.condition, env);
            if (!is_truthy(cond)) {
                fprintf(stderr, "Assertion failed at line %d, column %d\n", stmt->line, stmt->column);
                exit(1);
            }
            return create_void();
        }

        case AST_STRUCT_DEF:
            /* Struct definitions are handled at program level (typechecker) */
            return create_void();
        
        case AST_ENUM_DEF: {
            /* Register enum in interpreter environment for enum variant access */
            if (!stmt->as.enum_def.name) {
                fprintf(stderr, "Error: Enum definition has NULL name\n");
                return create_void();
            }
            
            EnumDef edef;
            edef.name = strdup(stmt->as.enum_def.name);
            edef.variant_count = stmt->as.enum_def.variant_count;
            
            if (edef.variant_count <= 0) {
                fprintf(stderr, "Error: Enum '%s' has invalid variant count: %d\n", edef.name, edef.variant_count);
                free(edef.name);
                return create_void();
            }
            
            /* Duplicate variant names */
            edef.variant_names = malloc(sizeof(char*) * edef.variant_count);
            if (!edef.variant_names) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant names\n");
                free(edef.name);
                return create_void();
            }
            
            for (int j = 0; j < edef.variant_count; j++) {
                if (stmt->as.enum_def.variant_names && stmt->as.enum_def.variant_names[j]) {
                    edef.variant_names[j] = strdup(stmt->as.enum_def.variant_names[j]);
                    if (!edef.variant_names[j]) {
                        fprintf(stderr, "Error: Failed to duplicate variant name at index %d\n", j);
                        edef.variant_names[j] = NULL;
                    }
                } else {
                    fprintf(stderr, "Warning: Enum '%s' has NULL variant name at index %d\n", edef.name, j);
                    edef.variant_names[j] = NULL;
                }
            }
            
            /* Duplicate variant values */
            edef.variant_values = malloc(sizeof(int) * edef.variant_count);
            if (!edef.variant_values) {
                fprintf(stderr, "Error: Failed to allocate memory for enum variant values\n");
                free(edef.name);
                for (int j = 0; j < edef.variant_count; j++) {
                    free(edef.variant_names[j]);
                }
                free(edef.variant_names);
                return create_void();
            }
            
            if (stmt->as.enum_def.variant_values) {
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = stmt->as.enum_def.variant_values[j];
                }
            } else {
                /* No explicit values - use index as value */
                for (int j = 0; j < edef.variant_count; j++) {
                    edef.variant_values[j] = j;
                }
            }
            
            env_define_enum(env, edef);
            return create_void();
        }
        
        case AST_UNION_DEF: {
            /* Register union in interpreter environment for union construction */
            if (!stmt->as.union_def.name) {
                fprintf(stderr, "Error: Union definition has NULL name\n");
                return create_void();
            }
            
            UnionDef udef;
            udef.name = strdup(stmt->as.union_def.name);
            udef.variant_count = stmt->as.union_def.variant_count;
            
            /* Copy generic parameters */
            udef.generic_param_count = stmt->as.union_def.generic_param_count;
            if (udef.generic_param_count > 0) {
                udef.generic_params = malloc(sizeof(char*) * udef.generic_param_count);
                for (int j = 0; j < udef.generic_param_count; j++) {
                    udef.generic_params[j] = strdup(stmt->as.union_def.generic_params[j]);
                }
            } else {
                udef.generic_params = NULL;
            }
            udef.is_pub = stmt->as.union_def.is_pub;
            udef.module_name = NULL;
            
            if (udef.variant_count <= 0) {
                fprintf(stderr, "Error: Union '%s' has invalid variant count: %d\n", udef.name, udef.variant_count);
                free(udef.name);
                return create_void();
            }
            
            /* Duplicate variant names */
            udef.variant_names = malloc(sizeof(char*) * udef.variant_count);
            udef.variant_field_counts = malloc(sizeof(int) * udef.variant_count);
            udef.variant_field_names = malloc(sizeof(char**) * udef.variant_count);
            udef.variant_field_types = malloc(sizeof(Type*) * udef.variant_count);
            
            for (int j = 0; j < udef.variant_count; j++) {
                udef.variant_names[j] = strdup(stmt->as.union_def.variant_names[j]);
                udef.variant_field_counts[j] = stmt->as.union_def.variant_field_counts[j];
                
                /* Duplicate field names and types for this variant */
                int field_count = udef.variant_field_counts[j];
                if (field_count > 0) {
                    udef.variant_field_names[j] = malloc(sizeof(char*) * field_count);
                    udef.variant_field_types[j] = malloc(sizeof(Type) * field_count);
                    
                    for (int k = 0; k < field_count; k++) {
                        udef.variant_field_names[j][k] = strdup(stmt->as.union_def.variant_field_names[j][k]);
                        udef.variant_field_types[j][k] = stmt->as.union_def.variant_field_types[j][k];
                    }
                } else {
                    udef.variant_field_names[j] = NULL;
                    udef.variant_field_types[j] = NULL;
                }
            }
            
            env_define_union(env, udef);
            return create_void();
        }
        
        case AST_FUNCTION:
        case AST_SHADOW:
            /* Function and shadow definitions are handled at program level */
            return create_void();

        default:
            /* Expression statements */
            return eval_expression(stmt, env);
    }
}

/* Check if an AST node contains calls to extern functions */
static bool contains_extern_calls(ASTNode *node, Environment *env) {
    if (!node) return false;
    
    switch (node->type) {
        case AST_CALL: {
            const char *func_name = node->as.call.name;
            Function *func = env_get_function(env, func_name);
            if (func && func->is_extern) {
                return true;
            }
            /* Check arguments recursively */
            for (int i = 0; i < node->as.call.arg_count; i++) {
                if (contains_extern_calls(node->as.call.args[i], env)) {
                    return true;
                }
            }
            return false;
        }
        case AST_BLOCK:
            for (int i = 0; i < node->as.block.count; i++) {
                if (contains_extern_calls(node->as.block.statements[i], env)) {
                    return true;
                }
            }
            return false;
        case AST_IF:
            if (contains_extern_calls(node->as.if_stmt.condition, env)) return true;
            if (contains_extern_calls(node->as.if_stmt.then_branch, env)) return true;
            if (node->as.if_stmt.else_branch && contains_extern_calls(node->as.if_stmt.else_branch, env)) return true;
            return false;
        case AST_WHILE:
            if (contains_extern_calls(node->as.while_stmt.condition, env)) return true;
            if (contains_extern_calls(node->as.while_stmt.body, env)) return true;
            return false;
        case AST_RETURN:
            if (node->as.return_stmt.value && contains_extern_calls(node->as.return_stmt.value, env)) return true;
            return false;
        case AST_PREFIX_OP:
            for (int i = 0; i < node->as.prefix_op.arg_count; i++) {
                if (contains_extern_calls(node->as.prefix_op.args[i], env)) return true;
            }
            return false;
        case AST_ARRAY_LITERAL:
            for (int i = 0; i < node->as.array_literal.element_count; i++) {
                if (contains_extern_calls(node->as.array_literal.elements[i], env)) return true;
            }
            return false;
        case AST_FIELD_ACCESS:
            return contains_extern_calls(node->as.field_access.object, env);
        case AST_LET:
            return contains_extern_calls(node->as.let.value, env);
        case AST_SET:
            return contains_extern_calls(node->as.set.value, env);
        default:
            return false;
    }
}

/* Run shadow tests */
bool run_shadow_tests(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program for shadow tests\n");
        return false;
    }

    /* Shadow test output goes to stdout (filtered by test scripts) */
    fprintf(stdout, "Running shadow tests...\n");

    bool all_passed = true;

    /* First pass: Evaluate top-level constants */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_LET) {
            eval_statement(item, env);  /* Evaluate the constant */
        }
    }

    /* Second pass: Register all enum definitions so they're available in shadow tests */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_ENUM_DEF) {
            eval_statement(item, env);  /* This will register the enum */
        }
    }

    /* Third pass: Register all union definitions so they're available in shadow tests */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_UNION_DEF) {
            eval_statement(item, env);  /* This will register the union */
        }
    }

    /* Fourth pass: Run each shadow test */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];
        
        if (item->type == AST_SHADOW) {
            const char *func_name = item->as.shadow.function_name;
            Function *func = env_get_function(env, func_name);
            
            /* Check if shadow test or function body uses extern functions */
            bool uses_extern = false;
            if (func && func->body && contains_extern_calls(func->body, env)) {
                uses_extern = true;
            }
            if (contains_extern_calls(item->as.shadow.body, env)) {
                uses_extern = true;
            }
            
            if (uses_extern) {
                fprintf(stdout, "Testing %s... SKIPPED (uses extern functions)\n", func_name);
                /* Note: Shadow tests with extern functions are skipped in interpreter mode.
                 * These tests work fine when compiled (nanoc), but the interpreter cannot
                 * execute them. Consider testing extern functionality via compiled tests. */
                continue;
            }
            
            fprintf(stdout, "Testing %s... ", func_name);
            
            /* Execute shadow test */
            eval_statement(item->as.shadow.body, env);

            fprintf(stdout, "PASSED\n");
        }
        /* Note: We do NOT execute non-shadow items here - they're already registered
         * in the environment by the type checker. Only shadow test bodies need execution. */
    }

    if (all_passed) {
        fprintf(stdout, "All shadow tests passed!\n");
    }

    return all_passed;
}

/* Run the entire program (interpreter mode) */
bool run_program(ASTNode *program, Environment *env) {
    if (!program || program->type != AST_PROGRAM) {
        fprintf(stderr, "Error: Invalid program\n");
        return false;
    }

    /* First pass: evaluate top-level constants before functions */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Evaluate top-level constants */
        if (item->type == AST_LET) {
            eval_statement(item, env);
        }
    }

    /* Second pass: execute all other top-level items (functions, statements, etc.) */
    for (int i = 0; i < program->as.program.count; i++) {
        ASTNode *item = program->as.program.items[i];

        /* Skip constants - already processed */
        if (item->type == AST_LET) {
            continue;
        }

        /* Skip shadow tests in interpreter mode - they're for compiler validation */
        if (item->type == AST_SHADOW) {
            continue;
        }

        /* Skip imports - they're handled separately before execution */
        if (item->type == AST_IMPORT) {
            continue;
        }

        /* Execute the item */
        eval_statement(item, env);
    }

    return true;
}

/* Call a function by name with arguments */
Value call_function(const char *name, Value *args, int arg_count, Environment *env) {
    /* Check if this is a generic list function (List_TypeName_new, List_TypeName_push, etc.) */
    if (strncmp(name, "List_", 5) == 0) {
        /* Extract element type name and operation from function name */
        /* Format: List_TypeName_new, List_TypeName_push, List_TypeName_get, List_TypeName_length */
        const char *type_start = name + 5;  /* Skip "List_" */
        const char *func_suffix = strrchr(name, '_');
        if (func_suffix && func_suffix > type_start) {
            func_suffix++;  /* Skip '_' */
            int type_name_len = (int)(func_suffix - type_start - 1);
            char *type_name = malloc(type_name_len + 1);
            strncpy(type_name, type_start, type_name_len);
            type_name[type_name_len] = '\0';
            
            /* Check which operation */
            if (strcmp(func_suffix, "new") == 0) {
                /* List_TypeName_new() -> List<TypeName> */
                /* Use DynArray with ELEM_INT to store struct pointers as int64_t */
                DynArray *list = dyn_array_new(ELEM_INT);
                free(type_name);
                return create_int((long long)list);
            } else if (strcmp(func_suffix, "push") == 0) {
                /* List_TypeName_push(list, value) -> void */
                if (arg_count < 2) {
                    fprintf(stderr, "Error: %s requires 2 arguments\n", name);
                    free(type_name);
                    return create_void();
                }
                DynArray *list = (DynArray*)args[0].as.int_val;
                /* Store struct value as pointer (int64_t) */
                /* args[1] should be a struct value */
                if (args[1].type != VAL_STRUCT) {
                    fprintf(stderr, "Error: %s_push expects struct value\n", name);
                    free(type_name);
                    return create_void();
                }
                int64_t value_ptr = (int64_t)args[1].as.struct_val;  /* Store StructValue* as int64_t */
                dyn_array_push_int(list, value_ptr);
                free(type_name);
                return create_void();
            } else if (strcmp(func_suffix, "get") == 0) {
                /* List_TypeName_get(list, index) -> TypeName */
                if (arg_count < 2) {
                    fprintf(stderr, "Error: %s requires 2 arguments\n", name);
                    free(type_name);
                    return create_void();
                }
                DynArray *list = (DynArray*)args[0].as.int_val;
                int index = (int)args[1].as.int_val;
                if (index < 0 || index >= list->length) {
                    fprintf(stderr, "Error: Index %d out of bounds\n", index);
                    free(type_name);
                    return create_void();
                }
                int64_t value_ptr = dyn_array_get_int(list, index);
                /* Cast back to StructValue* and return as struct value */
                StructValue *sv = (StructValue*)value_ptr;
                Value result;
                result.type = VAL_STRUCT;
                result.is_return = false;
                result.as.struct_val = sv;
                free(type_name);
                return result;
            } else if (strcmp(func_suffix, "length") == 0) {
                /* List_TypeName_length(list) -> int */
                if (arg_count < 1) {
                    fprintf(stderr, "Error: %s requires 1 argument\n", name);
                    free(type_name);
                    return create_void();
                }
                DynArray *list = (DynArray*)args[0].as.int_val;
                free(type_name);
                return create_int(list->length);
            }
            free(type_name);
        }
    }
    
    Function *func = env_get_function(env, name);
    if (!func) {
        fprintf(stderr, "Error: Function '%s' not found\n", name);
        return create_void();
    }

    /* Check argument count */
    if (arg_count != func->param_count) {
        fprintf(stderr, "Error: Function '%s' expects %d arguments, got %d\n",
                name, func->param_count, arg_count);
        return create_void();
    }

    /* Save original symbol count to restore environment after function call */
    int original_symbol_count = env->symbol_count;

    /* Add function parameters to environment with copies of string values */
    for (int i = 0; i < arg_count; i++) {
        Value param_value = args[i];

        /* Make a deep copy of string values to avoid memory corruption */
        if (param_value.type == VAL_STRING) {
            param_value = create_string(args[i].as.string_val);
        }

        env_define_var(env, func->params[i].name, func->params[i].type, false, param_value);
    }

    /* Execute the function body */
    Value result = eval_statement(func->body, env);

    /* Make a copy of the result if it's a string BEFORE cleaning up parameters */
    Value return_value = result;
    if (result.type == VAL_STRING) {
        return_value = create_string(result.as.string_val);
    }
    
    /* Clear is_return flag - we've exited the function */
    return_value.is_return = false;

    /* Clean up parameter strings and restore environment */
    for (int i = original_symbol_count; i < env->symbol_count; i++) {
        free(env->symbols[i].name);
        if (env->symbols[i].value.type == VAL_STRING) {
            free(env->symbols[i].value.as.string_val);
        }
    }
    env->symbol_count = original_symbol_count;

    return return_value;
}