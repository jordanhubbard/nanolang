/* eval_io.c - IO and file system built-in functions for interpreter
 * Extracted from eval.c for better organization
 */

#define _POSIX_C_SOURCE 200809L

#include "eval_io.h"
#include "../nanolang.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <libgen.h>
#include <time.h>
#include <assert.h>
#include <spawn.h>
#include <sys/wait.h>

/* Helper to create a dynamic array Value */
static Value create_dyn_array(DynArray *arr) {
    Value val;
    val.type = VAL_DYN_ARRAY;
    val.is_return = false;
    val.is_break = false;
    val.is_continue = false;
    val.as.dyn_array_val = arr;
    return val;
}

/* ==========================================================================
 * Built-in OS Functions Implementation
 * ========================================================================== */

/* File Operations */
Value builtin_file_read(Value *args) {
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

Value builtin_file_read_bytes(Value *args) {
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

Value builtin_bytes_from_string(Value *args) {
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

Value builtin_string_from_bytes(Value *args) {
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

Value builtin_file_write(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "w");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

Value builtin_file_append(Value *args) {
    const char *path = args[0].as.string_val;
    const char *content = args[1].as.string_val;
    FILE *f = fopen(path, "a");
    if (!f) return create_int(-1);

    fputs(content, f);
    fclose(f);
    return create_int(0);
}

Value builtin_file_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(remove(path) == 0 ? 0 : -1);
}

Value builtin_file_rename(Value *args) {
    const char *old_path = args[0].as.string_val;
    const char *new_path = args[1].as.string_val;
    return create_int(rename(old_path, new_path) == 0 ? 0 : -1);
}

Value builtin_file_exists(Value *args) {
    const char *path = args[0].as.string_val;
    return create_bool(access(path, F_OK) == 0);
}

Value builtin_file_size(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_int(-1);
    return create_int(st.st_size);
}

Value builtin_tmp_dir(Value *args) {
    (void)args;
    const char *tmp = getenv("TMPDIR");
    if (!tmp || tmp[0] == '\0') tmp = "/tmp";
    return create_string(tmp);
}

Value builtin_mktemp(Value *args) {
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

Value builtin_mktemp_dir(Value *args) {
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
Value builtin_dir_create(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(mkdir(path, 0755) == 0 ? 0 : -1);
}

Value builtin_dir_remove(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(rmdir(path) == 0 ? 0 : -1);
}

Value builtin_dir_list(Value *args) {
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

Value builtin_dir_exists(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

Value builtin_getcwd(Value *args) {
    (void)args;  /* Unused */
    char buffer[1024];
    if (getcwd(buffer, sizeof(buffer)) == NULL) {
        return create_string("");
    }
    return create_string(buffer);
}

Value builtin_chdir(Value *args) {
    const char *path = args[0].as.string_val;
    return create_int(chdir(path) == 0 ? 0 : -1);
}

/* Path Operations */
Value builtin_path_isfile(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISREG(st.st_mode));
}

Value builtin_path_isdir(Value *args) {
    const char *path = args[0].as.string_val;
    struct stat st;
    if (stat(path, &st) != 0) return create_bool(false);
    return create_bool(S_ISDIR(st.st_mode));
}

Value builtin_path_join(Value *args) {
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

Value builtin_path_basename(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *base = basename(path_copy);
    Value result = create_string(base);
    free(path_copy);
    return result;
}

Value builtin_path_dirname(Value *args) {
    const char *path = args[0].as.string_val;
    char *path_copy = strdup(path);
    char *dir = dirname(path_copy);
    Value result = create_string(dir);
    free(path_copy);
    return result;
}

char* nl_path_normalize(const char* path) {
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

Value builtin_path_normalize(Value *args) {
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

Value builtin_fs_walkdir(Value *args) {
    const char* root = args[0].as.string_val;
    DynArray* out = dyn_array_new(ELEM_STRING);
    if (root && root[0] != '\0') {
        nl_walkdir_rec(root, out);
    }
    return create_dyn_array(out);
}

/* Process Operations */
Value builtin_system(Value *args) {
    const char *command = args[0].as.string_val;
    return create_int(system(command));
}

Value builtin_exit(Value *args) {
    int code = (int)args[0].as.int_val;
    exit(code);
    return create_void();  /* Never reached */
}

Value builtin_getenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = getenv(name);
    return create_string(value ? value : "");
}

Value builtin_setenv(Value *args) {
    const char *name = args[0].as.string_val;
    const char *value = args[1].as.string_val;
    int overwrite = (int)args[2].as.int_val;
    return create_int(setenv(name, value, overwrite) == 0 ? 0 : -1);
}

Value builtin_unsetenv(Value *args) {
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

Value builtin_process_run(Value *args) {
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

Value builtin_result_is_ok(Value *args) {
    if (args[0].type != VAL_UNION) return create_bool(false);
    UnionValue *uv = args[0].as.union_val;
    return create_bool(uv && strcmp(uv->variant_name, "Ok") == 0);
}

Value builtin_result_is_err(Value *args) {
    if (args[0].type != VAL_UNION) return create_bool(false);
    UnionValue *uv = args[0].as.union_val;
    return create_bool(uv && strcmp(uv->variant_name, "Err") == 0);
}

Value builtin_result_unwrap(Value *args) {
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

Value builtin_result_unwrap_err(Value *args) {
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

Value builtin_result_unwrap_or(Value *args) {
    if (args[0].type != VAL_UNION) return args[1];
    UnionValue *uv = args[0].as.union_val;
    if (uv && strcmp(uv->variant_name, "Ok") == 0 && uv->field_count >= 1) {
        return uv->field_values[0];
    }
    return args[1];
}

Value builtin_result_map(Value *args, Environment *env) {
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
    mapped.is_break = false;
    mapped.is_continue = false;

    char *field_names[1] = { "value" };
    Value field_values[1] = { mapped };
    return create_union(uv->union_name, 0, "Ok", field_names, field_values, 1);
}

Value builtin_result_and_then(Value *args, Environment *env) {
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
    next.is_break = false;
    next.is_continue = false;
    if (next.type != VAL_UNION) {
        fprintf(stderr, "panic: result_and_then callback did not return a Result\n");
        exit(1);
    }
    return next;
}

/* ==========================================================================
 * End of Built-in OS Functions
 * ========================================================================== */
