#define _POSIX_C_SOURCE 200809L

#include "nano_eval.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#define NANO_SESSION_NAME "libnano_session.dylib"
#else
#define NANO_SESSION_NAME "libnano_session.so"
#endif

typedef int64_t (*fn_create)(void);
typedef void (*fn_destroy)(int64_t);
typedef const char *(*fn_string)(int64_t, const char *);
typedef const char *(*fn_cstr)(int64_t);
typedef void (*fn_bind)(int64_t, const char *, int64_t);
typedef int64_t (*fn_i64)(int64_t);
typedef int64_t (*fn_i64_i64)(int64_t, int64_t);
typedef const char *(*fn_arg)(int64_t, int64_t);
typedef void (*fn_clear)(int64_t);

static void *g_lib = NULL;
static fn_create p_create = NULL;
static fn_destroy p_destroy = NULL;
static fn_string p_string = NULL;
static fn_cstr p_error = NULL;
static fn_bind p_bind = NULL;
static fn_cstr p_buffer = NULL;
static fn_i64 p_point = NULL;
static fn_i64 p_cmd_count = NULL;
static fn_i64_i64 p_cmd_kind = NULL;
static fn_arg p_cmd_arg = NULL;
static fn_clear p_cmd_clear = NULL;
static int g_load_failed = 0;

static int try_open(const char *path) {
    g_lib = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    return g_lib != NULL;
}

static int load_lib(void) {
    const char *root;
    char path[1024];
    if (g_lib != NULL) {
        return 1;
    }
    if (g_load_failed) {
        return 0;
    }
    if (try_open("bin/" NANO_SESSION_NAME)) {
        goto resolve;
    }
    if (try_open("./" NANO_SESSION_NAME)) {
        goto resolve;
    }
    root = getenv("NANO_MODULE_PATH");
    if (root && root[0]) {
        snprintf(path, sizeof(path), "%s/../bin/%s", root, NANO_SESSION_NAME);
        if (try_open(path)) {
            goto resolve;
        }
    }
    fprintf(stderr, "nano_eval: cannot load %s (build with make libnano-session)\n",
            NANO_SESSION_NAME);
    g_load_failed = 1;
    return 0;

resolve:
    p_create = (fn_create)dlsym(g_lib, "nano_eval_create");
    p_destroy = (fn_destroy)dlsym(g_lib, "nano_eval_destroy");
    p_string = (fn_string)dlsym(g_lib, "nano_eval_string");
    p_error = (fn_cstr)dlsym(g_lib, "nano_eval_error");
    p_bind = (fn_bind)dlsym(g_lib, "nano_eval_bind_buffer");
    p_buffer = (fn_cstr)dlsym(g_lib, "nano_eval_buffer");
    p_point = (fn_i64)dlsym(g_lib, "nano_eval_point");
    p_cmd_count = (fn_i64)dlsym(g_lib, "nano_eval_cmd_count");
    p_cmd_kind = (fn_i64_i64)dlsym(g_lib, "nano_eval_cmd_kind");
    p_cmd_arg = (fn_arg)dlsym(g_lib, "nano_eval_cmd_arg");
    p_cmd_clear = (fn_clear)dlsym(g_lib, "nano_eval_cmd_clear");
    if (!p_create || !p_string) {
        fprintf(stderr, "nano_eval: missing symbols in %s\n", NANO_SESSION_NAME);
        g_load_failed = 1;
        return 0;
    }
    return 1;
}

int64_t nano_eval_create(void) {
    if (!load_lib()) {
        return 0;
    }
    return p_create();
}

void nano_eval_destroy(int64_t session) {
    if (p_destroy) {
        p_destroy(session);
    }
}

const char *nano_eval_string(int64_t session, const char *source) {
    if (!p_string) {
        return "";
    }
    return p_string(session, source);
}

const char *nano_eval_error(int64_t session) {
    if (!p_error) {
        return "eval session is not loaded";
    }
    return p_error(session);
}

void nano_eval_bind_buffer(int64_t session, const char *text, int64_t point) {
    if (p_bind) {
        p_bind(session, text, point);
    }
}

const char *nano_eval_buffer(int64_t session) {
    if (!p_buffer) {
        return "";
    }
    return p_buffer(session);
}

int64_t nano_eval_point(int64_t session) {
    if (!p_point) {
        return 0;
    }
    return p_point(session);
}

int64_t nano_eval_cmd_count(int64_t session) {
    if (!p_cmd_count) {
        return 0;
    }
    return p_cmd_count(session);
}

int64_t nano_eval_cmd_kind(int64_t session, int64_t index) {
    if (!p_cmd_kind) {
        return 0;
    }
    return p_cmd_kind(session, index);
}

const char *nano_eval_cmd_arg(int64_t session, int64_t index) {
    if (!p_cmd_arg) {
        return "";
    }
    return p_cmd_arg(session, index);
}

void nano_eval_cmd_clear(int64_t session) {
    if (p_cmd_clear) {
        p_cmd_clear(session);
    }
}
