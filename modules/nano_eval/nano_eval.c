#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef _DARWIN_C_SOURCE
#define _DARWIN_C_SOURCE
#endif

#include "nano_eval.h"

#include "../../src/nanolang.h"
#include "../../src/interpreter_ffi.h"
#include "../../src/runtime/ffi_loader.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NANO_EVAL_MAX_CMD 64
#define NANO_EVAL_MAX_AST 128

typedef struct {
    int kind;
    char *arg;
} NanoEvalCmd;

typedef struct {
    Environment *env;
    ASTNode *asts[NANO_EVAL_MAX_AST];
    int ast_count;
    char *buffer;
    int64_t point;
    char *last_result;
    char *last_error;
    NanoEvalCmd cmds[NANO_EVAL_MAX_CMD];
    int cmd_count;
} NanoEvalSession;

static NanoEvalSession *g_active = NULL;

static void free_cmds(NanoEvalSession *s) {
    int i;
    for (i = 0; i < s->cmd_count; i++) {
        free(s->cmds[i].arg);
        s->cmds[i].arg = NULL;
        s->cmds[i].kind = NANO_EVAL_CMD_NONE;
    }
    s->cmd_count = 0;
}

static void queue_cmd(int kind, const char *arg) {
    NanoEvalSession *s = g_active;
    if (s == NULL || s->cmd_count >= NANO_EVAL_MAX_CMD) {
        return;
    }
    s->cmds[s->cmd_count].kind = kind;
    s->cmds[s->cmd_count].arg = arg ? strdup(arg) : strdup("");
    s->cmd_count++;
}

static void set_error(NanoEvalSession *s, const char *msg) {
    free(s->last_error);
    s->last_error = strdup(msg ? msg : "");
}

static void set_result(NanoEvalSession *s, const char *msg) {
    free(s->last_result);
    s->last_result = strdup(msg ? msg : "");
}

static char *stringify_value(Value v) {
    char buf[128];
    switch (v.type) {
        case VAL_INT:
            snprintf(buf, sizeof(buf), "%lld", (long long)v.as.int_val);
            return strdup(buf);
        case VAL_FLOAT:
            snprintf(buf, sizeof(buf), "%g", v.as.float_val);
            return strdup(buf);
        case VAL_BOOL:
            return strdup(v.as.bool_val ? "true" : "false");
        case VAL_STRING:
            return strdup(v.as.string_val ? v.as.string_val : "");
        default:
            return strdup("");
    }
}

static Parameter *make_params(int count, Type type) {
    Parameter *params;
    int i;
    if (count <= 0) {
        return NULL;
    }
    params = calloc((size_t)count, sizeof(Parameter));
    if (params == NULL) {
        return NULL;
    }
    for (i = 0; i < count; i++) {
        params[i].name = "arg";
        params[i].type = type;
    }
    return params;
}

static void register_host(Environment *env, const char *name, int nparams,
                          Type ptype, Type rtype) {
    Function func;
    memset(&func, 0, sizeof(func));
    func.name = (char *)name;
    func.param_count = nparams;
    func.params = make_params(nparams, ptype);
    func.return_type = rtype;
    func.is_extern = true;
    env_define_function(env, func);
}

static void register_hosts(Environment *env) {
    register_host(env, "ed_message", 1, TYPE_STRING, TYPE_VOID);
    register_host(env, "ed_insert", 1, TYPE_STRING, TYPE_VOID);
    register_host(env, "ed_buffer_string", 0, TYPE_VOID, TYPE_STRING);
    register_host(env, "ed_point", 0, TYPE_VOID, TYPE_INT);
    register_host(env, "ed_goto_char", 1, TYPE_INT, TYPE_VOID);
    register_host(env, "ed_find_file", 1, TYPE_STRING, TYPE_VOID);
    register_host(env, "ed_save_buffer", 0, TYPE_VOID, TYPE_VOID);
    register_host(env, "ed_split_window", 0, TYPE_VOID, TYPE_VOID);
    register_host(env, "ed_other_window", 0, TYPE_VOID, TYPE_VOID);
}

static void register_host_image(void) {
    Dl_info info;
    memset(&info, 0, sizeof(info));
    if (dladdr((void *)nano_eval_create, &info) == 0 || info.dli_fname == NULL) {
        return;
    }
    (void)ffi_loader_open("nano_eval_host", info.dli_fname);
}

static NanoEvalSession *session_from(int64_t handle) {
    return (NanoEvalSession *)(intptr_t)handle;
}

int64_t nano_eval_create(void) {
    NanoEvalSession *s = calloc(1, sizeof(NanoEvalSession));
    if (s == NULL) {
        return 0;
    }
    (void)ffi_init(false);
    register_host_image();
    s->env = create_environment();
    s->buffer = strdup("");
    s->point = 0;
    s->last_result = strdup("");
    s->last_error = strdup("");
    register_hosts(s->env);
    g_active = s;
    return (int64_t)(intptr_t)s;
}

void nano_eval_destroy(int64_t session) {
    NanoEvalSession *s = session_from(session);
    int i;
    if (s == NULL) {
        return;
    }
    if (g_active == s) {
        g_active = NULL;
    }
    for (i = 0; i < s->ast_count; i++) {
        free_ast(s->asts[i]);
    }
    if (s->env) {
        free_environment(s->env);
    }
    free_cmds(s);
    free(s->buffer);
    free(s->last_result);
    free(s->last_error);
    free(s);
}

void nano_eval_bind_buffer(int64_t session, const char *text, int64_t point) {
    NanoEvalSession *s = session_from(session);
    size_t len;
    if (s == NULL) {
        return;
    }
    free(s->buffer);
    s->buffer = strdup(text ? text : "");
    len = strlen(s->buffer);
    if (point < 0) {
        point = 0;
    }
    if (point > (int64_t)len) {
        point = (int64_t)len;
    }
    s->point = point;
}

const char *nano_eval_buffer(int64_t session) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL || s->buffer == NULL) {
        return "";
    }
    return s->buffer;
}

int64_t nano_eval_point(int64_t session) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL) {
        return 0;
    }
    return s->point;
}

const char *nano_eval_error(int64_t session) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL || s->last_error == NULL) {
        return "";
    }
    return s->last_error;
}

int64_t nano_eval_cmd_count(int64_t session) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL) {
        return 0;
    }
    return s->cmd_count;
}

int64_t nano_eval_cmd_kind(int64_t session, int64_t index) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL || index < 0 || index >= s->cmd_count) {
        return NANO_EVAL_CMD_NONE;
    }
    return s->cmds[index].kind;
}

const char *nano_eval_cmd_arg(int64_t session, int64_t index) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL || index < 0 || index >= s->cmd_count) {
        return "";
    }
    return s->cmds[index].arg ? s->cmds[index].arg : "";
}

void nano_eval_cmd_clear(int64_t session) {
    NanoEvalSession *s = session_from(session);
    if (s == NULL) {
        return;
    }
    free_cmds(s);
}

const char *nano_eval_string(int64_t session, const char *source) {
    NanoEvalSession *s = session_from(session);
    Token *tokens;
    int token_count = 0;
    ASTNode *program;
    ModuleList *mods;
    int i;
    char *printed;
    Value last;

    if (s == NULL) {
        return "";
    }
    g_active = s;
    set_error(s, "");
    set_result(s, "");
    if (source == NULL || source[0] == '\0') {
        return s->last_result;
    }

    tokens = tokenize(source, &token_count);
    if (tokens == NULL) {
        set_error(s, "lexing failed");
        return s->last_result;
    }
    program = parse_repl_input(tokens, token_count);
    free_tokens(tokens, token_count);
    if (program == NULL) {
        set_error(s, "parse failed");
        return s->last_result;
    }
    if (program->type != AST_PROGRAM || program->as.program.count == 0) {
        free_ast(program);
        return s->last_result;
    }

    typecheck_set_current_file("<emacs>");
    mods = create_module_list();
    (void)process_imports(program, s->env, mods, "<emacs>");
    free_module_list(mods);
    if (!type_check_module(program, s->env)) {
        set_error(s, "type check failed");
        free_ast(program);
        return s->last_result;
    }

    last = create_void();
    for (i = 0; i < program->as.program.count; i++) {
        ASTNode *node = program->as.program.items[i];
        if (node->type == AST_IMPORT) {
            continue;
        }
        last = repl_eval_node(node, s->env);
    }

    printed = stringify_value(last);
    set_result(s, printed);
    free(printed);

    if (s->ast_count < NANO_EVAL_MAX_AST) {
        s->asts[s->ast_count++] = program;
    } else {
        free_ast(program);
    }
    return s->last_result;
}

int64_t ed_message(const char *text) {
    queue_cmd(NANO_EVAL_CMD_MESSAGE, text);
    return 0;
}

int64_t ed_insert(const char *text) {
    NanoEvalSession *s = g_active;
    size_t n;
    size_t old;
    char *nb;
    if (s == NULL || text == NULL) {
        return 0;
    }
    n = strlen(text);
    old = s->buffer ? strlen(s->buffer) : 0;
    if (s->point < 0) {
        s->point = 0;
    }
    if (s->point > (int64_t)old) {
        s->point = (int64_t)old;
    }
    nb = malloc(old + n + 1);
    if (nb == NULL) {
        return 0;
    }
    memcpy(nb, s->buffer ? s->buffer : "", (size_t)s->point);
    memcpy(nb + s->point, text, n);
    memcpy(nb + s->point + n,
           (s->buffer ? s->buffer : "") + s->point,
           old - (size_t)s->point);
    nb[old + n] = '\0';
    free(s->buffer);
    s->buffer = nb;
    s->point += (int64_t)n;
    return 0;
}

int64_t ed_buffer_string(void) {
    NanoEvalSession *s = g_active;
    if (s == NULL || s->buffer == NULL) {
        return (int64_t)(intptr_t)"";
    }
    return (int64_t)(intptr_t)s->buffer;
}

int64_t ed_point(void) {
    NanoEvalSession *s = g_active;
    if (s == NULL) {
        return 0;
    }
    return s->point;
}

int64_t ed_goto_char(int64_t pos) {
    NanoEvalSession *s = g_active;
    size_t len;
    if (s == NULL) {
        return 0;
    }
    len = s->buffer ? strlen(s->buffer) : 0;
    if (pos < 0) {
        pos = 0;
    }
    if (pos > (int64_t)len) {
        pos = (int64_t)len;
    }
    s->point = pos;
    return 0;
}

int64_t ed_find_file(const char *path) {
    queue_cmd(NANO_EVAL_CMD_FIND_FILE, path);
    return 0;
}

int64_t ed_save_buffer(void) {
    queue_cmd(NANO_EVAL_CMD_SAVE, "");
    return 0;
}

int64_t ed_split_window(void) {
    queue_cmd(NANO_EVAL_CMD_SPLIT, "");
    return 0;
}

int64_t ed_other_window(void) {
    queue_cmd(NANO_EVAL_CMD_OTHER_WINDOW, "");
    return 0;
}
