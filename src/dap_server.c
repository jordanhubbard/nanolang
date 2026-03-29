/* dap_server.c — nanolang Debug Adapter Protocol server
 *
 * DAP JSON over stdio with Content-Length framing (identical to LSP).
 * Runs the nanolang interpreter in-process, pausing execution at breakpoints
 * or after step commands by entering an inner event loop inside the hook.
 *
 * Supported requests:
 *   initialize, launch, setBreakpoints, configurationDone,
 *   threads, stackTrace, scopes, variables,
 *   continue, next, stepIn, stepOut,
 *   pause, disconnect
 *
 * Supported events sent to client:
 *   initialized, stopped, continued, output, terminated, thread
 */

#define _POSIX_C_SOURCE 200809L

#include "nanolang.h"
#include "module_builder.h"
#include "interpreter_ffi.h"
#include "cJSON.h"
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>

/* =========================================================================
 * Runtime globals (required by nanolang modules linked into this binary)
 * ========================================================================= */

int g_argc = 0;
char **g_argv = NULL;

char g_project_root[PATH_MAX] = "";

const char *get_project_root(void) {
    return g_project_root[0] ? g_project_root : ".";
}

static void resolve_project_root(const char *argv0) {
    char exe_path[PATH_MAX];
    if (realpath(argv0, exe_path) == NULL) {
        if (getcwd(g_project_root, sizeof(g_project_root)) == NULL)
            strcpy(g_project_root, ".");
        return;
    }
    char *slash = strrchr(exe_path, '/');
    if (slash) {
        *slash = '\0';
        slash = strrchr(exe_path, '/');
        if (slash) *slash = '\0';
    }
    strncpy(g_project_root, exe_path, sizeof(g_project_root) - 1);
    g_project_root[sizeof(g_project_root) - 1] = '\0';
}

/* =========================================================================
 * Debug state
 * ========================================================================= */

#define DAP_MAX_BREAKPOINTS 256
#define DAP_MAX_STACK       64
#define DAP_MAX_VAR_REF     1024

typedef enum {
    STEP_NONE,     /* running freely */
    STEP_OVER,     /* step over (next) */
    STEP_IN,       /* step into */
    STEP_OUT       /* step out */
} StepMode;

typedef struct {
    char file[PATH_MAX];
    int  line;
    bool enabled;
    char condition[256];   /* hit condition expression (currently checked by count) */
    int  hit_count;
} Breakpoint;

typedef struct {
    char name[256];
    char source_file[PATH_MAX];
    int  line;
    int  column;
} StackFrame;

typedef struct {
    /* Breakpoint table */
    Breakpoint  bp_table[DAP_MAX_BREAKPOINTS];
    int         bp_count;

    /* Step state */
    StepMode    step_mode;
    int         step_depth;   /* call depth at the time step_over was initiated */

    /* Current execution position */
    int         current_line;
    char        current_file[PATH_MAX];

    /* Call stack (maintained by hook push/pop helpers) */
    StackFrame  call_stack[DAP_MAX_STACK];
    int         call_stack_depth;

    /* Server state */
    bool        initialized;      /* initialize request handled */
    bool        launched;         /* launch request handled */
    bool        config_done;      /* configurationDone received */
    bool        running;          /* program currently executing */
    bool        paused;           /* paused at breakpoint/step */
    bool        disconnected;     /* disconnect received */
    bool        program_ended;    /* program finished */

    /* Sequence counter for outgoing messages */
    int         seq;

    /* Variable reference store: varref_id → Environment* snapshot (shallow) */
    /* We use a simple mapping: varref 1 = locals, varref 2 = globals */
    Environment *pause_env;       /* env snapshot when last paused */

    /* The program being debugged */
    char        program_path[PATH_MAX];
} DebugState;

static DebugState g_dap = {0};

/* =========================================================================
 * Content-Length framing
 * ========================================================================= */

static char *dap_read_message(void) {
    char header[256];
    int content_length = -1;

    while (fgets(header, (int)sizeof(header), stdin)) {
        size_t len = strlen(header);
        while (len > 0 && (header[len - 1] == '\r' || header[len - 1] == '\n'))
            header[--len] = '\0';
        if (len == 0)
            break;
        if (strncmp(header, "Content-Length: ", 16) == 0)
            content_length = atoi(header + 16);
    }

    if (content_length <= 0) return NULL;

    char *buf = malloc((size_t)content_length + 1);
    if (!buf) return NULL;

    size_t total = 0;
    while (total < (size_t)content_length) {
        size_t n = fread(buf + total, 1, (size_t)content_length - total, stdin);
        if (n == 0) { free(buf); return NULL; }
        total += n;
    }
    buf[content_length] = '\0';
    return buf;
}

static void dap_send(const char *json) {
    fprintf(stdout, "Content-Length: %d\r\n\r\n%s", (int)strlen(json), json);
    fflush(stdout);
}

/* =========================================================================
 * DAP message builders
 * ========================================================================= */

static void dap_send_response(int req_seq, const char *command, bool success,
                               cJSON *body) {
    cJSON *resp = cJSON_CreateObject();
    cJSON_AddNumberToObject(resp, "seq", ++g_dap.seq);
    cJSON_AddStringToObject(resp, "type", "response");
    cJSON_AddNumberToObject(resp, "request_seq", req_seq);
    cJSON_AddBoolToObject(resp, "success", success);
    cJSON_AddStringToObject(resp, "command", command);
    if (body)
        cJSON_AddItemToObject(resp, "body", body);
    char *s = cJSON_PrintUnformatted(resp);
    if (s) { dap_send(s); free(s); }
    cJSON_Delete(resp);
}

static void dap_send_error_response(int req_seq, const char *command,
                                     const char *msg) {
    cJSON *body = cJSON_CreateObject();
    cJSON *err  = cJSON_CreateObject();
    cJSON_AddNumberToObject(err, "id", 1);
    cJSON_AddStringToObject(err, "format", msg);
    cJSON_AddItemToObject(body, "error", err);
    dap_send_response(req_seq, command, false, body);
}

static void dap_send_event(const char *event, cJSON *body) {
    cJSON *ev = cJSON_CreateObject();
    cJSON_AddNumberToObject(ev, "seq", ++g_dap.seq);
    cJSON_AddStringToObject(ev, "type", "event");
    cJSON_AddStringToObject(ev, "event", event);
    if (body)
        cJSON_AddItemToObject(ev, "body", body);
    char *s = cJSON_PrintUnformatted(ev);
    if (s) { dap_send(s); free(s); }
    cJSON_Delete(ev);
}

static void dap_output(const char *category, const char *msg) {
    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "category", category);
    cJSON_AddStringToObject(body, "output", msg);
    dap_send_event("output", body);
}

/* =========================================================================
 * Breakpoint helpers
 * ========================================================================= */

/* Canonicalize file path: strip file:// prefix if present, resolve to realpath */
static void canonicalize_path(const char *in, char *out, size_t outsz) {
    const char *p = in;
    if (strncmp(p, "file://", 7) == 0) p += 7;
    if (realpath(p, out) == NULL)
        strncpy(out, p, outsz - 1);
    out[outsz - 1] = '\0';
}

static bool bp_matches(int bp_idx, const char *file, int line) {
    Breakpoint *bp = &g_dap.bp_table[bp_idx];
    if (!bp->enabled) return false;
    if (bp->line != line) return false;

    /* Compare canonicalized paths */
    char canon[PATH_MAX];
    canonicalize_path(bp->file, canon, sizeof(canon));
    char input_canon[PATH_MAX];
    canonicalize_path(file, input_canon, sizeof(input_canon));
    return strcmp(canon, input_canon) == 0;
}

static bool should_stop_at(const char *file, int line) {
    if (g_dap.step_mode != STEP_NONE) {
        /* Step-in: stop at every statement */
        if (g_dap.step_mode == STEP_IN) return true;
        /* Step-over: stop when we're back at the same or lower call depth */
        if (g_dap.step_mode == STEP_OVER &&
            g_dap.call_stack_depth <= g_dap.step_depth) return true;
        /* Step-out: stop when call depth is lower than when step-out was issued */
        if (g_dap.step_mode == STEP_OUT &&
            g_dap.call_stack_depth < g_dap.step_depth) return true;
    }

    /* Breakpoint check */
    for (int i = 0; i < g_dap.bp_count; i++) {
        if (bp_matches(i, file, line)) return true;
    }
    return false;
}

/* =========================================================================
 * Value formatting for variable inspection
 * ========================================================================= */

static void format_value(Value val, char *buf, size_t bufsz) {
    switch (val.type) {
        case VAL_INT:
            snprintf(buf, bufsz, "%lld", val.as.int_val);
            break;
        case VAL_FLOAT:
            snprintf(buf, bufsz, "%g", val.as.float_val);
            break;
        case VAL_BOOL:
            snprintf(buf, bufsz, "%s", val.as.bool_val ? "true" : "false");
            break;
        case VAL_STRING:
            snprintf(buf, bufsz, "\"%s\"", val.as.string_val ? val.as.string_val : "");
            break;
        case VAL_VOID:
            snprintf(buf, bufsz, "void");
            break;
        case VAL_ARRAY:
            snprintf(buf, bufsz, "[array len=%d]",
                     val.as.array_val ? val.as.array_val->length : 0);
            break;
        case VAL_DYN_ARRAY:
            snprintf(buf, bufsz, "[dyn_array]");
            break;
        case VAL_STRUCT:
            snprintf(buf, bufsz, "{%s}",
                     val.as.struct_val && val.as.struct_val->struct_name
                         ? val.as.struct_val->struct_name : "struct");
            break;
        case VAL_GC_STRUCT:
            snprintf(buf, bufsz, "{gc_struct}");
            break;
        case VAL_UNION:
            snprintf(buf, bufsz, "%s::%s",
                     val.as.union_val && val.as.union_val->union_name
                         ? val.as.union_val->union_name : "union",
                     val.as.union_val && val.as.union_val->variant_name
                         ? val.as.union_val->variant_name : "?");
            break;
        case VAL_FUNCTION:
            snprintf(buf, bufsz, "<fn %s>",
                     val.as.function_val.function_name
                         ? val.as.function_val.function_name : "?");
            break;
        case VAL_TUPLE:
            snprintf(buf, bufsz, "(tuple len=%d)",
                     val.as.tuple_val ? val.as.tuple_val->element_count : 0);
            break;
        default:
            snprintf(buf, bufsz, "<unknown>");
            break;
    }
}

static const char *type_name(Type t) {
    switch (t) {
        case TYPE_INT:    return "int";
        case TYPE_U8:     return "u8";
        case TYPE_FLOAT:  return "float";
        case TYPE_BOOL:   return "bool";
        case TYPE_STRING: return "string";
        case TYPE_VOID:   return "void";
        case TYPE_ARRAY:  return "array";
        case TYPE_STRUCT: return "struct";
        case TYPE_ENUM:   return "enum";
        case TYPE_UNION:  return "union";
        case TYPE_FUNCTION: return "fn";
        case TYPE_TUPLE:  return "tuple";
        default:          return "unknown";
    }
}

/* =========================================================================
 * Inner event loop (called when execution is paused)
 * =========================================================================
 * Reads and dispatches DAP messages until a "continue" / step command
 * or "disconnect" is received.
 * ========================================================================= */

/* Forward declarations */
static void handle_request(cJSON *msg);

static void dap_pause_loop(const char *reason, const char *description) {
    g_dap.paused = true;
    g_dap.step_mode = STEP_NONE;

    /* Send stopped event */
    cJSON *sbody = cJSON_CreateObject();
    cJSON_AddStringToObject(sbody, "reason", reason);
    if (description)
        cJSON_AddStringToObject(sbody, "description", description);
    cJSON_AddNumberToObject(sbody, "threadId", 1);
    cJSON_AddBoolToObject(sbody, "allThreadsStopped", true);
    dap_send_event("stopped", sbody);

    /* Inner event loop — handle requests while paused */
    while (g_dap.paused && !g_dap.disconnected) {
        char *raw = dap_read_message();
        if (!raw) break;
        cJSON *msg = cJSON_Parse(raw);
        free(raw);
        if (!msg) continue;
        handle_request(msg);
        cJSON_Delete(msg);
    }
}

/* =========================================================================
 * Request handlers
 * ========================================================================= */

static void handle_initialize(cJSON *msg, cJSON *args) {
    (void)args;
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    cJSON *caps = cJSON_CreateObject();
    cJSON_AddBoolToObject(caps, "supportsConfigurationDoneRequest", true);
    cJSON_AddBoolToObject(caps, "supportsHitConditionalBreakpoints", true);
    cJSON_AddBoolToObject(caps, "supportsTerminateRequest", true);

    dap_send_response(seq, "initialize", true, caps);
    g_dap.initialized = true;

    /* Send initialized event so client can send setBreakpoints */
    dap_send_event("initialized", NULL);
}

static void handle_launch(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    cJSON *prog = args ? cJSON_GetObjectItem(args, "program") : NULL;
    if (!prog || !prog->valuestring) {
        dap_send_error_response(seq, "launch", "Missing 'program' in launch args");
        return;
    }

    canonicalize_path(prog->valuestring, g_dap.program_path, sizeof(g_dap.program_path));
    g_dap.launched = true;

    dap_send_response(seq, "launch", true, NULL);
    dap_output("console", "nanolang-dap: program loaded, waiting for configurationDone\n");
}

static void handle_set_breakpoints(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    cJSON *source = args ? cJSON_GetObjectItem(args, "source") : NULL;
    cJSON *bps    = args ? cJSON_GetObjectItem(args, "breakpoints") : NULL;

    char src_path[PATH_MAX] = "";
    if (source) {
        cJSON *path = cJSON_GetObjectItem(source, "path");
        if (path && path->valuestring)
            canonicalize_path(path->valuestring, src_path, sizeof(src_path));
    }

    /* Remove existing breakpoints for this file */
    int new_count = 0;
    for (int i = 0; i < g_dap.bp_count; i++) {
        char canon[PATH_MAX];
        canonicalize_path(g_dap.bp_table[i].file, canon, sizeof(canon));
        if (strcmp(canon, src_path) != 0) {
            g_dap.bp_table[new_count++] = g_dap.bp_table[i];
        }
    }
    g_dap.bp_count = new_count;

    /* Add new breakpoints */
    cJSON *result_bps = cJSON_CreateArray();
    if (bps) {
        cJSON *bp;
        cJSON_ArrayForEach(bp, bps) {
            cJSON *line_item = cJSON_GetObjectItem(bp, "line");
            int line = line_item ? (int)line_item->valuedouble : 0;

            cJSON *rbp = cJSON_CreateObject();
            if (g_dap.bp_count < DAP_MAX_BREAKPOINTS && line > 0) {
                Breakpoint *entry = &g_dap.bp_table[g_dap.bp_count++];
                snprintf(entry->file, sizeof(entry->file), "%s", src_path);
                entry->line      = line;
                entry->enabled   = true;
                entry->hit_count = 0;
                entry->condition[0] = '\0';

                cJSON_AddBoolToObject(rbp, "verified", true);
                cJSON_AddNumberToObject(rbp, "line", line);
            } else {
                cJSON_AddBoolToObject(rbp, "verified", false);
            }
            cJSON_AddItemToArray(result_bps, rbp);
        }
    }

    cJSON *body = cJSON_CreateObject();
    cJSON_AddItemToObject(body, "breakpoints", result_bps);
    dap_send_response(seq, "setBreakpoints", true, body);
}

static void handle_configuration_done(cJSON *msg) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    dap_send_response(seq, "configurationDone", true, NULL);
    g_dap.config_done = true;
    /* Execution will begin when dap_run_program() is called from main loop */
}

static void handle_threads(cJSON *msg) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    cJSON *threads = cJSON_CreateArray();
    cJSON *t = cJSON_CreateObject();
    cJSON_AddNumberToObject(t, "id", 1);
    cJSON_AddStringToObject(t, "name", "main");
    cJSON_AddItemToArray(threads, t);

    cJSON *body = cJSON_CreateObject();
    cJSON_AddItemToObject(body, "threads", threads);
    dap_send_response(seq, "threads", true, body);
}

static void handle_stack_trace(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    cJSON *frames = cJSON_CreateArray();

    /* Top of stack = current position */
    {
        cJSON *f = cJSON_CreateObject();
        cJSON_AddNumberToObject(f, "id", 0);
        cJSON_AddStringToObject(f, "name",
            g_dap.call_stack_depth > 0
                ? g_dap.call_stack[g_dap.call_stack_depth - 1].name
                : "<top>");
        cJSON *src = cJSON_CreateObject();
        cJSON_AddStringToObject(src, "path",
            g_dap.current_file[0] ? g_dap.current_file : g_dap.program_path);
        cJSON_AddItemToObject(f, "source", src);
        cJSON_AddNumberToObject(f, "line",   g_dap.current_line);
        cJSON_AddNumberToObject(f, "column", 1);
        cJSON_AddItemToArray(frames, f);
    }

    /* Remaining call stack frames */
    for (int i = g_dap.call_stack_depth - 1; i >= 0; i--) {
        StackFrame *sf = &g_dap.call_stack[i];
        cJSON *f = cJSON_CreateObject();
        cJSON_AddNumberToObject(f, "id", g_dap.call_stack_depth - i);
        cJSON_AddStringToObject(f, "name", sf->name);
        if (sf->source_file[0]) {
            cJSON *src = cJSON_CreateObject();
            cJSON_AddStringToObject(src, "path", sf->source_file);
            cJSON_AddItemToObject(f, "source", src);
        }
        cJSON_AddNumberToObject(f, "line",   sf->line);
        cJSON_AddNumberToObject(f, "column", sf->column);
        cJSON_AddItemToArray(frames, f);
    }

    cJSON *body = cJSON_CreateObject();
    cJSON_AddItemToObject(body, "stackFrames", frames);
    cJSON_AddNumberToObject(body, "totalFrames", g_dap.call_stack_depth + 1);
    dap_send_response(seq, "stackTrace", true, body);
}

static void handle_scopes(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    cJSON *scopes = cJSON_CreateArray();

    /* Locals scope (varRef 1) */
    cJSON *locals = cJSON_CreateObject();
    cJSON_AddStringToObject(locals, "name", "Locals");
    cJSON_AddNumberToObject(locals, "variablesReference", 1);
    cJSON_AddBoolToObject(locals, "expensive", false);
    cJSON_AddItemToArray(scopes, locals);

    /* Globals scope (varRef 2) */
    cJSON *globals = cJSON_CreateObject();
    cJSON_AddStringToObject(globals, "name", "Globals");
    cJSON_AddNumberToObject(globals, "variablesReference", 2);
    cJSON_AddBoolToObject(globals, "expensive", false);
    cJSON_AddItemToArray(scopes, globals);

    cJSON *body = cJSON_CreateObject();
    cJSON_AddItemToObject(body, "scopes", scopes);
    dap_send_response(seq, "scopes", true, body);
}

static void handle_variables(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;

    cJSON *ref_item = args ? cJSON_GetObjectItem(args, "variablesReference") : NULL;
    int var_ref = ref_item ? (int)ref_item->valuedouble : 0;

    cJSON *vars = cJSON_CreateArray();

    Environment *env = g_dap.pause_env;
    if (env && (var_ref == 1 || var_ref == 2)) {
        /* For simplicity: varRef 1 = all symbols visible at current scope.
         * varRef 2 = global functions (names only). */
        if (var_ref == 1) {
            for (int i = 0; i < env->symbol_count; i++) {
                Symbol *sym = &env->symbols[i];
                if (!sym->name) continue;

                char val_str[512];
                format_value(sym->value, val_str, sizeof(val_str));

                cJSON *v = cJSON_CreateObject();
                cJSON_AddStringToObject(v, "name", sym->name);
                cJSON_AddStringToObject(v, "value", val_str);
                cJSON_AddStringToObject(v, "type", type_name(sym->type));
                cJSON_AddNumberToObject(v, "variablesReference", 0);
                cJSON_AddItemToArray(vars, v);
            }
        } else {
            /* Global functions */
            for (int i = 0; i < env->function_count; i++) {
                Function *fn = &env->functions[i];
                if (!fn->name) continue;
                cJSON *v = cJSON_CreateObject();
                cJSON_AddStringToObject(v, "name", fn->name);
                cJSON_AddStringToObject(v, "value", "<function>");
                cJSON_AddStringToObject(v, "type", "fn");
                cJSON_AddNumberToObject(v, "variablesReference", 0);
                cJSON_AddItemToArray(vars, v);
            }
        }
    }

    cJSON *body = cJSON_CreateObject();
    cJSON_AddItemToObject(body, "variables", vars);
    dap_send_response(seq, "variables", true, body);
}

static void handle_continue(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    cJSON *body = cJSON_CreateObject();
    cJSON_AddBoolToObject(body, "allThreadsContinued", true);
    dap_send_response(seq, "continue", true, body);

    cJSON *cbody = cJSON_CreateObject();
    cJSON_AddNumberToObject(cbody, "threadId", 1);
    cJSON_AddBoolToObject(cbody, "allThreadsContinued", true);
    dap_send_event("continued", cbody);

    g_dap.step_mode = STEP_NONE;
    g_dap.paused = false;   /* exits dap_pause_loop() */
}

static void handle_next(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "next", true, NULL);
    g_dap.step_mode  = STEP_OVER;
    g_dap.step_depth = g_dap.call_stack_depth;
    g_dap.paused = false;
}

static void handle_step_in(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "stepIn", true, NULL);
    g_dap.step_mode  = STEP_IN;
    g_dap.step_depth = g_dap.call_stack_depth;
    g_dap.paused = false;
}

static void handle_step_out(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "stepOut", true, NULL);
    g_dap.step_mode  = STEP_OUT;
    g_dap.step_depth = g_dap.call_stack_depth;
    g_dap.paused = false;
}

static void handle_pause(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "pause", true, NULL);
    /* Request a pause on next statement */
    g_dap.step_mode = STEP_IN;
}

static void handle_disconnect(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "disconnect", true, NULL);
    g_dap.disconnected = true;
    g_dap.paused = false;
}

static void handle_terminate(cJSON *msg, cJSON *args) {
    cJSON *id = cJSON_GetObjectItem(msg, "seq");
    int seq = id ? (int)id->valuedouble : 0;
    (void)args;

    dap_send_response(seq, "terminate", true, NULL);
    g_dap.disconnected = true;
    g_dap.paused = false;
}

/* Dispatch a single DAP request (used both from outer loop and inner pause loop) */
static void handle_request(cJSON *msg) {
    cJSON *type_item = cJSON_GetObjectItem(msg, "type");
    if (!type_item || !type_item->valuestring) return;
    if (strcmp(type_item->valuestring, "request") != 0) return;

    cJSON *cmd_item = cJSON_GetObjectItem(msg, "command");
    if (!cmd_item || !cmd_item->valuestring) return;
    const char *cmd = cmd_item->valuestring;

    cJSON *args = cJSON_GetObjectItem(msg, "arguments");

    if      (strcmp(cmd, "initialize")       == 0) handle_initialize(msg, args);
    else if (strcmp(cmd, "launch")           == 0) handle_launch(msg, args);
    else if (strcmp(cmd, "setBreakpoints")   == 0) handle_set_breakpoints(msg, args);
    else if (strcmp(cmd, "configurationDone")== 0) handle_configuration_done(msg);
    else if (strcmp(cmd, "threads")          == 0) handle_threads(msg);
    else if (strcmp(cmd, "stackTrace")       == 0) handle_stack_trace(msg, args);
    else if (strcmp(cmd, "scopes")           == 0) handle_scopes(msg, args);
    else if (strcmp(cmd, "variables")        == 0) handle_variables(msg, args);
    else if (strcmp(cmd, "continue")         == 0) handle_continue(msg, args);
    else if (strcmp(cmd, "next")             == 0) handle_next(msg, args);
    else if (strcmp(cmd, "stepIn")           == 0) handle_step_in(msg, args);
    else if (strcmp(cmd, "stepOut")          == 0) handle_step_out(msg, args);
    else if (strcmp(cmd, "pause")            == 0) handle_pause(msg, args);
    else if (strcmp(cmd, "disconnect")       == 0) handle_disconnect(msg, args);
    else if (strcmp(cmd, "terminate")        == 0) handle_terminate(msg, args);
    else {
        /* Unknown request — send a generic success response to avoid client hanging */
        cJSON *id = cJSON_GetObjectItem(msg, "seq");
        int seq = id ? (int)id->valuedouble : 0;
        dap_send_response(seq, cmd, true, NULL);
    }
}

/* =========================================================================
 * eval.c statement hook
 * ========================================================================= */

/* This function is installed as g_dap_statement_hook in eval.c */
static void dap_eval_hook(ASTNode *stmt, Environment *env) {
    if (g_dap.disconnected) return;
    if (!stmt) return;

    /* Track current position */
    g_dap.current_line = stmt->line;
    /* Use program_path as file (single-file debug for now) */
    snprintf(g_dap.current_file, sizeof(g_dap.current_file), "%s", g_dap.program_path);

    /* Save current env for variable inspection */
    g_dap.pause_env = env;

    if (should_stop_at(g_dap.current_file, g_dap.current_line)) {
        /* Determine reason */
        const char *reason = "breakpoint";
        if (g_dap.step_mode != STEP_NONE) reason = "step";

        dap_pause_loop(reason, NULL);
    }

    if (g_dap.disconnected) {
        /* Client disconnected — terminate gracefully by longjmp or just exit */
        exit(0);
    }
}

/* =========================================================================
 * Program runner (called after configurationDone)
 * ========================================================================= */

/* Declared in eval.c — the hook function pointer */
extern void (*g_dap_statement_hook)(ASTNode *stmt, Environment *env);

static int dap_run_program(void) {
    /* Install the hook */
    g_dap_statement_hook = dap_eval_hook;

    /* Read source */
    FILE *f = fopen(g_dap.program_path, "r");
    if (!f) {
        char msg[PATH_MAX + 64];
        snprintf(msg, sizeof(msg), "Cannot open program: %s\n", strerror(errno));
        dap_output("stderr", msg);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *source = malloc((size_t)fsize + 1);
    if (!source) { fclose(f); dap_output("stderr", "out of memory\n"); return 1; }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    fread(source, 1, (size_t)fsize, f);
#pragma GCC diagnostic pop
    source[fsize] = '\0';
    fclose(f);

    dap_output("console", "nanolang-dap: starting program\n");

    /* Thread started event */
    cJSON *te = cJSON_CreateObject();
    cJSON_AddNumberToObject(te, "threadId", 1);
    cJSON_AddStringToObject(te, "reason", "started");
    dap_send_event("thread", te);

    /* Lex */
    int token_count = 0;
    Token *tokens = tokenize(source, &token_count);
    if (!tokens) { free(source); dap_output("stderr", "lex failed\n"); return 1; }

    /* Parse */
    ASTNode *program = parse_program(tokens, token_count);
    if (!program) {
        free_tokens(tokens, token_count); free(source);
        dap_output("stderr", "parse failed\n"); return 1;
    }

    /* Environment + imports */
    clear_module_cache();
    Environment *env = create_environment();
    ModuleList *modules = create_module_list();
    if (!process_imports(program, env, modules, g_dap.program_path)) {
        free_ast(program); free_tokens(tokens, token_count);
        free_environment(env); free_module_list(modules); clear_module_cache();
        free(source);
        dap_output("stderr", "module loading failed\n"); return 1;
    }

    /* Type check */
    typecheck_set_current_file(g_dap.program_path);
    if (!type_check(program, env)) {
        free_ast(program); free_tokens(tokens, token_count);
        free_environment(env); free_module_list(modules); clear_module_cache();
        free(source);
        dap_output("stderr", "type check failed\n"); return 1;
    }

    /* FFI */
    (void)ffi_init(false);
    for (int i = 0; i < modules->count; i++) {
        const char *mp = modules->module_paths[i];
        char *mdir = strdup(mp);
        char *sl   = strrchr(mdir, '/');
        if (sl) *sl = '\0'; else { free(mdir); mdir = strdup("."); }
        ModuleBuildMetadata *meta = module_load_metadata(mdir);
        char mod_name[256];
        if (meta && meta->name) {
            snprintf(mod_name, sizeof(mod_name), "%s", meta->name);
        } else {
            const char *base = sl ? (sl + 1) : mp;
            snprintf(mod_name, sizeof(mod_name), "%s", base);
            char *dot = strrchr(mod_name, '.'); if (dot) *dot = '\0';
        }
        (void)ffi_load_module(mod_name, mp, env, false);
        if (meta) module_metadata_free(meta);
        free(mdir);
    }

    /* Run */
    g_dap.running = true;
    if (!run_program(program, env)) {
        dap_output("stderr", "runtime error\n");
    } else {
        /* Call main() if present */
        Function *main_fn = env_get_function(env, "main");
        if (main_fn) call_function("main", NULL, 0, env);
    }
    g_dap.running = false;

    /* Cleanup */
    free_ast(program);
    free_tokens(tokens, token_count);
    free_environment(env);
    free_module_list(modules);
    clear_module_cache();
    free(source);

    /* Remove hook */
    g_dap_statement_hook = NULL;
    g_dap.program_ended  = true;

    dap_output("console", "nanolang-dap: program exited\n");

    /* Send exited + terminated events */
    cJSON *xbody = cJSON_CreateObject();
    cJSON_AddNumberToObject(xbody, "exitCode", 0);
    dap_send_event("exited", xbody);
    dap_send_event("terminated", NULL);

    return 0;
}

/* =========================================================================
 * Main entry point
 * ========================================================================= */

int main(int argc, char *argv[]) {
    g_argc = argc;
    g_argv = argv;
    resolve_project_root(argv[0]);

    /* Outer event loop: run until launch + configurationDone, then run program,
     * then keep running until disconnect */
    while (!g_dap.disconnected) {
        char *raw = dap_read_message();
        if (!raw) break;

        cJSON *msg = cJSON_Parse(raw);
        free(raw);
        if (!msg) continue;

        handle_request(msg);
        cJSON_Delete(msg);

        /* Once launch + configurationDone received, execute the program */
        if (g_dap.launched && g_dap.config_done && !g_dap.running && !g_dap.program_ended) {
            dap_run_program();
            /* After program ends, keep reading until disconnect */
        }
    }

    return 0;
}
