/*
 * coroutine.c — nanolang cooperative coroutine scheduler
 *
 * See coroutine.h for design notes.
 */

#include "coroutine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Global context flag ────────────────────────────────────────────────── */

int g_current_coroutine_id = -1;   /* -1 = synchronous context */

/* ── Scheduler state ────────────────────────────────────────────────────── */

static Coroutine  s_coroutines[COROUTINE_MAX];
static int        s_count       = 0;
static int        s_next_id     = 0;
static bool       s_initialized = false;

/* ── Internal helpers ───────────────────────────────────────────────────── */

static Coroutine *find_by_id(int id) {
    for (int i = 0; i < s_count; i++) {
        if (s_coroutines[i].id == id) return &s_coroutines[i];
    }
    return NULL;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

void coroutine_init(void) {
    if (s_initialized) return;
    memset(s_coroutines, 0, sizeof(s_coroutines));
    s_count       = 0;
    s_next_id     = 0;
    s_initialized = true;
}

int coroutine_spawn(const char *fn_name, Value *args, int arg_count, Environment *env) {
    if (!s_initialized) coroutine_init();

    if (s_count >= COROUTINE_MAX) {
        fprintf(stderr, "[coroutine] run_queue full (max %d coroutines)\n", COROUTINE_MAX);
        return -1;
    }

    Coroutine *c = &s_coroutines[s_count++];
    memset(c, 0, sizeof(Coroutine));

    c->id     = s_next_id++;
    c->status = CORO_PENDING;
    c->env    = env;

    /* Copy function name */
    c->fn_name = fn_name ? strdup(fn_name) : strdup("<async>");

    /* Copy argument array */
    c->arg_count = arg_count;
    if (arg_count > 0 && args) {
        c->args = malloc(sizeof(Value) * arg_count);
        memcpy(c->args, args, sizeof(Value) * arg_count);
    } else {
        c->args = NULL;
    }

    c->result.type = VAL_VOID;
    return c->id;
}

Value coroutine_run_to_completion(int id) {
    Coroutine *c = find_by_id(id);
    if (!c) {
        fprintf(stderr, "[coroutine] run_to_completion: unknown id %d\n", id);
        return (Value){ .type = VAL_VOID };
    }

    /* Already finished — return cached result */
    if (c->status == CORO_DONE)  return c->result;
    if (c->status == CORO_ERROR) return (Value){ .type = VAL_VOID };

    /* Mark as running and set context flag */
    c->status = CORO_RUNNING;
    int prev_id = g_current_coroutine_id;
    g_current_coroutine_id = c->id;

    /*
     * Execute the async function via the public call_function() API.
     * call_function() handles parameter binding, env scoping, and body
     * execution — so we don't need to touch eval internals.
     */
    c->result = call_function(c->fn_name, c->args, c->arg_count, c->env);

    /* Restore context */
    g_current_coroutine_id = prev_id;
    c->status = CORO_DONE;

    return c->result;
}

int coroutine_run_all(void) {
    if (!s_initialized) return 0;

    int completed = 0;
    for (int i = 0; i < s_count; i++) {
        Coroutine *c = &s_coroutines[i];
        if (c->status == CORO_PENDING) {
            coroutine_run_to_completion(c->id);
            completed++;
        }
    }
    return completed;
}

Value coroutine_get_result(int id) {
    Coroutine *c = find_by_id(id);
    if (!c || c->status != CORO_DONE) return (Value){ .type = VAL_VOID };
    return c->result;
}

bool coroutine_has_pending(void) {
    for (int i = 0; i < s_count; i++) {
        if (s_coroutines[i].status == CORO_PENDING) return true;
    }
    return false;
}

void coroutine_reset(void) {
    for (int i = 0; i < s_count; i++) {
        Coroutine *c = &s_coroutines[i];
        if (c->fn_name) { free(c->fn_name); c->fn_name = NULL; }
        if (c->args)    { free(c->args);    c->args    = NULL; }
        /* env is owned by the caller — do not free it here */
    }
    memset(s_coroutines, 0, sizeof(s_coroutines));
    s_count                = 0;
    s_next_id              = 0;
    g_current_coroutine_id = -1;
}
