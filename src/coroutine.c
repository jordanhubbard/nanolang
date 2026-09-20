/*
 * coroutine.c — nanolang cooperative coroutine scheduler
 *
 * This implements a simple cooperative coroutine scheduler based on
 * deferred function execution. Each coroutine is spawned as a pending
 * call that runs to completion when the scheduler processes it.
 *
 * Design notes:
 * - "spawn(fn, args)" creates a coroutine entry in the run_queue
 * - "scheduler_run()" or "await coro_handle" drains the run queue
 * - "yield()" is currently a no-op; it does not re-queue or suspend a call
 * - No setjmp/longjmp needed; each coroutine runs as a normal function call
 * - "await coro_val" runs the scheduler until the target coroutine is done
 *
 * I do not implement resumable async/await here. My CPS walker does not
 * create continuations. Await may execute nested callbacks on the C stack.
 */

#include "coroutine.h"
#include "nanolang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* ── Global scheduler instance ─────────────────────────────────────────── */
NanoScheduler g_scheduler = { .initialized = false };

/* ── Initialization ─────────────────────────────────────────────────── */
void nano_scheduler_init(void) {
    if (g_scheduler.initialized) return;
    memset(&g_scheduler, 0, sizeof(g_scheduler));
    g_scheduler.current = -1;
    g_scheduler.initialized = true;
    for (int i = 0; i < MAX_COROUTINES; i++) {
        g_scheduler.coroutines[i].id = -1;
        g_scheduler.coroutines[i].status = CORO_DONE;
        g_scheduler.coroutines[i].awaiting_id = -1;
    }
}

/* ── Internal: find coroutine by id ───────────────────────────────────── */
static NanoCoroutine *coro_by_id(int id) {
    if (!g_scheduler.initialized || id < 0) return NULL;
    for (int i = 0; i < MAX_COROUTINES; i++) {
        if (g_scheduler.coroutines[i].id == id) {
            return &g_scheduler.coroutines[i];
        }
    }
    return NULL;
}

/* ── Spawn ─────────────────────────────────────────────────────────────── */
static int coro_spawn(CoroFn fn, void *arg, CoroArgDropFn arg_drop,
                      CoroResultDropFn result_drop, CoroResultCloneFn result_clone) {
    if (!g_scheduler.initialized) nano_scheduler_init();
    if (!fn || g_scheduler.count == INT_MAX) return -1;

    /* Find a free slot */
    int slot = -1;
    for (int i = 0; i < MAX_COROUTINES; i++) {
        if (!g_scheduler.coroutines[i].active && g_scheduler.coroutines[i].id < 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        fprintf(stderr, "[coroutine] Error: MAX_COROUTINES (%d) exceeded\n", MAX_COROUTINES);
        return -1;
    }

    int id = g_scheduler.count++;
    NanoCoroutine *coro = &g_scheduler.coroutines[slot];
    coro->id = id;
    coro->status = CORO_READY;
    coro->fn = fn;
    coro->arg = arg;
    coro->arg_drop = arg_drop;
    coro->result_drop = result_drop;
    coro->result_clone = result_clone;
    coro->awaiting_id = -1;
    coro->error_msg = NULL;
    memset(&coro->result, 0, sizeof(Value));
    coro->result.type = VAL_VOID;

    return id;
}

int nano_coro_spawn(CoroFn fn, void *arg) {
    return coro_spawn(fn, arg, NULL, NULL, NULL);
}

int nano_coro_spawn_owned(CoroFn fn, void *arg, CoroArgDropFn arg_drop,
                         CoroResultDropFn result_drop, CoroResultCloneFn result_clone) {
    if (!arg_drop || !result_drop || !result_clone) return -1;
    return coro_spawn(fn, arg, arg_drop, result_drop, result_clone);
}

static void coro_drop_argument(NanoCoroutine *coro) {
    CoroArgDropFn drop = coro->arg_drop;
    if (!drop) return; /* Legacy borrowed arguments stay observable until release. */
    void *arg = coro->arg;
    coro->arg_drop = NULL;
    coro->arg = NULL;
    coro->fn = NULL;
    if (drop) drop(arg);
}

static void coro_fail(NanoCoroutine *coro, const char *message) {
    coro->status = CORO_ERROR;
    coro->error_msg = message ? strdup(message) : NULL;
}

bool nano_coro_release(int id) {
    NanoCoroutine *coro = coro_by_id(id);
    if (!coro || coro->active ||
        (coro->status != CORO_DONE && coro->status != CORO_ERROR)) return false;
    /* My active latch also protects trusted cleanup hooks from slot reuse. */
    coro->active = true;
    CoroResultDropFn drop = coro->result_drop;
    Value result = coro->result;
    coro->result_drop = NULL;
    coro->result_clone = NULL;
    memset(&coro->result, 0, sizeof(coro->result));
    coro->result.type = VAL_VOID;
    coro_drop_argument(coro);
    if (drop) drop(result);
    free(coro->error_msg);
    memset(coro, 0, sizeof(*coro));
    coro->id = -1;
    coro->status = CORO_DONE;
    coro->awaiting_id = -1;
    coro->result.type = VAL_VOID;
    return true;
}

bool nano_coro_cancel(int id) {
    NanoCoroutine *coro = coro_by_id(id);
    if (!coro || coro->active || coro->status != CORO_READY) return false;
    coro->active = true;
    coro_fail(coro, "I cancelled this pending task.");
    coro_drop_argument(coro);
    coro->active = false;
    return true;
}

/* Both public execution paths use the same active-frame and ownership rule.
 * Early complete/error is terminal, but does not make my callback reclaimable. */
static bool coro_run(int slot) {
    NanoCoroutine *coro = &g_scheduler.coroutines[slot];
    if (coro->active || coro->status != CORO_READY || !coro->fn) return false;
    int previous = g_scheduler.current;
    g_scheduler.current = slot;
    coro->status = CORO_RUNNING;
    coro->active = true;
    Value result = coro->fn(coro->arg, coro->id);
    if (coro->status == CORO_RUNNING) {
        coro->result = result;
        coro->status = CORO_DONE;
    } else {
        if (coro->status != CORO_DONE && coro->status != CORO_ERROR)
            coro_fail(coro, "I require a terminal state when a task callback returns.");
        CoroResultDropFn drop = coro->result_drop;
        if (drop) drop(result);
    }
    coro_drop_argument(coro);
    g_scheduler.current = previous;
    coro->active = false;
    if (g_scheduler.step_count < INT_MAX) ++g_scheduler.step_count;
    return true;
}

/* ── Yield ─────────────────────────────────────────────────────────────── */
/*
 * yield() — cooperative hint to allow other coroutines to run.
 * In our simple non-setjmp scheduler, yield is a no-op for the current
 * coroutine (it continues executing). Await can execute another callback
 * on the same C stack; spawn only queues work.
 *
 * For a full preemptive/cooperative yield, ucontext_t or fibers would be needed.
 * I do not claim suspension or fairness from this no-op.
 */
void nano_coro_yield(void) {
    /* In the simple scheduler: yield is a no-op.
     * The current coroutine continues to its next await or return. */
}

/* ── Scheduler step ─────────────────────────────────────────────────── */
bool nano_scheduler_step(void) {
    if (!g_scheduler.initialized) return false;

    /* Find next READY coroutine */
    int found = -1;
    for (int i = 0; i < MAX_COROUTINES; i++) {
        NanoCoroutine *c = &g_scheduler.coroutines[i];
        if (c->id < 0) continue;

        /* Wake up coroutines that were waiting for a now-completed dependency */
        if (c->status == CORO_SUSPENDED && c->awaiting_id >= 0) {
            NanoCoroutine *awaited = coro_by_id(c->awaiting_id);
            if (awaited && (awaited->status == CORO_DONE || awaited->status == CORO_ERROR)) {
                c->status = CORO_READY;
                c->awaiting_id = -1;
            }
        }

        if (c->status == CORO_READY) {
            found = i;
            break;
        }
    }

    if (found < 0) return false; /* Nothing ready */

    return coro_run(found);
}

/* ── Run until done ─────────────────────────────────────────────────── */
void nano_scheduler_run_until_done(void) {
    if (!g_scheduler.initialized) return;

    int max_steps = MAX_COROUTINES * 10000; /* Safety limit */
    int steps = 0;
    while (nano_scheduler_pending_count() > 0 && steps++ < max_steps) {
        if (!nano_scheduler_step()) break;
    }
    if (steps >= max_steps) {
        fprintf(stderr, "[coroutine] Warning: scheduler safety limit reached (%d steps)\n", max_steps);
    }
}

/* ── Await a specific coroutine ─────────────────────────────────────── */
Value nano_coro_await_id(int coro_id) {
    NanoCoroutine *target = coro_by_id(coro_id);
    if (!target) {
        nano_coro_error("I cannot await an invalid or released task handle.");
        Value v;
        memset(&v, 0, sizeof(v));
        v.type = VAL_VOID;
        return v;
    }

    /* If already done, return result immediately */
    if (target->status == CORO_DONE) {
        return target->result;
    }

    /* In this run-to-completion scheduler every active target is on the
     * current C call stack. Waiting for it would wait on myself or an ancestor. */
    if (target->active && target->status != CORO_ERROR) {
        nano_coro_error("I cannot await myself or an active ancestor task.");
        Value result = {0};
        result.type = VAL_VOID;
        return result;
    }

    /* Run the scheduler until this coroutine is done */
    int max_steps = MAX_COROUTINES * 1000;
    int steps = 0;
    while (target->status != CORO_DONE && target->status != CORO_ERROR
           && steps++ < max_steps) {
        /* Try to run the target coroutine directly if it's ready */
        if (target->status == CORO_READY) {
            int slot = (int)(target - g_scheduler.coroutines);
            if (!coro_run(slot)) break;
        } else {
            /* Run other ready coroutines that might unblock this one */
            if (!nano_scheduler_step()) break;
        }
    }

    if (target->status == CORO_DONE) return target->result;

    nano_coro_error("I could not complete the awaited task.");

    Value v;
    memset(&v, 0, sizeof(v));
    v.type = VAL_VOID;
    return v;
}

/* ── Helpers ─────────────────────────────────────────────────────────── */
Value nano_coro_result(int coro_id) {
    NanoCoroutine *c = coro_by_id(coro_id);
    if (!c) {
        Value v;
        memset(&v, 0, sizeof(v));
        v.type = VAL_VOID;
        return v;
    }
    return c->result;
}

bool nano_coro_result_copy(int id, Value *out) {
    NanoCoroutine *coro = coro_by_id(id);
    if (!coro || !out || out == &coro->result || coro->status != CORO_DONE || !coro->result_clone) return false;
    bool active = coro->active;
    coro->active = true;
    Value copy;
    bool ok = coro->result_clone(coro->result, &copy);
    coro->active = active;
    if (ok) *out = copy;
    return ok;
}

bool nano_coro_await_copy(int id, Value *out) {
    NanoCoroutine *coro = coro_by_id(id);
    if (!coro || !out || out == &coro->result || !coro->result_clone) return false;
    (void)nano_coro_await_id(id);
    return nano_coro_result_copy(id, out);
}

bool nano_coro_is_done(int coro_id) {
    NanoCoroutine *c = coro_by_id(coro_id);
    return c && (c->status == CORO_DONE || c->status == CORO_ERROR);
}

int nano_scheduler_pending_count(void) {
    int count = 0;
    for (int i = 0; i < MAX_COROUTINES; i++) {
        NanoCoroutine *c = &g_scheduler.coroutines[i];
        if (c->id >= 0 &&
            c->status != CORO_DONE &&
            c->status != CORO_ERROR) {
            count++;
        }
    }
    return count;
}

void nano_coro_complete(Value result) {
    if (g_scheduler.current < 0) return;
    NanoCoroutine *c = &g_scheduler.coroutines[g_scheduler.current];
    if (c->status != CORO_RUNNING) return;
    if (c->result_clone) {
        Value copy;
        if (!c->result_clone(result, &copy)) {
            coro_fail(c, "I cannot copy an early task result.");
            return;
        }
        c->result = copy;
    } else c->result = result;
    c->status = CORO_DONE;
}

void nano_coro_error(const char *msg) {
    if (g_scheduler.current < 0) return;
    NanoCoroutine *c = &g_scheduler.coroutines[g_scheduler.current];
    if (c->status != CORO_RUNNING) return;
    coro_fail(c, msg);
}

int nano_coro_current_id(void) {
    if (g_scheduler.current < 0) return -1;
    return g_scheduler.coroutines[g_scheduler.current].id;
}
