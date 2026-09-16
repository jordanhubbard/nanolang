/*
 * coroutine.h — nanolang cooperative coroutine scheduler
 *
 * Implements a simple cooperative task scheduler for async/await.
 * Each coroutine is a deferred function call in a run_queue.
 * Coroutines run to completion when stepped; scheduler_run() drains all.
 *
 * Design:
 *   - NanoCoroutine: a pending/running/done task with fn pointer + arg
 *   - NanoScheduler: round-robin run_queue (fixed-size array, max 64)
 *   - nano_coro_spawn(): enqueue a coroutine, returns handle (id)
 *   - nano_coro_yield(): cooperative hint (no-op in simple scheduler)
 *   - nano_coro_await_id(): run scheduler until target coro is done
 *   - nano_scheduler_step(): run one READY coroutine to completion
 *   - nano_scheduler_run_until_done(): drain all pending coroutines
 */

#ifndef COROUTINE_H
#define COROUTINE_H

#include <stdbool.h>
#include <stdint.h>

/* Include full Value type from nanolang.h */
#include "nanolang.h"

/* ── Coroutine status ───────────────────────────────────────────────────── */
typedef enum {
    CORO_READY,       /* In queue, not yet started */
    CORO_RUNNING,     /* Currently executing */
    CORO_SUSPENDED,   /* Waiting on another coroutine (awaiting_id >= 0) */
    CORO_DONE,        /* Completed, result available */
    CORO_ERROR        /* Terminated with error */
} CoroStatus;

/* ── Coroutine entry function type ─────────────────────────────────────── */
/* Called with (arg, coro_id); returns the coroutine's result Value */
typedef Value (*CoroFn)(void *arg, int coro_id);

/* ── NanoCoroutine ─────────────────────────────────────────────────────── */
#define MAX_COROUTINES   64

typedef struct NanoCoroutine {
    int id;
    CoroStatus status;
    CoroFn fn;
    void *arg;
    Value result;        /* Result value when CORO_DONE */
    char *error_msg;     /* Error message when CORO_ERROR */
    int awaiting_id;     /* -1 = not awaiting; >= 0 = waiting for this coro */
} NanoCoroutine;

/* ── NanoScheduler ─────────────────────────────────────────────────────── */
typedef struct {
    NanoCoroutine coroutines[MAX_COROUTINES];
    int count;           /* Total coroutines ever allocated (monotonic) */
    int current;         /* Slot index of running coroutine (-1 = scheduler) */
    int step_count;      /* Total scheduler steps taken */
    bool initialized;
} NanoScheduler;

/* ── Global scheduler ─────────────────────────────────────────────────── */
extern NanoScheduler g_scheduler;

/* ── API ──────────────────────────────────────────────────────────────── */

/* Initialize the global scheduler */
void nano_scheduler_init(void);

/* Spawn a new coroutine: returns coroutine id (>= 0) or -1 on failure */
int nano_coro_spawn(CoroFn fn, void *arg);

/* Cooperative yield hint — allows other coroutines to run.
 * In the simple run-to-completion scheduler, this is a no-op.
 * With ucontext/fiber support, this would switch contexts. */
void nano_coro_yield(void);

/* Run the scheduler until a specific coroutine is done. Returns its result. */
Value nano_coro_await_id(int coro_id);

/* Step the scheduler once: pick the next READY coroutine and run it.
 * Returns false if no more coroutines are ready/running. */
bool nano_scheduler_step(void);

/* Run the scheduler until all coroutines are DONE or ERROR. */
void nano_scheduler_run_until_done(void);

/* Get the result of a completed coroutine */
Value nano_coro_result(int coro_id);

/* Check if a coroutine is done (DONE or ERROR) */
bool nano_coro_is_done(int coro_id);

/* Get count of pending (non-done, non-error) coroutines */
int nano_scheduler_pending_count(void);

/* Mark current coroutine as done with result (called from within a coro) */
void nano_coro_complete(Value result);

/* Mark current coroutine as errored */
void nano_coro_error(const char *msg);

/* Get id of currently running coroutine (-1 if in scheduler context) */
int nano_coro_current_id(void);

#endif /* COROUTINE_H */
