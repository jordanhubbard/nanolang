/*
 * coroutine.h — nanolang cooperative coroutine scheduler
 *
 * Provides a simple run-queue of suspended continuations for async/await.
 *
 * Design
 * ──────
 * Each coroutine wraps a pending call to an async function.  The scheduler
 * holds it in a run_queue and executes it cooperatively:
 *
 *   - coroutine_spawn(fn_name, args, arg_count, env) — enqueue a coroutine
 *   - coroutine_run_to_completion(id)                — run one coroutine now
 *   - coroutine_run_all()                            — drain the run_queue
 *
 * await semantics
 * ───────────────
 * Inside an async function body, `await expr` evaluates `expr` immediately
 * (transparent/eager).  The g_current_coroutine_id global marks that we are
 * executing inside a coroutine context.  This enables future extensions
 * (true suspension, I/O integration) without changing the eval interface.
 *
 * Backward compatibility
 * ──────────────────────
 * When g_current_coroutine_id == -1 (synchronous context), async fn calls
 * and await both behave exactly as before — no scheduler overhead.
 */
#pragma once
#include "nanolang.h"

/* Maximum number of concurrent coroutines in the run queue */
#define COROUTINE_MAX 64

/* Coroutine status */
typedef enum {
    CORO_PENDING,    /* Spawned, not yet started */
    CORO_RUNNING,    /* Currently executing */
    CORO_DONE,       /* Completed — result valid */
    CORO_ERROR       /* Failed */
} CoroutineStatus;

/* A coroutine entry in the run queue */
typedef struct {
    int              id;
    CoroutineStatus  status;
    /* Call descriptor — used to invoke the async function */
    char            *fn_name;
    Value           *args;
    int              arg_count;
    Environment     *env;
    /* Result (valid when status == CORO_DONE) */
    Value            result;
} Coroutine;

/* ── Scheduler API ──────────────────────────────────────────────────────────── */

/* Initialize the scheduler (idempotent). */
void  coroutine_init(void);

/* Spawn a new coroutine.
 * fn_name   : name of the async function to call
 * args      : argument array (caller-owned; scheduler copies)
 * arg_count : number of arguments
 * env       : execution environment
 * Returns coroutine id (>= 0) or -1 on failure.
 */
int   coroutine_spawn(const char *fn_name, Value *args, int arg_count, Environment *env);

/* Run a single coroutine to completion and return its result.
 * If already done, returns cached result.
 */
Value coroutine_run_to_completion(int id);

/* Drain all PENDING coroutines in FIFO order.
 * Returns number of coroutines completed.
 */
int   coroutine_run_all(void);

/* Fetch the result of a DONE coroutine.  Returns void if not done. */
Value coroutine_get_result(int id);

/* True if any coroutine is still PENDING. */
bool  coroutine_has_pending(void);

/* Free all coroutine state (call between test runs or at program exit). */
void  coroutine_reset(void);

/* ── Context flag ────────────────────────────────────────────────────────────
 * Set to the running coroutine id while inside a coroutine body.
 * -1 = synchronous (non-coroutine) context.
 * eval.c reads this to decide cooperative-await vs transparent-await.
 */
extern int g_current_coroutine_id;
