/* nano_dispatch.h — Automatic guarded mutables for NanoLang
 *
 * Every `let mut` variable in NanoLang is backed by one of these structs.
 * All writes go through NL_GUARDED_WRITE (dispatch_async, fire-and-forget).
 * All reads  go through NL_GUARDED_READ  (dispatch_sync,  waits for queue).
 * Callers have zero awareness of locks, joins, or thread state.
 *
 * Requires: clang -fblocks, -ldispatch (macOS: system; Linux: libdispatch-dev)
 */
#ifndef NANO_DISPATCH_H
#define NANO_DISPATCH_H

#ifdef __APPLE__
#  include <dispatch/dispatch.h>
#else
#  include <dispatch/dispatch.h>   /* swift-corelibs-libdispatch on Linux */
#endif
#include <stdint.h>
#include <stdbool.h>

/* ---------- Guarded-value structs (one per NanoLang primitive type) ---------- */

typedef struct { dispatch_queue_t _q; int64_t  _v; } NlGuarded_int;
typedef struct { dispatch_queue_t _q; double   _v; } NlGuarded_float;
typedef struct { dispatch_queue_t _q; bool     _v; } NlGuarded_bool;
typedef struct { dispatch_queue_t _q; char*    _v; } NlGuarded_str;

/* For struct types the transpiler emits an inline typedef, e.g.:
 *   typedef struct { dispatch_queue_t _q; MyStruct _v; } NlGuarded_MyStruct;
 */

/* ---------- Init macro ----------
 * Usage: NL_GUARDED_INIT(g, "nano.main.counter", 0);
 *   Creates a serial queue named after the variable + scope for Instruments.
 */
#define NL_GUARDED_INIT(g, name, init)                                         \
    do {                                                                        \
        (g)._q = dispatch_queue_create((name), DISPATCH_QUEUE_SERIAL);         \
        (g)._v = (init);                                                        \
    } while (0)

/* ---------- Sync read — blocks until queue drains, then returns value ----------
 * Usage:  int64_t x = NL_GUARDED_READ(counter, int64_t);
 * Implemented as a GNU/clang statement expression so it works inside any expr.
 */
#define NL_GUARDED_READ(g, T)                                                   \
    ({                                                                          \
        __block T _nl_tmp;                                                      \
        dispatch_sync((g)._q, ^{ _nl_tmp = (g)._v; });                         \
        _nl_tmp;                                                                \
    })

/* ---------- Async write — fire-and-forget, enqueues on the variable's queue ---
 * Usage:  NL_GUARDED_WRITE(counter, counter._v + 1);
 * The rhs expression is evaluated on the *caller* thread before enqueuing, so
 * cross-variable reads that appear in `expr` must be wrapped in NL_GUARDED_READ
 * by the transpiler before this macro is expanded.
 */
#define NL_GUARDED_WRITE(g, expr)                                               \
    do {                                                                        \
        __typeof__((g)._v) _nl_val = (expr);                                   \
        dispatch_async((g)._q, ^{ (g)._v = _nl_val; });                        \
    } while (0)

/* ---------- Cleanup ----------
 * Drain the queue (barrier sync) and release it.
 * The transpiler emits this at end-of-scope for every guarded local.
 */
#define NL_GUARDED_DESTROY(g)                                                   \
    do {                                                                        \
        dispatch_barrier_sync((g)._q, ^{});                                     \
        dispatch_release((g)._q);                                               \
    } while (0)

#endif /* NANO_DISPATCH_H */
