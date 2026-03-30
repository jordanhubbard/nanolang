/* nano_dispatch.h — Automatic guarded mutables for NanoLang
 *
 * On Apple platforms: uses Grand Central Dispatch (libdispatch) for thread-safe
 * guarded access.  Writes are async, reads are sync.
 *
 * On non-Apple platforms: sequential fallback.  par {} blocks are an
 * *independence annotation*, not a concurrency primitive — sequential execution
 * of independent blocks is always semantically correct per the NanoLang spec.
 *
 * Requires on Apple: clang -fblocks (applied via cflags_macos in module.json)
 */
#ifndef NANO_DISPATCH_H
#define NANO_DISPATCH_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __APPLE__

#include <dispatch/dispatch.h>

/* ---------- Guarded-value structs (one per NanoLang primitive type) ---------- */

typedef struct { dispatch_queue_t _q; int64_t  _v; } NlGuarded_int;
typedef struct { dispatch_queue_t _q; double   _v; } NlGuarded_float;
typedef struct { dispatch_queue_t _q; bool     _v; } NlGuarded_bool;
typedef struct { dispatch_queue_t _q; char*    _v; } NlGuarded_str;

/* For struct types the transpiler emits an inline typedef, e.g.:
 *   typedef struct { dispatch_queue_t _q; MyStruct _v; } NlGuarded_MyStruct;
 */

/* ---------- Init macro ---------- */
#define NL_GUARDED_INIT(g, name, init)                                         \
    do {                                                                        \
        (g)._q = dispatch_queue_create((name), DISPATCH_QUEUE_SERIAL);         \
        (g)._v = (init);                                                        \
    } while (0)

/* ---------- Sync read — blocks until queue drains, then returns value ---------- */
#define NL_GUARDED_READ(g, T)                                                   \
    ({                                                                          \
        __block T _nl_tmp;                                                      \
        dispatch_sync((g)._q, ^{ _nl_tmp = (g)._v; });                         \
        _nl_tmp;                                                                \
    })

/* ---------- Async write — fire-and-forget, enqueues on the variable's queue --- */
#define NL_GUARDED_WRITE(g, expr)                                               \
    do {                                                                        \
        __typeof__((g)._v) _nl_val = (expr);                                   \
        dispatch_async((g)._q, ^{ (g)._v = _nl_val; });                        \
    } while (0)

/* ---------- Cleanup ---------- */
#define NL_GUARDED_DESTROY(g)                                                   \
    do {                                                                        \
        dispatch_barrier_sync((g)._q, ^{});                                     \
        dispatch_release((g)._q);                                               \
    } while (0)

#else /* Non-Apple: sequential fallback (no GCD/Blocks required) */

/* Without GCD, guarded structs hold the value directly.
 * par {} blocks are an independence annotation — sequential execution is correct.
 */
typedef struct { int64_t  _v; } NlGuarded_int;
typedef struct { double   _v; } NlGuarded_float;
typedef struct { bool     _v; } NlGuarded_bool;
typedef struct { char*    _v; } NlGuarded_str;

#define NL_GUARDED_INIT(g, name, init)   do { (void)(name); (g)._v = (init); } while (0)
#define NL_GUARDED_READ(g, T)            ((T)(g)._v)
#define NL_GUARDED_WRITE(g, expr)        do { (g)._v = (expr); } while (0)
#define NL_GUARDED_DESTROY(g)            do { (void)(g); } while (0)

#endif /* __APPLE__ */

#endif /* NANO_DISPATCH_H */
