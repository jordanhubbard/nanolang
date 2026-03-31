/*
 * ringbuf.h — Portable single-header typed SPSC lock-free ring buffer
 *
 * Designed for seL4 agentOS protection-domain shared-memory channels,
 * but usable in any C99 environment.
 *
 * Usage:
 *   #include "ringbuf.h"
 *
 *   // Declare a ring buffer type that holds my_event_t with capacity 128
 *   RINGBUF_DEFINE(my_rb, my_event_t, 128)
 *
 *   // Expands to: my_rb_t, my_rb_init, my_rb_push, my_rb_pop, ...
 *   my_rb_t rb;
 *   my_rb_init(&rb);
 *
 *   my_event_t ev = { ... };
 *   my_rb_push(&rb, &ev);   // producer side — returns 0 ok, -1 full
 *
 *   my_event_t out;
 *   my_rb_pop(&rb, &out);   // consumer side — returns 0 ok, -1 empty
 *
 * The default generic type (ringbuf_t / ringbuf_push / ...) is created via:
 *   RINGBUF_DEFINE(ringbuf, ringbuf_entry_t, RINGBUF_SIZE)
 * where ringbuf_entry_t is a 64-byte opaque byte array.  Override by defining
 * RINGBUF_NO_DEFAULT before including this header.
 *
 * Guarantees:
 *   - SPSC only: one writer thread, one reader thread
 *   - Power-of-2 capacity — index wrap via bit-mask (no modulo)
 *   - Producer head / consumer tail on separate 64-byte cache lines
 *   - Acquire/release memory barriers; no locks, no CAS
 */

#ifndef RINGBUF_H
#define RINGBUF_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */

#ifndef RINGBUF_SIZE
#  define RINGBUF_SIZE 64u
#endif

/* Compile-time power-of-2 check — evaluates to 1 or 0 */
#define RINGBUF_IS_POW2(n) (((n) != 0u) && (((n) & ((n) - 1u)) == 0u))

/* -------------------------------------------------------------------------
 * Cache-line alignment
 * ------------------------------------------------------------------------- */

#define RINGBUF_CACHE_LINE 64u

#if defined(__GNUC__) || defined(__clang__)
#  define RINGBUF_ALIGNED __attribute__((aligned(RINGBUF_CACHE_LINE)))
#elif defined(_MSC_VER)
#  define RINGBUF_ALIGNED __declspec(align(64))
#else
#  define RINGBUF_ALIGNED
#endif

/* -------------------------------------------------------------------------
 * Memory barrier macros (GCC / Clang built-ins; safe fallback otherwise)
 * ------------------------------------------------------------------------- */

#if defined(__GNUC__) || defined(__clang__)
#  define RINGBUF_STORE_RELEASE()       __atomic_thread_fence(__ATOMIC_RELEASE)
#  define RINGBUF_LOAD_ACQUIRE()        __atomic_thread_fence(__ATOMIC_ACQUIRE)
#  define RINGBUF_LOAD_RELAXED(x)       __atomic_load_n(&(x), __ATOMIC_RELAXED)
#  define RINGBUF_STORE_RELAXED(x, v)   __atomic_store_n(&(x), (v), __ATOMIC_RELAXED)
#else
#  define RINGBUF_STORE_RELEASE()       do { } while (0)
#  define RINGBUF_LOAD_ACQUIRE()        do { } while (0)
#  define RINGBUF_LOAD_RELAXED(x)       (x)
#  define RINGBUF_STORE_RELAXED(x, v)   ((x) = (v))
#endif

/* -------------------------------------------------------------------------
 * RINGBUF_DEFINE(name, entry_type, capacity)
 *
 * Emits:
 *   typedef struct name##_s  name##_t;
 *   void   name##_init (name##_t *rb);
 *   int    name##_push (name##_t *rb, const entry_type *entry);  0/-1
 *   int    name##_pop  (name##_t *rb, entry_type *out);          0/-1
 *   int    name##_peek (name##_t *rb, entry_type *out);          0/-1
 *   size_t name##_count(name##_t *rb);
 *   int    name##_full (name##_t *rb);
 *   int    name##_empty(name##_t *rb);
 *
 * capacity MUST be a power-of-2 compile-time constant.
 * A negative-array-size error is raised if it is not.
 * ------------------------------------------------------------------------- */

#define RINGBUF_DEFINE(name, entry_type, capacity)                              \
                                                                                \
/* power-of-2 compile-time assertion (C99: negative array size = error) */     \
typedef char name##_pow2_assert[(RINGBUF_IS_POW2(capacity)) ? 1 : -1];         \
                                                                                \
typedef struct name##_s {                                                       \
    /* --- producer-owned cache line --- */                                     \
    RINGBUF_ALIGNED volatile uint32_t head;                                     \
    uint8_t _pad_head[RINGBUF_CACHE_LINE - sizeof(uint32_t)];                   \
    /* --- consumer-owned cache line --- */                                     \
    RINGBUF_ALIGNED volatile uint32_t tail;                                     \
    uint8_t _pad_tail[RINGBUF_CACHE_LINE - sizeof(uint32_t)];                   \
    /* --- data array (kept away from hot indices) --- */                       \
    entry_type buf[(capacity)];                                                 \
} name##_t;                                                                     \
                                                                                \
static inline void name##_init(name##_t *rb) {                                  \
    RINGBUF_STORE_RELAXED(rb->head, 0u);                                        \
    RINGBUF_STORE_RELAXED(rb->tail, 0u);                                        \
}                                                                               \
                                                                                \
/* push — producer only; returns 0 on success, -1 when full */                 \
static inline int name##_push(name##_t *rb, const entry_type *entry) {         \
    uint32_t h = RINGBUF_LOAD_RELAXED(rb->head);                                \
    uint32_t t;                                                                 \
    RINGBUF_LOAD_ACQUIRE();                                                     \
    t = RINGBUF_LOAD_RELAXED(rb->tail);                                         \
    if (((h - t) & 0xFFFFFFFFu) >= (uint32_t)(capacity)) {                     \
        return -1;                                                              \
    }                                                                           \
    rb->buf[h & ((uint32_t)(capacity) - 1u)] = *entry;                         \
    RINGBUF_STORE_RELEASE();                                                    \
    RINGBUF_STORE_RELAXED(rb->head, h + 1u);                                    \
    return 0;                                                                   \
}                                                                               \
                                                                                \
/* pop — consumer only; returns 0 on success, -1 when empty */                 \
static inline int name##_pop(name##_t *rb, entry_type *out) {                   \
    uint32_t t = RINGBUF_LOAD_RELAXED(rb->tail);                                \
    uint32_t h;                                                                 \
    RINGBUF_LOAD_ACQUIRE();                                                     \
    h = RINGBUF_LOAD_RELAXED(rb->head);                                         \
    if (h == t) {                                                               \
        return -1;                                                              \
    }                                                                           \
    *out = rb->buf[t & ((uint32_t)(capacity) - 1u)];                            \
    RINGBUF_STORE_RELEASE();                                                    \
    RINGBUF_STORE_RELAXED(rb->tail, t + 1u);                                    \
    return 0;                                                                   \
}                                                                               \
                                                                                \
/* peek — like pop but does not advance tail */                                 \
static inline int name##_peek(name##_t *rb, entry_type *out) {                  \
    uint32_t t = RINGBUF_LOAD_RELAXED(rb->tail);                                \
    uint32_t h;                                                                 \
    RINGBUF_LOAD_ACQUIRE();                                                     \
    h = RINGBUF_LOAD_RELAXED(rb->head);                                         \
    if (h == t) {                                                               \
        return -1;                                                              \
    }                                                                           \
    *out = rb->buf[t & ((uint32_t)(capacity) - 1u)];                            \
    return 0;                                                                   \
}                                                                               \
                                                                                \
/* count — approximate when read from a different core than the writer */      \
static inline size_t name##_count(name##_t *rb) {                               \
    uint32_t h, t;                                                              \
    RINGBUF_LOAD_ACQUIRE();                                                     \
    h = RINGBUF_LOAD_RELAXED(rb->head);                                         \
    t = RINGBUF_LOAD_RELAXED(rb->tail);                                         \
    return (size_t)((h - t) & 0xFFFFFFFFu);                                    \
}                                                                               \
                                                                                \
static inline int name##_full(name##_t *rb) {                                   \
    return name##_count(rb) >= (size_t)(capacity);                              \
}                                                                               \
                                                                                \
static inline int name##_empty(name##_t *rb) {                                  \
    return name##_count(rb) == 0u;                                              \
}

/* -------------------------------------------------------------------------
 * Default generic ring buffer
 *
 * Provides:  ringbuf_t, ringbuf_init, ringbuf_push, ringbuf_pop,
 *            ringbuf_peek, ringbuf_count, ringbuf_full, ringbuf_empty
 *
 * Entry type is a fixed-size byte array; cast your struct pointer to
 * ringbuf_entry_t* when calling push/pop, or use RINGBUF_DEFINE for a
 * fully type-safe variant.
 *
 * Suppress with:  #define RINGBUF_NO_DEFAULT
 * ------------------------------------------------------------------------- */

#ifndef RINGBUF_NO_DEFAULT

typedef struct {
    uint8_t bytes[64];
} ringbuf_entry_t;

RINGBUF_DEFINE(ringbuf, ringbuf_entry_t, RINGBUF_SIZE)

#endif /* RINGBUF_NO_DEFAULT */

#endif /* RINGBUF_H */
