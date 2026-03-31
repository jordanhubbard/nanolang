/*
 * refcount_gc.h — Reference-counting GC for nanolang WASM-compiled programs
 *
 * Provides a lightweight reference-counting allocator for heap-allocated
 * strings and closures in nanolang programs compiled to WASM via the C
 * transpiler path (nanoc → clang --target wasm32-unknown-unknown).
 *
 * Unlike the interpreter-side gc.h (which uses a doubly-linked list and
 * full object graph), this GC is designed for the constrained WASM linear
 * memory environment:
 *
 *   - No OS allocator; uses a bump-pointer pool backed by wasm_heap[].
 *   - Reference counts are stored in a 4-byte prefix before each object.
 *   - nl_rc_retain(ptr) / nl_rc_release(ptr) bump the count.
 *   - nl_rc_release drops to zero → zeroes the region (no free list yet).
 *   - nl_rc_str_new(bytes, len) allocates a heap string and returns ptr.
 *   - nl_rc_str_concat(a, b) returns a new heap string.
 *   - Closure captures retain their upvalue pointers on creation and
 *     release on drop.
 *
 * Integration with the WASM backend / transpiler:
 *   - Include this header in the generated C preamble when --target wasm
 *     is combined with string or closure expressions.
 *   - The DCE pass marks variables whose last use triggers nl_rc_release.
 *   - The par-let pass treats RC operations as atomic (no cross-binding deps).
 *
 * WASM linear memory layout (assumed):
 *   [0x00000 .. WASM_HEAP_BASE)  — stack, globals
 *   [WASM_HEAP_BASE .. WASM_HEAP_END)  — refcount_gc managed heap
 *
 * USAGE
 * -----
 *   // Allocate a new string
 *   char *s = nl_rc_str_new("hello", 5);
 *   nl_rc_retain(s);          // extra reference
 *   nl_rc_release(s);         // drop extra ref
 *   nl_rc_release(s);         // last ref: object zeroed
 *
 *   // String concatenation (returns newly allocated string, rc=1)
 *   char *r = nl_rc_str_concat(s1, s2);
 *
 * Thread safety: none. WASM execution in agentOS is single-threaded per slot.
 */

#ifndef NANOLANG_REFCOUNT_GC_H
#define NANOLANG_REFCOUNT_GC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

/* ── Heap configuration ─────────────────────────────────────────────────── */

/*
 * WASM_HEAP_SIZE — total heap bytes managed by refcount_gc.
 * Default 256KB, sufficient for typical agentOS WASM slots.
 * Override with -DNL_WASM_HEAP_SIZE=<bytes> at compile time.
 */
#ifndef NL_WASM_HEAP_SIZE
#  define NL_WASM_HEAP_SIZE (256 * 1024)
#endif

/* Alignment for all allocations (8 bytes covers i64 + pointers) */
#define NL_WASM_ALIGN 8

/* ── RC header prepended to every allocation ────────────────────────────── */

typedef struct {
    uint32_t refcount;   /* reference count; 0 = free */
    uint32_t size;       /* usable payload size in bytes (not including header) */
} NLRCHeader;

#define NL_RC_HEADER_SIZE  ((uint32_t)sizeof(NLRCHeader))

/* Given a user pointer, get its header */
#define NL_RC_HDR(ptr)  ((NLRCHeader *)((char *)(ptr) - NL_RC_HEADER_SIZE))

/* ── Heap state ─────────────────────────────────────────────────────────── */

/* The heap is a static array; the bump pointer advances on each alloc.
 * Freed objects are zeroed but not reclaimed into a free list (v1.0).
 * A compacting pass is planned for v2.0. */
static uint8_t  _nl_wasm_heap[NL_WASM_HEAP_SIZE];
static uint32_t _nl_wasm_bump  = 0;   /* next allocation offset */
static uint32_t _nl_wasm_alloc_count = 0;
static uint32_t _nl_wasm_live_bytes  = 0;

/* ── Core allocator ──────────────────────────────────────────────────────── */

/*
 * nl_rc_alloc(size) — allocate `size` bytes with refcount=1.
 * Returns NULL if the heap is exhausted.
 */
static inline void *nl_rc_alloc(uint32_t size) {
    /* Round up to alignment */
    uint32_t aligned = (NL_RC_HEADER_SIZE + size + (NL_WASM_ALIGN - 1))
                       & ~(uint32_t)(NL_WASM_ALIGN - 1);
    if (_nl_wasm_bump + aligned > NL_WASM_HEAP_SIZE) {
        /* OOM: attempt no-op; caller gets NULL */
        return NULL;
    }
    NLRCHeader *hdr = (NLRCHeader *)(_nl_wasm_heap + _nl_wasm_bump);
    _nl_wasm_bump += aligned;

    hdr->refcount = 1;
    hdr->size     = size;
    _nl_wasm_alloc_count++;
    _nl_wasm_live_bytes += size;

    return (char *)hdr + NL_RC_HEADER_SIZE;
}

/* ── Retain / Release ────────────────────────────────────────────────────── */

/*
 * nl_rc_retain(ptr) — increment reference count.
 * No-op on NULL.
 */
static inline void nl_rc_retain(void *ptr) {
    if (!ptr) return;
    NL_RC_HDR(ptr)->refcount++;
}

/*
 * nl_rc_release(ptr) — decrement reference count.
 * When count reaches zero: zero the payload (helps leak detection) and
 * update live_bytes counter.  The bump pointer is NOT retracted (v1.0).
 * No-op on NULL.
 */
static inline void nl_rc_release(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (hdr->refcount == 0) return;   /* already freed (double-release guard) */
    hdr->refcount--;
    if (hdr->refcount == 0) {
        _nl_wasm_live_bytes -= (hdr->size < _nl_wasm_live_bytes)
                                ? hdr->size : _nl_wasm_live_bytes;
        /* Zero payload to help mem_profiler leak detection */
        memset(ptr, 0, hdr->size);
        hdr->size = 0;
    }
}

/*
 * nl_rc_refcount(ptr) — query current reference count (for debug/test).
 */
static inline uint32_t nl_rc_refcount(const void *ptr) {
    if (!ptr) return 0;
    return NL_RC_HDR((void *)ptr)->refcount;
}

/* ── String helpers ──────────────────────────────────────────────────────── */

/*
 * nl_rc_str_new(data, len) — allocate a heap string (null-terminated).
 * The returned pointer points directly to the character data (C-string
 * compatible).  RC = 1 on return.
 * Returns NULL on OOM.
 */
static inline char *nl_rc_str_new(const char *data, uint32_t len) {
    char *p = (char *)nl_rc_alloc(len + 1);
    if (!p) return NULL;
    if (data && len) memcpy(p, data, len);
    p[len] = '\0';
    return p;
}

/*
 * nl_rc_str_from_cstr(s) — convenience: create heap string from C literal.
 * Returns NULL on OOM.
 */
static inline char *nl_rc_str_from_cstr(const char *s) {
    if (!s) return nl_rc_str_new("", 0);
    uint32_t len = (uint32_t)strlen(s);
    return nl_rc_str_new(s, len);
}

/*
 * nl_rc_str_concat(a, b) — concatenate two heap strings.
 * Returns a new heap string with RC=1.
 * Retains a and b only for the duration of the call (does not bump their RC).
 * Returns NULL on OOM.
 */
static inline char *nl_rc_str_concat(const char *a, const char *b) {
    uint32_t la = a ? (uint32_t)strlen(a) : 0;
    uint32_t lb = b ? (uint32_t)strlen(b) : 0;
    char *out = nl_rc_str_new(NULL, la + lb);
    if (!out) return NULL;
    if (a && la) memcpy(out,      a, la);
    if (b && lb) memcpy(out + la, b, lb);
    out[la + lb] = '\0';
    return out;
}

/*
 * nl_rc_str_eq(a, b) — null-safe string equality (for == operator on strings).
 */
static inline bool nl_rc_str_eq(const char *a, const char *b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

/* ── Closure support ─────────────────────────────────────────────────────── */

/*
 * NLClosure — heap-allocated closure record.
 *
 * Closures capture upvalues by reference (each upvalue is itself a heap
 * pointer managed by the RC system).  On closure creation, each captured
 * pointer is retained; on release, each is released.
 *
 * fn_ptr is an opaque function pointer index for indirect WASM calls.
 */
typedef struct {
    uint32_t  fn_index;          /* WASM table index for indirect call */
    uint32_t  upvalue_count;
    void     *upvalues[];        /* flexible array: upvalue_count pointers */
} NLClosure;

/*
 * nl_rc_closure_new(fn_index, uv_count, upvalues[]) — allocate a closure.
 * Each upvalue pointer is retained.  Returns NULL on OOM.
 */
static inline NLClosure *nl_rc_closure_new(uint32_t fn_index,
                                            uint32_t uv_count,
                                            void   **upvalues) {
    uint32_t sz = (uint32_t)sizeof(NLClosure) + uv_count * (uint32_t)sizeof(void *);
    NLClosure *cl = (NLClosure *)nl_rc_alloc(sz);
    if (!cl) return NULL;
    cl->fn_index      = fn_index;
    cl->upvalue_count = uv_count;
    for (uint32_t i = 0; i < uv_count; i++) {
        cl->upvalues[i] = upvalues[i];
        nl_rc_retain(upvalues[i]);
    }
    return cl;
}

/*
 * nl_rc_closure_release(cl) — release closure and all its upvalues.
 */
static inline void nl_rc_closure_release(NLClosure *cl) {
    if (!cl) return;
    NLRCHeader *hdr = NL_RC_HDR(cl);
    if (hdr->refcount <= 1) {
        /* About to free: release upvalues first */
        for (uint32_t i = 0; i < cl->upvalue_count; i++) {
            nl_rc_release(cl->upvalues[i]);
        }
    }
    nl_rc_release(cl);
}

/* ── Heap stats (for mem_profiler integration) ───────────────────────────── */

/*
 * nl_rc_heap_stats(live_allocs_out, live_bytes_out) — fill stats for
 * the agentOS mem_profiler PD.  Called from the WASM module's
 * __agentos_heap_stats export (hooked by wasm3_host.c).
 */
static inline void nl_rc_heap_stats(uint32_t *live_allocs_out,
                                     uint32_t *live_bytes_out) {
    /* Count live allocs by scanning the heap (O(n) but called infrequently) */
    uint32_t live = 0;
    uint32_t pos  = 0;
    while (pos + NL_RC_HEADER_SIZE <= _nl_wasm_bump) {
        NLRCHeader *hdr = (NLRCHeader *)(_nl_wasm_heap + pos);
        if (hdr->refcount > 0) live++;
        /* Advance by minimum alignment unit to scan every slot.
         * The actual slot size is hdr->size + NL_RC_HEADER_SIZE rounded up. */
        uint32_t slot = (NL_RC_HEADER_SIZE + hdr->size + (NL_WASM_ALIGN - 1))
                         & ~(uint32_t)(NL_WASM_ALIGN - 1);
        if (slot < NL_WASM_ALIGN) slot = NL_WASM_ALIGN;  /* safety */
        pos += slot;
    }
    if (live_allocs_out) *live_allocs_out = live;
    if (live_bytes_out)  *live_bytes_out  = _nl_wasm_live_bytes;
}

/*
 * nl_rc_heap_reset() — reset the bump pointer (for test harnesses only).
 * NOT safe to call at runtime — all live pointers become dangling.
 */
static inline void nl_rc_heap_reset(void) {
    memset(_nl_wasm_heap, 0, _nl_wasm_bump);
    _nl_wasm_bump        = 0;
    _nl_wasm_alloc_count = 0;
    _nl_wasm_live_bytes  = 0;
}

#endif /* NANOLANG_REFCOUNT_GC_H */
