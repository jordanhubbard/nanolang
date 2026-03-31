/*
 * refcount_gc.h — Reference-counting GC + tri-color mark-sweep cycle collector
 *
 * v2.0: Generational/hybrid GC for nanolang WASM-compiled programs.
 *
 * Architecture
 * ────────────
 * Fast path: reference counting with immediate reclamation (RC → 0 = free).
 *
 * Cycle collection: Bacon-Rajan purple-buffer approach (PLDI 2001).
 *   When RC drops but doesn't reach 0, the object is added to the "suspect
 *   cycle buffer" (it may be part of a cycle).  nl_gc_collect_cycles() runs
 *   a tri-color mark-sweep over the suspect buffer to find and reclaim
 *   objects that are only reachable from other objects in the cycle set.
 *
 * Tri-color marking (per-object in NLRCHeader.gc_word):
 *   WHITE (0b00) — not yet visited / unreachable
 *   GRAY  (0b01) — discovered, children not yet traced
 *   BLACK (0b10) — fully traced (reachable, keep)
 *   PURPLE(0b11) — suspected cycle root (on _nl_cycle_buf)
 *
 * Pointer maps: closures carry an upvalue pointer array at a known offset
 * (via NLClosure.upvalues[]).  Objects that contain child pointers register
 * a pointer-count in NLRCHeader.gc_word bits [8..15].  Pure data objects
 * (strings) have child_count = 0.
 *
 * GC trigger: automatic on nl_rc_alloc OOM (collect + retry), or explicit
 * via nl_gc_collect_cycles().
 *
 * WASM linear memory layout:
 *   [0x00000 .. WASM_HEAP_BASE)   — stack, globals
 *   [WASM_HEAP_BASE .. HEAP_END)  — nl_rc managed heap (bump-pointer)
 *
 * Object header layout (12 bytes):
 *   [0..3]  refcount  (uint32_t)
 *   [4..7]  size      (uint32_t)  — usable payload bytes
 *   [8..11] gc_word   (uint32_t):
 *     bits [1:0]  — tri-color (WHITE=0, GRAY=1, BLACK=2, PURPLE=3)
 *     bits [9:2]  — child_count (max 255 child pointers in payload)
 *     bits [10]   — on_cycle_buf flag
 *     bits [11]   — object_type: 0=data/string, 1=closure
 *     bits [31:12] — reserved
 */

#ifndef NANOLANG_REFCOUNT_GC_H
#define NANOLANG_REFCOUNT_GC_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>

/* ── Heap configuration ─────────────────────────────────────────────────── */

#ifndef NL_WASM_HEAP_SIZE
#  define NL_WASM_HEAP_SIZE (256 * 1024)
#endif

#define NL_WASM_ALIGN 8

/* Cycle suspect buffer capacity */
#ifndef NL_CYCLE_BUF_SIZE
#  define NL_CYCLE_BUF_SIZE 256
#endif

/* ── GC header (12 bytes) ────────────────────────────────────────────────── */

typedef struct {
    uint32_t refcount;   /* reference count */
    uint32_t size;       /* usable payload size (bytes, excl. header) */
    uint32_t gc_word;    /* tri-color + child_count + flags (see above) */
} NLRCHeader;

#define NL_RC_HEADER_SIZE  ((uint32_t)sizeof(NLRCHeader))  /* 12 */

/* GC word field accessors */
#define NL_GC_COLOR(hdr)          ((hdr)->gc_word & 0x3u)
#define NL_GC_SET_COLOR(hdr, c)   ((hdr)->gc_word = ((hdr)->gc_word & ~0x3u) | ((c) & 0x3u))
#define NL_GC_CHILD_COUNT(hdr)    (((hdr)->gc_word >> 2) & 0xFFu)
#define NL_GC_SET_CHILDREN(hdr,n) ((hdr)->gc_word = ((hdr)->gc_word & ~(0xFFu<<2)) | (((uint32_t)(n)&0xFFu)<<2))
#define NL_GC_ON_CYCBUF(hdr)      (((hdr)->gc_word >> 10) & 1u)
#define NL_GC_SET_ON_CYCBUF(hdr,v) ((hdr)->gc_word = ((hdr)->gc_word & ~(1u<<10)) | (((uint32_t)(v)&1u)<<10))
#define NL_GC_IS_CLOSURE(hdr)     (((hdr)->gc_word >> 11) & 1u)
#define NL_GC_SET_CLOSURE(hdr)    ((hdr)->gc_word |= (1u<<11))

/* Tri-color constants */
#define NL_GC_WHITE   0u  /* unvisited / unreachable */
#define NL_GC_GRAY    1u  /* discovered, children pending */
#define NL_GC_BLACK   2u  /* fully traced */
#define NL_GC_PURPLE  3u  /* suspected cycle root */

/* Given user pointer → header */
#define NL_RC_HDR(ptr)  ((NLRCHeader *)((char *)(ptr) - NL_RC_HEADER_SIZE))

/* ── Heap state ─────────────────────────────────────────────────────────── */

static uint8_t  _nl_wasm_heap[NL_WASM_HEAP_SIZE];
static uint32_t _nl_wasm_bump        = 0;
static uint32_t _nl_wasm_alloc_count = 0;
static uint32_t _nl_wasm_live_bytes  = 0;

/* Cycle suspect buffer: pointers of RC>0 objects that might be in a cycle */
static void    *_nl_cycle_buf[NL_CYCLE_BUF_SIZE];
static uint32_t _nl_cycle_count = 0;

/* Gray worklist for tri-color marking (stack-based) */
static void    *_nl_gray_stack[NL_CYCLE_BUF_SIZE * 4];
static uint32_t _nl_gray_top = 0;

/* ── Forward declarations ────────────────────────────────────────────────── */

static inline void nl_gc_collect_cycles(void);

/* ── Core allocator ──────────────────────────────────────────────────────── */

static inline void *nl_rc_alloc(uint32_t size) {
    uint32_t aligned = (NL_RC_HEADER_SIZE + size + (NL_WASM_ALIGN - 1))
                       & ~(uint32_t)(NL_WASM_ALIGN - 1);
    if (_nl_wasm_bump + aligned > NL_WASM_HEAP_SIZE) {
        /* OOM — run cycle collector and retry once */
        nl_gc_collect_cycles();
        if (_nl_wasm_bump + aligned > NL_WASM_HEAP_SIZE)
            return NULL;
    }
    NLRCHeader *hdr = (NLRCHeader *)(_nl_wasm_heap + _nl_wasm_bump);
    _nl_wasm_bump += aligned;

    hdr->refcount = 1;
    hdr->size     = size;
    hdr->gc_word  = NL_GC_BLACK;  /* new objects start BLACK (definitely live) */
    _nl_wasm_alloc_count++;
    _nl_wasm_live_bytes += size;

    return (char *)hdr + NL_RC_HEADER_SIZE;
}

/* ── Suspect-cycle buffer ────────────────────────────────────────────────── */

static inline void _nl_cycle_buf_add(void *ptr) {
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (NL_GC_ON_CYCBUF(hdr)) return;
    if (_nl_cycle_count >= NL_CYCLE_BUF_SIZE) {
        /* Buffer full — trigger collection immediately */
        nl_gc_collect_cycles();
    }
    if (_nl_cycle_count < NL_CYCLE_BUF_SIZE) {
        NL_GC_SET_COLOR(hdr, NL_GC_PURPLE);
        NL_GC_SET_ON_CYCBUF(hdr, 1);
        _nl_cycle_buf[_nl_cycle_count++] = ptr;
    }
}

/* ── Retain / Release ────────────────────────────────────────────────────── */

static inline void nl_rc_retain(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    hdr->refcount++;
    NL_GC_SET_COLOR(hdr, NL_GC_BLACK);  /* retained objects are live */
}

static inline void nl_rc_release(void *ptr);  /* forward decl */

/* Decrement RC of all children of a closure */
static inline void _nl_release_children(void *ptr) {
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (!NL_GC_IS_CLOSURE(hdr)) return;

    /* NLClosure layout: fn_index (u32) + upvalue_count (u32) + upvalues[] */
    typedef struct { uint32_t fn_index; uint32_t upvalue_count; void *uvs[]; } _NLCl;
    _NLCl *cl = (_NLCl *)ptr;
    for (uint32_t i = 0; i < cl->upvalue_count; i++)
        nl_rc_release(cl->uvs[i]);
}

static inline void nl_rc_release(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (hdr->refcount == 0) return;  /* double-release guard */
    hdr->refcount--;
    if (hdr->refcount == 0) {
        /* Immediate reclamation — release children first */
        _nl_release_children(ptr);
        _nl_wasm_live_bytes -= (hdr->size < _nl_wasm_live_bytes)
                                ? hdr->size : _nl_wasm_live_bytes;
        NL_GC_SET_ON_CYCBUF(hdr, 0);
        NL_GC_SET_COLOR(hdr, NL_GC_WHITE);
        memset(ptr, 0, hdr->size);
        hdr->size = 0;
    } else {
        /* RC > 0 but decremented — may be part of a cycle, add to suspect buf */
        if (NL_GC_COLOR(hdr) != NL_GC_PURPLE)
            _nl_cycle_buf_add(ptr);
    }
}

static inline uint32_t nl_rc_refcount(const void *ptr) {
    if (!ptr) return 0;
    return NL_RC_HDR((void *)ptr)->refcount;
}

/* ── Tri-color cycle collector (Bacon-Rajan) ─────────────────────────────── */
/*
 * Algorithm (simplified from Bacon-Rajan PLDI 2001):
 *
 * 1. MARK_GRAY: for each object on _nl_cycle_buf, recursively decrement
 *    children's RC and mark objects gray.
 *    (This simulates "what would happen if this root were removed from the graph")
 *
 * 2. SCAN: depth-first over gray objects:
 *    - If RC > 0 after gray-traversal → still externally reachable → color BLACK
 *    - If RC == 0 → only reachable from cycle set → color WHITE (garbage)
 *
 * 3. COLLECT_WHITE: free all WHITE objects, restore BLACK/PURPLE to their
 *    original RC.
 *
 * We use the gray worklist (_nl_gray_stack) to avoid deep recursion in WASM.
 */

/* Helper: get child pointers for an object */
static inline void **_nl_children(void *ptr, uint32_t *count_out) {
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    *count_out = 0;
    if (!NL_GC_IS_CLOSURE(hdr)) return NULL;
    typedef struct { uint32_t fn_index; uint32_t upvalue_count; void *uvs[]; } _NLCl;
    _NLCl *cl = (_NLCl *)ptr;
    *count_out = cl->upvalue_count;
    return cl->uvs;
}

/* Phase 1: mark_gray — decrement RC of children, color gray */
static inline void _gc_mark_gray(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (NL_GC_COLOR(hdr) == NL_GC_GRAY) return;
    NL_GC_SET_COLOR(hdr, NL_GC_GRAY);
    if (_nl_gray_top < NL_CYCLE_BUF_SIZE * 4)
        _nl_gray_stack[_nl_gray_top++] = ptr;

    uint32_t nc = 0;
    void **children = _nl_children(ptr, &nc);
    for (uint32_t i = 0; i < nc; i++) {
        if (!children[i]) continue;
        NLRCHeader *chdr = NL_RC_HDR(children[i]);
        if (chdr->refcount > 0) chdr->refcount--;
        _gc_mark_gray(children[i]);
    }
}

/* Phase 2: scan — decide reachability */
static inline void _gc_scan(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (NL_GC_COLOR(hdr) != NL_GC_GRAY) return;
    if (hdr->refcount > 0) {
        /* Still externally referenced — make black (and restore children) */
        NL_GC_SET_COLOR(hdr, NL_GC_BLACK);
        uint32_t nc = 0;
        void **children = _nl_children(ptr, &nc);
        for (uint32_t i = 0; i < nc; i++) {
            if (!children[i]) continue;
            NLRCHeader *chdr = NL_RC_HDR(children[i]);
            chdr->refcount++;  /* restore decremented RC */
            _gc_scan(children[i]);
        }
    } else {
        /* RC == 0 — only reachable from cycle, mark white for collection */
        NL_GC_SET_COLOR(hdr, NL_GC_WHITE);
        uint32_t nc = 0;
        void **children = _nl_children(ptr, &nc);
        for (uint32_t i = 0; i < nc; i++)
            _gc_scan(children[i]);
    }
}

/* Phase 3: collect white — free garbage objects */
static inline void _gc_collect_white(void *ptr) {
    if (!ptr) return;
    NLRCHeader *hdr = NL_RC_HDR(ptr);
    if (NL_GC_COLOR(hdr) != NL_GC_WHITE) return;
    NL_GC_SET_COLOR(hdr, NL_GC_BLACK);  /* prevent re-visit */
    /* Recurse into children before freeing self */
    uint32_t nc = 0;
    void **children = _nl_children(ptr, &nc);
    for (uint32_t i = 0; i < nc; i++)
        _gc_collect_white(children[i]);
    /* Free */
    _nl_wasm_live_bytes -= (hdr->size < _nl_wasm_live_bytes)
                            ? hdr->size : _nl_wasm_live_bytes;
    NL_GC_SET_ON_CYCBUF(hdr, 0);
    memset(ptr, 0, hdr->size);
    hdr->size = 0;
    hdr->refcount = 0;
}

/* Public: run the cycle collector */
static inline void nl_gc_collect_cycles(void) {
    if (_nl_cycle_count == 0) return;

    /* Phase 1: mark gray */
    for (uint32_t i = 0; i < _nl_cycle_count; i++) {
        void *p = _nl_cycle_buf[i];
        if (!p) continue;
        NLRCHeader *hdr = NL_RC_HDR(p);
        if (hdr->refcount > 0 && NL_GC_COLOR(hdr) == NL_GC_PURPLE)
            _gc_mark_gray(p);
    }

    /* Phase 2: scan */
    for (uint32_t i = 0; i < _nl_cycle_count; i++) {
        void *p = _nl_cycle_buf[i];
        if (p) _gc_scan(p);
    }

    /* Phase 3: collect white */
    for (uint32_t i = 0; i < _nl_cycle_count; i++) {
        void *p = _nl_cycle_buf[i];
        if (!p) continue;
        NLRCHeader *hdr = NL_RC_HDR(p);
        NL_GC_SET_ON_CYCBUF(hdr, 0);
        if (NL_GC_COLOR(hdr) == NL_GC_WHITE)
            _gc_collect_white(p);
        else
            NL_GC_SET_COLOR(hdr, NL_GC_BLACK);  /* restore */
    }

    /* Clear cycle buffer and gray stack */
    _nl_cycle_count = 0;
    _nl_gray_top    = 0;
}

/* ── String helpers ──────────────────────────────────────────────────────── */

static inline char *nl_rc_str_new(const char *data, uint32_t len) {
    char *p = (char *)nl_rc_alloc(len + 1);
    if (!p) return NULL;
    if (data && len) memcpy(p, data, len);
    p[len] = '\0';
    /* Strings have no children */
    NL_GC_SET_CHILDREN(NL_RC_HDR(p), 0);
    return p;
}

static inline char *nl_rc_str_from_cstr(const char *s) {
    if (!s) return nl_rc_str_new("", 0);
    return nl_rc_str_new(s, (uint32_t)strlen(s));
}

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

static inline bool nl_rc_str_eq(const char *a, const char *b) {
    if (a == b) return true;
    if (!a || !b) return false;
    return strcmp(a, b) == 0;
}

/* ── Closure support ─────────────────────────────────────────────────────── */

typedef struct {
    uint32_t  fn_index;
    uint32_t  upvalue_count;
    void     *upvalues[];
} NLClosure;

static inline NLClosure *nl_rc_closure_new(uint32_t fn_index,
                                            uint32_t uv_count,
                                            void   **upvalues) {
    uint32_t sz = (uint32_t)sizeof(NLClosure) + uv_count * (uint32_t)sizeof(void *);
    NLClosure *cl = (NLClosure *)nl_rc_alloc(sz);
    if (!cl) return NULL;
    cl->fn_index      = fn_index;
    cl->upvalue_count = uv_count;
    /* Mark header as closure so GC can trace children */
    NLRCHeader *hdr = NL_RC_HDR(cl);
    NL_GC_SET_CLOSURE(hdr);
    NL_GC_SET_CHILDREN(hdr, uv_count);
    for (uint32_t i = 0; i < uv_count; i++) {
        cl->upvalues[i] = upvalues[i];
        nl_rc_retain(upvalues[i]);
    }
    return cl;
}

static inline void nl_rc_closure_release(NLClosure *cl) {
    if (!cl) return;
    /* nl_rc_release will call _nl_release_children which releases upvalues */
    nl_rc_release(cl);
}

/* ── Heap stats ──────────────────────────────────────────────────────────── */

static inline void nl_rc_heap_stats(uint32_t *live_allocs_out,
                                     uint32_t *live_bytes_out) {
    uint32_t live = 0, pos = 0;
    while (pos + NL_RC_HEADER_SIZE <= _nl_wasm_bump) {
        NLRCHeader *hdr = (NLRCHeader *)(_nl_wasm_heap + pos);
        if (hdr->refcount > 0) live++;
        uint32_t slot = (NL_RC_HEADER_SIZE + hdr->size + (NL_WASM_ALIGN - 1))
                         & ~(uint32_t)(NL_WASM_ALIGN - 1);
        if (slot < NL_WASM_ALIGN) slot = NL_WASM_ALIGN;
        pos += slot;
    }
    if (live_allocs_out) *live_allocs_out = live;
    if (live_bytes_out)  *live_bytes_out  = _nl_wasm_live_bytes;
}

static inline void nl_rc_heap_reset(void) {
    memset(_nl_wasm_heap, 0, _nl_wasm_bump);
    _nl_wasm_bump        = 0;
    _nl_wasm_alloc_count = 0;
    _nl_wasm_live_bytes  = 0;
    _nl_cycle_count      = 0;
    _nl_gray_top         = 0;
}

#endif /* NANOLANG_REFCOUNT_GC_H */
