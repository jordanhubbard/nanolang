/*
 * heap_cycles.c — cycle collection for the NanoVM heap
 *
 * Reference counting frees an object the moment its count reaches zero. For a
 * cycle that never happens: each member's count includes the reference held by
 * the next, so the whole group stays live with nothing outside pointing at it.
 *
 * That is not a theoretical concern here. A cycle is constructible from
 * ordinary NanoLang without unsafe code or a self-referential value:
 *
 *     struct Node { label: string, children: array<Node> }
 *     let mut kids: array<Node> = []
 *     let n: Node = Node { label: "root", children: kids }
 *     set kids (array_push kids n)      // kids now holds n, n holds kids
 *
 * `array_push` mutates in place and returns the same array, so `n.children`
 * and `kids` are one object -- and it now contains `n`. Forbidding that shape
 * in the type system would mean forbidding recursive types reached through an
 * array, which rules out ordinary trees and graphs. So the answer is
 * collection rather than restriction, and this is it.
 *
 * The algorithm is Bacon and Rajan's synchronous cycle collector (PLDI 2001),
 * the same one the generated-C runtime uses in src/runtime/refcount_gc.h --
 * the two backends should not disagree about whether a program leaks.
 *
 * The idea is trial deletion. Take every object whose count dropped without
 * reaching zero: those are the only possible cycle roots, because an object
 * whose count never dropped is still held by whoever held it before. Subtract
 * the references those objects hold among themselves. Anything left with a
 * count above zero is reachable from outside the group, so restore it and
 * everything it reaches; anything at zero was only kept alive by the group,
 * and is garbage.
 */

#include <stdlib.h>
#include <string.h>
#include "heap.h"
#include "value.h"
#include "../nanoisa/isa.h"

/* Only these can hold references to other heap objects, so only these can be
 * part of a cycle. A string is a leaf: it can be in a cycle's payload but can
 * never close one. */
static bool can_hold_children(uint8_t tag) {
    switch (tag) {
    case TAG_ARRAY: case TAG_STRUCT: case TAG_UNION:
    case TAG_TUPLE: case TAG_HASHMAP: case TAG_CLOSURE:
        return true;
    default:
        return false;
    }
}

static VmHeapHeader *header_of(NanoValue v) {
    return val_is_heap_obj(v) && v.as.obj ? (VmHeapHeader *)v.as.obj : NULL;
}

/* Walk the heap references an object holds. Kept in one place: a child this
 * misses is a reference the collector will not restore, which turns a live
 * object into a freed one -- the one failure mode of this algorithm that is
 * worse than the leak it fixes. */
typedef void (*ChildFn)(VmHeap *heap, NanoValue child, void *ctx);

static void for_each_child(VmHeap *heap, NanoValue v, ChildFn fn, void *ctx) {
    switch (v.tag) {
    case TAG_ARRAY: {
        VmArray *a = v.as.array;
        /* Unboxed arrays store raw payloads, not values: no references. */
        if (!a || a->unboxed) return;
        for (uint32_t i = 0; i < a->length; i++) fn(heap, a->elements[i], ctx);
        return;
    }
    case TAG_STRUCT: {
        VmStruct *s = v.as.sval;
        if (!s) return;
        for (uint32_t i = 0; i < s->field_count; i++) fn(heap, s->fields[i], ctx);
        /* field_names are strings the struct owns; they are leaves, but they
         * are still references and the counts have to balance. */
        if (s->field_names)
            for (uint32_t i = 0; i < s->field_count; i++)
                if (s->field_names[i]) fn(heap, val_string(s->field_names[i]), ctx);
        return;
    }
    case TAG_UNION: {
        VmUnion *u = v.as.uval;
        if (!u) return;
        for (uint32_t i = 0; i < u->field_count; i++) fn(heap, u->fields[i], ctx);
        return;
    }
    case TAG_TUPLE: {
        VmTuple *t = v.as.tuple;
        if (!t) return;
        for (uint32_t i = 0; i < t->count; i++) fn(heap, t->elements[i], ctx);
        return;
    }
    case TAG_HASHMAP: {
        VmHashMap *m = v.as.hashmap;
        if (!m || !m->entries) return;
        for (uint32_t i = 0; i < m->bucket_count; i++) {
            if (m->entries[i].state != 1) continue;   /* 1 == filled */
            fn(heap, m->entries[i].key, ctx);
            fn(heap, m->entries[i].value, ctx);
        }
        return;
    }
    case TAG_CLOSURE: {
        VmClosure *c = v.as.closure;
        if (!c) return;
        for (uint16_t i = 0; i < c->capture_count; i++) fn(heap, c->captures[i], ctx);
        return;
    }
    default:
        return;
    }
}

/* ── Suspect buffer ─────────────────────────────────────────────────────── */

void vm_gc_note_suspect(VmHeap *heap, NanoValue v) {
    VmHeapHeader *h = header_of(v);
    if (!h || !can_hold_children(v.tag)) return;
    if (h->colour == VM_GC_PURPLE && h->buffered) return;
    h->colour = VM_GC_PURPLE;
    if (h->buffered) return;

    if (heap->cycle_count == heap->cycle_capacity) {
        uint32_t next = heap->cycle_capacity ? heap->cycle_capacity * 2 : 64;
        void **grown = realloc(heap->cycle_buf, (size_t)next * sizeof(*grown));
        /* Losing a suspect costs a leak, not correctness, so a failed growth
         * is survivable: drop this one rather than abort the program. */
        if (!grown) return;
        heap->cycle_buf = grown;
        heap->cycle_capacity = next;
    }
    heap->cycle_buf[heap->cycle_count++] = v.as.obj;
    h->buffered = 1;

    /* Collect when the candidate set has grown enough to be worth a pass. The
     * guard is not optional: collection releases objects, and vm_release calls
     * back into here, so without it a collection would recurse into itself
     * while the buffer is mid-rewrite. */
    if (!heap->gc_running && heap->cycle_count >= VM_GC_CYCLE_THRESHOLD)
        vm_gc_collect_cycles(heap);
}

/* Rebuild a NanoValue from a buffered pointer. The header records the tag, so
 * the buffer can hold bare pointers. */
static NanoValue value_from_obj(void *obj) {
    NanoValue v = {0};
    VmHeapHeader *h = (VmHeapHeader *)obj;
    v.tag = h->obj_type;
    v.as.obj = obj;
    return v;
}

/* ── Trial deletion ─────────────────────────────────────────────────────── */

static void mark_gray(VmHeap *heap, NanoValue v);

static void mark_gray_child(VmHeap *heap, NanoValue child, void *ctx) {
    (void)ctx;
    VmHeapHeader *h = header_of(child);
    if (!h) return;
    if (h->ref_count > 0) h->ref_count--;   /* the trial deletion itself */
    mark_gray(heap, child);
}

static void mark_gray(VmHeap *heap, NanoValue v) {
    VmHeapHeader *h = header_of(v);
    if (!h || h->colour == VM_GC_GRAY) return;
    h->colour = VM_GC_GRAY;
    for_each_child(heap, v, mark_gray_child, NULL);
}

static void scan(VmHeap *heap, NanoValue v);

static void scan_black(VmHeap *heap, NanoValue v);

static void scan_black_child(VmHeap *heap, NanoValue child, void *ctx) {
    (void)ctx;
    VmHeapHeader *h = header_of(child);
    if (!h) return;
    h->ref_count++;                          /* undo the trial deletion */
    if (h->colour != VM_GC_BLACK) scan_black(heap, child);
}

static void scan_black(VmHeap *heap, NanoValue v) {
    VmHeapHeader *h = header_of(v);
    if (!h) return;
    h->colour = VM_GC_BLACK;
    for_each_child(heap, v, scan_black_child, NULL);
}

static void scan_child(VmHeap *heap, NanoValue child, void *ctx) {
    (void)ctx;
    scan(heap, child);
}

static void scan(VmHeap *heap, NanoValue v) {
    VmHeapHeader *h = header_of(v);
    if (!h || h->colour != VM_GC_GRAY) return;
    if (h->ref_count > 0) {
        /* Something outside the candidate set still points here, so this and
         * everything it reaches must live. */
        scan_black(heap, v);
        return;
    }
    h->colour = VM_GC_WHITE;
    for_each_child(heap, v, scan_child, NULL);
}

/* Freeing the garbage set needs a mutating walk, not the read-only one above.
 *
 * The obvious shape -- recurse into the children, then release this object --
 * frees every member twice: the recursion frees a child, and then the object's
 * own release cascades into the same child. Detaching each slot before
 * following it is what makes each object's memory released exactly once, and
 * it also stops the walk from arriving back at an object it is already tearing
 * down. */
typedef void (*SlotFn)(VmHeap *heap, NanoValue *slot, void *ctx);

static void for_each_child_slot(VmHeap *heap, NanoValue v, SlotFn fn, void *ctx) {
    switch (v.tag) {
    case TAG_ARRAY: {
        VmArray *a = v.as.array;
        if (!a || a->unboxed) return;
        for (uint32_t i = 0; i < a->length; i++) fn(heap, &a->elements[i], ctx);
        return;
    }
    case TAG_STRUCT: {
        VmStruct *s = v.as.sval;
        if (!s) return;
        /* field_names are leaf strings the struct owns; release_struct still
         * releases them, so they are deliberately not detached here. */
        for (uint32_t i = 0; i < s->field_count; i++) fn(heap, &s->fields[i], ctx);
        return;
    }
    case TAG_UNION: {
        VmUnion *u = v.as.uval;
        if (!u) return;
        for (uint32_t i = 0; i < u->field_count; i++) fn(heap, &u->fields[i], ctx);
        return;
    }
    case TAG_TUPLE: {
        VmTuple *t = v.as.tuple;
        if (!t) return;
        for (uint32_t i = 0; i < t->count; i++) fn(heap, &t->elements[i], ctx);
        return;
    }
    case TAG_HASHMAP: {
        VmHashMap *m = v.as.hashmap;
        if (!m || !m->entries) return;
        for (uint32_t i = 0; i < m->bucket_count; i++) {
            if (m->entries[i].state != 1) continue;
            fn(heap, &m->entries[i].key, ctx);
            fn(heap, &m->entries[i].value, ctx);
        }
        return;
    }
    case TAG_CLOSURE: {
        VmClosure *c = v.as.closure;
        if (!c) return;
        for (uint16_t i = 0; i < c->capture_count; i++) fn(heap, &c->captures[i], ctx);
        return;
    }
    default:
        return;
    }
}

/* The garbage found by one pass, gathered before any of it is freed.
 *
 * Freeing during the walk is the obvious shape and it does not work: the walk
 * reads each object's header to decide what to do with it, and an object can
 * be reached a second time -- from a later root, or from the buffer -- after
 * an earlier branch already freed it. Gathering first means every header read
 * happens while the object is still allocated, and the frees then happen in a
 * pass that reads nothing. */
typedef struct {
    void   **items;
    uint32_t count;
    uint32_t capacity;
} DeadSet;

static bool dead_push(DeadSet *dead, void *obj) {
    if (dead->count == dead->capacity) {
        uint32_t next = dead->capacity ? dead->capacity * 2 : 32;
        void **grown = realloc(dead->items, (size_t)next * sizeof(*grown));
        if (!grown) return false;
        dead->items = grown;
        dead->capacity = next;
    }
    dead->items[dead->count++] = obj;
    return true;
}

static void collect_white(VmHeap *heap, NanoValue v, DeadSet *dead);

static void collect_white_slot(VmHeap *heap, NanoValue *slot, void *ctx) {
    NanoValue child = *slot;
    /* Detach first: nothing that follows may reach this object through here. */
    *slot = val_void();
    VmHeapHeader *ch = header_of(child);
    if (ch && ch->colour == VM_GC_WHITE && !ch->buffered)
        collect_white(heap, child, (DeadSet *)ctx);
    else
        vm_release(heap, child);   /* a survivor, or a leaf like a string */
}

static void collect_white(VmHeap *heap, NanoValue v, DeadSet *dead) {
    VmHeapHeader *h = header_of(v);
    if (!h || h->colour != VM_GC_WHITE || h->buffered) return;
    /* Black before descending, so a cycle cannot bring the walk back here,
     * and so a second root reaching this object stops at the check above. */
    h->colour = VM_GC_BLACK;
    for_each_child_slot(heap, v, collect_white_slot, dead);
    if (!dead_push(dead, v.as.obj)) {
        /* Out of memory gathering the set. Leaving the object alone leaks it,
         * which is what refcounting was already doing. */
        h->colour = VM_GC_WHITE;
    }
}

uint64_t vm_gc_collect_cycles(VmHeap *heap) {
    if (!heap || heap->cycle_count == 0 || heap->gc_running) return 0;
    heap->gc_running = true;

    /* Take the buffer. Releasing an object during a pass can note new
     * suspects, and appending to the same array being compacted in place
     * aliases it -- so the heap gets a fresh empty one and anything noted
     * during this pass is collected by the next. */
    void **buf = heap->cycle_buf;
    uint32_t count = heap->cycle_count;
    heap->cycle_buf = NULL;
    heap->cycle_count = 0;
    heap->cycle_capacity = 0;

    uint64_t freed = 0;

    /* Phase 1a: partition, and free the entries that are already dead.
     *
     * An object whose count reached zero while buffered was left unfreed by
     * vm_release, precisely because the buffer holds a bare pointer to it.
     * Freeing it is this pass's job, and it has to happen before any trial
     * deletion: a deletion lowers counts, so deciding "already dead" after
     * one has run would free an object the graph still holds, and the scan
     * would then walk into it. Colour is the discriminator, not the count --
     * vm_release marks a deferred object BLACK, while a candidate is PURPLE. */
    uint32_t kept = 0;
    for (uint32_t i = 0; i < count; i++) {
        NanoValue v = value_from_obj(buf[i]);
        VmHeapHeader *h = header_of(v);
        if (!h) continue;
        if (h->colour == VM_GC_PURPLE && h->ref_count > 0) {
            buf[kept++] = buf[i];
            continue;
        }
        h->buffered = 0;
        if (h->ref_count == 0) {
            h->ref_count = 1;      /* hand the last reference to the release */
            freed++;
            vm_release(heap, v);
        }
    }

    /* Phase 1b: subtract the references the candidate set holds among itself.
     *
     * A candidate may have dropped to zero during the frees above; it was not
     * freed, because it is buffered. It is garbage either way, and phase 2
     * reaches that conclusion on its own. */
    for (uint32_t i = 0; i < kept; i++)
        mark_gray(heap, value_from_obj(buf[i]));

    /* Phase 2: restore anything still reachable from outside the set. */
    for (uint32_t i = 0; i < kept; i++)
        scan(heap, value_from_obj(buf[i]));

    /* Phase 3: what is still white was only holding itself up. The buffered
     * flags come off first: collect_white refuses to free a buffered object,
     * which is what would otherwise stop it freeing a root at all. */
    for (uint32_t i = 0; i < kept; i++) {
        VmHeapHeader *h = (VmHeapHeader *)buf[i];
        if (h) h->buffered = 0;
    }
    DeadSet dead = {0};
    for (uint32_t i = 0; i < kept; i++)
        collect_white(heap, value_from_obj(buf[i]), &dead);

    /* Every gathered object has had its child slots emptied, so releasing one
     * frees exactly it and cascades into nothing. */
    for (uint32_t i = 0; i < dead.count; i++) {
        VmHeapHeader *h = (VmHeapHeader *)dead.items[i];
        NanoValue v = {0};
        v.tag = h->obj_type;
        v.as.obj = dead.items[i];
        h->ref_count = 1;
        freed++;
        vm_release(heap, v);
    }
    free(dead.items);

    free(buf);
    heap->cycles_collected += freed;
    heap->gc_running = false;
    return freed;
}

void vm_gc_cycle_buffer_free(VmHeap *heap) {
    if (!heap) return;
    free(heap->cycle_buf);
    heap->cycle_buf = NULL;
    heap->cycle_count = 0;
    heap->cycle_capacity = 0;
}
