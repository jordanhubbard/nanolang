/* I keep synchronous handler state shared across native compilation units. */
#ifndef NANO_EFFECT_RUNTIME_H
#define NANO_EFFECT_RUNTIME_H

#include <setjmp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gc.h"

typedef struct NlEffectGC NlEffectGC;
struct NlEffectGC {
    NlEffectGC *previous;
    void **value;
    bool owned;
    bool active;
};
extern _Thread_local NlEffectGC *nl_effect_gc_top;

static inline void nl_effect_gc_pop(NlEffectGC *guard) {
    if (!guard->active) return;
    guard->active = false;
    nl_effect_gc_top = guard->previous;
    if (guard->owned) {
        void *value;
        memcpy(&value, guard->value, sizeof(value));
        gc_release(value);
    }
}

typedef struct NlEffectFrame NlEffectFrame;
typedef struct {
    const char *name;
    void (*call)(NlEffectFrame *, void **, void *);
} NlEffectEntry;
struct NlEffectFrame {
    NlEffectFrame *previous;
    NlEffectEntry *entries;
    int count;
    void **captures;
    void *lexical_result;
    NlEffectGC *cleanup_mark;
    unsigned foreign_depth;
    jmp_buf escape;
};
extern _Thread_local NlEffectFrame *nl_effect_top;
extern _Thread_local unsigned nl_effect_foreign_depth;

static inline void nl_effect_pop(NlEffectFrame *frame) {
    nl_effect_top = frame->previous;
}

/* I preserve a returned owned value before abandoning its local owner. */
static inline void nl_effect_preserve_return(void *slot) {
    void *value;
    memcpy(&value, slot, sizeof(value));
    for (NlEffectGC *guard = nl_effect_gc_top; guard; guard = guard->previous) {
        void *owned;
        memcpy(&owned, guard->value, sizeof(owned));
        if (guard->owned && owned == value) {
            gc_retain(value);
            return;
        }
    }
}

static inline void nl_effect_escape(NlEffectFrame *frame) {
    if (nl_effect_foreign_depth != frame->foreign_depth) {
        fputs("I cannot return from a handler across a foreign callback boundary.\n", stderr);
        abort();
    }
    /* I release abandoned locals while their stack storage is still live. */
    while (nl_effect_gc_top != frame->cleanup_mark)
        nl_effect_gc_pop(nl_effect_gc_top);
    nl_effect_top = frame->previous;
    longjmp(frame->escape, 1);
}

/* I isolate setjmp from user locals: captures live in the installing caller,
 * whose ordinary C variables do not acquire setjmp's indeterminate-value rule. */
static inline int nl_effect_run(NlEffectFrame *frame,
                               void (*body)(NlEffectFrame *, void **, void *),
                               void *result) {
    nl_effect_top = frame;
    if (setjmp(frame->escape)) return 1;
    body(frame, NULL, result);
    nl_effect_top = frame->previous;
    return 0;
}

static inline void nl_effect_dispatch(const char *name, void **args, void *result) {
    for (NlEffectFrame *frame = nl_effect_top; frame; frame = frame->previous) {
        for (int i = 0; i < frame->count; ++i) {
            if (strcmp(frame->entries[i].name, name) == 0) {
                frame->entries[i].call(frame, args, result);
                return;
            }
        }
    }
    fprintf(stderr, "I cannot perform unhandled effect %s.\n", name);
    abort();
}
#endif
