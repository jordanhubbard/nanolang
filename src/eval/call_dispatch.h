#ifndef NANO_EVAL_CALL_DISPATCH_H
#define NANO_EVAL_CALL_DISPATCH_H
#include <string.h>
/* I classify each call name once; numeric branches preserve the existing
 * dispatch order and argument evaluation. Unknown names keep user/FFI lookup. */
typedef enum {
    EVAL_CALL_unknown,
#define EVAL_CALL_NAME(name) EVAL_CALL_##name,
#include "call_names.inc"
#undef EVAL_CALL_NAME
    EVAL_CALL_count
} EvalCallName;

static inline EvalCallName eval_call_name(const char *name) {
    static const char *const names[] = {
        "",
#define EVAL_CALL_NAME(name) #name,
#include "call_names.inc"
#undef EVAL_CALL_NAME
    };
    if (!name) return EVAL_CALL_unknown;
    int first = 1, last = EVAL_CALL_count;
    while (first < last) {
        int middle = first + (last - first) / 2;
        int order = strcmp(name, names[middle]);
        if (!order) return (EvalCallName)middle;
        if (order < 0) last = middle;
        else first = middle + 1;
    }
    return EVAL_CALL_unknown;
}

#endif
