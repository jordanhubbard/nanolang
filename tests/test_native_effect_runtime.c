#include "runtime/effect_runtime.h"
#include <assert.h>

static int finalized;
static void finalize(void *value) { (void)value; ++finalized; }

static void leave_handler(NlEffectFrame *frame, void **args, void *result) {
    (void)args; (void)result;
    *(int64_t *)frame->lexical_result = 42;
    nl_effect_escape(frame);
}

static void allocating_body(NlEffectFrame *frame, void **args, void *result) {
    (void)frame; (void)args; (void)result;
    void *owned = gc_alloc_opaque(8, finalize);
    assert(owned);
    NlEffectGC guard __attribute__((cleanup(nl_effect_gc_pop))) = {
        nl_effect_gc_top, &owned, true, true
    };
    nl_effect_gc_top = &guard;
    nl_effect_dispatch("Exit.leave", NULL, NULL);
    assert(!"I do not resume after a lexical return.");
}

int main(void) {
    gc_init();
    for (int i = 0; i < 100; ++i) {
        int64_t value = 0;
        NlEffectEntry entry = {"Exit.leave", leave_handler};
        NlEffectFrame frame = {.previous=nl_effect_top, .entries=&entry,
            .count=1, .lexical_result=&value, .cleanup_mark=nl_effect_gc_top};
        assert(nl_effect_run(&frame, allocating_body, NULL) == 1);
        assert(value == 42);
        assert(finalized == i + 1);
        assert(nl_effect_top == NULL);
        assert(nl_effect_gc_top == NULL);
    }
    gc_shutdown();
    return 0;
}
