#include "effect_runtime.h"

_Thread_local NlEffectGC *nl_effect_gc_top;
_Thread_local NlEffectFrame *nl_effect_top;

_Thread_local unsigned nl_effect_foreign_depth;
