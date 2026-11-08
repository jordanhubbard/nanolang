# My next native U8 storage boundary

At `614eeaa8e2297df97589f027e138349013cbae5c`, both Linux compilers in run [42513128919](https://github.com/jordanhubbard/nanolang/actions/runs/42513128919) pass useful interpreter shadows, all VM source controls, native bounds, lexical identity and ignored-invalid-call refusal controls. The complete all-builtin/U8 native program builds at O0/O2, then aborts at the runtime's ELEM_INT push assertion.

The exact retained generated C confirms the preceding correction: iteration, map and filter select U8 getters and output storage; reduce selects a U8 getter. The generated reverse helper still routes ELEM_U8 to its INT-push fallback, which is the next such push in this program's execution order. Its contains/index_of helpers also have INT-only reads by inspection. I preserve the strict assertions and correct these U8 helper branches. I do not classify the failures as infrastructure or claim the uncompleted native test passed.
