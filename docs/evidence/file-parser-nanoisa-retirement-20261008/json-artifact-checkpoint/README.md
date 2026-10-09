# My Json artifact boundary checkpoint

I extend self-hosted artifact registration from homogeneous strings to exact per-argument signatures for the real `std/json` exports. Json remains a declared opaque type, and imports retain the actual owning artifact. I preserve existing artifact signatures and refuse incompatible result, parameter and arity declarations.

My first canonical component build advances the real schema generator to the remaining string-only argument guard. The argument correction advances it to actual module shadows, which fail because opaque-null comparison uses `I64_NE`. I retain both failures. My subsequent correction emits generic equality with exact opaque identity and literal-zero validation, refusing ordering and nonzero integers. Its final canonical rebuild and real probes are pending at this checkpoint.

The new end-to-end ownership test keeps both producers, VM verification/execution and sanitized native execution required. Its C-seed VM path passes, while native translation refuses unsupported Json artifact imports. Its self-hosted path initially refuses the symbol, then the argument guard, then shadow comparison. Four wrong-ABI controls preserve existing output. No failing route is removed or converted to expected failure.

The existing artifact regression suite passes all ten methods after building `nano_aot_runtime.o` and `test_nanoisa_src_nano`, which were missing in the first direct run. I preserve the prerequisite failure and corrected result. Native opaque/Json adapters, real-caller cleanup of owned copies, fresh installed stages and the complete original parser corpus remain open under #978.
