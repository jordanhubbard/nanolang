# My managed primitive numeric formatting evidence

I implement `task_b029d365b4334e7d9399e7230e37eeac` under my pre-code
[primitive format contract](../NANOISA_MANAGED_PRIMITIVE_FORMAT.md), after
merged VM prerequisite4ba/PR677. Frozen source `9b420a57` became `1e2dda84`
on main `96733629`. The complete nanoisa/nanovm source and affected primitive/
profile tests compare byte-identically. The intervening reconstruction change
does not alter this runtime acceptance.

My new ordinary target methods cover seven integer boundary values and 64
native C-locale/default-rounding binary64 references, including signed zero,
nonfinite values, subnormals, rounding transitions and finite extremes. Each
exact numeric tag uses its value; the opposite numeric tag and bool/U8/void/
enum/string tags use zero. Literal and dynamic owned strings take fallback,
not CAST_STRING identity. Typed helper calls, committed globals, repeated
entry and disposal preserve the original aliases.

Actual VM, ASan-instrumented emitted native LLVM, and import-free Node/Wasmtime
agree on expected byte strings. Native emitted allocation failure in a called
string-fallback helper releases transient owners while retaining the global;
subsequent successful entry and disposal pass. The existing portable formatter
core's 2077-reference native/Wasm controls remain part of the adjacent gate.
I do not change locale or claim arbitrary rounding/locale agreement, nor equal
physical handles/allocation events with VM interning.

I passed `make test-llvm-literal-strings test-llvm-managed-strings
 test-verifier-profiles`: 11 global, 9 literal, 2 target packaging, 3 core,
36 managed methods and shared profile/publication acceptance. The managed
methods took 34.063 seconds. Native gates used
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
The initial focused module run also passed, but discovered the imported
formatter TestCase class and reran three existing tests; I changed that import
to a module before the final gate, which runs each method once.

Logs: `/tmp/nanolang-managed-primitive-focused.log` and
`/tmp/nanolang-managed-primitive-full.log`. Parent independent production
review found no scoped blocker. Only the managed profile gains these two
opcodes; explicit scalar/literal refusals and remaining unsupported-operation
previous-output controls pass. Full runtime51da, Darwin7ba and evaluator791a
remain open. No historical failed artifact was replayed, and no new compiler
bootstrap or full-language target acceptance is claimed.
