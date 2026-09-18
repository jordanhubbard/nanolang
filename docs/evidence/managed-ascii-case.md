# My managed ASCII case-conversion evidence

I implement `task_31acd26441144d52a82a533929669763` under my pre-code
[case contract](../NANOISA_MANAGED_CASE.md), following checked VM allocation
prerequisite543/PR674. Frozen source `a471150e` became `dcfa300c` on main
`01b6cd24`; managed runtime, LLVM/profile implementation and the new test
compare byte-identically. Main's native-C carrier changes prompted an
additional rebuilt integration gate below.

My three focused methods passed in 3.371 seconds on actual Linux ARM64 native
LLVM and VM, plus import-free Node/Wasmtime. Five inputs in both modes include
all 256 bytes, empty/unchanged text, embedded NUL/high bytes and a 1200-byte
mixed-case input. Literal and computed results match the exact ASCII mapping;
called helpers, globals, aliases, repeated entry and disposal preserve owners.
Native emitted IR is explicitly ASan-instrumented before linking its harness.

My native ASan/UBSan and Wasm core checks establish fresh unshared managed
handles and unchanged source aliases. Allocation failure, invalid mode and
null output consume exactly the transferred owner while preserving the output
sentinel and other references. A full eight-slot table exercises bytes-success/
table-allocation-failure rollback, followed by successful growth to 16 slots;
all eight original handles and byte contents survive. Emitted native failure
unwinds the called helper, retains the global and recovers on later entry.
Native/Wasm wrong-source-tag controls preserve committed globals and dispose
cleanly. VM interning may reuse equal objects; these tests do not establish
allocation-event or physical-identity equality between runtimes.

I passed `make test-llvm-literal-strings test-llvm-managed-strings
 test-verifier-profiles`: 11 global, 9 literal, 2 target packaging, 3 core,
34 managed methods and the shared profile/publication test. The 34 managed
methods took 34.624 seconds. After restacking, rebuilt tools and 5 integrated
methods passed 4.035 seconds: the 3 new methods, existing direct/boxed generated
C binary64 conversion and shared profiles. Native gates used
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.

Logs: `/tmp/nanolang-managed-case-focused.log`,
`/tmp/nanolang-managed-case-full.log`,
`/tmp/nanolang-managed-case-integrated.log`. Parent independent production
review found no scoped blocker. I updated obsolete unsupported-case fixtures
to still-unadmitted split/replace; previous-output controls and explicit
scalar/literal profile refusals remain. No historical failed artifact was
replayed. Full runtime51da, Darwin7ba and evaluator791a remain open; no new
compiler-bootstrap or full-language target acceptance is claimed here.
