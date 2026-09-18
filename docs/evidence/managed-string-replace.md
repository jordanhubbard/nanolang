# My consuming managed replacement evidence

I implement `task_4930450b7e4f483f91d47f2e24707f3e` under my pre-code
[replacement contract](../NANOISA_MANAGED_REPLACE.md), after VM sizing and
allocation6b3/PR680. Frozen production `af8ca608` plus fixture correction
`17949066` became `3948b849`/`e54ec8c5` on main `8f7fc062`. Complete nanoisa/
nanovm source and affected tests compare byte-identically after that restack;
the intervening reconstruction changes do not alter these gates.

Ten ordinary byte triples cover shrinking/growing results, nonoverlapping
matches, deletion, no match, empty source/needle, embedded NUL/high bytes and
600 source bytes growing to 1200 output bytes. Actual VM, ASan-instrumented
emitted native LLVM and import-free Node/Wasmtime agree. Literal and dynamic
operands, called helpers, all pairwise handle-alias arrangements, all-three-same
owners, committed globals, repeated entry and disposal retain exact bytes.
My managed result is fresh; VM interning can reuse equal storage.

Native ASan/UBSan and Wasm core tests fill all eight descriptor slots, then
separately fail scratch allocation, result-byte allocation and replacement-table
allocation. Output sentinel, all input aliases and live allocation counts
remain unchanged. Successful retry grows the table to sixteen slots without
changing source byte views. Null-output consumes the transferred owners while
preserving external aliases. Final disposal reports no live allocations.
Emitted native scratch/result allocation failures unwind a called helper and
recover on subsequent entry; native/Wasm source, needle and replacement type
errors each release the two transient string owners and retain the global.

I passed 11 global and 9 literal methods, then the corrected managed gate's
2 target packaging, 3 core, 39 managed methods and shared profile/publication
test. The final 39 methods took 39.282 seconds. Native gates used
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`.
Parent independent production review found no scoped blocker.

Two test-fixture corrections are retained: an initial duplicate `empty`
constant was refused by assembly; a later obsolete replacement-refusal fixture
was changed to unsupported ARR_NEW but initially carried an extra stack value.
I removed that unused value and reran the complete managed/profile gate.
Neither issue changed production code or required execution of an invalid
module. Logs: `/tmp/nanolang-managed-replace-focused.log`,
`/tmp/nanolang-managed-replace-focused-r2.log`,
`/tmp/nanolang-managed-replace-full.log`,
`/tmp/nanolang-managed-replace-full-r2.log`.

The scratch buffer and final copy are explicit costs, not a performance claim.
Only the managed profile gains replacement; scalar/literal selectors and
unsupported split/aggregate output preservation remain checked. Full51da,
Darwin7ba, evaluator791a and aggregate-dependent split remain open. No historical
failed artifact was replayed or full-language target acceptance claimed.
