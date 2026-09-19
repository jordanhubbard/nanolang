# I qualify private Samples composition without execution

I implement task_1bf5b051b828444c989382f8cc7702b6 under mixed parent4be.
My [contract](../NANOISA_MIXED_SAMPLES_COMPOSITION.md) distinguishes private
shape/affine/scalar analysis from executable admission. Production3b004056
received independent root and release_audit review before tests. No production
correction was needed during the following qualification.

At frozen56653e66 my first GCC run, including library compilation, passed5.681s;
Clang ASan/UBSan/LSan passed3.273s. Each reported110,177 assertions, including
allocation bookkeeping rather than110,177 distinct programs. Five adjacent
suites passed2.521s: shape339, descriptor138, affine-state/bytecode and verifier96.

I then added only test controls: a synthetic entry retaining both close and main
shadow calls in one graph; both predecessor orders at one repeated allocation
site; and all public verifier refusal paths. Frozen805ff5c4 passed GCC0.866s and
Clang3.322s,112,454 assertions each. The production source remained unchanged.
I did not repeat unrelated adjacent suites after these fixture-only changes.

The allocation sweep instruments malloc/calloc/free in mixed_float_proof,
affine_state (including its private constructor/walker), retained_layouts,
nvm_v2_layouts and nvm_v2_cursor. Every budget from0 through first success320 is
checked, followed by another successful terminal. Failed outputs retain their
sentinel, immutable code/layout/ownership bytes stay unchanged and live tracked
allocations return to zero. Failures include489 attempted Facts/frame allocations,
15 layout-decode,30 shape and2 view allocations; more than one attempt can fail
at a zero budget. Instruction decoding itself allocates no memory. This is not
an allocation-injection claim for the whole VM or all linked library code.

My Clang instrumentation covers those five translation units and the C query
fixture; remaining linked objects are the ordinary library build. Actual resolved
compiler/tool/wrapper hashes are retained, as are the linked object inputs.
Across each run,1,615 native-source/test/build input hashes and the host tools
remain unchanged. The31 linked library objects are sealed after the initial GCC
build and remain unchanged through the subsequent checks. Temporary instrumented
query executables are deleted by the harness; I retain their exact sources,
commands and flags, not a surviving-executable hash claim.

Controls cover zero-iteration/uninitialized ordinary locals, whole receiver
alternatives, lower-index callees, unused leaking helpers, owner mismatched joins,
observations reordered before moves, each F64 operand's FLOAT-or-VOID check,
generic EQ/NE policies and unchanged nominal mappings. Public general, owned,
function, max-stack and zero-link verifier APIs still refuse the pending module
before and after the private query. No VM or generated native program executes
pending data. These handwritten modules model the complete source/shadow graph;
they do not claim paired source compilation or original Samples execution.

The [manifest](mixed-samples-composition/manifest.json) seals26 reports from both
runs. All reports are original terminal outcomes; neither run failed. Public
runtime admission, checked scalar implementation and source publication remain
separate required dependencies. Full ownership, managed-value and release parents
remain open.

I integrate main5d660625 in a fresh tree at b8cc31f7, preserving the first qualified
tree and objects. Only additive roadmap content conflicted; both sides are kept.
NanoISA analysis source and the query fixtures remain identical to805ff5c4.
The integrated GCC run (including fresh library compilation) passed5.880s and
Clang sanitizer passed3.272s,112,454 assertions each. All8 resolved tools/wrapper,
31 linked objects and1,619 native-source/test/build inputs match their recorded
before/after and sealing identities. The expanded manifest seals39 reports.
I do not repeat the unchanged neighboring suites or bootstrap a source compiler.
