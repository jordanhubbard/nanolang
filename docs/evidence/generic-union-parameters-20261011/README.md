# My generic union parameter evidence

I continue #976 from 6dcf57cbb. I retain complete concrete union arguments during
C generic inference, admit checked union parameters in NanoVirt specialization,
and substitute union constructors, declarations and match views in native
specialized bodies. I normalize qualified union aliases against their loaded
declaration. This does not establish independent owners with equal names.

My self-hosted emitter now admits nested ordinary union payloads and retains
recursive callable boundaries while refusing direct recursive value layouts.
All eight callable-union methods also pass with the rebuilt emitter through its three entry points, including recursive callable signatures, with instrumented translation (recursive-union-components.log). The rebuilt component executes nested-only.nano through whole, program and raw
shadow entry points, NanoVM verification/execution, C AOT, and strict native
ASan/UBSan execution (nested-fresh-results.json).

I retain scalar producer masks separately from storage shapes. My existing
interprocedural provenance graph carries these masks through calls and copied
values; an uninterrupted exact tag comparison selects one arm's field facts.
A branch into the middle of that comparison prevents refinement. Unknown or
unobserved producers retain runtime checks. I do not replace runtime payload
guards with inferred tags or infer actual tags from declared return types.

My focused C/NanoISA and native suites each pass eight methods, including
scalar, array, callable, nested and imported payloads, generic factories and
identity calls, and incompatible local/return/assignment/argument boundaries.
The native suite also passes the added nested wrong-instance control in the
broader audit. My adjacent C/NanoISA audit passes 96 methods; the callback
suite in that command also exercises its configured native compiler stages.

My full translator gate passes 2,443 execution checks, 464 provenance checks
and 3,113 shape checks. Instrumented translator objects pass all eight retained
positive fixtures, and the instrumented provenance and shape tests pass the
same unit counts. I check ASan/UBSan instrumentation in all three changed
translator implementation objects (build-translator-sanitizer.py).

I retain failed intermediate builds and checks. The first selected-arm test
used a bool entry result, which my native entry contract refuses; I corrected
the fixture to return int and assert the bool result. Earlier tag propagation
omitted array tags, causing a variant-int-array storage conflict; the corrected
implementation retains array provenance. The first broad gate used Apple cc
and stopped because its sanitizer does not support leak detection. Setting CC
only in the environment was insufficient because Make selects cc explicitly.
The passing run passes Homebrew LLVM through the Make command line without
removing leak checks or assertions.

My 104-method native regression audit reports ten failed subcases: eight used
a bytecode-only flag in two tuple refusal methods, and two expose actual native
tuple emission gaps. I corrected the four tuple refusal commands to use the
shared compiler mode; all four refusal methods now pass through native and
NanoISA routes. Imported tuple results and structural generic tuple return
literals remain open roadmap requirements. I retain the initial failure log.

Two modules exporting Box<T> still collide during C module loading (owners).
That failure prevents a cross-owner generic identity audit; I do not claim
module-owned union parity from the single-module success.

My unchanged-source compiler-product audit passes all eleven stages in 512.702 seconds (product/manifest.json). It builds and executes the canonical emitter, verifies and translates its compiler module, links the resulting native compiler, runs the unchanged compiler-product acceptance methods, and compiles and executes Hello in the VM and native code. The resulting compiler passes all eight new union methods and both tuple methods that remain failing in the C-native route (native-product-unions.log). I do not claim a
release candidate, bootstrap equality, Linux qualification or full 5.1 scope
completion from these focused checks.

I trim trailing whitespace in retained text logs. The audit patch is stored as a JSON string; source-input hashes and diagnostic content retain the qualification boundary.
