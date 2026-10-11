# Aggregate union payloads

I extend my self-hosted union payload admission through the existing ordinary
value contracts. I retain complete instantiated types through construction,
returns and matching, including records, arrays, lists, tuples and the admitted
string-keyed maps. Record shape checking now follows union fields, allowing
acyclic alternating record/union chains. I exclude resource storage from the
ordinary payload contract and retain the separate ownership profile.

My C constructor argument parser previously discarded nominal names by passing
NULL name outputs to parse_type_with_element. I now retain those names and
callback signatures, including record array element identities, and free the
complete annotation trees on parser errors.

I test return/escape, shared collection mutation, imported same-named record
owners, wrong nominal constructor arguments and preserved prior artifacts.
The component wrapper emits production assembly without the full frontend or
driver shadow selection; those component results are not compiler-product
qualification. I retain separate full-driver results below.

My initial resource probe placed all resources in an unused function. Dead
code elimination skipped it, so it was not refusal evidence. My retained
resource-controls.py uses reachable construction and calls; all five cases
refuse through the ownership or ordinary-type gates and preserve prior output.

My direct legacy C-native control still fails Box<(int,string)> because its
tuple definition follows the concrete union field. The same source passes
C-frontend NanoISA, VM and C AOT. I record that ordering gap separately in
the roadmap and retain the failing native suite.

Initial fixture failures also include missing spaces between generic closing
brackets and assignment. I corrected the fixtures and retain the original
logs as test-development history, not additional product defects.

I pass 45 C-frontend NanoISA regression methods plus two additional collection
methods, both parser/typechecker gates and 19 methods with parser.c instrumented
by Homebrew Clang ASan/UBSan. Other linked compiler objects remain ordinary;
this is not a whole-compiler sanitizer claim.

My fresh C-built self-hosted seed passes 47 NanoISA/VM/sanitized-C-AOT methods,
24 native methods and six layout/resource refusal methods. Resource tests use
reachable values and require E0036 diagnostics while preserving both prior
native and NanoISA outputs. The direct emitter helper has separate refusal
diagnostics, which I retain in its component evidence.

I reproduce the shared suite with `python3 -m unittest
tests.test_union_aggregate_payloads` and select producers through
NANO_GENERIC_CSEED_COMPILER (NanoISA) or NANO_CONTEXTUAL_NATIVE_COMPILER (native).
NANO_NATIVE_TEST_CC selects the C AOT sanitizer compiler. The layout/resource
suites use NANOLANG_SELFHOST_COMPILER.

My additional 16-method nominal-order audit passes 15 methods and fails the
List<int> ordinary-record field control at C AOT AGG_PACK shape resolution.
The previous immutable compiler reproduces that same failure. I retain both
logs and keep the ordinary-record shape task open; union-list payloads pass.

My full eleven-step compiler-product audit passes in 523.927 seconds, with
all recorded source inputs unchanged. I retain its manifest, source patch and
step outputs under product-audit/. The audit builds a fresh seed, emits the
compiler to NanoISA, translates and links its native product, runs existing
compiler regression tests and compares VM/native hello execution. This is a
local Darwin product audit, not the exact release-candidate Linux/Darwin gate.

The resulting native self-hosted compiler passes 48 NanoISA/VM/sanitized-C-AOT
regression methods, 25 native methods and six resource/layout methods. The
added record-list payload mutation control passes through both C-frontend and
self-hosted NanoISA routes. I have not changed source after the product audit.
