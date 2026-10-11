# Callable union storage

I extend the generic callable-storage work under #976 from 1fc528f2a.
My retained producer baseline rejects ordinary callable union payloads in
native C, C NanoISA shadows and the self-hosted product. Generic callable
payloads pass C NanoISA but fail native declarations and self-hosted lowering.
I preserve the valid sources and complete diagnostics in more-baseline.log.

I carry callable and tuple payload annotations into native typedef discovery
and union field declarations. I recursively substitute tuple/callable payload
metadata while retaining the complete and flattened parameter/result views.
My C checker applies declared ordinary callable field contexts before lowering.
My self-hosted emitter parses arrows and nested parameter lists within generic
arguments, admits checked callable payloads with recursive-signature guards,
and specializes generic function values from the selected payload context.

My negative audit finds that the previous native self-hosted compiler publishes
an unused body containing an incompatible union callback (selfhost-negative.log).
I now apply the existing union-constructor validator to declared bindings,
returns and assignments, including unused bodies. I also validate inferred,
global and nested-record constructors; unchecked-constructor-baseline.json
retains the three accepted invalid sources before that correction. My fresh checker rejects all
six ordinary/generic local/return/assignment negative cases; both C routes preserve their prior output.
The checker fixture returns a status without printing its diagnostic list. I
retain the initial harness failure that incorrectly expected printed text, then
assert its exact typecheck-failure status, distinct from parse failure.

My multi-arm test then exposes C AOT merging different callback signatures from
separate variants. I retain separate callable provenance for each packed tag.
I narrow only the surviving aggregate on fallthrough of an uninterrupted
DUP / AGG_TAG / PUSH_I64 / EQ-or-I64_EQ / JMP_FALSE sequence. Any branch entry
into that sequence after DUP disables narrowing. Other paths retain ordinary
conservative target sets. Mutable-array alias traversal also visits variant
provenance; a selected view retains tag maps for subsequent matches.
I do not loosen callable signatures or remove runtime target checks.

My eight source methods pass. Six execution methods exercise C-native execution,
C NanoISA, and fresh self-hosted checker/emitter components. Those six fixtures use
whole-program, selected-program and shadow emission, followed by verification,
NanoVM and strict C11 ASan/UBSan execution. I cover ordinary/generic payloads,
generic signature fields, nested and recursive signatures, returned unions,
different signatures in separate arms, and six incompatible local/return/assignment contexts. I also reject inferred,
global and nested-record constructors while retaining their valid controls.
I validate concrete constructors at expression entry and check global
initializers after collecting every function signature.
My first corrected runs and subsequent discoveries remain beside the final log.

All eight methods also pass with ten C-compiler components instrumented by
ASan/UBSan and all three translator/shape/callable-analysis objects instrumented.
Leak detection is disabled for those compiler/translator processes. The
instrumented callable constraint suite passes 419 checks, including exact
selected-arm targets and refusal to narrow when an alternate branch enters
at the comparison's final jump. My full LLVM C AOT gate passes 2,443 execution,
419 callable and 3,098 shape checks. Its first Apple-Clang run fails because
that runtime does not support LeakSanitizer; I retain that terminal and use
Homebrew LLVM without removing leak assertions from the gate.

I add test-callable-union-storage to the compiler-product Make gate. Optional
NANO_UNION_CHECKER, NANO_UNION_EMITTER and NANO_UNION_TRANSLATOR paths run the
same source assertions through fresh components and an instrumented translator.
I built the source components from tests/nanoisa/fixtures/generic_checker_driver.nano.txt
and generic_emitter_driver.nano.txt with bin/nanoc_c and default shadow policy.
These component checks do not refresh my installed Stage 1/2 products or
establish their raw fixed point or final Linux/Darwin release qualification.
My full 5.1 scope stays open.

My 37 adjacent union metadata, literal-context, generic-union and contextual
callback methods pass in neighbors.log. Their installed-stage paths use
existing compiler products; these are regression evidence, not rebuilt stages.
My unchanged compiler-product audit passes in 513.322 seconds with unchanged
source inputs and all eleven stages exiting successfully (product/manifest.json).
My C seed builds a native compiler in 202.2 seconds; that compiler compiles its
own source to NanoISA in 279.4 seconds. I translate the result to strict C11,
build its native executable, pass twelve capture/record/scope acceptance
methods, and emit Hello for exact VM and native execution. Original deadlines
and assertions remain intact. This qualifies the local compiler-product route,
not the separate raw bootstrap and final candidate/platform requirements.

My newly emitted native compiler passes all eight source methods in 7.974 seconds
(fresh-product-corrected.log). It also rejects the three inferred/global/nested
invalid sources with exit 1 and preserves prior output bytes
(fresh-product-negative.json). My first run expected only the C seed diagnostic;
I retain that harness failure and accept either producer’s exact diagnostic.
My first Make integration names a nonexistent nanoisa target; I correct its
dependency to the actual bin/nanoisa product and retain the failed terminal.
