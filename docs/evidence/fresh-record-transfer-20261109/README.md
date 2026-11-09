# My rejected fresh record transfer candidate

I reject this optimization after its fixed whole-source comparison and restore
production bindings byte-for-byte from52933101c. I retain the actual evaluator
regressions and experiment history below; the transfer is not adopted.

I follow the measured binding experiment from PR #1137. I change only the
evaluated-let binding path: a completed `AST_STRUCT_LITERAL` whose value is
a non-null `VAL_STRUCT` transfers its fresh carrier. Return, break and continue
values retain the old copy path. Ordinary binding APIs, assignments, arguments
and nonliteral results retain their existing record-copy and borrow behavior.
I change no carrier layout or public API.

The evaluator's ordinary and spread record constructors already create owned
name storage and recursively copied nested-record/STRING payloads. Arrays and
other referenced fields keep the constructor's borrowed identity. Union
construction can share the literal AST kind but produces `VAL_UNION`, so it
does not take this transfer path. I do not derive argument ownership from
inherited callback source locations.

My complete environment unit passes under actual ASan/UBSan with stable source,
compiler and executable hashes. New controls require fresh binding identity,
copied nested/string independence, borrowed-array identity, metadata, empty
records, both borrowed types, nonliteral/unknown-expression copies, all three
control-flow flags and unchanged ordinary APIs. The receipt retains exact
sources selected by hash, compiler command, streams and collector source in
`environment-unit.json.gz`. The actual native image remains at the receipt's
local output path. Darwin leak detection remains disabled as required by the
installed runtime; managed usage and object counts are checked explicitly.

My evaluator tests use the actual lexer, parser, typechecker and interpreter.
They cover fresh literals, spreads, nested/string values, shared array fields,
independent copies, returned records and record arrays, named/returned
callbacks and the existing raw generated-list consumer. The unchanged original
interpreter suite runs with these tests in the component workflow. My first full component plans fail in the new C witness before the later source
provider steps: I assumed declaration order for a spread carrier, but its rows
field precedes the overridden item field. The retained Apple/GCC/Clang logs
show the same line2744 failure. I correct only the witness to find names and
check tags before access; production spread and transfer behavior are unchanged.
Both new evaluator functions then pass in a bounded isolated diagnostic with
fresh env/eval/io objects and hash-bound cached dependencies. This diagnostic
does not replace a fresh full-provider or full original interpreter gate.

I retain the first evaluator fixture's undeclared `dyn_array_free` build
refusal. I correct the fixture to use the actual existing `gc_release` API,
leave production dynamic-array ownership unchanged and pass strict C11/Werror
syntax checks. Both syntax outputs remain in this directory.

I predeclare the original complete Linux component plan with GCC and Clang,
the same Apple component plan and fixed six-attempt alternating original GCC
source creation against baseline `52933101c15be6fd189c5cecc2189cbbb29a23ba`.
I preserve every result and existing source inputs, build flags, selected
shadows, ten-second supervision and CLI budgets. I do not repeat an unchanged
failed plan for green. The physical kernel experiment remains completed; this
candidate does not launch another privileged observer.

This transfer removes a redundant copy. It does not account for or reclaim
the remaining escaped/unreachable frame carriers under #1116. Whole-source
improvement, reliable creation, full component/fresh-stage/platform qualification,
security disposition, NativeSDK closure and release acceptance remain open.

I retain the diagnostic collector's initial unused-static-test build refusal.
My corrected collector preserves the original main as an unexecuted global
function, retains its references, and runs the two new unchanged tests from a
separate diagnostic main. I preserve the first positional witness image
losslessly. The corrected full component plans require a new test source
checkpoint; the original fixed six-attempt GCC comparison remains on its live
handle and original product source pin.

## My global initialization scope

I verify that run_program and run_shadow_tests use the same API for top-level
let initializers. My transfer therefore covers completed fresh record literals
at global initialization as well as inside functions. I do not restrict the
implementation to a local-only call path. A new actual evaluator control keeps
global record/STRING snapshots independent through reassignment and three
shadow initialization cycles. All three new test functions pass in the bounded
diagnostic recorded in global-evaluator-diagnostic.json.gz, with fresh
env/eval/io objects and stable cached dependencies. I preserve the preceding
corrected diagnostic image losslessly. This is still not a complete fresh
provider gate.

I predeclare the existing Instrumented shadow calls GCC/Clang ASan/UBSan
workflow at my final test pin for the complete original interpreter suite,
including this global control. I retain its exact original flags, selected
tests and budgets. The running complete component and six-attempt comparison
plans remain on their original pins and handles; production bytes are unchanged.

## My failed whole-source adoption gate and retirement

Fixed GCC run42598208925 completes all six declared attempts with valid
comparison protocol. Baseline attempts1/5 create native compilers and attempt4
fails the unchanged ten-second shadow rule. Candidate attempts2/3/6 all fail
that rule: attempt2 completes1270 of1271 started bodies, while3/6 complete all
1286 before the supervision failure. Every completed assertion count is zero.
The full-corpus candidate body sums are9.874466s and9.845426s; baseline full sums
are9.544708s,9.690298s and9.509501s. Body sums do not establish actual reap time
or a historical failure cause. No complete successful pair supports a
whole-factory percentage.

My standalone audit verifies all55 archive entries,197 shared Git inputs,
67 stable source/provider inputs per factory and both actual C seed images
against their recorded hashes. Two baseline-created ELF compilers match their
before/after product hashes; this factory does not execute them. I retain all
four actual ELF images losslessly under .test_output/fresh-transfer-comparison-native/.
Other provider and host-tool hashes are reports without uploaded native SDK
bytes. The report and complete text/inventory records are in comparison-audit/.

I reject adoption and restore src/env.c and the optimization-specific environment
API test byte-for-byte from52933101c. I retain the meaningful evaluator tests
for record literals, spreads, copies, global initialization, callbacks and
escapes, and the workflow's complete interpreter invocation. I do not rerun
this completed comparison for a better sample. Complete Clang/Apple component
passes describe the tested candidate pin; they do not replace the failed fixed
comparison, qualify the restored final head or close source reliability/frame
reclamation.
