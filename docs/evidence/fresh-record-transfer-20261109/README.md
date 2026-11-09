# My fresh local record transfer candidate

I follow the measured binding experiment from PR #1137. I change only the
local evaluated-binding path: a completed `AST_STRUCT_LITERAL` whose value is
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
