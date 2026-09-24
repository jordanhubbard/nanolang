# My copied native representations

I retain three gaps in my directed storage graph. An uncalled array parameter
can lose its declared container kind; an unconstrained projected scalar loses
its tagged provenance across local copies; an array read copied into a record
local does not receive that consumer's container requirement.

I retain declared array containers without choosing an element representation.
After forward conversion facts settle, I propagate only record and array
container requirements backward through unresolved copies. I keep source and
destination layouts separate. Scalar types, optional payloads and field types
do not propagate backward. Conflicting producer fields still fail. Finally,
I propagate unconstrained projected-value provenance through directed copies
without widening exact consumers.

My diagnostic trace identifies the C-seed compiler's `build_field_metadata_index`
(function 325, offset 37): its unknown array element flows into record storage
at nodes 287242/287243. The private prototype accepts the retained full compiler
module, compiles it under strict Clang, and its compiler produces a hello
module that executes. The final implementation places this demand rule in the
shape solver; the prototype alone is not final qualification.

My unchanged twelve generic-array controls and two scalar-local cases pass
after the container/provenance correction. I then extend scalar projections to
three-copy chains and record-array readers to zero, one and two local copies
in both declaration orders, with actual generated native execution. Shape
controls retain distinct layouts, refuse conflicting fields/container kinds
and leave scalar and optional source kinds unconstrained.

The intermediate complete One IR run reaches another fixture defect: the
compiler's legitimate string literal `bin/nano_vm` trips a substring check.
I strip only C string literals before applying that same refusal to executable
source, with controls for escaped literals and actual forbidden identifiers.
My final compiler paths must still translate, compile and execute their
products. The complete owning run and final sanitizer qualification are being
retained separately; intermediate passes do not qualify the final source.

My earlier complete translator run passes 2,524 assertions ordinarily and
with fresh private ASan/UBSan objects before the final solver change. Those
runs retain the sanitizer driver's existing leak exclusion. The full One IR
run uses the existing CI shadow limit of 60 seconds. I do not claim that an
instrumented full compiler fits the separate ten-second default shadow limit.

My final ordinary and private ASan/UBSan translator gates both pass **2,558
assertions**. The final shape suite passes **1,614 checks**, including strict
GCC 13 at `-O3 -Wall -Wextra -Werror` in the isolated Linux ARM64 checkout.
GCC also compiles the final translator translation unit at those strict flags.
The complete fourteen-module Python owning run subsequently finishes with
all 31 One IR methods passing and 60 neighboring leak-runtime failures. My
[selected-compiler evidence](../selected-native-retention/README.md) retains
that terminal and the corrected 45-method rerun. Final hosted acceptance
remains incomplete.

The earlier d48 VM/native fixed points precede this compiler change. Final
fixed-point qualification must be repeated after the outstanding compiler
repairs, including separately tracked setter and nominal-order failures.
