# Legacy independent binding groups

I preserve legacy `par-let name=value ... in expression` as a distinct AST block.
I infer immutable bindings, check each initializer without sibling names, then
make the bindings visible to the body and the rest of the enclosing scope.
An initializer may read its own same-spelling outer binding. I emit ordered
initializer and body effects and discard the body result. I preserve each source
name after its initialization store, including `_` and shadowed names.

My C checker previously printed sibling-dependency errors but returned success
and overwrote an existing artifact. I propagate the error to compilation status.
My refusal fixture checks direct, outer-shadowed and aggregate sibling references,
malformed groups, nonzero exit and unchanged prior output.

My initial selfhost parser stored the block without setting its statement kind.
The checker skipped that statement and reported the following return's binding
as missing. I preserve that diagnostic and the corrected harness result. I assert
the returned statement kind in the parser shadow.

My fifteen C-route methods pass with sanitized translated products, including
ordered effects, arrays/strings, same-name outer reads and discarded scalar,
array, tuple and record results. Component gates and schema-generator parity pass.
The first product audit rejects my new shadow's unqualified `diag_list_new`
call. I correct it to `Diagnostics.diag_list_new`; the seed was built immediately
before that shadow-only correction. The resulting compiler must build the corrected
source and execute its dependency shadows. I retain the failed audit alongside the
final product evidence. My corrected product audit passes all seven steps: self-compilation with dependency
shadows, verification, translation without a VM wrapper, strict native linking,
help execution, fifteen sanitized metadata/legacy methods and 22 nominal/layout
methods. All eleven production input hashes remain unchanged during the audit.
I retain an additional builtin-shadowing probe: a later `array_length` binding
causes earlier builtin-call diagnostics in C while the NanoISA driver still
returns success. The selfhost route accepts it; C-native also diagnoses it. This
separate source-order/status defect remains open in the roadmap. I do not claim
release readiness.
Named nested functions and the full producer/reconstruction/platform gates remain open.
