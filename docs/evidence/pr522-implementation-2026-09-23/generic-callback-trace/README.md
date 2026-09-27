# I isolate generic callback translation failures

I capture six original accepted fixtures from `tests.test_generic_function_values`
at `d6d3d60b5`, using both `nano_virt` and fresh `nanoc_stage1` to emit bytecode.
All twelve emissions succeed and all twelve modules execute successfully in
NanoVM. These checks supplement the full 18-method suite already retained in
`../factory-locals`; I do not claim to qualify Stage2 or imported callbacks here.

| Original fixture | C frontend module translation | Stage1 module translation |
| --- | --- | --- |
| generic result | conflicting record/integer parameter kinds | conflicting integer/record parameter kinds |
| generic parameter | passes | no exact scalar target |
| local function value | conflicting record/integer parameter kinds | conflicting integer/record parameter kinds |
| forwarded function value | conflicting record/integer parameter kinds | conflicting integer/record parameter kinds |
| nested array callback | no exact scalar target | no exact scalar target |
| indirect record literal | passes | no exact scalar target |

The two translated C-frontend modules compile with strict warnings and
ASan/UBSan under Homebrew Clang, then execute successfully. Compiler binaries
and external libraries are ordinary. I retain exact commands and terminals in
`results.json.gz` and `additional-results.json.gz`.

## I separate target provenance from inference order

My `FUNCREF` classifier records only the function representation and invalidates
unique argument variants. My `CALL_INDIRECT` classifier does not use the runtime
function's identity: it scans every module function with matching arity and
result count, and excludes aggregate results through `scalar_kind_for_tag`.
For a zero-argument union factory, an unrelated integer-returning function can
therefore supply the inferred result kind. The later direct union consumer
reports conflicting storage kinds. Merely adding record result storage would
still leave unrelated targets contributing their shapes.

The record-argument failures have an additional, demonstrated order dependency.
I reorder only the Stage1 function tables in the generic-parameter and
record-literal modules and remap every function index, entry and parameter row.
Bodies and signatures stay unchanged. Both reordered modules assemble, execute
in NanoVM, translate, compile with ASan/UBSan and strict warnings, and execute
successfully. Their original orders refuse translation. I retain both original
and reordered assembly plus exact terminals in `reordered-results.json.gz`.
The first harness placed parameter directives before their functions; its
assembler rejection is retained separately as `reordered-harness-before.json.gz`.
It is a harness error, not evidence of a compiler defect.

My next repair must converge callback facts before rejecting unresolved targets,
carry actual target provenance through parameters, locals, returns and joins,
and propagate aggregate argument/result shapes through the selected calls.
I require function-table permutation coverage, unrelated same-arity decoys,
VM/native parity and the original positive and negative acceptance suites.
Reordering emitted functions is diagnostic evidence, not the product repair.
Resource ownership transfer remains a separate requirement. #522 stays draft.
