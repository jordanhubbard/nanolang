# My guarded call-scoped borrow annotations

I retain named-parameter `&T` and `&mut T` annotations in both parsers as a
prerequisite to my existing affine contract. I do not yet accept borrowed
programs: both production frontends explicitly refuse borrowed parameters
before publication. Shared reads, exclusive mutation, borrow overlap,
reference escape and NanoISA ownership facts remain implementation work under
`task_71821d84befc46e198795122c1112a27`.

I append `TOKEN_AMPERSAND` to the shared schema without changing existing token
numbers. My C parser retains a distinct shared/exclusive TypeInfo wrapper with
an independently owned referent tree. My self-hosted parser retains canonical
`&owned.Handle` and `&mut Box<array<int>>` parameter strings. Neither representation
is a runtime pointer ABI. Borrowed results, stored fields, nested references,
callback signature arguments and argument expressions remain unsupported with
explicit diagnostics. Ordinary by-value resource semantics remain unchanged.

My C parser regression inspects qualified and nested generic referents, copies
an annotation, changes its mode to prove inequality, frees the original AST,
and reads the surviving copy before freeing it. A stronger borrowed-function
referent case exposed a missing nested signature in my initial implementation;
`bfd19dc7` retains an owned signature copy and tests its array parameter and
string result. I do not claim compiled-module or IR borrow metadata acceptance.

My six driver methods run on C seed, Stage 1 and Stage 2. They inspect the actual
self-hosted parser's stored function parameters, require explicit refusal of
four parameter forms and an extern declaration, reject five unsupported type
contexts and borrowed arguments, preserve an existing output artifact after
rejection, and execute an ordinary resource-consumption control. All six methods
passed in 52.321 seconds; these are metadata/refusal tests, not borrow execution
acceptance. My initial parser-inspection fixture needed an explicit lexer import;
I corrected that dependency and reran it without changing compiler behavior.

The full C parser and typechecker checks passed at the first source checkpoint.
The generated schema check passes after committing the generated token files.
A fresh default-budget three-stage native bootstrap passed at `ff390c43`.
I also retained two Stage 1 failures at the default ten-second shadow deadline:
`/tmp/nanolang-borrow-bootstrap.log` and
`/tmp/nanolang-borrow-parser-final.log`. The successful same-budget run is
`/tmp/nanolang-borrow-bootstrap-r2.log`. I do not infer the cause from the retry.
The separate Linux deadline task `task_628759a2daf743b9bf13c9a7fea2ced0` holds this
additional evidence; neither its gate nor full release acceptance closes here.

At final source `bfd19dc7`, a fresh three-stage native bootstrap and complete
C parser, typechecker and module-metadata targets passed with the explicit
bounded diagnostic setting `NANO_SHADOW_TIMEOUT_SECONDS=60`. The generated-list
metadata initialization check also passed. The log is
`/tmp/nanolang-borrow-final-gates.log`. I do not change the product default or
remove mandatory shadows.

The regenerated final stages then passed all six methods again in 51.694
seconds without a timeout override: 33 guarded rejection decisions and six
executable ordinary/parser controls across the three drivers. Final paired log:
`/tmp/nanolang-borrow-paired-final.log`. I reran these because the final C parser
repair rebuilt the stages; this is exact-source evidence for the prerequisite.
