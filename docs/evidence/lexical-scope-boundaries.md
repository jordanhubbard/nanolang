# My lexical scope repair

I keep type metadata for later emission without treating an exited block as
still visible. Parsed ordinary and unsafe blocks now retain the position of
their closing brace. My checker attaches that exclusive bound to new local
symbols, loop variables and function parameters. An inner scope keeps its own
earlier bound. Source-aware lookup skips an exited symbol instead of deleting
the metadata.

My native emitter uses source-aware lookup for expression reads, array helpers
and assignments. Parameters reintroduced during emission receive their
function's source location and scope bound. Without that last step, my first
broader bootstrap attempt selected older metadata for several compiler array
parameters and failed native compilation; that failure was not a passing gate.
I also retain declared metadata for non-array parameters: discarding `HashMap`
type arguments while introducing a located parameter broke `map_get` emission.

My interpreter restores block-local bindings on normal, return, break and
continue exits. I preserve a yielded local string before releasing local
string bindings. This repair does not establish general aggregate or closure
lifetime safety.

## Regression scope

`tests/test_lexical_scope_boundaries.py` checks same-line and multiline scalar
shadowing, float-array metadata after integer-array shadowing, same-named array
parameters in distinct functions, HashMap parameters, loop exits, returned strings, and rejection
of references to exited blocks, sibling branches and another function's
parameter. Positive cases run shadows and the produced native executable.
Negative cases require a static undefined-name diagnostic and preserve the
previous output artifact.

`tests/test_env_scoping.c` also checks the exact closing-brace boundary and
verifies that both inner and outer symbols remain retained.

## Verification

My final run of `make bootstrap3 test-env-scoping test-eval test-typechecker
test-parser test-transpiler test-one-ir-compiler` exits successfully. It includes
bootstrap smoke checks, 36 environment checks, eight focused scope methods,
the evaluator/typechecker/parser suites, transpiler boundary checks, two
assert-emission methods and 21 native compiler acceptance methods. The two
earlier failed bootstrap attempts above are superseded by this fresh run.
This is not canonical NanoISA fixed-point evidence or full release acceptance.

My ownership matrix no longer fails its C-seed ordinary-shadow case. Its 26
remaining failing subcases are in the two self-hosted stages. These repairs do
not implement the complete ownership contract, stored closure lifetimes, or
unlocated/generated-node scope identity. My release gate remains open.
