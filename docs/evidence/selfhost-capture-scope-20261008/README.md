# My self-hosted lexical-scope checkpoint

I distinguish anonymous declarations with `PNODE_LAMBDA`, appended to the shared
enum without changing the AST record layout. The creation identifier retains its
source position and resolves that explicit marker. A generated-looking name by
itself does not establish an anonymous declaration or creation site.

I check each anonymous body where its expression occurs, using the current
lexical symbols and its own parameters. I no longer check it independently with
only globals. Nested bodies inherit intermediate scopes; ordinary block copies
preserve shadowing. I retain resource checks on anonymous bodies. The parser now
recognizes an immediate zero-argument anonymous call instead of returning only a
grouped function value.

My regression harness compiles the checker itself, runs selected dependency and
root shadows, and executes the same assertions in NanoVM and strict C11 native
code under ASan/UBSan/LSan. It covers nested captures, parameter/block shadowing,
mutable captures, shadow-local captures, immediate invocation, inaccessible later
and block-local names, immutable mutation, wrong returns and resource cleanup.
This tests checker behavior, not execution of captured source by my emitter.

I rebuild the complete native compiler from its C-seed-produced module. The
unchanged canonical capture source now passes checking and reaches the emitter's
`undefined binding n` refusal, with prior output preserved. Transitive capture
metadata and source `CLOSURE_NEW`/upvalue emission remain open.

I retain the first missing-public-export build failure and the immediate-call
fixture failures. The initial fixture used double parentheses, then a single
pair; both exposed grouping instead of invocation. The debug log reports a
function-valued return where an integer was required. Separately, my C seed
accepts the immediate-call type but refuses its shadow bytecode with an undefined
generated lambda. `cseed-immediate.nano` and its log retain that still-open defect.

My focused qualification passes one method in 27.250 seconds after
retaining resource checking. The freshly rebuilt compiler passes all 69 parity/CLI/product/scope methods in
55.910 seconds. All four generated schema artifacts regenerate
byte-identically. Full release, raw fixed-point, import/capture lifetime and
cross-platform acceptance remain separate obligations.
