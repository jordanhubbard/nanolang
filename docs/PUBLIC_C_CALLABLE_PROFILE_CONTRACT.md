# I refuse callable values without a public C callable ABI

I execute `task_9ba42264f6694f66bc95552f3a834940` under original6ade, after qualified array8026 (PR779 pending canonical merge). This contract precedes code. Static review finds TYPE_FUNCTION still reaches c_type's int64 fallback in ordinary locals/parameters/results/fields; calls can then render f(...) on that integer representation. Anonymous metadata was separately refused in777, and lifted computed calls already refuse. I do not treat those narrower guards as general named-function-value handling. No failed artifact was executed.

I choose checked refusal, not a new function-pointer or captured-environment ABI. Explicit TYPE_FUNCTION locals/globals, function parameters/results (including externs), record fields and union payloads are outside this public C profile. Retained TypeInfo nesting or nested function annotations cannot erase that boundary. I preserve ordinary function declarations whose parameter/result types remain qualified scalar/string/exact nominal types: the existence of a declaration or its descriptive signature is not itself a first-class callable value.

I refuse identifiers used as values when exact lexical metadata says FUNCTION or when an unbound identifier denotes a declared function. A lexical scalar with the same spelling retains its scalar identity. Direct AST_CALL with a declared name and no expression callee retains its existing exact declaration and argument/result checks. Builtins continue after actual declaration/binding precedence; source builtin reservations remain the shared checker policy, and same-spelled AST API declarations are tested separately. Function-typed call arguments and result signatures refuse before emission; a checked signature is inspected for callable parameter/result values rather than indiscriminately rejected merely because signature metadata exists.

Expression/computed callees and calls through local bindings have no qualified callable storage ABI here. I refuse them explicitly instead of selecting a builtin or a same-spelled global declaration. I do not change direct named scalar calls, module-qualified existing direct calls with retained supported facts, recursion, private source main identity, scalar helper ordering, string ownership or the underlying source language's function-value support in other producers. I do not infer ABI support from a C expression that happens to compile.

The guards apply before all public declaration/storage emission paths and reachable value/call consumers. Both path and FILE entrypoints preserve prior bytes on semantic refusal, retain the first error, release private staging state and recover on a later valid invocation in the same process. Existing external FILE I/O partial-write limitations remain explicit. No invocation retains AST/type ownership or global option/error state. Header and context inventory describe the actual direct-call profile without promising unnamed/first-class callable admission.

I send production for independent review before frozen execution. Ordinary source controls cover named function assignment, callable parameters/results and fields where accepted by the shared checker; direct API controls exercise exact retained annotations, nested signatures, identifier/function-value and expression/local-bound callee refusals at both public entrypoints, then recovery. I distinguish frontend refusal from backend refusal. Positive source/API controls retain direct scalar/string/union calls, recursion, exact declared-name precedence and actual lexical scalar identity. Strict GCC/Clang C99/C11 O0/O2 ASan/UBSan plus affected adjacent profile/array/lifting tests and seven existing programs precede closure. First outcomes, immutable source/harness/tool identities and API/source distinctions remain recorded.

Original6ade closes only after a clause-to-evidence audit establishes its actual inventory/portable-or-refused boundary. I add no universal-admission, new capture ABI or platform claim. Loop/global lifted insertion and shared wildcard/no-success policies remain accurately classified independent obligations rather than reasons to invent source support in this backend.

## I update old private rendering controls to the explicit refusal boundary

My frozen f5855788 focused three-method gates pass under GCC and Clang, and all seven ordinary programs pass. Both adjacent fifty-method gates finish with46 passes and four failures: private concat/int/length/format fixtures assert that TYPE_FUNCTION-bound or expression callees emit text successfully. Their prior success checked spelling only, not a qualified callable ABI. This contradicts the newly reviewed checked-refusal contract. I retain both original terminal logs and all46 file/three compiler identities; I do not execute those failed artifacts again.

Before any fixture change, I record a harness-only correction: these exact local-bound and expression-call controls must require the callable diagnostic and no emitted expression bytes, then reset invocation-local error state for later independent controls. Existing direct declaration and descriptive scalar checked-signature success controls, scalar conversion behavior, output publication and recovery remain unchanged. I freeze the corrected four fixtures before fresh focused and adjacent gates. Production remains byte-identical; this is not a new callable admission or a claim that the first aggregate gate passed.

## I retain checked scalar import identities in source-only emission

My C seed's `--target c` path retains the checked dependency function closure.
I resolve each direct call in its declaring module's namespace, including
transitive aliases and private helpers. Same-spelled functions in distinct
modules receive distinct private C names. I derive qualified result types from
the resolved declaration. I emit prototypes before definitions and preserve
my root entry identity. This path does not execute dependency or root shadows.

This import closure admits scalar function declarations (int, u8, float, bool,
string and void) within my existing C expression profile. Imported storage and
nominal declarations receive checked refusal before publication; this does not
establish an imported layout or callable-value ABI. My direct AST API retains
its existing flattened qualified-name convention when no checked resolver is
provided. An optional resolver borrows its context and declarations for both
planning and emission; each returned definition must occur in the retained root.

My imported-shadow suite compiles and executes the resulting C, including two
transitive dependencies with same-spelled helpers and string results. Deliberately
failing shadows remain unexecuted. Unsupported imported storage preserves the
previous artifact. My standalone API checks use the same service-declaration
predicate as the parser, without requiring the parser to link into each fixture.
