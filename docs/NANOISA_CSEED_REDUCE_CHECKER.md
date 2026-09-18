# My exact C-seed reduce checker prerequisite

I record task_65b8b96f489b4e268a0a79b0052cbe61 under collection callback75b340 and named producerfd15,
with scalar callbackd099 and policy5009 still open. This contract precedes code.
My fresh23224572 negative gate accepted a FLOAT initializer with a declared
fn(int,float)->int callback. I preserve that log and emitted artifact; I do not
execute the mismatched callback. Static review locates the incomplete branch
in typechecker.c: it checks each operand independently, then returns A.

I require exactly three operands: a checked array<E>, a non-void known initializer
A, and an actual known fn(A,E)->A. The callback signature comes from the visible
value binding before a declaration, or from the result signature of a computed
callback expression. A same-named function cannot override a lexical/global
callable. Existing ordinary call/builtin dispatch remains unchanged.

I compare exact known kinds before identity; permissive types_match/types_equal
cannot justify int/float, enum/int or UNKNOWN compatibility. Primitive kinds
need no artificial heap annotation. Nominal/array/tuple/function types require
their retained complete annotation or an exact resolved declaration identity.
My record and enum declarations have no generic parameter list; their exact
identities require zero type arguments. Generic union definitions provide the
authoritative argument count.
I preserve recursive generic arguments and reject missing required components;
a source name alone cannot erase generic arguments. I use bounded recursion and
borrow existing metadata without publishing partially allocated state.

Typed empty arrays retain E. A fresh empty literal can use callback E only if
existing checked contextual rules establish it; I do not infer E from an
unrelated initializer. Array literal leaves must match the expected exact E,
including nested leaves, rather than relying on a widened container base kind.
For source expressions with complete retained array metadata, I compare that
actual element identity. The initial expression must match A and callback result
must match its first parameter, including full retained identity where needed.

Failure emits a precise reduce arity/type/signature diagnostic and returns
UNKNOWN through the existing checker error path before output publication.
I do not change map/filter compatibility, general types_match, runtime calls,
function-value/native admission or the independently reviewed direct producer.
Unknown required identity gets an explicit refusal; I do not substitute VOID.
I report any valid formerly accepted shape that lacks sufficient retained facts
as a separate prerequisite rather than silently claiming whole-language parity.

I freeze reviewed production and fresh controls before qualification. Positives
cover int/float/bool/string, typed empty arrays, local/computed callback values
and fully retained nominal/tree identities at the checker boundary. Negatives
cover too few/many operands, non-array, non-function, callback arity, accumulator,
element/result mismatches, enum/int, distinct nominal/generic identities and
missing required type facts. Cseed public publication preserves prior output.
Existing paired source reduce and named producer gates must retain their actual
supported runtime boundaries; no old failed artifacts are replayed. I run
focused checker, scalar-reduce and NanoVirt controls; sourcebootstrap evidence
retains its exact pin when only the C checker changes.

I close only this prerequisite after reviewed canonical merge. The broader
collection callback, function-value, scalar policy and release work stays open.
