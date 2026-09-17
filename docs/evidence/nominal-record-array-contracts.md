# I check nominal record-array boundaries

I previously accepted a homogeneous array of one record type where another
record type was declared. I now compare declaration identity at local and
global initializers, mutable assignment, ordinary and qualified call arguments,
record fields, fixed union payload fields and function returns.

I preserve empty array context and nominal identity through literals, aliases,
record fields, declared producers, count/fill construction and slicing. Module
aliases resolve to the owning declaration, so identically shaped records from
different modules remain distinct. I retain function return element metadata in
nested and module checking contexts.

Three regression methods exercise ten wrong-type contexts with both the C seed
and NanoVirt, plus matching execution and imported declaration identity. Failed
checks preserve an existing output artifact. Matching programs execute natively
and in NanoVM. Focused record-literal and projection checks, the typechecker
suite and a fresh three-stage bootstrap pass on the final source. Four C union
payload integration methods also pass: fixed, generic and nested ordinary
record-array payloads execute, and a wrong fixed nominal payload is rejected
without replacing prior output.

I do not infer generic union constructor substitution from fixed payload checks.
An inline `Box<T>.Some` expression needs the enclosing concrete instantiation
before I can compare `array<T>` with `array<Plain>`. This remains a separate
contextual typing prerequisite. I do not claim new nested-array or callback
signature equivalence from these fixed nominal checks.
