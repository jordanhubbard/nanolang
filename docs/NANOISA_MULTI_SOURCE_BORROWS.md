# My bounded multi-parameter source borrow contract

I extend source child505 after PR #606 under MAC
`task_f209d694d3be415982456b00b4df5ad9`. My existing
[multi-caller runtime](NANOISA_MULTI_CALLER_REFERENCE.md) remains unchanged.

I retain exactly entry0 and one nonrecursive helper1. The helper accepts one
through eight borrowed parameters, each naming an exact finite resource record
with int/bool leaves and an explicit shared or exclusive mode. All remaining
helper locals and the result remain scalar under my existing source profile.

I prepare explicit root-local arguments in source order into reference slots
0 through N-1 inside one region. Earlier holds remain active while later
arguments are prepared. Equal roots may repeat only as shared arguments;
any overlapping pair containing an exclusive argument is refused. Distinct
roots may use either mode. Every argument must match its own formal's exact
nominal identity, mode and mutable authority. I never infer identity from shape.

I emit exact arity, STRUCT parameter tags and per-position ownership modes and
layouts. Helper value parameter slots remain non-authoritative; REF_GET and
REF_SET address the corresponding parameter reference slot. The existing
verifier and runtime resolve each slot to its actual caller origin. Advisory
names retain each formal's physical index and begin at zero; they grant no
access. Region end and scalar-owner disposal keep their existing ordering.

I lower every selected shadow in the same bounded entry/helper profile or
refuse publication. Nested paths, value/reference mixtures, control flow,
imports, globals, deeper calls, aggregate results and escaping references remain
unsupported. This work does not complete the broad affine or borrow parents.

I require exact paired canonical metadata/instructions, shared aliases,
disjoint exclusive roots, mixed modes and distinct nominal records, eight
arguments, repeated calls, observable caller mutation and names for every
formal. I retain ordinary source refusals for arity, mode, nominal identity,
exclusive overlap, more than eight parameters and unsupported shapes. Passing
and failing selected shadows preserve publication semantics. Existing source,
multi-caller runtime, VM and sanitized native gates remain required.
