# I require observable generic-union source parity

I continue `task_a18a9f752536469faafc4d3ebec01dfd` after the bounded
identity and exact-value refinement in PR893. That checkpoint proves structural
facts and executes constructors and relays. Its source fixture uses
`assert true` shadows and never matches a concrete payload. I do not call that
semantic source parity.

## My admitted source

I use one root-module declaration with at least two simultaneous concrete
instances whose substituted payloads differ. The first acceptance source uses
`Choice<int,string>` and `Choice<float,bool>` together. Every constructor,
parameter, local, result and match is resolved to the declaration plus its
recursive type arguments; a display spelling or shared variant ordinal is not
authority.

I admit exact scalar payloads already covered by the PR893 carrier. I do not
add imported aliases, nested aggregates, resource payloads, wildcard patterns,
new guard policy or mixed `ARRAY_FIELDS` authority here. Kind 2 remains
mandatory-understanding and unavailable until its independently owned validator
and the shared mixed-envelope conjunction are qualified.

## My statement and value behavior

A statement match evaluates its scrutinee once, tries arms in source order and
binds only the selected variant's exact concrete payload. A `return` in an arm
exits the enclosing function. A falling-through arm does not invent a value or
leak its payload binding.

A match expression evaluates its scrutinee once. Its selected arm's final
expression supplies exactly one `int`, `bool`, `float`, or `string` result.
Every reaching arm has the same exact result type. The result must be observed
by ordinary assertions and deterministic program output; an `assert true`
shadow does not establish payload transport.

I exercise both variants and the empty variant where declared. I distinguish
the two concrete generic instances in the same module and observe payloads whose
values would expose swapped layout, shared ordinal, wrong field tag or default
zero behavior. I retain lexical payload scope, source-order guarded-arm rules,
once-only scrutinees, exact `MATCH_TAG` edge proofs and the terminal unmatched
invariant already established by PR889 and PR893.

## My independent routes

The unchanged positive source is compiled independently by:

- C-seed `nano_virt --emit-nvm`;
- installed Stage 1 `--emit-nvm`;
- installed Stage 2 `--emit-nvm`;
- C-seed `nanoisa_emit` text assembly; and
- the self-hosted `nanoisa_emit` product.

Each artifact is assembled where needed, structurally verified, executed by
NanoVM, translated by nvm2c, compiled as strict C with the platform's supported
ASan/UBSan/LSan toolchain, and executed natively. Every route must produce the
same exact stdout and successful assertions. I inspect each dump independently
for both concrete type names, match instructions and payload projections. I do
not infer one frontend's success from another frontend's artifact.

Selected dependency and root shadows remain mandatory. They assert real
results for both concrete instances and both statement/value paths. A failed
shadow must prevent publication.

## My refusal boundary

I retain prior output for every refused route. Controls include a payload field
used with the wrong substituted scalar type, a value assigned across concrete
instances, a result match whose arms disagree, an unknown or incomplete arm,
and a payload binding used outside its lexical arm. Each frontend must report a
checked match, concrete identity, payload type, coverage or scope diagnostic;
parser crashes, silent output replacement and generic fallback messages do not
qualify.

I preserve the first terminal from every new route before correction. I qualify
production and fixtures on fresh current-main tools on Linux and Darwin before
claiming this row. This bounded source work does not qualify mixed
`UNION_VARIANTS` plus `ARRAY_FIELDS`, imported identity, the full product, PR522
or a release.
