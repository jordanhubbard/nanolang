# My exact generic union instances

MAC `task_6550ccf97eb44f4d8c08f02cd4189cd7`, under my affine source and
cross-backend equivalence roadmap.

I retain a union's exact bound declaration identity and concrete type arguments
through construction, typed locals, parameters, calls, returns and match
payloads. Lexical generic formals are substituted before resolving any
same-named resource or module record. I use the existing token-aware type
substitution helper; I do not replace substrings or introduce unknown-type
wildcards. Arity, duplicate formals, unknown arguments and unresolved payload
spelling fail closed.

Each concrete instance has its own nominal identity. My executable union IDs
and `.types` bounds retain those distinctions; I do not merge differently
instantiated declarations because their current payload shapes happen to
match. Selected constructor context supplies missing arguments only when its
bound declaration agrees exactly. Explicit arguments must match that context.
Named fields remain complete and unique, evaluated in source order and packed
in declaration order. Match payload bindings retain the instantiated identity
and their original lexical endpoints.

This slice admits my existing plain scalar payloads and exact `array<int>`
payloads required by the retained source corpus. Resource-bearing unions,
unknown/unsupported argument or payload shapes and invalid copy/move paths
remain refused. I do not change the resource checker, reference authority,
owned runtime profile or generic resource disposal rules.

I require the unchanged `tests.test_affine_module_identity` and
`tests.test_affine_generic_identity` decisions, canonical Stage1/Stage2
NanoISA publication, mandatory selected shadows, VM/native values and preserved
negative outputs. Focused raw-producer checks retain instantiated nominal
metadata, same-declaration multiple instances, exact constructor context,
formal shadowing and wrong-instance refusals. I inspect C-seed instance
allocation before claiming exact paired IDs. Full frozen product acceptance
and release remain separate gates.
