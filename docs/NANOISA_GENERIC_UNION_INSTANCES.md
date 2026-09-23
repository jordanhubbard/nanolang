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

Each concrete instance retains its exact source identity. Existing wire union
IDs identify bound declarations; they do not encode concrete arguments. I keep
that format and the existing verifier boundary, with argument equality and
payload transport checked before emission. I do not claim that the wire ID
alone distinguishes `Box<int>` from `Box<string>`. The ordinary C-seed baseline
emits both under declaration0 and native translation succeeds. Selected
constructor context supplies missing arguments only when its bound declaration
agrees exactly. Explicit arguments must match that context.
Named fields remain complete and unique, evaluated in source order and packed
in declaration order. Match payload bindings retain the instantiated identity
and their original lexical endpoints.

For a complete selected pattern such as `let Box.Some { value } = payload`,
I retain `Box<string>.Some` from the lexical match binding. I check the exact
variant and every field once before publishing the hidden pattern local. The
pattern spelling alone cannot establish a selection or erase concrete arguments.

I admit ordinary scalar and exact `array<int>` payloads, and bounded nested
ordinary union instances. Known unused type arguments do not become stored
payloads. Resource-bearing unions,
unknown/unsupported argument or payload shapes and invalid copy/move paths
remain refused. I do not change the resource checker, reference authority,
owned runtime profile or generic resource disposal rules.

I require the unchanged `tests.test_affine_module_identity` and
`tests.test_affine_generic_identity` decisions, canonical Stage1/Stage2
NanoISA publication, mandatory selected shadows, VM/native values and preserved
negative outputs. Focused raw-producer checks retain instantiated source identity and declaration-level wire
metadata, same-declaration multiple instances, exact constructor context,
formal shadowing and wrong-instance refusals. My C-seed baseline retains the shared declaration ID; paired checks compare
that same contract. Full frozen product acceptance
and release remain separate gates.

## Nested instance implementation contract

I separate a known type argument from a stored payload. `Marker<Handle>` may
be ordinary when no payload uses `T`; `Box<Handle>` still contains an owner.
I validate argument spelling and arity before substituting fields, then walk
the actual payload types. Nested ordinary unions retain their exact source
identity and use the existing aggregate representation. I reject repeated
instance cycles and stop expansion at the existing checker depth bound of 128.
This is a bounded producer check, not a new ownership or wire-format guarantee.
VM/native execution, nested copies, parameters, returned values and unchanged
owner refusals must qualify the implementation before I mark it complete.
