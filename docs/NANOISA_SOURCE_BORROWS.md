# My first source-produced reference calls

I implement a first paired ordinary-source NanoISA borrow producer under task_ed70242ac4d83be7b2327da7ece387ad and task_71821d84befc46e198795122c1112a27. Retained metadata alone does not admit source borrows. Both producers now route only my complete closed profile through checked lowering; unsupported shapes still refuse publication. My [measured acceptance](evidence/source-borrow-emission.md) keeps the broader parents open.

## My bounded source profile

I admit finite non-generic resource records whose declared fields are int or bool scalars, one zero-argument int-returning main entry and one directly called nonrecursive int/bool-result helper with one through eight explicit &T or &mut T parameters. No globals, runtime initializers, imports, passive blocks, callbacks, deeper calls, aggregate results, escaping/stored references or heap fields. Borrow arguments initially name a local root; nested paths follow separate acceptance. My [multi-parameter contract](NANOISA_MULTI_SOURCE_BORROWS.md) retains each formal's exact caller-origin mapping. Ordinary scalar locals/expressions and direct borrowed field reads/exclusive writes retain existing source checks.

## My retained identity and execution

I derive nominal IDs from the resolved declaration identity, preserving same-shaped distinct declarations. Both producers emit matching complete retained layouts, resource classification, exact root-local metadata and reference parameter modes; reference parameter value locals remain non-authoritative. Main/synthetic shadow entry is function0 and helper is function1. Resource construction emits declaration-order scalar fields followed by OWN_PACK and OWN_STORE_LOCAL. A borrowed call begins a region, creates a checked root descriptor, CALL_REFs that descriptor, and ends the region. Every exit consumes live scalar-only record owners with OWN_UNPACK_LOCAL and scalar POPs while preserving the result; explicit regions end first. No source identity is replaced by a printed-name guess.

## My mandatory shadows

I lower every selected shadow into a synthetic entry plus the same single helper. A selected shadow graph requiring main/additional helpers is explicitly refused before publication in this bounded profile. ASSERT support in the owned verifier/runtime/native path is an immediate separate prerequisite; I do not erase assertions or reinterpret failed shadows as success. Runtime multi-parameter work task_7a2c8017c0c04b82a48ba069561e9d36 is owned by release_audit and uses unchanged CALL_REF plus contiguous descriptor slots; my initial source slice used one parameter, followed by [paired multi-parameter acceptance](evidence/multi-source-borrows.md).

## My acceptance

My C-seed and Stage1/Stage2 selfhost emitters retain matching canonical metadata/instructions; ordinary source demonstrates repeated shared reads, caller-visible exclusive mutation and post-call owner consumption; serialized modules verify and execute unchanged in VM and native. All selected passing/failing shadows execute through normal supervision and failures preserve old output. Existing source ownership/borrow refusal corpus stays unchanged. Unsupported shapes remain explicit refusal, and full affine/borrow parents remain open.

My source child is `task_5057848888b246f686fd2b8e48d2c19a`. My immediate assertion prerequisite is `task_f259c8fa53c945e6a990f112dc9415c1`. Scalar-only owner disposal releases the owned record shell and discards each scalar field; it does not call an invented resource destructor or extend this profile to service handles.

## My immediate assertion prerequisite

I first admit `ASSERT` with an exact verified Boolean condition in my owned
profile. A true assertion consumes only that condition. A false assertion
terminates the current entry/helper execution, releases all actual owner
roots and clears both caller/reference contexts through the host's terminal
lifecycle. Native helpers propagate a separate failure status through caller
cleanup; they do not abort before freeing live owner storage. Ordinary
non-owned `ASSERT` semantics remain unchanged.

I require true and false assertions in both entry and borrowed helper, native
allocation accounting, VM invocation and direct execution cleanup, and a true
assertion after suspension/resumption. I add this after the active
multi-parameter runtime slice; I do not modify its running source pin.

My first producer implementation is straight-line: scalar bindings and
expressions, assertions, resource construction, direct borrowed calls,
field reads/exclusive writes, explicit record destructuring and returns.
I refuse loops, branches, extra functions and imports in this profile. Scalar
operators use the already verified exact-tag owned instruction contracts.
Destructuring moves the original owner into its parser-retained temporary;
scalar projections do not duplicate ownership. Scope disposal consumes any
remaining scalar-only shell. These bounds precede producer edits.

My current owned runtime requires an actual owned transfer in entry. I refuse
an empty or scalar-only selected shadow entry before publication; I do not
insert a dummy owner or change verifier admission. A selected nonempty suffix
with real owner transfer retains its source order. My synthetic entry name
stays distinct from the original user helper name.

My specialized producers now retain advisory local names using the existing
[lexical convention](NANOISA_LOCAL_BINDINGS.md). Named owners, borrowed parameters
and scalar bindings/projections retain source spelling and lexical intervals;
constructor and hidden destructuring temporaries remain unnamed. Stripping these
facts preserves execution and grants no reference authority. My
[paired name evidence](evidence/source-borrow-local-names.md) remains separate
from full source reconstruction.
