# My bounded multi-parameter caller-origin contract

I extend the merged one-parameter contract under MAC
`task_7a2c8017c0c04b82a48ba069561e9d36`, with affine parent ed702 and borrow
parent718 still open. My base is `ce8cc3ee` (merged PR581). I pass my
[bounded paired gates](evidence/nanoisa-multi-caller-reference.md).

## My existing instruction and metadata

I retain CALL_REF (0x0f) with its existing u32 callee and u16 reference operand.
The reference operand names the first slot of a contiguous descriptor range.
Its length is the helper's declared parameter count, already present in the
function signature and ownership metadata. One parameter retains exactly its
current meaning and encoding. I add neither an opcode nor a wire section.
Ownership formats 1 and 2 remain unchanged; older one-parameter consumers
continue refusing the newly admitted helper arities.

I keep exactly entry 0 and one nonrecursive helper 1, with exactly two bounded
reference activations. Entry has no parameters. The helper takes between one
and eight borrowed record parameters, each with an exact shared/exclusive mode
and scalar-leaf nominal referent. All remaining helper locals and the single
result are int/bool/u8. I continue refusing mixed value/reference parameters,
owned helper locals, deeper calls, imports, callbacks and aggregate results.
Eight parameters bound the pairwise conflict work to 28 pairs; my existing
256 reference slots and 32-field path bounds do not grow.

For helper arity N, CALL_REF callee,base reads caller slots base through
base+N-1 and initializes helper reference slots 0 through N-1. I check the
whole range in widened arithmetic before indexing. No reference enters a
NanoValue, value stack, local value or heap field. The corresponding helper
value locals stay void and non-authoritative.

## My actual caller-place substitution

I prepare each argument descriptor through the existing ordered borrow and
reborrow instructions. Their earlier holds remain active while later argument
preparation executes. CALL_REF itself evaluates no source expressions and
cannot reorder them. Future source production must preserve this preparation
order; this bytecode slice does not establish source-level acceptance.

I validate the entire argument batch before changing the helper state:

1. I require every caller descriptor to be live and to resolve through its
   authoritative owner layout. I retain its invocation, owner local and numeric
   path; I do not substitute a helper-local number for the caller root.
2. I require each declared parameter's exact nominal referent and requested
   mode. Shared authority cannot become exclusive.
3. I compare each requested call reborrow against every existing caller hold,
   excluding only that argument's own parent/ancestor chain. An active child
   that conflicts with the requested mode prevents the call. I never ignore a
   hold merely because another argument also names it.
4. I compare every pair of requested parameter places. Equal and prefix paths
   overlap based on their field vectors, independent of table indices. Shared
   aliases are compatible; any overlapping pair containing an exclusive mode
   is refused. Disjoint roots and sibling paths may both be exclusive.

For example, an exclusive parent and an existing shared child may both be
passed as shared parameters if all other holds permit it. Passing either as
exclusive conflicts with the shared alias. On return I retain the caller's
original child hold; I do not incorrectly restore a parent write that was
already suspended before the call.

I copy all required paths before publishing any parameter binding. Allocation
failure leaves the caller and unbound helper unchanged. Callee authority stores
immutable checked origin facts for every parameter, so subsequent reborrows
validate against the actual originating root. Exact joins include all parameter
origins, modes, child relationships and region obligations.

## My execution and escape boundary

NanoVM copies checked descriptors into the helper's existing bounded context.
Every descriptor retains its actual caller frame and generation, owner local
and immutable path. Field access resolves current caller storage after stack
resizing or core resumption. Helper-local value access remains refused for all
parameter indices, not just local 0.

My private native helper receives the same descriptor batch and original live
caller storage. Its references access those fields directly, with no record
copy or writeback surrogate. Both implementations retain earlier caller holds,
expire the helper's descriptors on return and clear both contexts on terminal
failure. Core yield preserves both contexts. Public helper entry and nested
host invocation remain refused, without erasing suspended descriptors.

## My required evidence

I require paired VM/native shared aliases, disjoint exclusive roots and nested
sibling paths, mixed modes on disjoint places, all eight parameters, repeated
calls, independent helper reborrows and exact caller restoration. I include
separate table indices denoting the same path and checked preparation order.

I retain refusal gates for overlapping exclusive/shared arguments, escalation,
wrong nominal identity, inactive slots, an invalid contiguous range, more than
eight parameters, conflicting existing children, value/OWN parameter access,
incompatible joins and escaping helper regions. I inject allocation failures
through batch binding, parameter facts and owner construction; verify unchanged
state or complete terminal cleanup; yield inside the helper; relocate the value
stack; and resume reads/mutation through multiple original caller places.

I preserve all one-parameter and standalone cases, both ownership formats,
assembly/codec/reconstruction, both dispatch modes, ASan/UBSan/native leak checks
and genuine canonical compiler-host linkage. LLVM/Wasm reference lowering,
broader call shapes and source frontend production are not admitted by this
paired VM/native slice. Full affine and v5.1 publication acceptance stay open.
