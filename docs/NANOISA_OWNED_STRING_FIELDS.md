# My retained STRING fields in unique owned records

I track `task_793523b9e4c04e389f9d63652758991a` under my managed lifetime,
affine equivalence and aggregate parents `task_51da49b39230468784da3481b893563b`,
`task_28f2fb4b1f3c8a5ce93df628bb569d76` and
`task_488a05eb5e2a417caf83a8353363a30d`. This is a preimplementation contract.
I have not qualified or admitted these fields. My broader parents stay open.

## My unchanged acceptance and dependencies

I preserve `PREFIX` and the complete
`test_nested_transfer_and_unsafe_scope` in `tests/test_owned_record_patterns.py`:
`Bundle` contains `file: Handle` and `label: string`; its caller constructs an
inline `Handle`, its callee destructures both fields inside `unsafe`, compares
`label` with `"ready"`, and consumes `file`. I preserve ordinary main and every
selected shadow, including the original `close` shadow. I do not replace the
label with an integer or remove the nested constructor, unsafe block or equality.

I depend on the separately reviewed ordered inline construction child
`task_f4a34feabe954cc2921151c2819edc66` and source unsafe/float child
`task_a490c997619140c78c6f150f40445062`. I preserve source-order staging followed
by declaration-order packing from the inline contract. I integrate their
actual canonical commits before claiming the unchanged original acceptance.
PR803 already supplies exact nested owned results; its query/runtime/source
qualification remains independently sealed. My starting canonical commit is
`d6cacdb9`.

The original `Bundle { file: Handle, samples: array<float> }` case is a
separate required dependency. I do not substitute string ownership for array
identity, mutable aliases or mixed provenance.

## My static admission audit

At my starting commit:

- `ownership_contracts.c` requires scalar-only resource layout trees. STRING
  fails that predicate even though the ordinary VM has managed strings.
- `affine_state.c` permits STRING only as a direct value-graph parameter/local
  fact. Its scalar definition and projected-field operations exclude STRING;
  its bounded owned-result tree also excludes STRING leaves.
- `affine_bytecode.c` permits literals, parameter loads and printing, but
  scalar local stores, stack copies/discards and comparisons exclude STRING.
- `verifier.c` and both source layout collectors independently retain narrower
  field guards. Broadening one query cannot establish executable admission.
- `nvm2c_owned.h` currently carries immutable static byte views. Retain/release
  manages records only; DUP, POP and PRINT rely on their old non-managed
  operands. The ordinary VM already retains string stack/local roots and
  recursively releases string fields in records.
- `value.c` equality checks identical pointers, then null pointers, then
  `vmstring_equal`; that helper compares lengths, hashes and bytes. Native
  equality must preserve its observable content result, not pointer identity.

I change these boundaries coherently and separately from numeric scalar
predicates. I audit every consumer of shared layout authority before widening
it: an accepted descriptor is not permission for a borrowed, linked, LLVM/Wasm
or other unqualified execution profile. I retain those profiles' explicit
refusals unless separately qualified. I need no new tag, opcode or wire version.

## My shared-authority consumers and retained refusals

I enumerated actual calls and downstream selectors at d6cacdb9:

| Consumer | My intended boundary after descriptor widening |
| --- | --- |
| `nvm_v2_convert.c`, both conversion directions | Validate and transport exact complete STRING-bearing resource facts; no execution permission follows. |
| `retained_layouts.c` through `nvm_ownership_layout_authorities` | Preserve RESOURCE identity and exact leaf tags; never classify the new owner as an ordinary copyable record. |
| `affine_state.c` state construction, result closure and field queries | Admit STRING only in the reviewed value graph operations; preserve exact owner authority and root liveness. |
| `ownership_contracts.c` borrowed parameter descriptors | Reject borrowed roots whose reachable layout contains STRING, even if the requested path ends at an INT leaf; retain existing scalar-only borrowed root behavior. |
| `reference_places.c` and path users in affine bytecode, VM and native | Preserve STRING leaf/projection/write refusal; valid path transport does not grant reference authority. |
| `verifier.c` structural and owned-module selection | Run complete owned admission after authority validation; STRING-field execution requires the value-call graph profile. Preserve linked ownership refusal and non-value-graph refusal. |
| `vm.c` invocation classification and result activation | Keep needs-ownership true and require full module proof, instantiated constants, exact result layout and existing cleanup. No ordinary VM fallback for an unsupported owned module. |
| `nvm2c.c` ordinary eligibility and `nvm2c_owned.h` | Keep ordinary native path refusal when ownership is required; only independently validated owned graph uses the new carrier. |
| `managed_array_shapes.c` managed shape selection | Preserve rejection when authority validation reports requires-verifier/ownership. No graph or array promotion of resource shells. |
| `verifier.c` closed/managed target profiles and `nvm2llvm.c` managed selector | Scalar profiles retain nominal/ownership refusal; managed record/array selection still rejects resource-bearing graphs. LLVM and Wasm gain no owned STRING admission. |
| `nanovirt/ordinary_authority.inc` | Preserve RESOURCE facts and ordinary-versus-owned selection; no false ordinary declaration from newly valid STRING fields. |
| C `borrow_codegen.inc` and Nano `nanoisa_borrows.nano` | Keep source field guards unchanged during runtime qualification, then admit only the reviewed exact source profile. |
| Nano `owner_selection.nano` and program dispatch | Keep resource-bearing wrappers on specialized routing; no scalar owner-free closure classification of these layouts. STRING support is not a reason to bypass owner selection. |

I keep the borrowed-root exclusion in shared descriptor validation so it
also protects disconnected helper descriptors and paths to otherwise scalar
siblings. I compute reachable STRING information with bounded prior-index
layout traversal, not recursive repeated expansion. I verify each downstream
profile's actual guard at implementation review; if a listed refusal is not
already enforced, I add the narrow guard before widening authority. I do not
remove ordinary complete nonresource STRING record support.

## My exact supported values

I retain the existing bounded acyclic owned value-call graph, complete exact
nominal layouts, prior-index child edges, mode-zero value parameters, selected
shadows, scalar public entry and existing frame/stack/layout bounds. I preserve
currently admitted INT/BOOL/U8 owner leaves; I add only STRING with no nested
layout. PR804 adds nonparameter FLOAT locals and operand operations, not FLOAT
owner fields or signatures. I preserve those FLOAT field/signature refusals
and keep its independently qualified local/operand behavior unchanged. A string field does not make
an otherwise ordinary record into a unique resource. A resource root and every
owned child keep their exact affine authority.

My new nonowner STRING values may come from existing module literals,
parameters, locals or admitted STRING fields. I support exact STRING locals
and assignment, LOAD/STORE, DUP/POP/SWAP where these stack operations are
already otherwise eligible, direct value-call parameters, OWN_PACK/OWN_UNPACK,
and checked field observation without transferring the unique shell. I admit
exact STRING/STRING EQ and NE to BOOL, and retain existing printing. I do not
make STRING arithmetic, ordered comparison, truthiness or numeric casts legal.

I permit these leaves inside an otherwise eligible owned return tree, keeping
PR803's bounded prior-index closure and exact result layout. I do not admit a
standalone STRING function result. STRING remains non-affine: a local copy can
survive a shell move or destruction. A field observation retains its string
before releasing its temporary shell observation. It never creates a second
consumable owner. Borrowed STRING projections and STRING reference writes stay
refused; I do not widen reference-place scalar predicates.

I retain the owned profile's embedded-NUL literal refusal and permit empty
strings. No general string factory, concat, substring, conversion, array/map,
owned union/generic, extern string, callback, indirect call, recursion, tail
call, global or linked-module admission follows from this contract. Existing
admitted cases stay admitted; exclusions here describe new STRING paths, not
removal of previously qualified independent behavior.

## My native storage and transfer proposal

I replace the private static-only STRING carrier with a retainable immutable
string cell: reference count, exact byte length and owned bytes with a trailing
NUL. The active tag remains STRING and carries no nominal layout. I retain the
existing record carrier separately. Generated literal bytes remain static
inputs; PUSH_STR creates a checked cell and publishes it to a stack slot only
after successful initialization. I apply this consistently to the owned
value-graph emitter, including its existing literal/print subset, and qualify
that subset again. This introduces a possible native allocation failure where
the old static view had none; I report it explicitly and use the existing
status/cleanup channel, never a fabricated value.

I use the emitter's existing NOWN_ALLOC/NOWN_FREE hooks, with overflow checks
before header+length+terminator size arithmetic. Allocation failure leaves all
prior roots intact. A fresh cell starts with one root. Copying retains before
publishing the new root; a count-overflow refusal creates no new root and
leaves the old roots intact. Moves clear their source exactly once. Release
frees a cell only after its last root and never frees module literal storage.
I require checked retain integration at each new STRING copy; I do not infer
safe STRING behavior from the existing record-only retain helper.

This private native representation implements the same retain/copy/transfer/
release contract as my managed-string work, but does not import its module
handle table, cycle collector, external allocator ABI or LLVM/Wasm admission.
My immutable strings have no outgoing edges; exact owned record children keep
the existing acyclic layout rule. No process-lifetime arena or leaked string
pool substitutes for release. A future shared carrier integration remains a
separate representation change, not a reason to weaken this lifetime contract.

| Boundary | My required root behavior |
| --- | --- |
| Literal | Allocate/init first; publish one root only on success. |
| LOAD/DUP/field observation | Retain exact STRING before publishing the copy. |
| STORE | Move prepared RHS, release old local, clear source; aliases survive. |
| POP/PRINT/EQ/NE | Consume and release every popped STRING on all exits. |
| SWAP | Move carriers without retaining or dropping roots. |
| Direct call | Prepare all operands in source order, then transfer each root once. |
| OWN_PACK | Allocate shell before detaching prepared operands; move fields by declaration order. |
| OWN_UNPACK | Move fields out, clear shell fields, then release the empty shell. |
| Owned return | Move the entire exact carrier through pending result; failure cleans pending root. |
| Terminal failure | Release every stack/local/prepared-argument/pending-result/frame root once. |

EQ/NE evaluates both operands once and preserves VM same-pointer/null/content
semantics, using length plus byte comparison for distinct nonnull cells.
Standalone helper null controls are defensive tests, not source null admission.
Printing uses the existing exact admitted bytes/newline behavior and releases
its consumed root. VM host print traps retain their existing release obligation.

## My staged checkpoints

1. I first implement exact layout/query/affine/verifier eligibility plus native
   storage and cleanup, retaining both source field guards. I independently
   review production before executing fresh ordinary API/runtime fixtures.
   Shared descriptor acceptance must not make unqualified profiles execute.
2. I qualify exact roots, aliases, pack/unpack, observations, parameter calls,
   nested owned returns, comparison/print and allocation failures. Query failure
   preserves output arguments; translator refusal preserves prior output.
   I retain the existing owned string, nested-result, authority and lifecycle
   controls. I freeze harness, source, compiler flags and tools before each gate.
3. After runtime evidence and another production review, I admit exact STRING
   field/local/equality paths in both C and Nano producers. I keep numeric,
   borrowed and unsupported managed guards distinct. I integrate canonical
   inline/unsafe prerequisites, rebuild all required tools, run fresh two-stage
   bootstrap, and qualify the unchanged original source and selected shadows.

## My evidence requirements

I use fresh valid ordinary modules and source programs, not old failure
artifacts. Positive runtime cases cover empty/nonempty strings, equal bytes
from distinct cells, unequal lengths/content, duplicate aliases, local
replacement, strings surviving consumed shells, nested pack/unpack and exact
owned factory/relay returns, repeated calls, both branch arms, early assertion
cleanup and printing before failure. I inspect exact STRING/no-layout metadata
and source-order side effects independently of output equality.

I inject deterministic required-capacity failures separately during VM setup,
VM invocation and native string/shell allocation. I report each phase and
attempt count; zero-capacity requests do not invent failed required storage.
Each refusal preserves unrelated aliases and prior content, frees only its
owned temporaries and permits a later valid invocation. I test failure after
string preparation but before parent allocation, after sibling preparation,
and across call/return cleanup. Leak/root counters and strict GCC/Clang
ASan/UBSan/available LeakSanitizer supplement observable values. I state exactly
which linked objects are instrumented. Darwin uses a separately recorded tool
and sanitizer configuration rather than inferring platform parity from Linux.

Negative controls check other field tags/layout mismatches, wrong local/call
STRING facts, dead owner reuse, unsupported STRING arithmetic/order/reference
paths, embedded NUL, standalone STRING results and retained graph bounds.
I verify rejected modules without executing them. Both source producers must
preserve old output on checked refusal and retain complete shadow selection.

My completion evidence contains canonical prerequisite ancestry, source/harness
and tool hashes before/after, commands, first terminal outcomes, fresh bootstrap,
C-seed/raw/Stage1/Stage2 VM/native results and ordinary legacy/interpreter
results for the unchanged original fixture. I distinguish source producer
coverage from runtime API coverage, and Linux from Darwin. Bounded success
closes only this child after merge; arrays, complete managed LLVM/Wasm owned
transport, the full ownership parents and release acceptance remain open.

## My first runtime production checkpoint

I keep both source producers unchanged. My shared authority validator now
accepts complete STRING resource leaves without the old scalar-tree rejection;
I remove its obsolete scalar-tree scratch allocation. Reference-place validation
instead checks the entire borrowed root, preserving an allocation-free scalar
leaf fast path and using one bounded prior-index work table for nested roots.
This also protects a borrowed scalar sibling below a STRING-bearing root.

I add distinct STRING define/field queries rather than changing numeric scalar
APIs. Initialization joins meet retainable STRING locals, and exact field
observation checks liveness and owner access. Affine opcode admission permits
only value-graph STRING copies/discards/stores/equality. The full verifier keeps
STRING fields value-graph-only and allows already-checked field observations
inside those helpers; borrowed helper restrictions remain unchanged. Owned
result closure gains exact STRING leaves, not standalone STRING results.

Native copies retain before publishing; literal allocation checks size before
addition; POP/PRINT/equality release consumed values. Both string and existing
record reference-count overflow now fail before a copy is published, through
the existing status path. Shared descriptor conversion and managed selector
refusals remain as enumerated above. I have run only a whitespace/diff check;
no build, fixture, bootstrap or runtime acceptance is claimed at this checkpoint.
