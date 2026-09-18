# I preserve ordinary managed values alongside owners

I record a static proposal under `task_4be28fef163f42069064357639b3b5cc`,
reviewed against canonical source `f761af4c`. I have not implemented or executed
this proposal. My full ownership equivalence parent
`task_28f2fb4b1f3c8a5ce93df628bb569d76`, portable managed lifetime parent
`task_51da49b39230468784da3481b893563b`, aggregate identity parent
`task_488a05eb5e2a417caf83a8353363a30d`, and installed-product affine parent
`task_e8d860a16da0464891dd32e91c42bef1` remain open.

## My unchanged acceptance source

I retain `tests/test_owned_record_patterns.py::test_ordinary_inferred_field_and_alias`
with its complete `PREFIX`: resource Handle, consuming close, and shadow close.
Samples has an ordinary `array<float>` field; main constructs it, projects the
array, copies its alias, and asserts element 1 equals 2.5. Shadow main calls main.
My selected-shadow entry must execute both original shadow bodies in order in
one admitted module. I do not split compilation, omit a shadow, drop declarations
before source checking, or change shared initialization to obtain acceptance.

I distinguish these required families:

| Source | Obligation |
| --- | --- |
| Samples plus PREFIX | Ordinary record/array identity and lifetime coexist with explicit owner transfer in the selected module. |
| Bundle containing Handle and array<float> | An affine owner contains a managed array field; destructive unpack transfers its managed root. |
| Bundle containing Handle and string | An affine owner contains a managed string field; unpack and comparison preserve its lifetime and bytes. |
| Connection containing FileHandle | A transitive ordinary resource wrapper, independently owned by the native-effects source lane. |

My first family does not prove the second or third. None closes general mutable
collections, cycles, callbacks, imports, or full LLVM/Wasm support.

## What my current code establishes

* `ownership_contracts.c::check_layout_facts` accepts complete scalar/string/record
  ordinary layouts, but rejects array fields. Resource layouts still require
  scalar trees. `descriptor` permits an ARRAY tag with no layout index, but that
  fact alone proves neither its element type nor absence of an owned child.
* `affine_state.c::supported` refuses ARRAY locals. Its parameter/result queries
  admit only the existing scalar/string/owned subsets. `affine_bytecode.c` has no
  array instructions or ordinary AGG_PACK; LOAD_LOCAL treats STRUCT as an owner
  observation, AGG_GET requires that observation, and CALL associates STRUCT
  with owned transfer. These are explicit admission boundaries, not permissions
  to infer ownership from the runtime tag in an expanded profile.
* The affine analyzer already handles PUSH_F64, float scalar locals, F64
  arithmetic/comparisons and matching scalar comparisons. The direct owned native
  emitter in `nvm2c_owned.h` has no PUSH_F64/F64 cases and refuses them through its
  default branch. Its carrier has integer scalar, owner record and static string
  view fields, with no ordinary array representation. Generic native comparisons
  currently read its integer scalar member.
* `vm.c` already retains/releases ordinary heap values in its common stack,
  local and return lifecycle. Owned CALL still performs positional checks before
  frame reservation/activation; owned STRUCT return validates the declared owner
  layout before detaching the pending root. Broader admission requires extending
  those checks deliberately, not bypassing them through ordinary dispatch.
* Both specialized source producers currently emit resource layouts and
  int/bool leaves. The selfhost `nb_tag`, `nb_parameter_tag`, layout loop and
  body lowering are distinct guards. Its synthetic entry retains all selected
  shadows and the eight-function complete acyclic graph bound.

## My dependency order

1. I first complete direct native lowering for the binary64 scalar operations
   already admitted by the owned verifier. I preserve bits in a distinct scalar
   carrier and select generic comparison semantics from exact proven operand
   tags, never by reading float bits as integers. I reuse the existing shared
   scalar arithmetic policy. I retain current parameter/result/resource-field
   admission and all ownership rules. This independently reviewable prerequisite
   adds no arrays, records, managed roots, source admission, or wire format.
   Its precise production contract and MAC child precede code.
2. I establish an explicit distinction among owned values, owner observations,
   ordinary managed records, ordinary arrays, and scalar values throughout
   descriptors, abstract stack/local state and native carriers. COMPLETE without
   RESOURCE must establish recursively owner-free record fields. ARRAY alone
   is insufficient: exact element provenance must survive construction, local
   copies, projections and joins. Existing managed-array shape analysis is a
   candidate proof source, not automatically trusted admission. I preserve
   unknown/incomplete refusal. I must settle a versioned descriptor/shape transport
   contract before changing any reserved bits or layout-index meaning; existing
   bytes must keep their current interpretation and old readers must refuse any
   required new feature. No format change is authorized by this proposal.
3. I add a bounded runtime/native ordinary flat float-array and ordinary-record
   intersection with the existing owned value graph. Its first opcode set covers
   the unchanged fixture's actual emitted construction, exact field projection,
   alias copies, indexed read, float comparison and cleanup; any mutation admitted
   in that set must preserve VM alias identity. I keep owner fields scalar-tree
   only, no ordinary-to-owned container laundering, and no new CALL_REF mixture.
4. After the runtime prerequisite qualifies, I add paired C-seed/selfhost source
   lowering for those exact descriptors and instructions, including inferred
   field/alias types, source identities, lexical names and every original shadow.
   Program and selected-shadow modules must verify, roundtrip canonically, and
   execute with matching VM/native results. I preserve full-source checking and
   output atomicity on every refusal.
5. I separately contract managed fields inside owners, including whole-owner
   moves, destructive unpack, partial-construction cleanup and managed child
   aliases. Bundle admission cannot come from clearing RESOURCE or treating an
   owner as an ordinary record. This remains a required dependent obligation.

## My lifetime and call requirements for the managed intersection

I preserve one root for each live stack/local/call-result reference. LOAD/DUP and
field projection retain ordinary managed identity; STORE replaces and releases
the old root only after the new value is safe; POP and scope/frame cleanup release
exactly once. OWN_MOVE/OWN_STORE/OWN_UNPACK remain consuming operations and cannot
copy an owner. Ordinary record destruction releases its managed children; it
never silently consumes a source owner. I preserve source-explicit owner exit
obligations and exact owner/reference/region joins. Definite initialization of
ordinary locals may meet only after their exact declared type/shape agrees.

I prepare arguments in source order with earlier owners and managed references
still rooted. I validate every positional type/nominal/shape before frame
activation or transfer; reservation failure cleans all prepared roots. The first
Samples slice needs only existing int/owned signatures, so ordinary managed
parameters/results remain refused until separately qualified. No common call
shortcut may confuse an ordinary STRUCT with an owned result. Later ordinary
managed returns must remain rooted until caller capacity and exact contract are
validated, then publish once after callee cleanup. Generation and caller-held
reference rules remain unchanged.

I require terminal assertion/allocation/index failures, repeated invocation and
all four public VM APIs to clean both ordinary managed roots and owners. Native
cleanup must cover stack temporaries, locals, prepared arguments, partially built
records/arrays and pending returns. I distinguish heap object injection from
frame/analysis allocation injection and report exact measured points. I do not
claim arbitrary cycle support from reference counting; the first shape is a
finite ordinary record containing a flat scalar array and cannot form a heap
cycle under its admitted operations.

## My acceptance before any full-family claim

I retain the original Samples source/PREFIX and original assertions across the
C seed and both freshly bootstrapped selfhost stages, canonical bytecode, VM and
native execution. I add ordinary alias mutation/overwrite and allocation-pressure
controls, owner preparation around managed allocations, failure cleanup, zero and
entered control-flow paths, and false mandatory shadows preserving prior output.
I preserve wrong shape/nominal, owned child in ordinary container, owner duplicate,
unknown metadata and unsupported graph refusals without executing refused output.
Canonical metadata equality includes names, intervals and full shadow closure.

I retain the earlier 950f product failure unchanged. Passing a new prerequisite
is not passing Samples, the mixed Bundle cases, the original full test suite, or
release acceptance. I run no historical failing compiler artifact for this audit.
