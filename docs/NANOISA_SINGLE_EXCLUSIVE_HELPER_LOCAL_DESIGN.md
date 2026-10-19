# I map single-exclusive helper-local ownership before admission

I record a design-only follow-on to #999 from `a1cef240f`. I have not changed
production code, emitted a new baseline, or measured reconstructed local counts.
My existing required scope is row9538 in `RELEASE_5.1_SCOPE.md` (preserve verified
ownership through reconstruction) and row10409 (recover structured if/while/return).
The helper-owner row of `NANOISA_RECONSTRUCTION_REQUIRED_SCOPE_MAP.md` and the
existing positive producer test identify this concrete composition. I do not
infer an unlimited new profile from those rows.

## I preserve the exact accepted-source construction

`tests/test_source_borrow_emission.py:test_helper_owned_locals_preserve_formals_and_shadows`
constructs its `former_refusal` case from
`tests/nanoisa/fixtures/source_borrow_resource_paths.nano` by replacing exactly:

```nano
set view.value (+ view.value 1)
```

with:

```nano
let owner: Leaf = Leaf { value: 1, active: true }
let Leaf { value, active } = owner
set view.value (+ view.value 1)
```

I keep the entire remaining fixture unchanged. Its helper is
`bump(view: &mut Leaf) -> int`; Leaf has INT value and BOOL active fields; Pair
owns two Leafs. The caller uses a moved.left projection and a direct moved Leaf
in a rank loop, branch-local consumption, and the final total7 assertion.
The existing source test requires three ordinary producer dumps to agree, raw
emitter products to execute through VM/native routes, and independently emitted
original shadows to agree and execute. Those source expectations are evidence
of accepted producer scope, not a new reconstruction success receipt.

Destructuring may synthesize local observations even without an explicit source
field read. I determine actual opcodes, descriptor use, instruction identities
and counts from the future retained baseline dump; I claim none from syntax alone.

## I keep the smaller authority boundary explicit

I propose only the existing ownership-v2, two-function shape: parameterless INT
entry plus one exclusive Leaf formal, modes `[2]`, and an INT helper result.
I retain depth2, nominal layout identity, existing instruction/function bounds
and the 128-local cap. The exact baseline must establish these properties.
Inherited descriptor0 remains exclusive and cannot be moved, overwritten or
ended by local ownership operations. If the actual producer emits a local
observation, it must use a complete ephemeral descriptor1 region after the
inherited formal, with matching begin/borrow/read/end and exact live owner.
That conditional obligation does not itself authorize new observation admission.

I separately prove helper-local construction, move and complete consumption,
excluding the borrowed formal from the owned-live set. Every successful return
must consume each local generation; traps are not evidence of successful cleanup.
I retain caller projected/direct path handling and its existing branch/loop
origin restoration. No escaping references, owned formal/result, deeper helper
calls, local-owner writes or arbitrary signatures are added.

## I distinguish this from the two-formal component

The pending `[2,1]` component does not enable this case. The single helper uses
`borrow_read(target, field)` and `borrow_write(field, value)`, not the multi-formal
IR tuples. Its origin and work checks need explicit scalar-event handling while
keeping formal0 out of owned transfers. Pack/unpack work must be counted before
accepting the existing total-helper witness.

The caller's first invocation can be inside a branch and can use a projected or
direct root. I cannot reuse the two-formal straight-line literal-prefix witness
by changing arity. The existing single-exclusive zero-tree consistency witness
is valid only after proving total, nontrapping helper behavior without unproved
assertion requirements. The inserted local resource is consumed; actual lowered
observations and any assertions must be checked before relying on that argument.
Single-path shadow generation remains distinct from two-root multi-shadow output.
Original-shadow products remain separate inputs; I do not claim recovery of
original mandatory shadows.

If finite rendering is needed, I review reuse only after immutable identity,
consumption and work proofs. I do not add an ADD scratch bank, drop assertions,
change caller order, introduce a dispatcher or raise limits to fit the fixture.

## I require a baseline before implementation

After the current helper component integrates and the serial slot is released,
I retain the exact substituted source, all three ordinary producers, one honestly
pinned original-shadow producer, verified modules, VM outcomes, dumps, local
counts, and actual reconstruction refusal with prior output preservation.
I inspect descriptor1 lowering and any helper-generated paths in those artifacts.
Only then do I propose the smallest production interface and first unchanged
anchor/count gate. Later checks must preserve formal identity after local
consumption, repeated caller mutation, exact failure/publication behavior and
old profile bytes. Fresh aggregate/platform evidence remains separate from
component qualification. This design closes no required scope row or issue.
