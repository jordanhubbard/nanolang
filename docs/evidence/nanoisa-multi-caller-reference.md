# My bounded multi-parameter caller evidence

I implement MAC `task_7a2c8017c0c04b82a48ba069561e9d36` under my
[batch contract](../NANOISA_MULTI_CALLER_REFERENCE.md). My compiler checkpoint
is `2c6b4439`; `ea9ef327` adds legacy-format and exact-capacity tests without
changing compiler source. My integrated compiler checkpoint is `f2f37940`,
including main `17186c46` and its underscore/artifact-string repairs.

I retain CALL_REF's existing bytes and interpret its reference operand as the
first contiguous descriptor slot. The declared helper arity supplies the count,
from one through eight. I validate the range without narrow-integer overflow.
Each helper parameter keeps the actual caller root, nominal layout and path;
no reference is carried in a NanoValue or ordinary parameter local.

I compare every requested authority with existing caller holds, excluding only
that argument's own ancestor chain, then compare the requested pairs. I allow
shared aliases and disjoint exclusive paths. I reject conflicting aliases and
active children even when another argument selects a related descriptor. I
allocate all path copies before publishing bindings, preserving both states
when any allocation fails. Exact joins retain ordered origins and paths.

I retain exactly two VM contexts and the existing native private helper model.
The helper accesses original caller storage and returns one scalar. Ending the
call expires only helper descriptors; pre-existing caller holds remain intact.
A core yield preserves both contexts across stack relocation. These permissions
come from verification; runtime descriptor fields alone do not authorize
unverified code.

## My measured gates

On the named compiler/test checkpoints I pass:

- 2,004 checks across thirteen paired VM/native programs and thirteen refusal
  cases. I cover shared aliases, equal paths with different table indices,
  disjoint sibling paths, reordered arguments, distinct roots with identical
  layouts, different nominal roots, mixed modes, repeated calls and reborrows;
- one-parameter compatibility, multi-parameter ownership format 1, format 2
  paths, all eight parameters and an eight-slot batch ending exactly at the
  caller's descriptor capacity;
- 93 atomic binding/parameter-allocation checks, including failure of each
  path copy after an earlier copy succeeds, unchanged caller/helper snapshots,
  reordered-origin join inequality and recovery after parameter-fact failure;
- 89 owner-allocation checks, with no surviving descriptors or owners;
- actual helper suspension, value-stack relocation and resumed mutation through
  two distinct paths of the original caller record;
- 1,548 prior caller checks with 43/55 allocation checks, 1,336 root-reference
  checks with 29 allocation checks, 1,662 nested-reference checks with 63
  allocation checks, and 1,051 owned-transfer checks with 32 allocation checks;
- 214/243 affine-state checks, 300/419 affine-bytecode checks, 144 ownership
  metadata checks, 2,856 ISA checks and 33 schema checks;
- 274,416 VM checks under both dispatch modes, plus computed-goto execution of
  the complete multi-parameter fixture;
- ASan/UBSan execution, atomic binding and allocation recovery, plus emitted
  native program leak checks and exact serialized/native reconstruction;
- 2,414 native translator checks, 1,092 shape checks, 16 LLVM methods and
  39 WebAssembly/scalar methods;
- a genuine canonical native compiler seed build, help, hello bytecode
  publication, verification and execution through its host module.

At `f2f37940` I repeat the full multi-parameter and prior caller gates,
ownership metadata controls and 33 schema checks. All eight artifact-string
cleanup methods also pass. I repeat the genuine canonical compiler-host seed,
help and verified hello publication/execution after those integrated source
changes. The broader translator/dispatch counts above remain
proof for their earlier named compiler checkpoint, not a claim that the full
release suite passed at this head.

My retained local logs use `/tmp/nanolang-multi-caller-` with these suffixes:
`bounds.log`, `focused.log`, `adjacent.log`, `computed-final.log`,
`asan-final.log`, `state-asan-final.log`, `translators.log`, `restack.log`,
`seed-build.log`, `seed.log`, `help.log`, `hello.log`,
`seed-integrated-build.log`, `seed-integrated.log`, `help-integrated.log` and
`hello-integrated.log`. My checked-in tests are
the durable repeatable evidence; these logs are not published release assets.

I do not admit mixed value/reference parameters, owned helper locals, deeper
calls, recursion, imports, callbacks, aggregate results or source borrow
production. LLVM/Wasm reference lowering and mandatory owned-shadow assertions
remain separate implementation boundaries. My affine/borrow parents and full
v5.1 publication hold remain open.
