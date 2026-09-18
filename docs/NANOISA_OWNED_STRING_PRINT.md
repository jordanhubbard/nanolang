# My owned immutable string and print contract

I track `task_badd6be9c31a6e2eac810b95913b4f84` as the next runtime
prerequisite under `task_c4351c720aee424ea9b90187e51a08f2`, after exact
owned and void value-call results merged in PR #738.

I extend only my verified standalone acyclic owned value-call profile. I keep
its eight-function and eight-frame bounds, exact direct calls, complete
ownership metadata, mode-zero parameters, scalar public entry result and
existing resource ownership rules. I admit immutable strings only where the
unchanged affine example needs them: helper parameters, their corresponding
local slots, module literals, and `PRINT` or `PRINTLN` operands. I do not admit
string results, string-bearing resource fields or general string operations.

This is a runtime contract. Source producers and the unchanged affine example
remain guarded until a separate change proves that both producers emit the
same accepted metadata and instructions.

## My existing VM boundary

My VM instantiates each module string constant as a heap `VmString` retained by
that module's constants table. `PUSH_STR` validates the index, retains that
object for the operand stack and pushes a normal `TAG_STRING` value. A call
moves its prepared argument references into the callee frame. Ordinary local,
stack and terminal cleanup release the corresponding VM roots.

`PRINT` and `PRINTLN` pop one value into `TRAP_PRINT`. The public execution
loop prints the trap value to the selected VM output stream, writes one newline
for `PRINTLN`, and releases the trap value exactly once before resuming. The
trap owns that popped value while execution is suspended. A caller of the
lower-level core interface inherits that same ownership obligation; I do not
weaken it for this profile.

I preserve those rules. The owned verifier admits a string only when its
function descriptor, ownership descriptor and actual operand agree on
`TAG_STRING` with mode zero and no layout. Every admitted function and call
edge is checked before execution. Runtime tag checks remain defensive; a tag
does not invent lifetime or layout authority.

The existing `borrow_tag` and `nb_tag` summaries also participate in record
field eligibility. I therefore keep string parameter acceptance separate from
the INT and BOOL resource-leaf guards. Admitting `TAG_STRING` for a parameter
does not make a string an eligible resource field, and the shared summaries
must not widen that field rule as a side effect.

## My bounded admission

I add these cases to the owned value-call graph and no others:

1. A helper parameter and its parameter-local slot may be mode-zero
   `TAG_STRING`. Copying an immutable string through `LOAD_LOCAL` retains the
   VM value; call activation moves the prepared argument root exactly once.
2. `PUSH_STR` may load a validated module literal. Its index must be in range
   and its instantiated constant must exist before execution begins.
3. `PRINT` and `PRINTLN` may consume one `TAG_STRING`. They produce no operand
   and do not change a function's declared result.
4. `CALL` may transport these immutable string parameters through the existing
   bounded acyclic graph. Its arity, parameter order and tags remain exact.

My shared affine and runtime-verifier opcode allowlists take the value-graph
classification explicitly. They refuse `PUSH_STR`, `PRINT` and `PRINTLN` when
that classification is false, before instruction-state analysis. The later
affine step and native emitter checks remain as defense in depth. A borrowed
`CALL_REF` helper therefore cannot inherit string output merely because it
uses the same affine machinery.

I retain the existing INT, BOOL, U8 and resource cases. This string slice does
not make a string an owned resource, scalar integer or layout-bearing record.
It does not admit string `RET`, string fields in `OWN_PACK`, concatenation,
comparison, conversion, arrays, maps, globals, imports, callbacks, indirect
calls, linked modules or tail calls.

My current VM string printer uses its NUL-terminated view. NanoISA constants
also carry an explicit byte length, but embedded NUL output does not yet have a
single general-language contract. I therefore reject an embedded NUL in every
string literal admitted by this owned profile before publication. Empty
strings remain valid. This is an explicit bounded refusal, not a claim that
NanoLang strings cannot contain NUL elsewhere.

## My native representation

My specialized native owned emitter currently represents each value with a
scalar field or an owned-record pointer. I extend that private carrier with an
immutable byte pointer and length. The function descriptor still determines
the active representation; I do not introduce an unverified dynamic tag.

Each admitted module literal becomes generated static bytes plus its exact
length. An empty literal has a valid zero-length view. Passing or storing a
string copies that view. The view owns no allocation, is never freed, and does
not outlive its generated module. Resource-record fields retain their current
scalar-leaf contract, so native record cleanup never interprets a string view
as a record owner.

Native `PRINT` writes the exact admitted byte length. `PRINTLN` performs the
same write and then writes exactly one newline. The existing output boundary
does not promote a host stream error into a language result; this contract
does not claim durable I/O. Rejecting embedded NUL keeps these exact native
bytes consistent with the current VM printer rather than silently choosing a
new language-wide string-output rule.

Each host `PRINT` or `PRINTLN` trap invalidates the current invocation proof
before execution resumes. The public execution loop may re-establish that
proof only through the existing checked-resume path. A lower-level core caller
still owns and must release the trapped VM value on every handled or abandoned
trap path; invalidating the proof does not transfer or erase that obligation.

Every predicate that can positively admit owned execution first requires the
active instantiated constant table. This includes the conservative
`vm_ownership_supported` predicate used by trace, callback and reference
fallbacks, not only the fast invocation-proof path. Public core entry and
proof-invalidated resume retain their explicit readiness checks as well.

At a call, the caller prepares every argument before activation. A string view
is copied into the callee's corresponding parameter carrier and the prepared
caller slot is cleared with the rest of the transferred arguments. On normal
return, assertion failure, validation failure or injected allocation failure,
cleanup releases actual record owners and clears every carrier. Clearing a
string view performs no free. It must not leave a stale pointer in an active
stack or local slot.

## My output and lifetime checks

I qualify corrected standalone modules, not a replay of a frozen product
artifact. Positive cases cover:

- a nonempty module literal printed with and without a newline;
- an empty literal and an empty string parameter;
- a string parameter forwarded through three helper frames;
- repeated and sibling calls using the same module constant;
- exact byte order and newline placement in a captured real stream;
- printing before a later successful return;
- printing before a later assertion failure, preserving prior output while
  releasing every stack, local, trap and frame root;
- repeated execution through `vm_invoke`, `vm_execute`, `vm_call_function`
  and `vm_invoke_callable` with unchanged module-constant ownership; and
- generated native execution under strict GCC and Clang with ASan and UBSan.
  Linux keeps leak detection where supported. The required Darwin gate uses
  available Homebrew LLVM with LeakSanitizer enabled. Apple Clang may provide
  optional additional ASan/UBSan evidence with explicit live-root counters;
  its unsupported LeakSanitizer runtime does not replace the required gate.

I instrument `VmHeap` allocation through `heap.c`. Setup reaches the intern
bucket and module-string objects; invocation reaches the owned-record header and
field storage. I report those attempt counts separately. I do not claim
call-frame, constant-table-array or general process allocation coverage. Every
injected invocation failure must leave the same VM reusable; every injected
setup failure must leave the module reusable in a fresh VM.

Refusal cases cover a bad string index, an absent instantiated literal,
embedded NUL, wrong parameter tag or mode, string result, string resource
field, unsupported string opcode, missing or extra print operand, recursive
graph, unsupported call kind and a graph beyond the retained bounds. I verify
refused modules without executing them and preserve prior output buffers.

The existing owned graph, owned/void result, consuming-call, borrowed-call and
authority gates remain required. Binary and text roundtrips must preserve the
existing string constants, function tags and ownership descriptors without a
wire-version change. If those existing formats cannot express an admitted
case exactly, I record a transport prerequisite before widening execution.

## My completion boundary

This task completes only when the bounded VM and native gates pass on reviewed
source and the existing ownership suites remain green. It does not complete
source-producer admission, the affine example, product PR #522, general string
semantics, full ownership, backend equivalence or release acceptance. Those
claims require their own evidence. A contract is useful. It is not executable.
