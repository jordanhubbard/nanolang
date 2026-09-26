# My capture initialization analysis

I record this source audit at f0a78cfc5 under
`task_ad64b875bbf14dbfbc3422c5d1f4f860`, inside shared-capture parent
`task_af8091f571a842bc90656e2c7f19b68e`. My
[binding contract](NANOISA_CAPTURE_BINDING_CONTRACT.md) defines the semantics.
This audit specifies the next analysis dependency; I have not implemented or
qualified it. My current execution consumers continue to refuse the feature.

## What my existing code establishes

`capture_bindings.c` checks payload modes, source and target slots, instruction
boundaries and exact closure sites. It does not follow initialized bindings
through control flow. `verifier.c` follows stack heights and explicit retain
balances. Its handler edge starts at the installed arm with zero operand height
and zero retain debt; that edge does not describe local initialization.

My execution model in `src/nanovm/vm.c` supplies these distinct transitions:

| Transition | Existing runtime behavior | Required binding analysis |
| --- | --- | --- |
| `HANDLER_PUSH` | Stores module, operation, target, owner frame and parameter range. | Retain handler identity and lexical ownership, not an installation-time copy of local state. |
| `PERFORM` | Finds the nearest matching operation by name, reserves a new activation, moves arguments into the handler range and clears its other physical locals. | Read the selected owner's current prefix state; initialize exactly the activation parameter range. |
| Prefix local access | `effect_local_index` follows `effect_owner` while the slot is below `effect_local_start`. | Resolve the same owner separately for each slot, including nested handlers with different starts. |
| `EFFECT_RESUME` | Destroys the handler activation and returns one value to the performing continuation. | Propagate surviving owner-binding changes to that continuation; discard activation-local facts. |
| Handler `RET` | Unwinds to the lexical root and returns that function's declared results. | Follow the nonlocal return, not the suspended performer's ordinary successor. |
| Ordinary call/return | Creates and destroys a distinct activation. | Preserve its own local identities while accounting for effects that reach a caller-owned handler. |

These are source observations. I have not executed capture-bearing versions of
these paths, and the table is not proof that new binding storage is wired into
them. The existing frame cleanup work keeps owned state teardown explicit;
initialization and execution admission remain separate obligations.

## Required state and transfers

I distinguish an unreachable program point from a reachable point with no
initialized locals. Ordinary entry initializes exactly the parameter range.
Handler entry owns a separate range and borrows prefix bindings through its
lexical owner chain. The analysis must retain that distinction rather than
flattening every slot into the handler's new physical array.

For each reachable context, `BIND_INIT_LOCAL` establishes initialization and a
fresh dynamic identity, even when it replaces a prior initialized binding.
`BIND_CLEAR_LOCAL` removes initialization and is idempotent. `LOAD_LOCAL`,
`STORE_LOCAL` and each local source of `CLOSURE_BIND` require initialization on
every incoming path. Store additionally requires shared mode. Upvalue sources
use the exact target environment contract; their shape cannot be inferred from
local initialization or capture count alone.

At ordinary branch joins I intersect definite initialization from reachable
predecessors. Loops require a fixed point, including the zero-iteration edge,
clears on backedges and a fresh declaration on each iteration. Dynamic identity
is not an ever-growing verifier counter: the runtime creates the actual cell
identity, while the analysis proves that each access names a live binding.

Handler entry depends on reachable performances while that handler is active.
A local may be initialized or cleared after installation. The owner state at
installation is therefore neither the complete set of entry states nor a safe
replacement for it. Direct, indirect, linked and callback calls may perform an
operation handled by their caller. Their continuations must account for a
handler clearing or reinitializing that caller's binding before resuming.

The next implementation must define finite effect/call summaries and their
composition with active handler ownership before publishing any initialization
proof. It must cover recursion, nearest matching handlers, module-relative
function identity and nested owner chains. I have not selected an algorithm or
claimed those summaries already exist. A local-only pass or permanent refusal
of effects is not completion of this task.

## Proof publication and limits

I compose initialization with existing structural, stack, type, environment
shape and affine checks; none replaces the others. A successful helper query
must not enable a public execution route by itself. Every supported entry and
backend must consume the same completed contract before I admit the feature.

I bound allocated state and charged work before allocation and iteration. A
limit or allocation refusal preserves the caller's prior proof/output and
releases every temporary owner. I require explicit diagnostics distinguishing
invalid initialization, unsupported composition, exhausted work and memory
failure. Exact limits and the owned result representation must be reviewed
before implementation; the transport budget is not automatically a dataflow
budget.

## Required acceptance

My eventual fixtures must cover ordinary parameters and clear nonparameters;
load/store/capture before initialization; branch joins with one or both arms
initializing; zero-iteration loops; loop clears and repeated declarations;
immutable-store refusal; repeated shared sources and forwarded upvalues.

Effect fixtures must separately cover initialization and clearing between
installation and performance, nonzero and empty parameter ranges, nested owner
starts, nearest-handler selection, resume with owner mutation, lexical return
without resumption, and recursive/direct/indirect/linked/callback performance.
Source producers must preserve these same distinctions in emitted metadata and
instructions. Positive ordinary and effect programs remain required alongside
checked refusals; accepting only the easiest subset does not close this work.

I require allocation-prefix and work-limit controls with independent recovery,
reviewed source before affected execution, and fresh Linux/Darwin ordinary and
sanitizer gates. Final acceptance also requires VM, C, LLVM and Wasm parity,
projection/link preservation and canonical bootstrap output. Transport tests,
frame teardown tests and an initialization query alone do not establish those
results.

## Direct entry prerequisite

My current direct CALL initializes a new frame without a closure. TAIL_CALL
reuses a frame but explicitly drops its owned callable and clears its closure.
FUNCREF creates a raw function identity with no captured environment. Before
initialization analysis, I therefore require each module-relative target of
these three instructions to exist and declare zero upvalues. CLOSURE_BIND
remains the explicit environment construction route, including zero-capture
closures. I validate this property in the nonexecuting structural capture
query; it does not establish stack safety, reachability or initialization.

I do not infer indirect or linked targets from these local indices. Their
module identity, exact environment and call/effect summaries remain required.
The finite analysis algorithm is still an open dependency. This direct entry
invariant neither enables capture execution nor closes initialization proof.
