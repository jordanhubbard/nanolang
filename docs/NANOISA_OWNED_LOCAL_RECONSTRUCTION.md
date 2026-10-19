# I reconstruct bounded resource transfers and direct calls

I advance required ownership reconstruction under #999 and rows 9538/10409 with
one closed, single-entry module with acyclic direct helpers: resource records whose ordered fields are exact
INT or BOOL, local construction, move, consuming unpack and exact scalar/resource calls and returns.
I retain every numeric nominal layout identity and resource declaration. Two
same-shaped layouts remain distinct. I do not claim the wider borrow, generic, nested, union, array, effect, cross-module call or control-flow requirement complete.

I require successful current `nvm_verify` executable admission before extracting
facts. Affine instruction analysis is additional consistency evidence, never a
replacement for execution admission; advisory frontend facts grant no authority.
I expose the exact retained layouts and local declarations and preserve complete
instruction order. My owned region statements are never duplicated, discarded
or folded through scalar temporary optimization.

My C output uses concrete uniquely named structs and explicit one-use transfer
temporaries. My Nano output uses `resource struct` and consuming `let Type {
field_names } = owner` syntax. That syntax does not support field aliases;
repeated same-name patterns are supported with distinct lexical slots. I snapshot
scalar results immediately after each pattern. Both outputs consume one checked region tree and contain no embedded VM, goto or
hidden bytecode dispatcher. Deterministic numeric names preserve layout identity
without claiming original source spelling recovery.

I admit only the actual producer's exact ephemeral shared-field read sequence
`REGION_BEGIN/BORROW_LOCAL_SHARED/REF_GET/REGION_END` on a live exact local
owner. I represent it as a read statement, require matching reference index and
valid scalar field, and forbid escape or overlapping/misnested regions. I retain
terminal consuming `OWN_UNPACK_LOCAL`. I refuse branches, loops, borrowed/indirect/tail/module/extern calls,
other references, globals, imports, nested records and unsupported instructions in this resource route. I retain the
existing scalar route unchanged. All refusals preserve a prior output file.

Acceptance requires actual C-hosted source producer constructor/move/destructure
fixtures, same-shaped distinct nominal records, field ordering and once-only
scalar computations. I compare original NanoVM with reconstructed native C and
reconstructed NanoVM/native execution, and inspect regenerated resource layout
correspondence. Missing/mutated authority, wrong nominal identities, duplicate or
lost owners and use after move must refuse. Fresh Stage1/Stage2, Linux/Darwin and
broader ownership paths remain separate unfinished qualification.

My first source probes disproved two artificial assumptions: a standalone block
is not accepted syntax, and `if true` does not erase branch ownership obligations.
Repeated patterns are accepted. Real source destructuring emits ephemeral reads
followed by a delayed consuming unpack, so an owned-opcode-only allowlist would
not qualify the real producer. I preserve those failed probe logs and expand
only the exact observed read sequence, without broad borrowed reconstruction.

## My direct-call boundary

I retain at most eight functions and 4096 instructions across the whole module,
128 locals per function, eight parameters, sixteen resource layouts and sixteen
scalar fields per layout. My entry remains argument-free with one INT result.
Helpers take exact mode-zero INT/BOOL/resource parameters and return one exact
INT/BOOL/resource value. Every retained function must satisfy the contract,
including unused helpers. Recursive and entry-directed calls remain refused.

I initialize owner parameters in their exact positional slots, consume call
arguments once, and snapshot each call result before another instruction. Scalar
instructions operate only on the scalar suffix above the last staged owner;
they cannot pop, duplicate or reorder that owner. An owned return resolves all
local owners and leaves no extra operand. Both emitters consume the same ordered
tree and retain direct calls rather than inlining them.

My Nano output names the retained entry `main` directly. Its arithmetic helpers
also count toward the source producer's function budget. I refuse publication
when the emitted source has more than seven functions, reserving the eighth for
shadow execution. C retains the eight-function input bound. A source-budget
refusal is not successful Nano reconstruction. Generated helper shadows remain
separate from original source shadows, which I do not recover.

I retain [direct-call qualification](evidence/owned-calls-reconstruction-20261018/README.md).
This extends the bounded local checkpoint; it does not close #999.
