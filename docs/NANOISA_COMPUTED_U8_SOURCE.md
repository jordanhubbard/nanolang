# My computed byte source and reconstruction checkpoint

I continue task_c6b2a040c1434fc784a9d46c02a4981e after raw CAST_U8 and generated
C/LLVM/Wasm qualification. My full source corpus is still required. I preserve
my reviewed modulo256 contract and exact runtime U8 result; literals outside
0..255 still refuse. This checkpoint changes no producer or fixture yet.

My C and Nano producers independently recognize a checked U8 destination. For
an exact computed INT expression I evaluate once and append CAST_U8. Existing
U8 expressions remain U8; BOOL, FLOAT, enum, unresolved and aggregate source
kinds do not become integer conversions through assignment compatibility.
The explicit cast does not change the declared result or parameter ABI.

I audit every destination: local and global initialization and reassignment,
lexically captured mutation, direct and qualified calls, exact checked indirect
signatures, ordinary returns and tail calls. A tail call that requires result
conversion cannot bypass it. I take a callable snapshot before arguments and
retain left-to-right argument evaluation. An address or same-spelled function
is not a checked signature. Existing contextual aggregate destinations must
retain their declared element/field identity and authority; a scalar conversion
is not permission to widen a closed aggregate or service profile. Any uncovered
source destination remains ledgered, not silently reported as complete.

My reconstruction consumes only an exact INT or U8 fact for CAST_U8 and creates
an exact U8 expression. C emits defined uint8_t narrowing. Nano emits an explicit
helper with an INT parameter and U8 return, using the same checked destination
lowering; it does not manufacture an out-of-range literal or mask into an INT
result. Helper generation is demand-driven and retains meaningful shadows.
Unknown/wrong tags and unsupported regions refuse before publication. Existing
output files and input module bytes remain unchanged on refusal.

Before execution I review complete production and fixtures. I retain original
`tests/test_u8_basic.nano` and its assertions, then cover all256 byte identities,
integer extrema and negative/modulo boundaries, every destination above,
once-only effects, tag assertions, literal/wrong-source refusals and callable
lookup precedence. I update the old computed-expression refusal expectation
only alongside a positive exact-tag control; literal refusal stays intact.

Fresh C seed, Stage1 and Stage2 compilers must produce and execute the original
source and new controls through the real canonical producer/VM. Native,
LLVM/Wasm and both reconstruction languages must agree on values and tags in
their supported profiles. I retain selected sanitizer scope, actual providers,
first failures, both-host results and the complete unchanged verifier corpus.
A focused conversion pass cannot close unrelated list or full5.1 acceptance.

My first source checkpoint adds contextual INT narrowing in both emitters and
exact scalar destination metadata for C locals, globals and captured bindings.
C direct/qualified arguments and retained checked indirect signatures use the
same conversion helper; callee snapshot order and result-tag tail-call checks
remain. Reconstruction now represents a narrowing expression and emits its
Nano helper only when required. I have not executed these changes. Contextual
aggregate destination audit, complete fixture review and fresh paired
qualification remain before this source work can be called complete. The Nano
emitter's existing indirect-call refusal remains visible full-graph work; this
scalar change does not manufacture an indirect call implementation.

My first retained fixture matrix has seven methods. It uses four explicit
canonical producers: nano_virt and separately freshly compiled Nano emitters
from C seed, Stage1 and Stage2. NANO_U8_EMITTERS must name those three distinct
executables; no missing producer is substituted. Actual source output is
verified/executed and reassembled for generated C/LLVM/Wasm O0/O2 comparison.
C-seed native output is compared independently. I retain the original byte
program unchanged, all256 integer-to-byte values and extrema, globals/locals,
qualified parameters/returns, once-only calls and differing-result tail calls.

The existing C producer's checked indirect/captured path has a separate actual
VM control, including a captured byte that shadows an integer global. That is
not independent Nano indirect or generated closure acceptance. Both remain
explicit full-graph work. Literal256 and negative literal refusals remain; old
computed-expression refusals are replaced by positive narrowing checks. Raw
CAST_U8 reconstruction covers both C and Nano with actual recompiled producer
outputs. No build, fixture discovery or execution has occurred at this point.

## My interpreted scalar destination prerequisite

I retain the first20bdb Linux and Darwin source failures under task_c4180e737bd042b1a2cfdb04a9c19906. My fresh builds, complete C-seed/Stage1/Stage2 bootstrap and three actual Nano emitter builds pass; my C-seed's interpreted shadows then leave a257 return unnarrowed in a U8-returning function. This is a demonstrated evaluator mismatch, not a failed generated CAST_U8 test.

My evaluator represents both INT and U8 with VAL_INT. At a checked exact TYPE_U8 scalar destination, I narrow its integer payload through uint8_t once. I preserve control-flow metadata, do not allocate, and leave other value kinds and destination types unchanged. The shared environment publication paths cover local/global initialization, parameters and reassignment using the retained destination type. Both evaluator function invocation paths narrow only a completed return belonging to that activation; a nonlocal return addressed to an outer handler must retain its original value. Indirect and qualified calls use the actual registered function's destination. Existing checker literal and wrong-source refusals remain authoritative.

I review this source before rebuilding any provider. The seven source methods remain unchanged and mandatory. Aggregate fields/elements, full callable graph and complete verifier corpus remain required under the parent; a passing scalar prerequisite does not close those obligations.

## I retain the next corrected-code failure

At70516, fresh Linux and Darwin build/bootstrap/runtime and all three actual
Nano emitter builds pass. My all-values/extrema method passes through all four
source producers and its backend checks. The next unchanged captured fixture
fails before publication; the remaining five methods do not run.

An independent scalar observation preserves the original source and compiler:
the closure returns2, the outer byte remains1, and the same-spelled global
remains900. The failed assertion is line13. The diagnostic reports line14
because the VM looks up its already-advanced instruction pointer. I retain
`/tmp/nanolang-u8-capture-diagnostic-u5oabd0n/audit.json` and both original
qualification directories `/tmp/nanolang-computed-u8-70516-linux` and
`/tmp/nanolang-computed-u8-70516-puck`. This observation does not establish a
shared cross-backend capture specification. I require the reviewed contract
and complete capture lifetime acceptance under task_af8091f571a842bc90656e2c7f19b68e;
source mapping is independently tracked by task_b7779ee196994432a18178145d93ac42.
