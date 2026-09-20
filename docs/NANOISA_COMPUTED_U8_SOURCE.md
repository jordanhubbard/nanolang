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
