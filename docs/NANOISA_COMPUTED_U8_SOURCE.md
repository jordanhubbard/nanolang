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

## My enum-to-byte source policy

I resolve the previously recorded enum boundary under
`task_c6b2a040c1434fc784a9d46c02a4981e` by preserving my existing checked source
compatibility. The C checker accepts ENUM/U8, and the Nano checker accepts
TYPE_ENUM with its TYPE_INT representation of u8. Native unsigned-byte
assignment already narrows the enum's integer value. Both NanoISA producers
must therefore append CAST_U8 for an actual checked enum expression at an
exact U8 destination, just as they do for INT. This supersedes the earlier
checkpoint's enum refusal; it does not make arbitrary named types integers.
The C producer uses TYPE_ENUM and the Nano producer uses its actual parser
enum declaration lookup. Enum members are expressions: a member valued -1 or
256 narrows to 255 or 0. Direct numeric literals retain the existing 0..255
rule. BOOL, FLOAT, unresolved names and aggregate values still refuse.

I require parser-backed emission shadows for enum members and a named enum
result, then fresh independent producer/runtime comparison and unchanged
literal/wrong-tag controls. Existing full byte destinations, captures, source
corpus, reconstruction and all required backend gates remain open. This
policy correction alone cannot close the parent task.

## My global enum prerequisite

My fresh938f Linux qualification builds the C seed, bootstraps and builds all three independent emitters successfully. The first enum fixture then stops before native publication: both C global checker entry points use strict tag equality, unlike local assignment compatibility. My earlier statement about existing checker compatibility applies to those local compatibility paths, not every global declaration. I extend only an exact declared U8 receiving an actual TYPE_ENUM in both global paths. I retain all other strict global comparisons, including existing numeric-literal checks, and leave the enum fixture unchanged. I require independent source review before the next fresh qualification. The original failure and six unreached selected methods remain recorded; shared captures are still open.

## My explicit native byte destination prerequisite

My9609 qualification passes fresh bootstrap and all three emitter builds. The unchanged enum source then reaches native C: assigning enum511 to a global byte and enum256 to the same byte produces strict overflow diagnostics. I retain the exact source and compiler terminal under task_91533a35bf774d3eb36907ef39b2cb73. I will emit `(uint8_t)(expression)` at exact checked byte global/local initializers, variable stores and ordinary/effect lexical returns. I capture a store destination type from the existing source-position-aware lookup before lowering its right-hand expression. I evaluate each expression once and preserve its order; other types remain unchanged. Ordinary/qualified/indirect calls already snapshot arguments into temporaries before the typed C call conversion; I retain that machinery. Aggregate field/element acceptance remains under the full parent rather than inferred from this scalar correction. I require additive direct enum-member local/return controls and the unchanged original once-only checks before closure.

## My source-level literal guard

Task_a2c8eaede4cb4dd7a08c6f120654f846 retains a static audit finding: C scalar compatibility alone does not distinguish a direct numeric literal from a computed integer. Before explicit byte casts can be qualified broadly, I must preserve literal range in the checker, independently of a host C overflow warning. I add an expression-aware destination predicate for exact U8 literal0..255 and leave other compatibility unchanged. Local/global initialization, stores, ordinary and lexical returns, checked direct/indirect/effect arguments receive this guard; aggregate contexts remain under the complete parent and the separate concrete-union work. Globals accept actualINT orENUM only at exact U8, with the same literal guard, while other global comparisons remain strict. Existing seven source methods stay required. New Cseed-native refusals retain output sentinels for each scalar route; direct0/255 and computed overflow remain positive controls.
