# My computed integer-to-byte conversion proposal

I record MAC `task_c6b2a040c1434fc784a9d46c02a4981e` before implementation.
Both corrected canonical full Make runs pass the original90 methods, then reject
`tests/test_u8_basic.nano` at `let c: u8 = (+ b 1)`. I retain the full original
program and assertions. This proposal is based on canonical PR922 and does not
change source, opcodes or tests yet.

## The boundary I have today

My checker permits INT/U8 assignment compatibility and returns INT for arithmetic
on integer-like operands (`typechecker.c` types_match and numeric expressions).
My C destination type is `uint8_t`. My canonical `compile_expected_tag` deliberately
accepts only exact U8 expressions or contextual literals in0..255. My current
[NanoISA reconstruction contract](NANOISA_RECONSTRUCT_U8.md) expressly excludes
computed integer coercion. The new refusal is thus an exposed whole-compiler
coverage gap, not a reason to remove that exact-tag check.

I have no existing scalar integer-to-U8 conversion opcode. I will not relabel an
INT stack value as a byte, allocate an array as a conversion trick, reinterpret
FLOAT/BOOL payloads, or broaden arbitrary byte arithmetic/comparison instructions.

## Proposed exact semantics for review

I propose preserving the existing C unsigned-byte assignment semantics: converting
an INT result yields its low8 bits, mathematically the unique value in0..255
congruent modulo256. Thus computed201 stays201, computed256 becomes0, and
computed-1 becomes255. I implement this through defined unsigned conversion,
not signed overflow or representation-dependent pointer casts. U8 input is an
identity conversion. FLOAT, BOOL, STRING, VOID and every aggregate tag refuse.

Canonical contextual numeric literals retain their current explicit0..255
restriction, including the existing refusal of literal256 and negative literals.
The computed-expression conversion is an explicit new operation; the literal
policy is not silently weakened. This existing literal/computed distinction is
intentional in this proposal and requires review. If a uniform checked-range
policy is preferred, that would also require changing and qualifying C-native
assignment semantics; I do not choose it implicitly from one failing fixture.

I propose operand-free `CAST_U8` in the currently unused cast slot0x8f, pending a
complete catalog/consumer audit before code. Its effect is one INT or U8 consumed,
one exact U8 produced; no allocation, host callback, global state or hidden
collection. Wrong-tag failure releases the consumed value through the existing
error cleanup. It does not admit a U8 executable entry result.

The source producer may emit this conversion only at a checked exact U8
destination for a computed INT expression: typed binding/assignment, direct
parameter or result. Every non-U8 destination keeps its original lowering.
Existing exact byte values keep their tags, and out-of-range literal checks
remain in place. Arrays and generic imports do not gain authority from that
context; each existing target still performs its complete ordinary/managed/owned
admission. Indirect call shapes must retain their explicit checked signature and
must not infer conversion from a runtime callable's address.

## Required consumers and staged implementation

1. Review the semantics, opcode allocation and source-context boundary first.
2. Audit/update the canonical specification and generated catalogs, C/Nano opcode
   decoders, assembly/disassembly, operand/stack effects and verifier type transfer.
   Runtime tag checks remain necessary when shared verification has incomplete
   advisory types. Older consumers reject the unknown opcode rather than
   interpreting a different instruction; existing serialized section versions and
   U8 tag number do not change.
3. Implement VM switch/computed-goto behavior and exact strict native C lowering.
   Reconstructed Nano uses a checked U8 destination whose canonical producer emits
   this conversion; C uses defined uint8_t conversion. LLVM uses truncation to i8
   and zero extension with the exact tag; Wasm masks the integer low8 bits and
   publishes the U8 tag. No byte is silently widened to INT in returned values.
4. Audit every closed target/classifier explicitly. A target stays refused until
   its complete conversion semantics are implemented and qualified. New ordinary
   scalar conversion does not select File, owner-array, mixed-record, passive,
   linked-service or other authority profiles. Preserve service-first ordering,
   output sentinels and unsupported-case refusals.
5. Add source and raw-bytecode controls before execution review. Cover values0,
   1,127,128,200,201,255,256,257,-1,-256,INT64_MIN/MAX; exact input/output tags;
   unchanged input roots; wrong-tag cleanup; once-only effects; direct argument,
   local overwrite and return contexts; non-U8 regression controls; literal
   out-of-range refusal; and the complete unchanged test_u8_basic program.
6. Qualify original raw and source programs through C seed, Stage1/Stage2 canonical
   producers, VM and native output, then LLVM/Wasm and reconstruction as their
   corresponding implementation checkpoints land. Compare exact bytes/tags and
   observable values, not just process exits. Include original U8 transport,
   truthiness, generic numeric, array and source-shadow neighbors, selected
   sanitizer coverage and explicit Linux/Darwin provider identities.

I require source and fixture review before each stage runs. The first raw
conversion checkpoint is not whole-source, all-backend or full5.1 acceptance.
Both original verifier failures and the full release roadmap stay visible.

## My catalog and consumer audit before opcode implementation

I inspected the current source at the c404/922 base. The active legacy catalog
in `spec/nanoisa.yaml` ends its cast group at F64_TO_BITS `0x8e` and resumes at
CLOSURE_NEW `0x90`; `0x8f` is unallocated in that catalog. This is a candidate
legacy byte assignment, not permission to invent a separate portable-v2 opcode.
`src/nanoisa/isa.h` supplies the symbolic enum; generated metadata comes only
from `scripts/gen_nanoisa_schema.py` into `src/nanoisa/generated_schema.h`.
`isa.c` builds its decoder/encoder metadata table from that generated catalog.
I must change the schema and enum coherently and retain schema validation.

| Consumer | Concrete required implementation or refusal boundary |
| --- | --- |
| Encoding and textual tools | `isa.c`, `assembler.c`, `disassembler.c` consume catalog metadata; test exact one-byte encoding, decode, mnemonic round trip and stack effect |
| Stack/type verification | `verifier_types.c` requires INT or U8 input and U8 output; `verifier.c` target whitelists must remain closed until matching emitters exist |
| VM | `vm.c` actual handler and explicit computed-goto label table must both change; wrong tags retain existing cleanup/error conventions |
| Native C | Both simulation/type-state and emitting cases in `nvm2c.c`; retain U8 in straight and merged/dynamic carriers without INT relabeling |
| LLVM | `nvm2llvm.c` scalar/managed routes and `nvm2llvm_managed.inc` must produce an exact U8-tagged value using defined integer narrowing |
| Wasm | `scripts/nvm2wasm.py` delegates to the real LLVM translator with wasm32 runtime, then llc/wasm-ld; it has no independent cast interpreter to patch |
| C producer | `src/nanovirt/codegen.c` `compile_expected_tag` and every checked binding/assignment/direct parameter/return caller |
| Nano producer | `src_nano/compiler/nanoisa_codegen.nano` `nisa_emit_expected_expr` and its contextual callers, with meaningful shadows |
| Reconstruction | `src/nanoisa/hl_facts_main.c`, `scripts/nanoisa_reconstruction.py`, `scripts/nvm2hl.py`; preserve explicit narrowing in both reconstructed C and Nano source |
| Passive/shape analyses | `passive.c`, `passive_calls.inc`, `managed_array_shapes.c`; decide explicit scalar transfer support rather than accepting a catalog entry automatically |
| Owned/mixed/service analyses | Existing closed switches in owned-array authority/origins, mixed samples, File code/flow and native-owned emission remain refusal boundaries until separately justified |

The existing reconstruction `SIMPLE` set and CAST_INT expression emitter do not
provide a U8 narrowing operation. I must not reconstruct it as an INT cast or
silently elide it. The existing owned native emitter also has no cast case; its
PUSH_U8 support is not computed-conversion authority. Adding one catalog row
cannot constitute complete backend support.

My acceptance sequence must inspect each destination profile's whitelist and
actual transfer path, then test both ordinary INT and already-U8 operands,
INT64 extrema, computed negative/256 boundaries and wrong-tag refusal. Literal
range failures remain unchanged. I will retain output sentinels in not-yet-
implemented translator/reconstruction profiles. No catalog or opcode source
has changed at this audit checkpoint; the complete list prerequisite remains
my first implementation lane.

## My first raw VM checkpoint

I first implement only the legacy catalog/enum, generated encoding metadata, shared type transfer and both VM dispatch forms. CAST_U8 consumes INT or U8 and returns exact U8 with defined unsigned modulo256 conversion. A definite wrong tag fails verification; an unknown advisory type is still checked at runtime and its consumed value is released before the existing type-error unwind. No allocation is introduced. Native C, LLVM/Wasm, reconstruction and source producers remain explicit later checkpoints and must refuse this opcode until implemented. I do not change any public profile whitelist or service/owned/mixed authority here.

I require independent source review before fixtures and execution. Fixtures must check encoding/mnemonic roundtrip, stack effect, exact result tags/bytes and INT extrema under both dispatch forms, runtime wrong-tag cleanup through a verifier-unknown local, and unchanged output sentinels for unsupported translators. Catalog consistency and existing scalar-byte neighbors remain required. This raw checkpoint cannot close the unchanged source corpus failure or task_c6b2a040c1434fc784a9d46c02a4981e.
