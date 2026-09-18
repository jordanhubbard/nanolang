# My exact binary64 bit transport contract

I record `task_eadc85b5000f43e3b375ce274ff5ac7a` before implementation.
This checkpoint proposes the shared surface and encoding for review. It does
not admit new executable behavior or reconstructed float source.

## My source operations

| Operation | Exact parameter | Exact result |
|---|---|---|
| `float_from_bits` | `int` | `float` |
| `float_to_bits` | `float` | `int` |

I evaluate the operand once. These pure builtins allocate nothing, perform no
floating arithmetic, import no host function, and do not alter floating-point
environment state. They do not accept implicit bool, u8, enum, string or mixed
numeric conversions. Existing `cast_int`, `cast_float` and string parsing keep
their current numeric semantics and errors.

My integer denotes a two's-complement 64-bit pattern. For `float_from_bits`,
conversion from signed int to uint64 is modulo 2^64, then `memcpy` copies those
bits into binary64 storage. For `float_to_bits`, `memcpy` reads uint64 bits;
the result is their signed interpretation. C implementations reconstruct a
representable negative result without an out-of-range unsigned-to-signed cast.
No pointer punning or host byte-order interpretation is permitted.

Every binary64 encoding is part of this contract: both zeros, all finite
encodings, both infinities, and quiet/signaling NaNs with their exact payload
and sign. I must not quiet or canonicalize NaNs during these operations or the
qualified scalar transport paths. My string parser intentionally sets the NaN
quiet bit and therefore cannot implement this API. I make no new promise about
payload bits produced by floating arithmetic.

## My proposed stable encoding

I reserve currently unused primary opcode values in the authoritative
`spec/nanoisa.yaml` and synchronized enum/metadata:

| Mnemonic | Code | Stack input | Stack output | Immediate operands |
|---|---|---|---|---|
| `F64_FROM_BITS` | `0x8d` | exact INT | exact FLOAT | none |
| `F64_TO_BITS` | `0x8e` | exact FLOAT | exact INT | none |

These values follow TYPE_CHECK (`0x8c`) and are unused in the inspected schema
and active enum at base `53b43377`. I add matching normative representation
operations, justified because a portable library cannot observe hidden value
bits. I do not repurpose an existing opcode or use the extension prefix.

I keep schema/ISA and container versions at 2: this is an additive instruction
allocation with no changed old encoding or semantics. Older readers encounter
unknown opcodes and refuse decode (`isa_decode` checks metadata before reading
operands); I do not claim older runtimes execute the new instructions. Schema
validation must detect collisions and generated-metadata drift. This version
choice is explicit and subject to review before production edits.

## My complete implementation boundary

I update the builtin registry and exact source type checks, C interpreter and
legacy C emission/runtime helpers, C-seed NanoVirt producer, selfhost source
checker/emitter and direct-return inline classification. No frontend may replace
these operations with numeric casts or decimal text. Existing name-resolution
and shadowing rules remain in force.

I add exact verifier rules, both VM dispatch routes and ordinary native C
classification/emission, including checked tagged scalar extraction. I retain
wrong-tag runtime errors where static tags are unknown; known wrong types are
refused before execution. I preserve all unrelated ownership/profile refusals.

Shared LLVM lowering and Wasm reuse exact integer payload transport or LLVM
`bitcast` with checked tags; neither may use numeric `sitofp`/`fptosi`, fast-math
flags or NaN canonicalization. I update each supported scalar/managed profile
coherently. Existing profiles that do not admit FLOAT transport remain explicit
refusals until their own contracts permit it; this is not heap/owner widening.
Wasm execution must be qualified in both Wasmtime and Node, with no new host
imports for these operations.

## My qualification order

1. I qualify schema allocation, assembly/disassembly and verifier tag rules.
2. I qualify exact VM/native/LLVM/Wasm transport on the same ordinary modules.
3. I qualify C interpreter/native, C-seed NanoVirt and selfhost producers on
   ordinary source programs, including fresh producer/bootstrap evidence as
   required by their changes.
4. Only after all transport routes pass may a separate reconstruction change
   consume `f64_bits` and emit exact source constants using this API.

Tests compare integer bit results, never NaN float equality. My corpus includes
signed zeros, finite/subnormal boundaries, infinities, multiple signed quiet and
signaling NaN payloads, each single-bit position and a deterministic mixed-bit
sample. I test both inverse directions, local/global storage, argument/result
transport, branch joins and once-only effectful operand evaluation. I retain
ordinary typed refusals and previous output, and existing numeric-cast controls.
Native GCC/Clang sanitizer gates and LLVM/Wasm execution are required; a C-only
helper or VM-only result cannot close this task. A finite corpus is tested
evidence, not exhaustive proof over all 2^64 patterns.

I preserve historical compiler-failure evidence without replay. The full
reconstruction parent remains open, and its current executable whitelist stays
unchanged throughout this transport task.
