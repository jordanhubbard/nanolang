# Portable NanoISA vs. Runtime Representations

I keep a hard line between the **portable NanoISA** — the durable, serializable
contract every tool agrees on — and the **runtime representations** the VM
derives from it to verify and to run a program. This document defines that line
so that a reader can reason about the portable ISA on its own, without knowing
anything about how my VM decodes, verifies, or dispatches it.

The short version: the portable ISA is the *specification*; the runtime
representations are *implementation projections* of a loaded module. A change to
the portable ISA is a change to the on-disk/on-wire contract that every
conforming producer and consumer must honor. A change to a runtime
representation is an internal optimization that must not alter observable
behavior.

## The three layers, and which is portable

| Layer | Kind | Source of truth | Portable? | Where it lives |
|-------|------|-----------------|-----------|----------------|
| Portable NanoISA | Specification | `spec/nanoisa.yaml` (+ generated `src/nanoisa/generated_schema.h`) | Yes — the durable contract | `src/nanoisa/` |
| Verified instruction IR | Runtime representation | Decoded from a loaded module | No — internal to a VM instance | `src/nanovm/vm_decode.*` |
| Optimized dispatch IR | Runtime representation | Projected from the verified IR | No — internal to the hot loop | `src/nanovm/vm_dispatch.*` |

Only the first layer is portable. The other two are private to a single VM
process, are rebuilt on every load, and never appear on disk or on the wire.

## Layer 1 — the portable NanoISA (the contract)

The portable NanoISA is everything a producer must emit and a consumer must
accept without seeing my VM internals. It is defined once, declaratively, in
`spec/nanoisa.yaml`, and I generate `src/nanoisa/generated_schema.h` from it so
the design and the code never drift (`make schema-check`).

The portable surface is exactly these things:

- **Instruction families and their meaning.** Every portable instruction has
  one comprehensible meaning; no opcode is overloaded to do two jobs depending
  on its operands. `i64.add` adds integers, `f64.add` adds floats, `const.i64`
  pushes an integer literal. The generator refuses to emit the schema if any
  meaning is missing or shared.
- **Operand kinds and forms.** Every operand names a kind declared in
  `operand_kinds`; no instruction hides a raw encoding inline. Mirror-image
  instructions take mirror-image operands (`local.get`/`local.set` both take a
  `local`; every `mem.*` load/store takes the same `[offset, align]`).
- **The one-byte opcode space and the extension prefix.** Opcode bytes are
  identifiers, not a running instruction count. The primary plane occupies
  `0x00..0xFE`; the byte `0xFF` is the reserved extension prefix that escapes
  into a second 256-opcode plane. This is a portable numbering rule: a
  conforming decoder must treat `0xFF` as an escape, never an instruction.
- **Value tags.** The tag byte that identifies each runtime value
  (`i64`, `f64`, `bool`, `string`, `array`, `record`, `variant`, …).
- **Compact encodings as encoding-only aliases.** A compact form
  (`const.i64.small`, `local.get.short`, `*.compact`) is a *bit-level alias* of
  a canonical family instruction with the same mnemonic family, stack effect,
  and ownership. It changes the bytes, never the meaning: a disassembler always
  renders the canonical operand. Compact encodings are portable because they are
  part of the byte contract, but they add no new instruction semantics.
- **The `.nvm` module format.** The 32-byte header, the section directory, and
  the required sections (`CODE`, `STRINGS`, `FUNCTIONS`, `STRUCTS`, `ENUMS`,
  `UNIONS`, `GLOBALS`, `IMPORTS`, `DEBUG`, `METADATA`, `MODULE_REFS`) are the
  serialized shape of a portable module. See
  [NanoISA Architecture](NANOISA.md#nvm-binary-format).

The full instruction reference, encoding tables, and format layout live in
[NanoISA Architecture](NANOISA.md); this document does not restate them. The
point here is only *what counts as portable*: if it is in `spec/nanoisa.yaml` or
the `.nvm` format, it is part of the contract; everything below is not.

### The portability rule

A byte stream that conforms to the portable NanoISA must produce the same
observable behavior on any conforming VM, regardless of how that VM decodes,
verifies, or dispatches it. The runtime representations in Layers 2 and 3 exist
to make execution correct and fast on *my* VM; they are free to change shape
release to release, and they carry no compatibility promise, precisely because
they are not portable.

## Layer 2 — verified instruction IR (runtime representation)

The verified instruction IR (`VmDecodedModule`, `src/nanovm/vm_decode.*`) is the
result of one decode pass per function performed when a module is loaded. It:

- establishes instruction boundaries over the byte-addressed code, and
- resolves every branch and direct call against a verified boundary map.

It is the representation the verifier reasons about, and it is byte-offset
addressed so that it lines up 1:1 with the portable bytecode it came from. It is
*derived* from Layer 1: it adds no instruction meanings and removes none. It
exists only inside a live VM and is rebuilt on every load. It never appears in a
`.nvm` file and has no portability contract.

## Layer 3 — optimized dispatch IR (runtime representation)

The optimized dispatch IR (`VmDispatchModule`, `src/nanovm/vm_dispatch.*`) is a
projection of the verified IR shaped for the hot fetch loop. Instructions live
in a flat, instruction-indexed array; the linear-path successor is a precomputed
instruction index; branch and call targets are precomputed as dispatch indices.
It is derived from — and validated against — the verified IR, and it is rebuilt
in lockstep whenever the verified IR is rebuilt.

`vm_core_execute` runs this representation. A dispatch cursor advances by
instruction index on the linear path and consults a byte-offset map only to
re-enter the stream after a jump, call, or return, which keeps the byte-addressed
`ip` contract that frames, traps, and returns depend on.

Layer 3 is the most implementation-specific of the three. Its array layout,
index scheme, and precomputation are pure optimizations governed by
[How I Optimize NanoISA](NANOISA_OPTIMIZATION_POLICY.md): a change here is
accepted on measured speed with unchanged behavior, never by altering the
portable ISA.

## Why the separation matters

- **Stability of the contract.** Producers (NanoLang, Forth) and consumers
  (loader, verifier, assembler, disassembler, other VMs) only need to agree on
  Layer 1. They must never depend on the internal shape of Layers 2 and 3.
- **Freedom to optimize.** Because Layers 2 and 3 are private and rebuilt per
  load, I can restructure them for speed without a format-version bump and
  without breaking any existing `.nvm` module, as long as behavior is preserved.
- **Independent testability.** Each layer is validated on its own terms:
  Layer 1 by schema generation and `.nvm` format tests, Layer 2 by the verifier,
  and Layer 3 by validation against the verified IR it was projected from.
- **A verified-then-optimized pipeline.** A loaded module flows Layer 1 →
  Layer 2 → Layer 3. Verification happens on Layer 2 (byte-addressed, 1:1 with
  the portable form), and optimization happens on Layer 3 (dispatch-indexed).
  Keeping "what is portable" apart from "what is verified" and "what is
  optimized" is what lets each stage stay simple.

## See also

- [NanoISA Architecture](NANOISA.md) — the full portable instruction set,
  encoding, and `.nvm` format reference.
- [How I Optimize NanoISA](NANOISA_OPTIMIZATION_POLICY.md) — the evidence and
  acceptance rules that govern changes to the optimized dispatch IR.
- `spec/nanoisa.yaml` — the single declarative source of the portable ISA.
