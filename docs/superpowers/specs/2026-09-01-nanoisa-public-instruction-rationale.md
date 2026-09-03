# Why every public instruction belongs in the ISA

Roadmap 4.0, Phase 12, Documentation and acceptance.

I keep an instruction in the portable ("public") NanoISA only when a runtime
library written in NanoLang and reached through `call.import` or a `trap` could
not provide it for itself. Everything a portable library can compute by
composing existing instructions is a library function, not an instruction. This
note records that rule for the whole public instruction set so future changes
have a written test to argue against.

The companion note
[NanoISA Primitive String and Aggregate Operations](2026-09-01-nanoisa-primitive-string-aggregate-ops.md)
applies the same rule to string and aggregate opcodes, and
[NanoISA v2 — string and collection algorithms live in runtime libraries](2026-09-01-nanoisa-runtime-library-algorithms.md)
lists the algorithms that were moved out. This note generalizes their rule to
every family in `spec/nanoisa.yaml`.

## Where the rationale lives

Every instruction in the `instruction_families` section of
`spec/nanoisa.yaml` now carries a `justification` field beside its `meaning`,
`operands`, stack effect, and `ownership`. I generate that field into
`src/nanoisa/generated_schema.h` (`NanoisaV2Family.justification`), and
`scripts/gen_nanoisa_schema.py` refuses to emit the schema unless every public
instruction claims one of the justifications below. So the design and the code
never drift, and `make schema-check` fails instead of shipping an instruction
that never said why it is an instruction.

## The six justifications

An instruction earns its place in the ISA by claiming exactly one:

- **representation** — it reads or builds a value's internal layout: a pooled
  constant, a typed load or store against my linear memory, or an aggregate
  `pack`, `get`, `set`, or `tag`. A library cannot reach the physical layout a
  portable ISA exists to hide, so removing the instruction would force every
  frontend and runtime to agree on private layout details.
- **core-semantics** — it is a primitive whose bit-exact result I must define
  once for every backend: a typed literal push, or integer and floating-point
  arithmetic. A library that reimplemented these would either duplicate the
  definition or disagree with it.
- **execution-substrate** — it operates on my own operand stack or call frame:
  stack shuffles (`drop`, `dup`, `swap`, `pick`, `roll`) and local, global, and
  upvalue access. A library runs *on* the stack and the frame; it has no
  portable way to address them from the outside.
- **control-flow** — it manipulates the instruction pointer, call frames, or
  signatures: branches, direct and indirect calls, tail calls, and return. A
  library is reached *through* a call; it cannot define what a call is.
- **host-boundary** — it *is* the effect boundary: the typed traps and host
  imports. A library cannot cross from pure computation to a side effect without
  one of these; they are the crossing.
- **encoding** — it is an encoding-only alias of a canonical instruction. It
  belongs for compact size, not for any new meaning, and it names the canonical
  instruction it decodes to. It adds no behavior a library would ever want.

## The classification

| Family | Instructions | Justification |
| --- | --- | --- |
| constants | `const.i64`, `const.f64` | core-semantics |
| constants | `const.pool` | representation |
| stack | `drop`, `dup`, `swap`, `pick`, `roll` | execution-substrate |
| variables | `local.get/set`, `global.get/set`, `upvalue.get/set` | execution-substrate |
| integer | `i64.*` (arithmetic, bitwise, shifts, wide/carry) | core-semantics |
| float | `f64.add`, `f64.sub`, `f64.mul`, `f64.div` | core-semantics |
| memory | `mem.load8..64`, `mem.store8..64` | representation |
| aggregate | `aggregate.pack/get/set/tag` | representation |
| control | `branch`, `branch.zero`, `branch.nonzero`, `call`, `call.indirect`, `tail.call`, `return` | control-flow |
| control | `call.import` | host-boundary |
| trap | `trap.print`, `trap.println`, `trap.assert`, `trap.halt`, `trap.host`, `trap.dispatch` | host-boundary |
| compact | `const.i64.small`, `local.get/set.short`, `pick/roll.compact`, `global.get/set.compact`, `aggregate.get/set.compact` | encoding |

There is no seventh justification. If a proposed instruction cannot claim one of
these six, it is a library function, and it belongs behind `call.import` or a
`trap` like every other host or module word.

## Rule for future changes

Before adding a public instruction, state its justification in
`spec/nanoisa.yaml`. A `representation` claim must name the layout detail no
composition of existing instructions can reach. A `core-semantics` claim must be
a primitive whose result I define once. An `execution-substrate`,
`control-flow`, or `host-boundary` claim must show that a library, running on the
stack and reached through a call or trap, cannot express the operation for
itself. An `encoding` claim must name the canonical instruction it aliases and
add no new meaning. An instruction that can prove none of these is a library
word, and `scripts/gen_nanoisa_schema.py` will reject it.
