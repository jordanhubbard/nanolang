# NanoLang 4.0

I am NanoLang 4.0. This release completes NanoISA v2 and NanoVM v2: a regular,
compositional, verified instruction set with a module format that can express
it, and a runtime whose safety argument is checked rather than assumed.

**This document describes a release I have not made yet.** It is written now so
that the boundary between what is done and what remains is legible while the
work is still in progress, and so the release itself is a matter of confirming
evidence rather than reconstructing it. Phase 12 stands at 60 of 78 items.

## What I Shipped

### A regular portable instruction set

- Every portable instruction has one comprehensible meaning, and operand forms
  are symmetric. The schema generator refuses to emit a design that breaks
  either rule, so this is enforced rather than reviewed.
- Every public instruction records *why* it belongs in the ISA rather than a
  runtime library. Higher-level string and collection algorithms moved out;
  only primitives justified by representation or measured cost remain.
- Language-specific struct, tuple and union lowering is replaced by
  layout-driven `AGG_PACK`, `AGG_GET`, `AGG_SET` and `AGG_TAG`.
- Direct function references and heap closures are separate value tags rather
  than one overloaded representation.
- Special print, assert and host operations are typed traps.
- Opcodes no frontend emits are gone: `GC_SCOPE_ENTER`, `GC_SCOPE_EXIT` and
  `CLOSURE_CALL`.

### A verifier whose result the runtime relies on

- Control-flow verification validates branch targets against a decoded
  instruction-boundary map, so a branch into the middle of an instruction is
  rejected at load.
- Stack heights propagate through every basic block and merge states must
  agree.
- Call arity, result shape, aggregate counts, local, global and upvalue bounds,
  type tags and import signatures are all checked.
- Where the proof permits it, verified operations dispatch to unchecked private
  handlers. The verifier's work is spent rather than duplicated.
- The decoder, loader, verifier, assembler, disassembler and co-process
  protocol are fuzzed against malformed input.

### Execution that was measured, not assumed

- Predecoded indexed dispatch, with serialized bytecode, verified instruction
  IR and optimized dispatch IR kept as three distinct representations.
- Profile-selected private superinstructions, with frontend bookkeeping kept
  out of the portable opcode space.
- Unboxed homogeneous arrays for integer, float, boolean and byte elements.
- Globals sized from serialized declarations rather than 4,096 values embedded
  in every VM.
- Module constants preinstantiated, so a string literal does not allocate and
  search an intern table on every execution.
- Contiguous hash-map storage, and transient-string interning replaced.

### A module format that can carry all of it

- A 40-byte header with three independent version axes: `magic[3]` for the
  format lineage, `format_version` for the layout, `isa_version` for the
  instruction semantics. Conflating those is what left v1 unextendable.
- Feature-bit negotiation that fails closed: a module using a capability the
  reader lacks is rejected at load rather than failing subtly during execution.
- A bounded 64-bit section directory carrying code, constants, signatures,
  globals, imports, layouts, links, metadata and optional debug sections.
- Separately linked modules with imports resolved once into typed call
  descriptors.
- Symbolic functions, imports, fields, types, constants and labels as
  first-class assembler operands.

### A typed FFI boundary

- Mixed integer and floating signatures dispatch through generated typed stubs,
  placing arguments in the register class the platform ABI requires. Before
  this, a `double` alongside an integer argument was passed in a
  general-purpose register and silently produced a wrong answer on arm64.
- One argument limit across imports, traps, direct FFI and co-process calls.
- Host work batched rather than crossing the process boundary per element.

## Evidence

Verified with:

- `make test-quick`, `make test-units` and `make examples-core` green.
- NanoISA, NanoVM, NanoVirt and verifier suites green under clang and GCC.
- AddressSanitizer and UndefinedBehaviorSanitizer clean. Both jobs had been
  failing to compile for long enough that nobody had seen what they reported;
  when they ran they found 21 undefined-behaviour errors and a heap overflow,
  all fixed here.
- Benchmark workloads recorded with distributions rather than single timings,
  under `docs/benchmarks/`.
- CI green across x64 and arm64, C, PTX and RISC-V backends, sanitizers,
  coverage, documentation, benchmarks and security.

## What I Fixed That Nobody Had Seen

Worth recording, because each was invisible rather than unreported:

- Reading uninitialized `_Bool` fields in the typechecker and transpiler.
  Undefined behaviour a compiler may optimize on, in a flag deciding whether
  the "program must define a main function" check runs at all.
- Every heap object under-aligned. The reference-counted header was 12 bytes
  and the heap base had alignment 1, so any object holding a pointer or a
  double was misaligned. Tolerated on x86-64 and arm64; a fault elsewhere.
- A heap-buffer-overflow in the assembler on a quoted string ending in a
  backslash, reachable from malformed input.
- `OP_CALL_EXTERN` silently truncating an over-long argument list, dropping
  arguments *and* desynchronizing the operand stack for everything after it.

The first three were found by sanitizer and fuzz jobs that had been red so long
they were treated as background noise. The fuzz tests in particular had been
merged in a state where CI could not compile them, so they had never executed;
given the chance to run, they found a real heap overflow within seconds.

## Boundary

I do not claim the following as complete, and they are documented as open in
[my roadmap](ROADMAP.md):

- The v2 module format is specified and its container is implemented, but v1
  remains the default on-disk format. The section encoders, serializer and
  loader are planned in
  `docs/superpowers/plans/2026-09-02-nanoisa-v2-module-format.md`.
- Computed-goto dispatch is not shipped. The portable switch is the only
  dispatch strategy.
- Maximum operand depth, frame depth, ownership effects and explicit
  termination are not verified. Linked-module calls are recognized but their
  signatures are checked at instantiation rather than at verification.
- The two-result `ARR_POP` convention remains, as does reference-ownership
  handling for array removal, closure calls and FFI trap arguments.
- Whether heap graphs use tracing collection or enforceable cycle restrictions
  is not decided for the VM.
- Symbolic assembly examples for NanoLang and Forth are not written.

Module signing is deliberately 5.0 work, not 4.0. The mechanism is nearly free,
but deciding where verification keys come from and who may issue them belongs
with the capability and policy work rather than ahead of it.

## Links

- [Current README](https://github.com/jordanhubbard/nanolang/blob/main/README.md)
- [4.0 changelog entry](../CHANGELOG.md#unreleased)
- [Roadmap](ROADMAP.md)
- [3.5 release presentation](RELEASE_3.5.md)
- [NanoISA v2 module format specification](superpowers/specs/2026-09-01-nanoisa-v2-module-format.md)
