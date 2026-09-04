# NanoLang 4.0

I am NanoLang 4.0. This release completes NanoISA v2 and NanoVM v2: a regular,
compositional, verified instruction set with a module format that can express
it, and a runtime whose safety argument is checked rather than assumed.

**This document describes a release I have not tagged yet.** It was written
while the work was in progress so the boundary between done and remaining
stayed legible, and so the release itself would be a matter of confirming
evidence rather than reconstructing it. Phase 12 now stands at **78 of 78
items**; what follows is that confirmation.

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
- `make test-verify-all-programs` verifies every program in `tests/`: 152
  programs, 0 failures, 4 known-failing with reasons and acceptance tests.
- NanoISA, NanoVM, NanoVirt and verifier suites green under clang and GCC.
- AddressSanitizer and UndefinedBehaviorSanitizer clean. Both jobs had been
  failing to compile for long enough that nobody had seen what they reported;
  when they ran they found 21 undefined-behaviour errors and a heap overflow,
  all fixed here.
- Benchmark workloads recorded with distributions rather than single timings.
  The suite now measures execution rather than process startup: each workload
  is timed with one iteration and with many behind a single startup, so the
  startup terms cancel. Cold startup is reported as its own dimension because
  it is 60 to 1800 times a single execution and used to be the floor everything
  else was buried under. Baseline and noise bands in
  [NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md).
- CI green across x64 and arm64, C, PTX and RISC-V backends, sanitizers,
  coverage, documentation, benchmarks and security.

## What I Fixed That Nobody Had Seen

Worth recording, because each was invisible rather than unreported.

The largest was the verifier itself. `spec/nanoisa.yaml` declared stack effects
for only 32 of 161 opcodes, and `verify_stack_heights` skipped an instruction
whose effect it did not know -- which also skipped enqueueing its successors,
so the walk stopped there and everything downstream went unverified while
`nvm_verify` still returned ok. The entire portable ISA was affected. This
verified clean:

    PUSH_I64 1
    I64_ADD      ; no declared effect -> the walk stops here
    POP
    POP
    POP          ; underflows a one-deep stack
    RET

Absence of data was indistinguishable from proof. Fixing it exposed a second
bug in the same walk -- branch successors resolved an absolute CODE offset
against a per-function index table, so only function 0, which starts at offset
0, ever worked -- and then eight latent codegen bugs that the truncated walk
had been hiding, each of which would have trapped at run time on the path the
verifier found. All are fixed here, and `make test-verify-all-programs` now
verifies every program in `tests/` so a program that compiles but does not
verify cannot sit unnoticed again.

Four reference-ownership leaks, each with a correct counterpart nearby to
compare against: `vm_array_remove` dropped an element without releasing it
while `vm_array_set` released what it overwrote; `CALL_INDIRECT` popped the
callable and nothing ever released it, so every closure call kept its closure
alive forever; the `OP_CALL_EXTERN` trap never released its arguments; and
`marshal_result` pushed each string element of a returned array without
releasing, though `vm_array_push` retains.

The benchmark suite could not detect an interpreter change of any size. It
timed one process per sample, so every workload took about 17 ms whether it
retired 78 instructions or 32,082 -- startup was the whole measurement and
execution under one percent of it. Every optimization decision it existed to
inform was still open, which is what that looks like from the outside.

And the smaller ones:

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

Every item listed here in earlier drafts is now closed. What remains is
genuinely outside 4.0, or is tracked with an acceptance test.

**Deliberately later.** Module signing is 5.0 work. The mechanism is nearly
free, but deciding where verification keys come from and who may issue them
belongs with the capability and policy work rather than ahead of it. LLVM and
WebAssembly return only as NanoISA translators, which is Phase 14.

**Declined on evidence, not deferred.** Computed-goto dispatch, private
superinstructions, split payload/tag operand stacks and trap stack ranges are
all measured and rejected in
[NANOISA_MEASUREMENTS.md](NANOISA_MEASUREMENTS.md). Three of them were
conditioned on measurement in the first place; the measurement now exists, and
each entry records what a future prototype would have to beat. The portable
`switch` remains the only dispatch strategy, which is the fallback the roadmap
asks for rather than a gap.

**Known and tracked.** Three programs in `tests/` contain a transitively
imported function whose float arithmetic lowers to integer opcodes, because
codegen resolves an imported function's parameter names against the main
program's symbol table by comparing line numbers across different files
(issue #223). The verifier is right and the bytecode is wrong: the VM traps on
`I64_ADD` with float operands, so those functions would fail if called. They
are on the `test-verify-all-programs` allowlist, and removing them from it is
the acceptance test for the fix. `Makefile.gnu` also tracks no header
dependencies, so a struct change makes an incremental build untrustworthy
(issue #211); CI always builds clean, which is why it only ever bites someone
working locally.

Nothing on this list blocks the release. The first group is scope, the second
is a decision with evidence behind it, and the third is defects that are
visible, reproducible and bounded -- which is the difference between a known
issue and an unknown one.

## Links

- [Current README](https://github.com/jordanhubbard/nanolang/blob/main/README.md)
- [4.0 changelog entry](../CHANGELOG.md#unreleased)
- [Roadmap](ROADMAP.md)
- [3.5 release presentation](RELEASE_3.5.md)
- [NanoISA v2 module format specification](superpowers/specs/2026-09-01-nanoisa-v2-module-format.md)
