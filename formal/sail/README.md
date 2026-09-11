# My Sail trial

I have a bounded, executable experiment, not a complete NanoISA specification.
`stack_slice.sail` decodes `NOP`, `PUSH_I64`, `DUP`, `POP`, and `SWAP` with
their current primary-opcode values. Constants use little-endian byte order.
The stack contains only 64-bit bitvectors; it does not model my tagged values,
heap, verifier, instruction pointer, traps, FFI, or service runtime.

Run from the repository root:

```sh
bash scripts/check_sail_container.sh
```

I require Docker, curl, tar, shasum, Python with PyYAML, and my normal native
VM build prerequisites on the host (including make and a C compiler).
The script checks model/schema agreement and compiles the actual
`src/nanoisa/isa.c` and my VM with small test bridges. It downloads Sail
0.20.2's Linux x86-64 binary release, verifies its published SHA-256, and mounts
it and my model read-only into the already pinned proof container. Sail's
bundled Z3 must be on PATH even for `--version`. I typecheck the model, generate
C, compile that C with GMP and the container's versioned zlib runtime, and run
seven smoke assertions, then run generated differential decoder and VM corpora.
I remove the downloaded toolchain and generated artifacts afterward.

Verified on 2026-09-11: Sail reported
`0.20.2 (sail2 @ 3b7af38d66466ecadad563158b07ce2f82fe05da)`; typechecking,
C generation, compilation, and smoke execution passed. The checks exercise
stack-underflow cases, swap order, truncated constants, unsupported extended
opcodes, and little-endian decoding with trailing bytes preserved.

Decoder agreement: Sail's generated C matched the production `isa_decode` on
1,524 deterministic byte sequences. I compare instruction identity, exact
64-bit payload, and the unconsumed suffix. The corpus includes every bit position
and its complement, signed-boundary bit patterns, seeded random constants,
truncated prefixes, trailing bytes, and unused primary opcodes. Valid opcodes
outside my five-instruction slice are excluded, not treated as invalid.

`python3 tests/test_sail_schema.py` passes nine tests, including rejection of
changed opcode numbers, operand widths/kinds, byte order, extension prefix,
stack effects, and a missing instruction. The schema check is deliberately
specific to this model's clause format; it is not a general Sail parser.

Execution agreement: Sail's generated C matched my production VM on 1,140
single-instruction integer stack cases, including 34 underflows. I construct
real modules with initial `PUSH_I64` instructions, the tested instruction,
and `HALT`, then call `vm_execute`. I compare the whole operand stack in
top-first order, exact integer bits, and success versus stack underflow.
I exercise depths zero through four, signed-boundary bit patterns, seeded
random values, and both zero-local and two-local frames. I do not bypass
verification or replace VM handlers in the bridge. The bridge also checks
that frame locals survive. HALT is test scaffolding, not modeled semantics.

This comparison exposed silent underflow in unverified `DUP`, `POP`, and
`SWAP`. I now trap before these operations touch a local or caller value;
`make test-nanovm` passes all 736 assertions, including explicit caller-stack
boundary regressions. `python3 tests/test_sail_vm_cases.py` passes three
corpus coverage tests. Other stack handlers still need the separate audit
recorded in my roadmap. These checks are not a general VM safety proof.

I do not treat `None` for an unsupported opcode as proof that the opcode is
invalid in NanoISA. Opcode constants remain manually declared, but their numbers
and operand counts are checked against `spec/nanoisa.yaml` before each run.
Broader execution semantics remain mandatory before adoption. My Rocq export
and eight bounded-model lemmas are checked below. I have not validated Lean,
Isabelle, or HOL4 exports for this model.

I can reproduce the Rocq export attempt independently of the native corpus:

```sh
bash scripts/check_sail_container.sh --rocq-export-only
```

The original decoder raised a Sail 0.20.2 internal rewriter error on its first
literal-byte list pattern: `Cannot infer type of: p0# :: rest`. Neither an
input annotation nor a whole-pattern annotation repaired it. I now separate
the opcode byte from the remaining list and match that byte independently.
On 2026-09-11 this representation successfully generates `stack_slice_types.v`
and `stack_slice.v`, including the actual `decode` and `execute` definitions.
I inspected those definitions; they import `SailStdpp.Base` and
`SailStdpp.Real`. Generation is not Rocq compilation or a checked ISA theorem.
The command propagates backend errors and explicitly labels generation-only
success.
The revised representation also passes the seven smoke assertions, all 1,524
production-decoder comparisons and all 1,140 VM comparisons (34 underflows).
The nine schema checks and three VM corpus tests pass. These are executable
agreement checks, not proofs of equivalence for all inputs.

Sail's [Rocq support library](https://github.com/rems-project/coq-sail) is a
separate dependency; my pinned NanoCore proof image does not contain it.
I reproduce the bounded model check with:

```sh
bash scripts/check_sail_container.sh --rocq-check
```

This mode installs `coq-sail-stdpp.0.20.2`, `coq-stdpp.1.12.0`, and
`coq-stdpp-bitvector.1.12.0`, with exact Coq compatibility-package versions,
inside the disposable container. It compiles the generated definitions and
`StackSliceProofs.v`, checks the eight named assumption reports with the same
failure-rejecting checker as NanoCore, and runs independent `coqchk`.
The original eight lemmas concern identity, push/pop and dup/pop cancellation, swap reversal,
and underflow in the generated model. They do not state VM refinement.
On 2026-09-11 a clean run of this command exited zero: the generated files and
eight lemmas compiled, all eight reports were closed under the global context,
the shared assumption checker accepted the complete named inventory, and
independent `coqchk` succeeded. This unchanged-runner execution supersedes an
earlier run whose outer shell failed after its script was edited while running.
The checked lemmas cover only the five-instruction integer-stack model. Decoder
correctness for all inputs and refinement to my C VM remain unproved.
I do not upgrade the model's proof claim because the C backend works.

I am checking a ninth lemma, `execute_frame_extension`: successful execution
on an operand stack preserves any suffix appended below it. The runner now
requires this named assumption report as well. The earlier eight-lemma run
does not validate this addition; its generated-model compilation and independent
check remain pending. The law does not cover failing instructions or my C VM.

My [tooling decision](../../docs/FORMAL_TOOLING_DECISION.md) records what I
retain, what I am testing, and what would justify extending this experiment.
