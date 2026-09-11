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

I require Docker, curl, tar, shasum, a C compiler, and Python with PyYAML on the
host. The script checks model/schema agreement and compiles the actual
`src/nanoisa/isa.c` with a small test bridge. It downloads Sail
0.20.2's Linux x86-64 binary release, verifies its published SHA-256, and mounts
it and my model read-only into the already pinned proof container. Sail's
bundled Z3 must be on PATH even for `--version`. I typecheck the model, generate
C, compile that C with GMP and the container's versioned zlib runtime, and run
seven smoke assertions, then runs a generated differential decoder corpus.
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

I do not treat `None` for an unsupported opcode as proof that the opcode is
invalid in NanoISA. Opcode constants remain manually declared, but their numbers
and operand counts are checked against `spec/nanoisa.yaml` before each run.
Differential VM execution, broader semantics and prover exports remain mandatory
before adoption. No Lean, Rocq, Isabelle, or HOL4 export has been
validated for this model yet.

My [tooling decision](../../docs/FORMAL_TOOLING_DECISION.md) records what I
retain, what I am testing, and what would justify extending this experiment.
