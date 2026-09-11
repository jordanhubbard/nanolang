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

I require Docker, curl, tar, and shasum on the host. The script downloads Sail
0.20.2's Linux x86-64 binary release, verifies its published SHA-256, and mounts
it and my model read-only into the already pinned proof container. Sail's
bundled Z3 must be on PATH even for `--version`. I typecheck the model, generate
C, compile that C with GMP and the container's versioned zlib runtime, and run
seven smoke assertions. I remove the downloaded toolchain afterward.

Verified on 2026-09-11: Sail reported
`0.20.2 (sail2 @ 3b7af38d66466ecadad563158b07ce2f82fe05da)`; typechecking,
C generation, compilation, and smoke execution passed. The checks exercise
stack-underflow cases, swap order, truncated constants, unsupported extended
opcodes, and little-endian decoding with trailing bytes preserved.

I do not treat `None` for an unsupported opcode as proof that the opcode is
invalid in NanoISA. Opcode constants are manually copied in this first trial;
schema agreement and differential tests against `isa_decode` and NanoVM remain
mandatory before adoption. No Lean, Rocq, Isabelle, or HOL4 export has been
validated for this model yet.

My [tooling decision](../../docs/FORMAL_TOOLING_DECISION.md) records what I
retain, what I am testing, and what would justify extending this experiment.
