# How I Optimize NanoISA

I do not accept a NanoISA optimization because one timing looked smaller. I
accept it when the behavior remains the same and repeated measurements show a
useful improvement across maintained workloads.

## Evidence

Run:

```sh
make benchmark-nanoisa
```

I record 20 samples by default. Set `NANOISA_BENCH_RUNS` to change that count.
Each run writes instruction, branch, call, stack, heap, trap, and FFI counters.
The summary records the commit, operating system, architecture, compiler,
median, interquartile range, retired instructions, and median nanoseconds per
retired instruction.

The suite includes recursion, the current Forth interpreter, arrays, strings,
hash maps, direct calls, indirect calls, and FFI. Forth compilation, compiled
Forth words, and Forth exceptions join the suite when that runtime exists.

## Acceptance

Before and after results must use the same machine, compiler, build flags,
workloads, and sample count. Every result must pass the relevant unit,
integration, example, and semantic-equivalence tests.

I accept a general execution optimization when:

- its median improvement is larger than the observed run-to-run noise;
- it improves or preserves both NanoLang and Forth workload groups;
- no maintained workload regresses by more than 5% without a documented
  reason and an explicit tradeoff decision;
- code size, allocated bytes, retained values, and FFI traffic do not hide the
  cost somewhere else;
- the simpler implementation wins when performance is statistically tied.

I accept a private superinstruction only when it accounts for at least 1% of
retired baseline instructions in a maintained workload and survives a second
measurement after other lowering waste is removed.

These are tested engineering criteria, not proof of performance on every
machine.

## Private superinstructions

A private superinstruction is a dispatch-only fusion: it collapses a short run
of already-verified instructions into one step in the optimized dispatch IR
(representation 3 in `src/nanovm/vm_dispatch.h`). It is never a portable
opcode. It does not appear in the serialized bytecode or the verified
instruction IR, it is not assigned an ISA opcode byte, and it never exposes
frontend bookkeeping to the portable program. The verified IR still owns
program meaning; a superinstruction only fuses steps the verifier already
proved safe and preserves the byte-addressed instruction pointer so frames,
returns, and traps behave exactly as they do on the unfused stream.

Fusion is profile-selected. `VmDispatchProfile` carries one opt-in flag per
candidate fusion, and every flag defaults off, so an unconfigured VM executes
the plain verified stream. `vm_dispatch_profile_none()` selects nothing;
`vm_dispatch_profile_all()` selects every implemented candidate and exists for
measurement and tests, not as a shipping default. A build enables a fusion by
default only after it satisfies the acceptance criteria below. Profiles are
applied when the dispatch IR is projected (`vm_dispatch_build_module`), so the
same portable module runs fused or unfused depending only on the profile.

I accept a private superinstruction only when it accounts for at least 1% of
retired baseline instructions in a maintained workload, survives a second
measurement after other lowering waste is removed, and passes the same
semantic-equivalence tests as the unfused stream (the fused and unfused
projections must return identical results). Evaluating and accepting specific
fusions — local-field load, local increment, compare-branch, union-tag branch,
and tail-call — remains ongoing roadmap work; the mechanism here keeps each one
private and opt-in until its workload justifies it.
