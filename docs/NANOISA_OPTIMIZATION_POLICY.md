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
