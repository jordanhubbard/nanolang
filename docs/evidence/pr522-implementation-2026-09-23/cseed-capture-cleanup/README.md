# My C-seed command-capture cleanup

My direct C-seed capture helper previously returned unowned malloc buffers.
I now retain each buffer in a translation-unit owner and register an exit
handler that frees them. Returned strings remain valid across subsequent calls
and through the program lifetime. This is process-exit cleanup, not a claim of
bounded retention in a long-running process. VM and canonical native consumers
keep their existing lifetime contracts.

I serialize owner registration/insertion with an atomic flag. Allocation and
exit-handler-registration failures return an empty string and leave no owned
allocation. The helper retains the existing 65,535-byte capture limit.

Two tests pass with Homebrew LLVM ASan/UBSan/leak/UAR checks: an allocation-counted
C99 harness exercises four concurrent callers, preserved earlier results,
empty captures, allocation/registration failure, explicit cleanup and actual
exit cleanup; a real C-seed-generated NanoLang product preserves both captures
and passes leak checks. I keep stricter leak roots
`LSAN_OPTIONS=use_stacks=0:use_registers=0`; the counted exit callback additionally
requires zero outstanding buffers after the registered cleanup runs.

I rebuild the instrumented Stage 1 compiler and rerun its original hello smoke
with ordinary leak roots. It still fails, now with 19,213 bytes in 95 allocations.
The retained baseline reports 150,285 bytes in 97 allocations: the exact
131,072-byte/two-allocation reduction removes both capture buffers. Path results,
hashmap storage and parser lists still leak under
`task_02204077a4d34d4aa5ce20ef7054e113`. I have not qualified full instrumented
bootstrap or installed publication by this partial repair.

The complete `make test-transpiler` command passes, including its fresh ordinary
bootstrap, C-seed-absence smoke, StringBuilder boundary checks and both assertion
literal tests. I retain its full log.
