# Linux sanitizer toolchain investigation

I investigate `task_73af686a85cf4585a3766ed22a07d361` after the GCC 13.3.0 full compiler shadows exceed the unchanged 60-second deadline. The preceding [planner/profile evidence](../linux-planner-and-shadow-profile/README.md) preserves the baseline, rejected constructor optimization and source hashes. I do not change production interpreter code or CI in this checkpoint.

## Controlled diagnosis

My named-local constructor assembly lacks saved fake-stack flag cleanup. [GCC upstream commit aac7bb773f7](https://github.com/gcc-mirror/gcc/commit/aac7bb773f7) repairs this code-generation omission on the GCC 13 branch. Its explanation matches the sampled allocation scans. A further diagnostic stack (`linux-profile-flags/stack-02.txt`) shows allocated flags and hint positions in the billions. Sampling is diagnostic, not a passing performance result.

I compile `fake_stack_probe.c` separately with GCC 13.3.0 and distribution Clang 18.1.3:

```sh
cc -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer fake_stack_probe.c -o fake_stack_gcc
clang -O0 -g -fsanitize=address,undefined -fno-omit-frame-pointer fake_stack_probe.c -o fake_stack_clang
ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1 ./fake_stack_gcc 100000
ASAN_OPTIONS=detect_leaks=0:detect_stack_use_after_return=1 ./fake_stack_clang 100000
```

Both non-error runs produce the expected sum. GCC takes 0.381191 seconds and Clang 0.001148 seconds in this local probe; this is not a whole-compiler speed claim. An extra `uar` argument intentionally dereferences the escaped local after return. Both compilers report `stack-use-after-return` after 100 calls. After 100,000 calls, GCC produces no ASan diagnosis and exits 224 from the invalid read, while Clang still reports the expected error. `fake-stack-canaries.json` and individual logs preserve all four outcomes. A nonzero exit alone is not counted as a detected canary.

## Full workload qualification remains required

I restore `src/env.c` to the committed implementation before building fresh private Clang ASan/UBSan objects. The first link fails on `__muloti4`; selecting Clang's matching compiler runtime with `--rtlib=compiler-rt` fixes that link. Both build terminals and tool identities are retained. I keep all other compile flags, shadows and the 60-second deadline, and explicitly enable `detect_stack_use_after_return=1`.

The first complete Clang attempt (`linux-shadow-clang.log`) exits 1 because the shadow child receives signal 9. The VM kernel (`linux-vm-kernel.log`) records the exact compiler child killed by the OOM killer in the isolated 4 GB machine, at approximately 3.4 GB child RSS. This establishes the local memory failure; it is not a timeout pass or a compiler correctness result. I retain it before increasing only the dedicated VM memory to 8 GB for a controlled retry.

The 8 GB retry (`linux-shadow-clang-8gb.log`) passes all 902 compiler shadows and publishes the native compiler, exit 0. Total command wall time is 76.72 seconds; the supervised shadow child retains its 60-second deadline. I use the original committed constructors, no sampling, and explicit use-after-return detection. This is local full-workload evidence, not hosted acceptance.

I then run the original GCC binary in that same 8 GB VM with the same explicit use-after-return setting, deadline and module cache. It still stops shadows after 60 seconds and exits 1 (`linux-shadow-gcc-8gb.log`, total command wall 89.02 seconds, maximum RSS 1,882,180 KB). Memory alone does not clear this failure. The standalone canary, emitted-code omission, sampled allocator state and matched-memory full workloads support qualifying a corrected sanitizer toolchain.

I have not yet adopted a sanitizer toolchain change. Final full-workload, hosted partition, ordinary GCC platform and canonical bytecode fixed-point gates remain required.
