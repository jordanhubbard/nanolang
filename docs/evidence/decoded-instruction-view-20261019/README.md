# I borrow my current decoded instruction

I base this isolated `perf/5.1-decoded-instruction-view` slice on `103d7e83c8482010c7f080ddc6369b9ff6ff496d`, and record [ownership before source edits](https://github.com/jordanhubbard/nanolang/issues/976#issuecomment-6262858540). My only production change replaces the main dispatch loop's DecodedInstruction value copy with a const pointer. The inverse mechanical transformation reproduces the exact base file; I retain that check in `mechanical-check.json`. Independent review found no blocker.

I borrow only after the existing module validity, function, cursor and byte-offset checks. I reacquire the pointer each iteration. FFI returns a value-only trap before host execution; queued callbacks run outside the core activation or after a yield. Tracing only reads and prints. Heap traversal does not invoke host finalizers. Rebuild/invalidate free cached instructions, so I do not retain this pointer across a trap or replace any existing validity check with an immutability assumption. Append-only function synchronization preserves existing instruction arrays but may relocate function tables; I preserve the current lookup behavior.

The retained ASan sample attributed 341/3604 stacks (9.46%) to the old copy, with 323 inside __asan_memcpy. This suggested a useful bounded candidate; it did not identify a complete explanation for the older Linux 7200-second Stage1 timeout. GC and cursor checks also cost time and remain unchanged.

## Completed Apple sanitizer A/B

I build fresh serial O3 ASan/UBSan objects in `obj/decoded-view-sanitized`. Baseline and candidate use the same object/link set except vm.o; I compile baseline vm.c directly from the exact base commit, and retain commands and binary hashes. Flags include `-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer`. I copy the same retained 4cfad compiler seed into each experiment, retain source hashes, and alternate A/B/B/A/A/B. Each invocation has a 180-second bound. Every completed compiler output is byte-identical within its experiment and passes original VM verification and program execution. ASAN_OPTIONS disables unsupported macOS leak checking only; no ASan/UBSan/ODR check is suppressed.

The no-import parser fixture completes six runs in about one second each and shows no reliable benefit; I retain `ab-small.json` and make no speedup claim for it. The existing `tests/token_value_bytes.nano` fixture includes the compiler lexer/diagnostics import closure. Its six completed runs give:

| Variant | CPU seconds, three runs | Median CPU | Median wall |
| --- | --- | --- | --- |
| Baseline | 5.643129, 5.628684, 5.626681 | 5.628684 | 5.680804 |
| Const view | 5.154446, 5.043574, 5.001745 | 5.043574 | 5.091732 |

I measure a 10.40% median CPU reduction and 10.37% median wall reduction for this bounded workload under existing worker contention. I retain the source closure inventory and verify it unchanged after the run. Native self-rebuild and the frozen full instrumented generation were independent active jobs. These measurements do not establish full-bootstrap timing, Linux performance or that the 7200-second timeout is solved.

## Lifetime gates and retained failures

Baseline and candidate Apple-instrumented VM suites each pass 274654 assertions. Callback lifecycle (40000 cross-thread invocations), callback allocation/cancellation, retained dispatch/allocation, substring, heap allocation and stack allocation checks pass. Switch/threaded ordinary admission checks pass with real VM execution and explicitly modeled host hooks, including nested invocation followed by module rebuild or function append. Actual VM FFI active native code mutation and scoped service controls pass before the separate retained-image failure below.

Both baseline and candidate originally overflow the O3-inlined test_vm main stack before the first VM call. I retain both failures and use identical `-fno-inline-functions` for the test_vm/test_vm_ffi harness translation units; linked runtime objects remain O3 instrumented. The ordinary observer harness includes vm.c and deliberately recompiles that TU at O1; it also needs no-inline to avoid its main-entry stack overflow, with ASan/UBSan still enabled. These are test construction boundaries, not pointer-runtime failures.

The full Apple FFI recipe remains blocked. Its unchanged artifact loader, which links no VM object, aborts on reload2 of the same first fixture/global/address with ASan ODR registration. Baseline and candidate direct VM FFI runs reproduce the same retained-image failure. This occurs before the simultaneous-handle part, and is not demonstrated collision between first and second libraries. The integration owner's independent exact-source experiments reproduce failure with Apple Clang21 both with image registration and globals-dead-stripping disabled; consistent Homebrew Clang23.1.2 passes all reloads and simultaneous-handle checks without suppression. I retain all three outcomes, exact commands, source hashes, compiler version and logs in `artifact-registration/`. This establishes toolchain-specific behavior, not an identified LLVM/runtime source root cause. No product workaround or sanitizer suppression is applied.

## Consistent Homebrew qualification

I rebuild all runtime/dependency objects separately in `obj/decoded-view-homebrew` with Homebrew Clang23.1.2 and the same O3 ASan/UBSan/no-recover flags. I do not mix Apple and Homebrew sanitizer runtimes or object sets. `providers.json` records dynamic sanitizer-library provenance. The complete scoped lifetime command sequence passes: 274654 VM assertions; all 30 actual VM FFI tests including active native mutation/reentry, retained-image reloads and ABI checks; callback lifecycle/allocation/cancellation; retained dispatch/allocation; heap/stack/substring allocation contracts; and switch/threaded ordinary admission. I retain `homebrew-lifetime-commands.sh`, `homebrew-lifetime-tests.log`, and the observer's source/command/status receipts. The same harness-only no-inline adjustments described above apply; all sanitizer checks and assertions stay enabled.

These results qualify this narrow pointer change locally. The Apple-specific retained-image ODR failure remains an explicit gate limitation, and the active full-generation experiment and fresh aggregate/platform checks remain the integration owner's work. I do not claim release readiness or closure of full #976.
