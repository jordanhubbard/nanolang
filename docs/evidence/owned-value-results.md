# My owned and void value-result qualification

I track task_a2797653878d4acba42423a689909ebf under the unchanged c435 affine
example blocker. My [contract](../NANOISA_OWNED_VALUE_RESULTS.md) precedes code
at 5bfe52cd. My descriptor checkpoint 4a135aa8 passes 502 ownership-v1/v2 result
query, binary conversion and canonical-text roundtrip checks, plus 314 ordinary
and 343 allocation affine-state checks. The existing wire format carries exact
result tags/counts/layouts; transport alone grants no execution authority.

Reviewed production e8d80f99 adds exact value-graph result admission, owned/void
CALL effects, live VM return layout checks before detach, and native pending
result carriers with constructor layout identity. Entry 0 stays scalar and the
borrowed CALL_REF profile stays separate. My fixtures cover factories,
conditional and nested forwarding returns, interleaved scalar/resource inputs,
VOID consumers, eight active frames, same-shaped distinct nominal types,
Boolean/byte resource leaves, caller holds across returned-owner allocation,
and failures before and after returned-owner publication. All four public VM
APIs run twice; ten static refusal controls are never executed.

My first harness build stops at a misleading-indentation warning; I separate
its test statements and retain /tmp/nanolang-owned-value-results-focused.log.
The corrected expanded fixture pin fc379453 passes 1552 lifecycle checks,
3485 heap-fault checks across 312 budgets with 272 actual injected failures, and
117 return-facts preflight/reentry checks. Native strict/generated GCC checks
include repeated entry, owner allocation sweeps and ASan/UBSan/LSan cleanup.
The frozen all-ten-case instrumented corpus passes 47.044 seconds at that pin;
its logs remain /tmp/nanolang-owned-results-sanitizer-parts. This does not
substitute for the adjacent borrowed gates, which reveal the following issue.

My first full adjacent run adds return-facts allocations to unchanged scalar
CALL_REF paths and reaches two existing allocation-test ceilings. I retain
/tmp/nanolang-owned-value-results-full.log and record
 task_af6dac9228e54700b98a72d852db8969 before correcting code. Reviewed correction
3bf82a22 queries layout facts only for STRUCT results. Existing scalar/VOID
count/tag checks, capacity preflight and context cleanup remain. I do not raise
any allocation ceiling. The corrected full log
/tmp/nanolang-owned-value-results-corrected-full.log passes the original 55 caller
and 93 multi-caller allocation checks, all new controls, existing scalar graph,
affine, borrowed, consuming and assertion gates, 274541 VM, 2422 native, 1365 shape,
96 verifier and 28 VM-FFI checks.

My fresh corrected instrumented build at
3bf82a223ce8f2260485486d55519263e119eefb passes all ten cases/all four APIs/two
repetitions in 46.747 seconds. I compile every linked NanoVM/NanoISA object and
the driver with ASan/UBSan at O2, enable leak checking, and retain ordinary common
legacy compiler/runtime objects. At most two cases run concurrently, each with
a 300-second bound. No sanitizer finding occurs. Executable SHA256 is
216c3e3996ead3a8f14b243f4f9f395fb0c2fb461c77ce9ed0e64a1bed88e075.
I retain /tmp/nanolang-owned-results-corrected-sanitizers-build and per-case
logs/status/summary under /tmp/nanolang-owned-results-corrected-sanitizer-parts.

A separate initial Clang qualification is incomplete: case 9 receives errno13
Permission denied when starting bin/nano_vm, before executing that case. The
harness overlapped a Make run that relinked the same tool; it was not an
immutable-tool qualification. I retain /tmp/nanolang-owned-value-results-clang.log
and task_26c8ef718fbf4cb7834412950f4a510d without inferring the exact permission
cause or a runtime defect. My fresh sequential hash-pinned Clang gate passes after final integration, as recorded below. A later pass will not relabel or explain the old incident.

My final integrated source is eb6e6b3a4b0c74550245b27059936e3edbf6720c,
including canonical f922299a. The reviewed result production is unchanged from
3bf82a22; integration also retains the independent incoming STRUCT_NEW allocation
failure guard. The full integrated log is
/tmp/nanolang-owned-value-results-final-integrated.log. It passes 502 descriptor,
1552 result lifecycle, 3485 heap-fault, 117 return-preflight, 1847 scalar graph,
338 graph-preflight, 529 invocation-proof and 69 verifier-reuse checks;
1548 caller, 2004 multi-caller, 959 assertion checks and unchanged 55/93 parameter
allocation controls; 274541 VM, 2422 native, 1365 shape and 96 verifier checks.
The VM-FFI target also passes. Generated-native GCC result qualification passes
in 7.003 seconds.

At that exact source, my fresh instrumented ten-case corpus passes in 46.753
seconds, with every case exercising all four APIs twice. The instrumentation,
300-second case bounds and maximum concurrency of two are unchanged. My binary
SHA256 is b60d75743d1bad4b124ca41f0f321b29c3769a62ef1060eb640d576e97a134fd.
I retain /tmp/nanolang-owned-results-final-sanitizers.log, the final build under
/tmp/nanolang-owned-results-final-sanitizers-build, and individual case statuses
plus summary under /tmp/nanolang-owned-results-final-sanitizer-parts.

After all tool rebuilds finish, my strict/generated Clang result gate passes in
2.554 seconds. I use the pinned wrapper selecting GCC 13's AArch64 installation.
Before/after SHA256 manifests match exactly for nano_vm, nvm2c, the result driver
and wrapper; I retain /tmp/nanolang-owned-results-final-clang-{before,after}.sha256
and /tmp/nanolang-owned-value-results-final-clang.log. This independent success
does not explain or relabel the historical errno13 incident.

My first independent Apple Clang run of the generated-native result harness at
PR738 head eb6e6b3a passes the descriptor, lifecycle, allocation and preflight
executables, then all ten generated-native cases abort before entry because
Apple ASan does not support `detect_leaks=1`. I preserve
/private/tmp/nanolang-pr738-eb6e6b3a-focused-darwin.log with SHA256
48c7c01722073d9afb1b6480f96146189d1ce54395f636d25f8b3669ca3430eb.
This is a harness-policy failure, not a runtime finding.

At code checkpoint 497dd96d0f039c9867bcb9087c4a1b33854b99bc, I first
selected `detect_leaks=0` from the Darwin host platform. The exact focused
target passes 502 descriptor, 1552 lifecycle, 3485 allocation checks across
312 budgets and 272 actual injected failures, 117 return-preflight checks and
all ten generated-native cases in 6 seconds. I retain
/private/tmp/nanolang-owned-result-lsan-497dd96d-darwin.log with SHA256
64171f8c08128a39b29eef0bfebdb3bf83f9a360deb62dba50a0173b26a85987. That
OS-wide fallback was too broad: Homebrew LLVM on Darwin supports LeakSanitizer.

At source dfbea5baf77ac96c8cec9afc0ed8d9ff2586fa4c, I select leak detection
from the compiler runtime identity instead. `/usr/bin/clang` reports Apple
Clang 21.0.0, so I use the narrow known-unsupported fallback
`detect_leaks=0`. The complete focused target passes in 10 seconds with the
same 502 descriptor, 1552 lifecycle, 3485 allocation, 312 budget, 272 injected
failure, 117 preflight and ten generated-native case counts. I retain
/private/tmp/nanolang-owned-result-apple-dfbea5ba.log with SHA256
b20a6571cf0db2f2004539aec8a6cc2e773474bf0b750d4eb21c47762600773c.

In a separate clean tree at the same source,
`/opt/homebrew/opt/llvm/bin/clang` reports Clang 23.1.1 and keeps
`detect_leaks=1`. The identical complete focused target passes in 13 seconds
with the same counts. I retain
/private/tmp/nanolang-owned-result-llvm23-dfbea5ba.log with SHA256
fed07eda1154c762622e1b623fed84ad0d9c248a946387320f76d952ee845d34.
ASan/UBSan, `halt_on_error=1` and every generated `live==0` assertion remain
enabled in both runs. I do not claim unchanged leak-sanitizer coverage: Apple
Clang uses the explicit zero-live-allocation boundary while Homebrew LLVM 23
also executes LeakSanitizer. The Homebrew LLVM 23 run is my required Darwin
qualification. The Apple Clang run is optional additional ASan/UBSan evidence;
it does not substitute for the required LeakSanitizer gate.

This portability correction changes only test policy and documentation; the
qualified owned-result production remains unchanged. Full source admission,
string/PRINT effects and the unchanged affine example remain open.

## My independent current-main LSan policy review

My final compiler-specific policy at2261d027 retains required Homebrew LLVM
leak checks and only the documented optional Apple-runtime fallback. Independent
review verified both retained Darwin report hashes and their ten generated
cases, explicit compiler identities and detect_leaks settings. The code at
qualifieddfbea5ba is unchanged in2261d027; only its evidence wording changed.

I integrated canonical main84bbc into a fresh Linux worktree at04df093a.
`make -j4 CC=cc test-owned-result-descriptors test-owned-value-results` passes
in18.751seconds with all tracked source/test hashes unchanged. The compiler
policy test and actual generated-native gate retain detect_leaks=1 on Linux.
I preserve the existing counts and zero-live-allocation assertions; no production
runtime changed for this policy. My [sealed reports](owned-result-lsan-current/report-sha256.json)
retain that integration and the hash-verified Darwin logs. This clears the
bounded original broad-OS-policy concern, not the separate product/release hold.
