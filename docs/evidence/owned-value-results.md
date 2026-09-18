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

At code checkpoint 497dd96d0f039c9867bcb9087c4a1b33854b99bc, Darwin
selects `detect_leaks=0`; every other platform retains `detect_leaks=1`.
ASan/UBSan, `halt_on_error=1` and every generated `live==0` assertion remain.
The exact Darwin focused target passes 502 descriptor, 1552 lifecycle, 3485
allocation checks across 312 budgets and 272 actual injected failures, 117
return-preflight checks and all ten generated-native cases in 6 seconds. I
retain /private/tmp/nanolang-owned-result-lsan-497dd96d-darwin.log with SHA256
64171f8c08128a39b29eef0bfebdb3bf83f9a360deb62dba50a0173b26a85987.

This portability correction changes only test policy and documentation; the
qualified owned-result production remains unchanged. Full source admission,
string/PRINT effects and the unchanged affine example remain open.
