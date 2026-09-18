# My owned and void value-result qualification

I track task_a2797653878d4acba42423a689909ebf under the unchanged c435 affine
example blocker. My [contract](../NANOISA_OWNED_VALUE_RESULTS.md) precedes code
at5bfe52cd. My descriptor checkpoint4a135aa8 passes502 ownership-v1/v2 result
query, binary conversion and canonical-text roundtrip checks, plus314 ordinary
and343 allocation affine-state checks. The existing wire format carries exact
result tags/counts/layouts; transport alone grants no execution authority.

Reviewed productione8d80f99 adds exact value-graph result admission, owned/void
CALL effects, live VM return layout checks before detach, and native pending
result carriers with constructor layout identity. Entry0 stays scalar and the
borrowed CALL_REF profile stays separate. My fixtures cover factories,
conditional and nested forwarding returns, interleaved scalar/resource inputs,
VOID consumers, eight active frames, same-shaped distinct nominal types,
Boolean/byte resource leaves, caller holds across returned-owner allocation,
and failures before and after returned-owner publication. All four public VM
APIs run twice; ten static refusal controls are never executed.

My first harness build stops at a misleading-indentation warning; I separate
its test statements and retain /tmp/nanolang-owned-value-results-focused.log.
The corrected expanded fixture pinfc379453 passes1552 lifecycle checks,
3485 heap-fault checks across312 budgets with272 actual injected failures, and
117 return-facts preflight/reentry checks. Native strict/generated GCC checks
include repeated entry, owner allocation sweeps and ASan/UBSan/LSan cleanup.
The frozen all-ten-case instrumented corpus passes47.044 seconds at that pin;
its logs remain /tmp/nanolang-owned-results-sanitizer-parts. This does not
substitute for the adjacent borrowed gates, which reveal the following issue.

My first full adjacent run adds return-facts allocations to unchanged scalar
CALL_REF paths and reaches two existing allocation-test ceilings. I retain
/tmp/nanolang-owned-value-results-full.log and record
 task_af6dac9228e54700b98a72d852db8969 before correcting code. Reviewed correction
3bf82a22 queries layout facts only for STRUCT results. Existing scalar/VOID
count/tag checks, capacity preflight and context cleanup remain. I do not raise
any allocation ceiling. The corrected full log
/tmp/nanolang-owned-value-results-corrected-full.log passes the original55 caller
and93 multi-caller allocation checks, all new controls, existing scalar graph,
affine, borrowed, consuming and assertion gates,274541 VM,2422 native,1365 shape,
96 verifier and28 VM-FFI checks.

My fresh corrected instrumented build at
3bf82a223ce8f2260485486d55519263e119eefb passes all ten cases/all four APIs/two
repetitions in46.747 seconds. I compile every linked NanoVM/NanoISA object and
the driver with ASan/UBSan atO2, enable leak checking, and retain ordinary common
legacy compiler/runtime objects. At most two cases run concurrently, each with
a300-second bound. No sanitizer finding occurs. Executable SHA256 is
216c3e3996ead3a8f14b243f4f9f395fb0c2fb461c77ce9ed0e64a1bed88e075.
I retain /tmp/nanolang-owned-results-corrected-sanitizers-build and per-case
logs/status/summary under /tmp/nanolang-owned-results-corrected-sanitizer-parts.

A separate initial Clang qualification is incomplete: case9 receives errno13
Permission denied when starting bin/nano_vm, before executing that case. The
harness overlapped a Make run that relinked the same tool; it was not an
immutable-tool qualification. I retain /tmp/nanolang-owned-value-results-clang.log
and task_26c8ef718fbf4cb7834412950f4a510d without inferring the exact permission
cause or a runtime defect. A fresh sequential hash-pinned Clang gate follows
final integration. A later pass will not relabel or explain the old incident.

Full source admission, string/PRINT effects and the unchanged affine example
remain open. Final canonical integration qualification follows this checkpoint.
