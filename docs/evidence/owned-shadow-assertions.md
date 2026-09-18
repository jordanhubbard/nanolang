# My owned-profile shadow assertions

At source `dc2d6284` (based on main `8ae08368` including multi-parameter
caller references), I admit `ASSERT` only with an exact verified Boolean in
my owned/reference profile. The existing ordinary ASSERT path is unchanged.
A true condition is consumed without changing reference authority. A false
condition reports assertion failure, not successful completion.

My native helper propagates its failure status through caller cleanup. It
does not abort before the caller releases its actual owner roots. Direct VM
owned calls now unwind their activation's stack and callable roots on failure;
the invocation wrapper already supplied that cleanup, but direct execution
also needs it. Both reference contexts are cleared by the terminal path.
Successful trap suspension retains them until normal resumption.

`make test-owned-assertions` passes 959 lifecycle checks and five paired
artifacts. I exercise true and false entry/helper assertions through
`vm_invoke`, `vm_execute`, `vm_call_function` and `vm_invoke_callable`, repeat each path eight times,
and check frame/stack/reference state and the heap-object baseline. A true
helper assertion also executes after yielding and relocating VM stack storage.
A false helper assertion after 1,200 NOPs uses an installed callback runtime
to exercise actual yielding through the host loop before failure. The core
ASSERT trap keeps contexts live until the host evaluates it; a false condition
then reaches the public call unwind, or the callable API's existing separate
context/stack/frame cleanup. Both passing and failing serialized artifacts go through normal VM execution
and `nano_vm --check-shadows`. Generated native code preserves failure through
helper/caller cleanup, leaves the output result untouched on failure and frees
all counted owner allocations on twenty repeated calls.

I ran the focused test with VM, heap, value, affine state/analysis, verifier and
native translator sources instrumented using `-fsanitize=address,undefined
-fno-omit-frame-pointer -O0 -g`, with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`.
All 959 checks and paired artifacts pass. Generated native harnesses also pass
GCC and Ubuntu Clang 18 ASan/UBSan with leak detection. My ambient
`/usr/local/bin/clang` refused compilation on its GCC-installation-selection
warning under `-Werror`; I retain that diagnostic and explicitly used
`/usr/bin/clang`, rather than suppressing a warning or blaming source code.

My adjacent gate passes:

- 274,416 VM checks;
- 2,414 native translator and 1,092 shape checks;
- 1,548 caller-reference and 2,004 multi-caller checks;
- 1,051 owned-runtime checks;
- the existing owned/caller allocation and atomic binding controls.

Logs are `/tmp/nanolang-owned-assertions-four-api.log`,
`/tmp/nanolang-owned-assertions-sanitizers-final.log`,
`/tmp/nanolang-owned-assertions-clang.log`,
`/tmp/nanolang-owned-assertions-clang18-final.log` and
`/tmp/nanolang-owned-assertions-adjacent.log`.

This completes bounded prerequisite `task_f259c8fa53c945e6a990f112dc9415c1`.
My source producer child `task_5057848888b246f686fd2b8e48d2c19a` remains open:
neither source borrow guard is removed, and retained metadata alone does not
establish source admission. [My source contract](../NANOISA_SOURCE_BORROWS.md)
retains full selected shadows or explicitly refuses an unsupported graph.
