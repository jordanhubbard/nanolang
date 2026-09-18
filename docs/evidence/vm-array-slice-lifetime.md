# My checked VM array-slice boundary

I applied the defensive five-line handler change before running new fault
controls. ARR_SLICE now releases popped start/end owners after their last use,
including receiver-type refusal, and checks the heap result before publishing
an array. A failed allocation reports VM_ERR_MEMORY after operand cleanup.
My endpoint fallback, uint32 conversion, clipping, declared storage and shallow
copy semantics are unchanged. No managed opcode/profile is added here.

On production/test pin a70124ef I passed the corrected ordinary lifecycle gate
and full VM suite: 274493 checks, zero failures, plus checked heap/stack
allocation recovery. Clang ASan/UBSan passed the focused fixture with both
computed-goto and switch dispatch. I used the explicit native GCC installation
flag and leak detection. This is Linux evidence, not a Darwin claim.

The new controls cover packed int and exact negative-zero float bits, boxed
NUL-bearing string children, lengths/capacities, empty/reversed/clipped/wrapped
endpoints, noninteger heap-bearing fallback bounds, aliased bound arguments,
receiver-type failure, both header/buffer allocation failures, clean frames,
unchanged input references/accounting, successful reentry, and a copied child
surviving source-array teardown.

My first corrected-code fixture stopped at its final object-count assertion.
Static inspection confirmed the existing suspect buffer defers destruction of
zero-ref arrays. I retained the original log and added explicit cycle-collector
draining at teardown, keeping the exact count and reference assertions. I did
not rerun the pre-fix handler or weaken a product assertion. The corrected
ordinary and sanitizer gates then passed.

Logs are `/tmp/nanolang-vm-array-slice-normal.log` (first fixture assertion),
`/tmp/nanolang-vm-array-slice-corrected.log`,
`/tmp/nanolang-vm-array-slice-sanitized.log`, and
`/tmp/nanolang-vm-array-slice-switch.log`. Task1e89 is bounded by this prerequisite;
managed literal/slice childb702, source convention efa11, aggregate488 and
managed51da remain open.

My clean restack onto main53b43377 (PR712/714) retains identical VM and fixture
files. The focused ordinary gate passed again on801df526; its log is
`/tmp/nanolang-vm-array-slice-integrated.log`. I did not repeat the unrelated
compiler bootstrap or claim full managed target acceptance.
