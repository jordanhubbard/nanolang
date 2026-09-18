# My referenced native branch labels

I record `task_3d5492ef63924b4d91a11364a80fde83` before implementation.
The unchanged range fixture at source `553a57bb` reaches strict GCC unused
labels in generated `nl_bounds`; `/tmp/nanolang-range-arity-paired.log`
retains the refusal before native binary execution.

The emitter prescan marks targets of every decoded jump, while terminated
unreached instructions are later skipped. A target with only a skipped
incoming jump may still be visited by fallthrough and emit an unused label.
I preserve the existing target/stack-join/termination algorithm. During body
emission I retain byte offsets and lengths of generated label prefixes, and
mark destinations of every actually emitted PC goto, including conditional,
forward, backward and end-of-body edges. After emission and before declaration
insertion, I blank only unreferenced prefixes, retaining their statement and
newline. Offsets, not pointers, survive output-buffer growth.

I add no false jumps, warning suppression or new control-flow reachability
rules. Every allocation is checked and released on success/refusal. I require
ordinary zero-trip/range, live backward/conditional joins, skipped terminal
following JMP controls, strict GCC/Clang execution and adjacent native/shape
checks. Historical compiler crashes and their artifacts remain excluded.

The first focused run passes forward/backward and end-of-body controls and
executes the growth case successfully, but its test-only length assertion
expected more than 16,000 bytes while the output is 10,370. I retain
`/tmp/nanolang-native-label-gcc.log`; I correct the assertion to the actual
8,192-byte growth boundary without changing production or fixture operations.

My first complete native run reports 2,421 passes and one failure in the
old unsupported-classifier list, which still requires ROT3 to be unsupported
after merged PR699 admitted exact INT/BOOL rotation. I retain
`/tmp/nanolang-native-label-full.log` and record test-maintenance child
`task_71ebb5716eb747daa10d66d0467bd612` before the correction. I replace only
that list member with still-unsupported ROLL, keeping the same instruction-local
refusal assertion and check count. ROT3's separate positive/other-tag/underflow
checks remain. The label production is unchanged; I require a new full result
rather than claiming the first run passed.

## My measured acceptance

Production `b9827aa7` passes four focused GCC methods in 0.815 seconds
with O0/O2 ASan/UBSan execution, and eight focused/adjacent Clang methods
in 1.830 seconds. The unchanged range fixture and invalid-operand method
pass GCC in 1.997 seconds and Clang in 1.713 seconds. I use stable range
source `37888716` (production `553a57bb`) and redirect only its native
translator command to this corrected source; the adapter edits no source.
Producer/translator before-after hashes, both modules and generated C are
retained in [my manifest](evidence/native-referenced-labels.json).

After the separately recorded stale-control correction `248fd407`, my full
native gate passes 2,422 checks and the shape gate passes 1,365 checks.
I preserve the first 2,421/1 outcome, do not reinterpret it as a pass, and
verify label production plus range-tool hashes unchanged afterward.
This is not a new compiler bootstrap or completion of the root-owned range
source contract. That lane will integrate my canonical translator separately.

I integrated main `c9e2a16f` additively after qualification. Native translator,
shape solver and focused/full native test sources are unchanged; the new
managed eligibility and specialized borrow-source work is separate. I retain
the original measured source/tool pins rather than claiming rebuilt-main
product acceptance.
