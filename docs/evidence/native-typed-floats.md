# My native typed float boundary

I track this scalar repair in MAC `task_fd4c63cf9f3e46f09ece380ce00c7a58`.
The paired scalar compiler fixture produces verified bytecode that executes in
my VM, but my previous native translator refuses `F64_NEG` at function 1,
offset 3. I preserve that bytecode rather than replacing typed operations with
generic arithmetic.

I lower all eleven typed F64 operations. I require float operands, return bool
tags from comparisons, and preserve IEEE double bits through local, function,
tail and tagged-global transport. I copy float bits into the existing tagged
integer payload with `memcpy`; no pointer punning or heap allocation is needed.
A typed consumer of a tagged global checks its tag before decoding those bits.
Division by either signed zero returns positive zero, matching my VM.

My `CAST_STRING` uses the VM's `%g` conversion and the existing owned string
allocator. Printing is a separate contract: whole floats within the VM's
bounded range retain one decimal place. I check the range before the integer
cast used to detect whole values. Generic comparisons and truthiness of tagged
floats use decoded numeric values, including negative zero. Generic ordering
retains `val_compare`: unordered NaN compares as zero, so generic `LE`/`GE`
return true. Typed F64 comparisons retain IEEE unordered behavior, and equality
remains false for NaN. My direct and tagged tests distinguish these contracts. I leave typed
integer-to-float coercion refused.

I retain a separate explicit float-to-int gap in
`task_b927827f37734658bce360d7ecf913aa`. Static float `CAST_INT` already refuses;
its tagged fallback previously returned zero. I now refuse that tagged case
instead of publishing a wrong result. My regression executes the finite case
successfully in VM and preserves the native refusal. NaN, infinity and range
policy need a checked cross-runtime conversion contract before that followup.
Float arrays and record fields are outside this scalar repair.

## Evidence

`tests/test_native_floats.py` exercises all typed operations and result tags,
NaN/infinity, signed zero, both zero divisors, local/global/call/return transport,
self and cross-function tail calls, exact printing, generic tagged comparisons,
and fresh string ownership under ASan/UBSan with leak detection. It checks all
eleven operations against three dynamically tagged nonfloat inputs in VM and
native execution. Static malformed float assembly preserves a prior output.

The retained real C-seed and selfhost artifacts
`/tmp/nanolang-float-seed.nvm` and `/tmp/nanolang-float-self.nvm` both execute in
VM and compile/run as standalone native programs with ASan, UBSan and leak
detection. They come from the emitter owner's current `scalar_floats.nano`
fixture, including a direct string-format return. Commands/results are retained
in `/tmp/nanolang-native-f64-paired-artifacts.log`. I leave source builtin
formatting parity and `JMP_TRUE` support on their independently recorded tasks.

My four-method focused matrix passes in 55.531 seconds. The additional direct
and tagged NaN contract extension passes with ASan/UBSan/LSan in 0.435 seconds.
My adjacent native suite passes 2390 checks and the shape solver passes 1092;
classifier/emitter opcode coverage and sanitizer-driver tests also pass. Logs:
`/tmp/nanolang-native-f64-tests-final.log`,
`/tmp/nanolang-native-f64-nan-parity.log`, and
`/tmp/nanolang-native-f64-regressions-final.log`. I do not infer complete native
self-compilation or release readiness from this scalar gate.
