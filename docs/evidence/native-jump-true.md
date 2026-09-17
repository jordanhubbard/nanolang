# My native true-branch contract

I track this repair in MAC `task_211f22859e164287a07a63cba74ace5b`.
My verified float-format bytecode uses `JMP_TRUE`; my previous native translator
refuses that opcode. I retain the emitted bytes rather than replacing them with
a different branch sequence.

I recognize both conditional opcodes in target discovery, local initialization,
classification and stack joins. Emission consumes the condition once, tests the
requested polarity, and performs the existing taken-edge parallel transfer.
Backward true edges retain the existing live-root publication and collection
safepoint. I do not change collection scheduling or ownership.

My shared condition helper follows VM truthiness for represented values:
zero integers, signed float zeros and void tagged globals are false; nonzero
numbers and NaN are true. Strings and aggregate handles test existence, not
length. My by-value record represents a constructed heap object and is true.
This also admits the same direct handle conditions for `JMP_FALSE`, rather than
assigning inconsistent truthiness to opposite branch polarities. Unsupported
bytecode representations remain refused.

## Validation

Four focused methods pass in 0.426 seconds under ASan/UBSan with leak detection. Seventeen
condition values select exactly one observable branch effect under each of the
two branch polarities (34 cases). A taken branch
transfers live integer/string stack values, and a 1000-iteration backward branch
preserves a retained global string while consuming newly owned string
conditions. Three malformed branches preserve a prior artifact. Skipping a record
initialization still yields void in VM and retains my native optional-record
storage refusal; I do not invent a constructed record on that path.

I also execute both retained actual compiler-produced formatting artifacts,
`/tmp/nanolang-format-seed.nvm` and `/tmp/nanolang-format-self.nvm`, in VM and
sanitized standalone native products. Their exact output is `inf`, `-inf`, and
`nan`, on separate lines. Their internal assertions retain whole-float suffix,
precision, exponent and once-only operand behavior. Commands and results are
in `/tmp/nanolang-jump-true-format-artifacts.log`; focused output is
`/tmp/nanolang-jump-true-tests-initialization.log`.

My adjacent suite passes 2390 native checks and 1092 shape checks, plus
classifier/emitter opcode coverage and sanitizer-driver checks. I retain its
output in `/tmp/nanolang-jump-true-native-regressions.log`. This branch companion does not claim full
native compiler convergence or completion of the release contract.
