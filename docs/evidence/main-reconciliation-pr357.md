# Main PR #357 comparison integration

I reconcile main `7b982db3` with release `4f81e5eb`. I retain the release's
boolean-array storage, checked call formatting, tagged locals and globals,
and generic comparison ordering. I give incoming float storage its own kind
and shape, distinct from boolean arrays and integers.

I preserve incoming float constants and numeric comparisons, including mixed
integer/float operands. Locals, stack duplication, joins and direct/tail-call
argument transport use double storage. I emit constants from their exact bits
so infinity, NaN and signed zero remain valid C inputs. Numeric ordering uses
my VM's three-way comparison, including its zero result for unordered NaN;
equality retains ordinary floating equality. Different existing scalar tags
retain the VM's tag order rather than the incoming compile-time rejection.

I adapt incoming fixtures to cast comparison booleans explicitly before an
integer return. My added executable regression covers float locals, duplicate
values, direct/tail-call arguments, both mixed numeric directions, infinity,
NaN ordering and equality. A shape regression rejects float/integer unification.

`make -j8 test-nvm2c` passes 1,761 translator checks and 1,076 shape checks on
Linux arm64 with strict GCC. Opcode coverage and sanitizer-driver prerequisites
also pass. Log: `/tmp/nanolang-main357-aot-final.log`.

This is bounded comparison and operand-transport support. It does not establish
float arithmetic, float function results, tagged float globals, float arrays,
or a complete native compiler bootstrap. Full release gates remain separate.
