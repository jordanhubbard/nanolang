# My source float-text boundary

I retain #1082 under the complete #1081 JUnit work. Current `src/eval.c` appends
`.0` to whole finite floats in generic string conversion, while source producer
calls to `to_string`/`cast_string` used bare `CAST_STRING`. The original full
f-string fixture expects `x=0.0`; I preserve that assertion and the entire file.

Raw `CAST_STRING` has a different, independently tested reference contract:
`tests/test_managed_binary64_format.py` compares its bytes to native C `%g`,
including whole floats, signed zero and explicit signed nonfinite spellings.
I preserve that opcode, VM heap formatter, C AOT raw helper and shared formatter.
My [existing raw VM probe](existing-raw-vm-baseline.json) assembles and executes
all ten expected primitive cases successfully. These existing tools have recorded
hashes; this probe is not fresh-source or native lifecycle admission.

I extend the C producer's existing `float_to_string` lowering to typed float
operands of source `to_string`/`cast_string`. I share the self-hosted float-text
emitter between the same paths, with a mandatory shadow checking conversion,
conditional suffixing and once-only argument emission. Int/u8/bool/string paths
and actual declared-function calls retain their existing dispatch.

I add source/VM/C AOT checks for positive/negative zero, positive/negative whole
floats, fractional/scientific values, signed infinity/NaN and once-only calls.
A separate direct-assembly VM/C AOT test retains raw primitive expectations.
The existing canonical suite also executes the complete original f-string
fixture, so a focused conversion claim cannot hide another original failure.

My [existing local seed observation](existing-seed-observation.json) prints raw
`0` for generic conversion but `0.0` for explicit `float_to_string`, including a
bit-constructed zero. I retain that binary/source distinction: this binary is
not fresh current-source qualification. I do not use it to override the current
interpreter source contract or to claim the repaired producer passes.

My [report/storage controls](report-storage-controls.log) pass 14 tests; the
existing actual lexer accepts seven static fixture fragments, and Python/YAML
syntax checks pass. I require the actual default fast gate, a fresh hosted bootstrap, case-level canonical
results through all three producers, complete original four-route reports and
unchanged source/provider checks before native acceptance. Both issues remain
open. Earlier run 42413840812 continues on its immutable source checkpoint.
