# My explicit generic constructor acceptance

MAC `task_952ac994c2684adc8944cb2373670456` and prerequisite
`task_cede96d8189b42cea8f5692810031bb4`.
My contract is [explicit generic constructors](../NANOISA_EXPLICIT_GENERIC_CONSTRUCTORS.md).

I stacked this slice on frozen generic lowering `43b753df`. Parser checkpoint
`eadce2bf` passed a fresh bootstrap. Its first 15-method gate retained twelve
passing inherited methods and failures in the three new methods. Valid
selfhost explicit locals and returns had a different internal representation
from their declared types; C context propagation replaced explicit arguments
and accepted three zero-payload instance mismatches. I preserved the log and
did not execute those mismatched outputs.

I recorded the prerequisite before changing either checker. At `2d4bf874`,
known selfhost explicit constructors use the existing declared union type
representation. C compares existing concrete argument trees after resolving
the same union declaration and before contextual replacement. Omitted
arguments retain their existing inference. I do not change unknown-type
handling, wire identity or ownership authority.

The corrected fresh bootstrap passed. At test checkpoint `9825acc7`, all
15 paired methods pass in 45.629 seconds. The gate includes C-seed, Stage1,
Stage2, two raw selfhost emitters, selected shadows, VM execution and sanitized
native execution. Explicit int/string and nested array arguments, multiple
arguments, constructor calls/returns and ordinary lowercase comparisons pass.
All three zero-payload mismatches receive type/lowering refusals and preserve
old output across the two raw emitters and all three canonical producers.
The inherited methods retain resource-payload constructor checks, expression
matches, field order, lexical scope and mandatory shadow controls.

The adjacent wrapper for all 25 unchanged affine module/generic identity
methods also passes in 43.939 seconds. That separate check uses the external
native carrier prerequisite `9a45a788` via `NVM2C`, since its mixed-variant
contract is not yet integrated into this stacked branch. The 15-method syntax
gate above uses this branch's own translator.

Logs:

- `/tmp/nanolang-explicit-generic-bootstrap.log`
- `/tmp/nanolang-explicit-generic-paired.log`
- `/tmp/nanolang-explicit-generic-types-bootstrap.log`
- `/tmp/nanolang-explicit-generic-types-paired.log`
- `/tmp/nanolang-explicit-generic-affine.log`

Commands: `make bootstrap`, then
`make -j4 -o bootstrap nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump` and
`python3 -m unittest -v tests.test_explicit_generic_constructors`.

After PR666 and the native prerequisite reached canonical main, I cleanly
restacked at `655fc22c`. Range comparison confirmed unchanged parser, checker
and test patches. The integrated fresh bootstrap passed, followed by all
15 paired methods in 46.611 seconds using my own rebuilt tools. The log is
`/tmp/nanolang-explicit-generic-integrated.log`.
The four distinct generic/context methods also pass in 47.377 seconds with
my own translator, including the wrapper for all 25 unchanged affine
module/generic identity methods. The adjacent log is
`/tmp/nanolang-explicit-generic-integrated-adjacent.log`. No external tool
override is used for either integrated gate.

Task completion still requires this slice's canonical merge. Full product
and release qualification remain separate.
