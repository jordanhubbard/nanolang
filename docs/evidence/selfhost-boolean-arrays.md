# My boolean-array lowering

I carry `array<bool>` through local and field types, literal inference,
construction, parameters, results, access and mutation. Empty and nonempty
literals and filled constructors use `TAG_BOOL` (4). Boolean reads retain
boolean comparison semantics rather than integer comparison tags.

My raw emitter previously accepted an integer array append containing a boolean
(`/tmp/nanolang-bad-array-push.nano` and `.nasm`). I now validate supported
container types, typed-list names and append value compatibility before emitting
`ARR_PUSH`. Existing enum/integer compatibility remains; bool/integer mixing
is refused.

Validation:

- `make -j8 test-nanoisa-src-nano`: 86 checks and 44 methods pass.
- Fourteen exact C-seed opcode comparisons cover six fixture functions.
- Both modules verify and execute under NanoVM and strict C11 AOT. The fixture
  checks empty, literal and filled arrays, returned arrays, record fields,
  shared aliases, replacement, append and boolean equality.
- Ten malformed element, index, container, typed-list and equality cases refuse
  publication. Former boolean-unsupported cases now have positive coverage;
  unsupported float-array cases retain their refusal checks.
- The generated boolean fixture passes unsuppressed ASan/UBSan, including
  allocation cleanup (`/tmp/nanolang-bool-array-sanitizer.log`).

The integration log is `/tmp/nanolang-boolean-arrays-targeted.log`.

`List<bool>` remains outside this claim: the C parser refuses its type token.
Task `task_9556b493260246be8e271b27e29cdf02` records that separate contract and
implementation boundary. Full compiler emission and bytecode bootstrap remain
unfinished.

A freshly built canonical compiler advances to `unsupported extern result or
symbol path_basename`. Task `task_81e682a57a3d431e845b0f41f140352e` records the exact
host-contract continuation. The probe is
`/tmp/nanolang-canonical-after-boolean-arrays-probe.log`; it publishes no module.
