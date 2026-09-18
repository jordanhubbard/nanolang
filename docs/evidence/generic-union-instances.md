# My exact generic union acceptance

MAC `task_6550ccf97eb44f4d8c08f02cd4189cd7`;
[bounded contract](../NANOISA_GENERIC_UNION_INSTANCES.md).

I retain declaration-level wire union IDs. Concrete arguments remain exact
source identities through contextual construction, locals, calls, returns and
match payloads. I substitute lexical formals before resolving module record
names. This slice does not make the wire ID alone distinguish generic
instances or change ownership authority.

My initial fresh bootstrap passed. The first 16-method paired run retained
three failing methods: native mixed-variant scalar inference refused accepted
values; one new negative expectation omitted the correct constructor-declaration
diagnostic; and the unchanged affine suite reached the same native scalar
carrier boundary in two producer cases. I preserved that log and the positive
fixtures. The other 13 methods passed.

I recorded native carrier prerequisite
`task_6697993e325847d9a27c4af9e6507e65` separately. Explicit generic constructor
source syntax remains continuation `task_952ac994c2684adc8944cb2373670456`;
I do not count a parse failure as proof of a type mismatch. An internal AST
shadow covers explicit zero-payload constructor argument equality at the
lowering boundary, while accepted source tests cover zero-payload values
passed under the wrong concrete instance.

The exact union return-call guard passed a fresh bootstrap at `934a3276`.
The corrected refusal method passed in 36.302 seconds, including direct
returns and returned calls with mismatched concrete arguments. It requires
the intended type/lowering diagnostic, excludes parse failures, and preserves
the previous output. The source and test adapter checkpoint is `78eadea0`.

Retained evidence:

- `/tmp/nanolang-generic-union-paired.log`
- `/tmp/nanolang-generic-union-return-bootstrap.log`
- `/tmp/nanolang-generic-union-return-refusals.log`
- `/tmp/nanolang-generic-choice-positive.nano`
- `/tmp/nanolang-generic-choice-positive.nvm`
- `/tmp/nanolang-generic-choice-positive.log`

The external native prerequisite checkpoint `1e640b59` then passed every
generic positive and all 25 unchanged affine-suite methods. The full inherited
16-method gate ran in 54.541 seconds: 14 methods passed; two previously passing
scalar expression-match methods received checked native boxed/concrete stack
join refusals across six producer routes each. I retained those fixtures and
reported the boundary to the native prerequisite owner. I did not execute
rejected output or weaken either method.

That command was
`NVM2C=/home/jkh/Src/nanolang-native-variant-scalar-carriers/bin/nvm2c python3 -m unittest -v tests.test_generic_union_emission`.
Its log is `/tmp/nanolang-generic-union-native-companion.log`; producer and
translator hashes are in
`/tmp/nanolang-generic-union-native-companion-hashes.txt`.

Final paired native acceptance and canonical integration are still pending.
Frozen product qualification and release remain separate gates.
