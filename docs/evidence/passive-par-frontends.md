# My bounded source `par` emission

I retain `ASTBlock.is_par` through my self-hosted schema and parser. Both source
checkers require a nonempty block of distinct immutable `let` bindings. I check
each initializer against the environment before the block, then publish its
bindings in source order. A sibling reference, mutable external binding,
aggregate value, or effectful call is outside this slice.

I inspect the resolved bodies of closed scalar calls. I admit scalar frame-local
mutation and `while` loops without changing my separate `pure fn` contract. I refuse
recursion, globals, foreign and indirect calls, and observable effects. A known
spelling does not excuse an incompatible user declaration. My self-hosted
checker and emitter share this body classifier; the bytecode verifier separately
checks actual scalar producers, definite local initialization, and call results.

My C-seed and self-hosted NanoISA emitters retain version-2 passive records.
External reads in this bounded lowering must be immutable scalar function
parameters with executable entry guards. The C emitter publishes final code
offsets; the self-hosted emitter uses structured assembler markers. Canonical
disassembly still retains the exact passive payload. Serial execution is
unchanged; this is eligibility metadata, not a parallel scheduler.

## Reproduction

```
make bootstrap nanoisa_emit
make test-nanoisa-src-nano
python3 -m unittest tests.test_passive_par_frontends
```

I retain the unchanged `tests/test_par_blocks.nano` and full
`examples/language/nl_pi_calculator.nano` on the C-seed, Stage 1 and Stage 2
native frontends. The paired NanoISA test extracts the original `int_to_float`,
`arctan_series`, and `calculate_pi_machin` definitions and their shadows directly
from that calculator. It compares C-seed/self-hosted bytecode and exact passive
payloads, then requires VM and native output `3.14159`. The helper remains its
original source loop; I do not replace it with a conversion opcode.

The scalar fixture separately exercises `int`, `bool`, `float`, and `string`
parameters, entry guards, exported bindings, and two blocks. Refusal cases retain
an existing output file across all four source drivers. A separate duplicate-name
module test retains each function owner and its private helper, including verified
canonical Stage 1/Stage 2 `--emit-nvm` output with passive metadata.

The final restacked bootstrap passed. The emitter gate passed 86 C comparison
checks and 84 methods (102.566 seconds). Five frontend methods passed in 37.554
seconds; the extended canonical-owner method passed separately. Schema validation
and 79 C typechecker checks also passed before the final additive restack.

I retain one unexplained failure of the existing artifact-publication test under
`task_6651883267424a2f907b439de9fba4c6`. It passed in the earlier run and in one
isolated rerun (23.438 seconds) with identical source, binary, and capture-helper
hashes before and after execution. A separate bootstrap ran concurrently with
the failed gate; that observation does not establish its cause. I preserved both
full-gate logs, the isolated log, hash manifests, and ordinary emitted artifacts.

## Remaining acceptance

The full calculator's raw self-hosted NanoISA emission still refuses its
declared `strlen` ABI before lowering. Its native frontend success and extracted
scalar closure do not establish full canonical calculator publication. I retain
that boundary under `task_20f6cb36fbf24bba987b4ea503529438`.

Immutable local captures, globals, aggregates, `u8`, resource permissions,
foreign purity summaries, `flow` extraction, and full passive conformance remain
open. I do not infer eligibility from signature annotations alone or silently
discard a `par` claim when its bytecode proof is unavailable.
