# Native effect execution baseline

I exercise the same 13 effect programs in my VM and canonical native C route.
My C producer, fresh self-hosted seed and regenerated native compiler each pass
60 source cases with VM execution: 13 executed programs, ten ordinary refusals
and 37 declaration controls. The regenerated compiler's same 13 modules all
fail native translation at unsupported `HANDLER_PUSH`. This is a failing
acceptance gate, not native support.

I include captured mutation, recursive arms, string and array transport, typed
ordered arguments, lexical returns, nesting, void results and local loops.
My initial array fixture had two syntax errors (`>=` at its annotation boundary
and a missing comma); I retain those failures and the corrected paired results.
I do not change compiler behavior to accept an invalid fixture.

I reproduce the paired target gate with:

```sh
NANOLANG_EFFECT_COMPILERS=/private/tmp/nl51-row-signatures/compiler \
NANOLANG_EFFECT_TARGETS=vm,native python3 tests/test_selfhost_effects.py
```

I default the existing frontend gate to VM execution; explicitly selecting
native requires successful translation, strict C11 compilation and the same
exit result. Native failure is an assertion failure, never a skip or expected
success. My compiler provenance is the nine-step audit in
`../row-effect-signatures-20261011` at production revision `72c85b770`.

My implementation still needs dynamic handler lookup, arm activation with
shared lexical locals, distinct resume versus lexical-return paths, and
cleanup of all bypassed calls. I must preserve live operands and managed
values across those transitions. `EFFECT_RESUME` returns to the performed
operation; `RET` in an arm exits its lexical owner, following nested arm owners.
I retain the existing VM semantics in `src/nanovm/vm.c` as the executable
comparison. No legacy AST-to-C path satisfies this acceptance gate.

I published the row implementation at `72c85b770` and the worker handoff at
coordination commit `a146ad2d1d458a4e275e9315007a39f24ec08a5e`. GitHub API
updates still failed; successful HTTPS git publication does not establish that
my issue comment was posted. I keep #976 and #989 open.
