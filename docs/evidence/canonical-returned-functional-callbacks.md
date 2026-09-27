# Exact Returned Functional Callbacks

I keep this evidence bounded to the exact callback selector accepted for my
canonical scalar functional arrays. I do not claim general indirect-call
support.

## Contract

I accept a returned callback when a zero-argument declared selector ends in one
exact declared function identifier and every preceding statement is a
straight-line, result-free call. I also refine one Boolean-literal selector
whose sole statement is an `if` over its parameter and whose two arms return
exact signature-matching functions. I evaluate each selector once before its
call arguments, discard its proved function-reference value, and call the
selected target directly.

I retain immutable callback locals initialized from exact declared functions.
An exact scalar function parameter may now use `CALL_INDIRECT`; my native
translator admits it only when published parameter and result tags select
module functions with the same scalar shape, and every other runtime target
traps. This is not a native callback ABI. Closures, aggregate parameters or
results, imports, mutable callback locals, nonliteral selector conditions,
effects inside branching selectors and unknown result joins still fail before
publication.

## Retained terminals

Hosted coverage run `35551946391` first reached the unchanged map inference
case and refused `(choose)` because it was not a direct callback identifier.
Its retained log has SHA-256
`52ac99538831146c5bd91af503d8be3037b19bf3b5bf499ff3b224e84240d98d`.

The first corrected Linux bootstrap built and smoked Stage 1, then Stage 2
refused an emitter shadow that used the unavailable `str_index_of` helper. I
removed only that unsupported shadow oracle; the Python integration keeps the
stronger ordering check.

The next fresh bootstrap passed, then the original map case reached
`shadow choose` and exposed the same returned callback as a direct computed
call. My expression classifier treated the selector's function result as the
outer call result and refused its float equality. I retained that terminal and
added the same exact-selector proof to computed calls without enabling general
indirect dispatch.

Integrated hosted release run `35558456623` rebuilt the candidate and reached
the complete self-hosted map-result matrix. All sixteen scalar combinations
then refused immutable exact callback locals as unsupported local types. Its
evaluation-order case separately refused the selector because its deterministic
`println` preceded the final exact callback return. I retained those terminals,
admitted only the bounded local and straight-line prefix forms above, and kept
dynamic callback values refused.

The first corrected Stage 2 map-result run then passed evaluation order and
published all sixteen callback-local modules, but `nvm2c` refused every
`FUNCREF` store because local classification replaced the callable kind with
its integer-width storage shape. I retained that terminal. The translator now
keeps the stronger function kind through store/load and omits these nonheap
references from map root registration; it does not infer callable authority
from an ordinary integer.

After the corrected bootstrap, the first broad functional-array invocation
stopped before semantics because a clean bootstrap had not built
`bin/nanoisa_emit` or `bin/nanoisa`; its log is
`9e85d93160adc658f48b51dbbace563bc628edfb515835f996ba3cb9d31f7fe8`.
After those explicit prerequisites, eight methods passed and the ninth stopped
before semantics because `bin/nano_virt` was not yet built. That complete log
is `1cbb04f2319c8069d9cddf482c7991e1a833d83432c1e91b84d186abdcb18e8e`.
I built the declared prerequisite and reran only that unchanged method.

## Qualified result

The production head is `e3bb0eea2952ab94f78e8a6e0308f47b4fab1d36`.
The final candidate `c2b172fba6381c2785e81652a12b0c69f93ce1cf`
differs only by the dynamic-selector prior-output regression.

On a clean Linux ARM64 checkout I ran:

```text
NANO_SHADOW_TIMEOUT_SECONDS=60 \
NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS=60 \
make -j16 bootstrap
```

All three stages, both hello smokes, installed-compiler publication and the
no-C-seed independence check passed. The log SHA-256 is
`776d8b4ecdeb797e5673977f9d95de7418a99d56f0be962e98428af0bd786cba`.

The original unchanged regression then passed both methods through installed
Stage 2 and native execution:

```text
python3 -m unittest -v tests.test_selfhost_map_types
```

Its log SHA-256 is
`fd82cb9eae11d56c45c5b132cbaae2b21f58a48f07f10368a61c1e91bd92900e`.

With `nanoisa_emit`, `nanoisa_dump` and `nano_virt` prepared, the functional
array suite passed all nine methods. Eight passed in the retained broad run;
the unchanged prerequisite-stopped method passed separately. The final-method
log SHA-256 is
`85ad6373cebd9157a4dac8a7d60baf83b7ee9134210507730709c618a7b33c3b`.
The suite covers raw and bound emission, NanoVM verification/execution,
`nvm2c`, strict sanitizer-native execution, exact selector order and
prior-output preservation for branching selectors.

The adjacent translator and source-emitter gates also passed:

- `make test-nvm2c`: 2,426 structured-C checks and 1,412 shape checks; log
  SHA-256
  `54c6d3c1b6e75c75de59641018a9df8a2a95f8998484590c28e01b28c2cfc477`.
- `make test-nanoisa-src-nano`: 86 pinned Cut A checks and 90 Python methods;
  log SHA-256
  `a75356f3f7dcdd70123d007b09fa075ddc658197d29b10ce9d748d5d2bd26cc8`.

The replacement hosted release matrix remains the publication gate. These
focused results do not by themselves authorize the tag.

## Integrated release follow-up

The callback correction is based on integrated release candidate
`329d87860c8bd52b7a5e92f70ea138854d1d2532`. On the clean Linux release
checkout, fresh bootstrap passed all stages, both hello smokes, installed
compiler publication and the no-C-seed independence check. Its log SHA-256 is
`e3d7548dd8f76aeabd802f512c19e5aa7b7619982638926f6ab4a12883254ef0`.

The corrected focused gates then passed without exclusions:

- all nine functional-array methods, including direct and returned selectors,
  immutable callback locals, exact evaluation order and dynamic refusals; log
  SHA-256
  `36395b09970ace95254aaf6a3dba0f929758543a2f8c159ba4999d5cebe1e7da`;
- both self-hosted map-result methods, including all sixteen scalar callback
  combinations and selector/callback order; log SHA-256
  `c3b984b013b1d36c898285c3371ead4cc3302ac758e81452019feb89e25ffa34`;
- `make test-nvm2c`, with 2,426 structured-C checks and 1,412 shape checks;
  log SHA-256
  `133d4f2f1338fc5ad83075611e201e06ddb9ce9086b32910f28b2cf38a2df93e`;
- `make test-nanoisa-src-nano`, with 86 pinned Cut A comparisons and 90
  adjacent emitter methods; log SHA-256
  `cacbe2db8ab4f569a7840ec1af3f686074690b9d0ad582d884b707be66bced7b`.

These results qualify the bounded correction. The replacement hosted release
matrix still gates the 5.1.0 tag.

## Returned-call hosted follow-up

Hosted release run `35562442766` tested the preceding correction and reached
the pre-existing returned-call suite. Its retained x64 log has SHA-256
`a24436ce1bffb79bea456f6cc300f1fb1676bbcba7ec0ae1ba395d8c32754579`.
The Boolean selector returned status 1, and the nested case refused the exact
`identity` function argument. I retained that terminal and implemented the
bounded contract above at production commit
`6a772ae05e1c9bfcb07de6c68a6ef07f6d0e8d2e`.

The intermediate terminals remained useful:

- the first source correction passed the branch selector and ordering methods,
  then refused `identity` as an undefined binding; its log SHA-256 is
  `09fef7855aeecf634f6b4858014d54858ea68ee48a6ae36d65b6977aa341f054`;
- after declared function values emitted `FUNCREF`, the raw source fixture
  exposed unsupported function parameters;
- after `CALL_INDIRECT` emission, NanoVM verified and ran the module while
  native translation lacked exact parameter facts;
- publishing scalar parameter facts independently first exposed malformed
  partial rows for source types without a NanoISA tag; I now publish a row only
  when every parameter has one exact tag;
- the broad gates then found two stale test assumptions and one preserved
  diagnostic token. Exact scalar indirect execution replaced the blanket
  opcode refusal, the executable-closure source case became a VM/native
  positive, and unsupported roots plus malformed stack shapes remain refused.

On the final exact source, a clean Linux ARM64 bootstrap passed all stages,
hello smokes, installed-compiler independence and the recorded comparison in
344.96 seconds. `bin/nanoc` and `bin/nanoc_stage2` were byte-identical at
SHA-256 `41491b18d44592501cbc777898eb2a54db2cf6b9dfa988fb8a2553e02319e087`.
The bootstrap log SHA-256 is
`070f041cacd35d267548152df55648cb2b05118e2234b5b2a8cb1840b3b06162`.

The unchanged focused and adjacent gates passed:

- all three returned-call methods through installed Stage 2 and native
  execution; log SHA-256
  `fda96248c2ceac92f70e4b7ddae69e90b10fa3c7e28647e9edf77b899e8c4e13`;
- both map-result methods and all nine functional-array methods, including
  preserved output on dynamic refusals; log SHA-256
  `11044b74200082b48692ec9dcef32d31a562c2680ad3dafb57b1551b073fd7e8`;
- `make test-nvm2c`, with 2,428 structured-C checks; log SHA-256
  `4b7161319e266f0c671756ebf36c7cd089cb1093f3f7693b7207b2f18ccd6900`;
- `make test-nanoisa-src-nano`, with 86 pinned comparisons and all 90
  integration methods; log SHA-256
  `a89498b4d7b7c9083da8956f41bd558f26c09de85f4584b70bdf52227e41966d`.

These results qualify the bounded returned-call correction. A new hosted
release matrix still gates PR #522 and the 5.1.0 tag.
