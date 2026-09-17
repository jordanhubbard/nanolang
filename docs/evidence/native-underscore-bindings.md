# My native underscore bindings

I treat `_` as an ordinary readable lexical identifier outside match patterns.
Each initializer reads the previous binding; the new binding becomes visible
only afterward. My interpreter and checker already retain that distinction,
but both native producers reused one C identifier for repeated declarations.
The original Stage2 failure is retained in
`/tmp/nanolang-flow-final2-gates.log`.

I now assign separate emitted names without changing source names or locations.
Reads, assignments, interpolation, callback names and cleanup bookkeeping use
the active emitted name. Nested blocks and loop bindings restore the outer
alias. My self-hosted environment copies preserve this alias explicitly.

My C-native and NanoVirt emitters also re-establish the exact checked loop
binding after evaluating its iterable or bounds. They copy declaration metadata
by source identity instead of inferring a new element type. This prevents a
later emission-time registration of an outer array from hiding the checked
integer loop variable. I preserve nominal ownership and resource classification;
I do not copy transient checker flow state or change global lookup policy.

`tests/test_native_underscore_bindings.py` exercises C seed, Stage1 and Stage2
native compilation, interpreter and verified VM execution. Its unchanged main
fixture covers repeated scalar/string/array names, initializer order, mutable
assignment, parameters, interpolation, nested scopes, internal-name collisions,
range and array loops, ordinary named-loop controls and map cleanup names. A
failed shadow must preserve a previous output in all three native compilers.
The C-only legacy `par-let` check uses a body call and verifies its existing
post-expression binding visibility.

The fresh default three-stage bootstrap passed at source `ef78f942`.
After the nominal cleanup review correction, the ten combined underscore,
range-bound and effect-execution methods passed in 33.639 seconds; the NanoVirt
unit gate passed 89 checks. Logs are
`/tmp/nanolang-discard-bootstrap-final.log`,
`/tmp/nanolang-discard-final-gates-r2.log` and
`/tmp/nanolang-discard-nanovirt.log`. The initial broader fixture exposed the
checked-loop prerequisite; I retain `/tmp/nanolang-discard-gates.log` rather
than claiming it was an infrastructure failure.

After rebasing onto main `ce8cc3ee`, source `86354f7b` also passes the
default bootstrap, typechecker and environment-scoping gates, 45 transpiler
checks, two assertion-literal methods and ten lexical-scope methods. The
integrated NanoVirt gate passes all 89 checks. I retain these logs at
`/tmp/nanolang-discard-core-gates.log` and
`/tmp/nanolang-discard-integrated-tools.log`. The ten integrated paired methods
pass again in 33.703 seconds (`/tmp/nanolang-discard-integrated-gates.log`).

## My remaining boundaries

- `task_55d61923e89447029bf447a58cf69024`: C-seed `--target c` still uses the
  separate retiring AST exporter. My cleanup-name test uses default native
  compilation with `--keep-c`; the product cutover routes explicit C through
  `nvm2c`, whose final release reachability must be reconciled separately.
- `task_fc39603161314568929719b36d25f14b`: explicit underscore union-payload
  pattern binding differs across producers. I do not redefine pattern semantics
  through this ordinary-local repair.
- `task_49fffa70e832474ebb2573198615dfac`: a legacy `par-let` bare arithmetic
  body becomes a no-effect C statement and strict compilation refuses it.
  `/tmp/nanolang-discard-final-gates.log` preserves that result. I do not suppress
  the warning or claim self-hosted legacy `par-let` syntax support.
- `task_911317d4234c46049c3dea1a2d0a153d`: inline/computed array loop-element
  inference is separate from restoring already checked metadata.
- `task_195cac35e7704e56805932977512ae02`: generated cleanup-name checks do not
  prove actual map deallocation. The existing map lifetime boundary remains.

These checks do not establish the cause of the separate historical product
export-shadow incident `task_dd74b033c3984805bc27ce5017096c3c` or replay its
preserved compiler/input.
