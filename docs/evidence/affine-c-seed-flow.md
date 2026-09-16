# My C-seed ownership-flow checkpoint

I replace shared-symbol ownership mutation with a separate function pass in
`src/resource_flow.c`. I retain lexical bindings in growable storage, register
parameter obligations, distinguish observation from moves, and check scope
exits, branch joins, early returns, assignment and loop edges. I inspect
resource-bearing collection annotations on ordinary and foreign signatures.
Foreign by-value consumption is a declared boundary, not verified C cleanup.

## Evidence

- `make test-resource-flow-allocations` checks 300 owners, rejects an unresolved
  final owner and capacity overflow, and injects each of 20 allocation failures
  in its fixture. Every injected failure reports an error and releases its
  ownership storage. The fixture stubs nominal classification; it does not test
  frontend type resolution.
- The same allocation fixture passes with AddressSanitizer and
  UndefinedBehaviorSanitizer enabled.
- `make bootstrap3 test-resource-classification test-one-ir-compiler` checks
  bootstrap smoke behavior, nested/cyclic/module-owned resource classification,
  and 21 native compiler acceptance methods. This is not canonical NanoISA
  fixed-point evidence or complete compiler equivalence.
- `python3 -m unittest discover -s tests -p test_affine_contract_boundaries.py`
  runs 15 methods across my C seed and both self-hosted stages. The expanded
  gate still fails 27 subcases: 13 per self-hosted stage and one C-seed lexical
  shadow case. Rejection probes require a static ownership diagnostic and
  preservation of the previous output artifact. Positive probes only establish
  source emission; they do not invoke the declared foreign consumer.
- After the fresh bootstrap, the combined frontend-parity and boundary suite
  runs 16 methods and reports the same 27 failures. All three original native
  rejection fixtures pass on all three compilers.

## Remaining work

My C typechecker retains an inner ordinary binding after its block, so a later
call resolves that binding instead of the outer resource parameter. I track
that repair in `task_a47320503e11474e8a4b51dab4b347a4`.

My self-hosted checker still needs branch joins, loop/early-exit obligations,
assignment state, indirect resource-result checking and resource collection
signature rejection. I retain those failures in the shared gate. The full
ownership task `task_c60a8d2e14b7494f8875e75b16e9b087` remains open.

Borrow syntax, whole-owner destructuring, resource match payloads, captures,
generic substitution, complete nominal module identity and NanoISA ownership
facts remain incomplete. My explicit rejection of forms needing lowering is
not their implementation. These tests do not establish complete ownership
soundness, backend parity or runtime cleanup. I have not cleared the release
gate.
