# My mutable record-array parameter failure

I reduce the diagnostic formatter failure to a shared empty array, a helper
that pushes a record containing a nested location, and a caller that packs
the returned location fields into another record. Direct mutation translates;
helper mutation fails through both source producers. My populated diagnostic
corpus fails through both as well.

My executable `regression.py` checks direct, helper, forwarded and indirect
mutation, including caller aliases, plus the unchanged diagnostic formatter.
It compares VM output with strict native C under ASan/UBSan/LSan for both source
producers. I can reproduce the current failure with:

```sh
PATH=/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:$PATH NANOLANG_SELFHOST_COMPILER=/private/tmp/nanolang-capture-ae92c0488-cow/bin/nanoc_stage2 python3 docs/evidence/mutable-record-array-baseline-20261008/regression.py
```

My unchanged translator fails seven subtests across two methods. I tested a
proposed equality constraint on record-array call parameters: the new corpus
passed, but the full native gate failed four existing tagged-field array-call
controls (2,423 checks passed). I retain that rejected patch and both logs.
I withdrew the production change; this is failure evidence, not a fix.

The call path passes a shared mutable array handle. My existing directed
shape conversion transfers caller facts into the callee, but does not carry
new nested field facts from callee writes back to caller reads. Equating all
fields loses supported optional/scalar storage views. I still need a sound
shared-handle constraint that retains those views and propagates write facts.
Task `task_f61d318c7c8640e58f47ffccd1abe4f4` remains open.
