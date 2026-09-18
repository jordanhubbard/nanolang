# My scalar union source evidence

I lower exact nongeneric unions with int, bool, float and string payloads,
including complete named construction and exhaustive statement-block matches.
I preserve source-order evaluation, declaration-order packing, nominal identity
through locals/calls/returns, scoped payload bindings and all selected shadows.

My production checkpoint is `a648c2087de7534b5550120f140ccc452a76fed3`, based
on main `3e0cedcd` with float-array PR643 and native padding PR645 integrated.
I use this tree's own freshly built native translator; no dependency override
is present in the final gate.

- Fresh `make -j8 bootstrap nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump`
  passes. Log: `/tmp/nanolang-core-union-integrated-bootstrap.log`.
- `make -j8 -o bootstrap test-scalar-union-emission test-nanovirt` passes:
  seven paired methods in 38.162 seconds and 89 NanoVirt checks. Log:
  `/tmp/nanolang-core-union-integrated-paired.log`.
- `make -o stage1 test-typechecker` passes all typechecker tests. Log:
  `/tmp/nanolang-core-union-integrated-typechecker.log`.

- Adjacent selected-variant ownership, generic selected ownership and resource
  collection suites pass all 43 methods in 142.401 seconds. Log:
  `/tmp/nanolang-core-union-integrated-ownership.log`.

The paired methods include both unchanged core control examples, canonical
Stage1/Stage2 publication, separately lowered selected shadows, VM execution,
and strict ASan/UBSan native execution. I also check raw C-seed-hosted and
Stage2-hosted emitter paths for supported fixtures. Constructor and scope
refusals preserve prior output through both canonical stages and raw emitters.

I repaired two C-seed prerequisites separately recorded before implementation:
named field packing/checking used source positions instead of declared
positions, and retained arm symbols escaped block scope or lost checked
metadata during emission. I retain the original logs at
`/tmp/nanolang-union-field-order.log`,
`/tmp/nanolang-core-union-first-tests.log`, and
`/tmp/nanolang-union-resource-adjacent.log`.

My resource-bearing reordered-constructor test is adjacent C-seed coverage;
it does not admit resource unions into the selfhost scalar union subset.
Generic, nested/resource payloads, broader match forms, and non-block
expression-arm scope remain outside this claim. Whole-source native emission
can still refuse unused union helpers without sufficient field evidence;
canonical reachability and selected-shadow acceptance do not remove that
separate shape-proof requirement.

MAC: `task_9a12d05fafaf4f92bea5919683f9ba32`,
`task_0cf9859eec354141815e9c84112c8b14`,
`task_c514d01489f84c17b8013458c9241758`.
