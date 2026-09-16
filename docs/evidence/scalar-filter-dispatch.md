# My scalar filter dispatch

I repair the release reconciliation's filter failure under MAC
`task_ac94d5cc420e481a896d5f1a2d37f595`.

My self-hosted emitter previously inspected only identifier inputs. It sent
float and boolean literals to an integer callback helper. I now use existing
expression-type inference and retain the contextual array type for an empty
literal. My boolean helper takes `bool (*)(bool)` and allocates boolean storage.

The expanded test exposed two more empty-literal boundaries. My C seed emitted
integer callback arguments for empty string input; my VM kept integer storage
and truncated a later appended float. Both paths now derive scalar empty-literal
storage from the predicate's unary, boolean-returning signature. I reuse the
existing callback-signature query, including variable and returned functions;
the direct empty-literal execution regression uses named predicates.

`tests/nl_functions_filter.nano` checks integer, float, boolean and string
literals and variable inputs; empty input and all-rejected results; append after
empty output; and unchanged source length. Its main shadow executes the same
assertions before artifact publication. Compiler shadows check emitted helper
selection for four scalar literals and the boolean helper's signature/storage.

I compile and execute that regression successfully with `bin/nanoc_c`, freshly
rebuilt `bin/nanoc_stage1`, `bin/nanoc_stage2`, and `bin/nano_virt` plus
`bin/nano_vm`. The VM run executes selected shadows before publishing bytecode.
These tests do not establish aggregate filtering, full callback type safety,
or evaluation-order parity between every backend.

`make test-one-ir-compiler test-c-backend` passes all 21 native-compiler
acceptance methods and all seven C-backend cases after the change. My bootstrap
passes its smoke tests but still reports different native binaries; canonical
NanoISA equality is a separate gate.

My complete `make test-quick` attempt exits nonzero at the existing
`test-affine-selfhost` gate. Before that stop, all 17 language cases and all
242 eligible VM examples pass; all six excluded examples remain ineligible on
both checked backends. The C seed still accepts `unresolved.nano` and reports
`use_after_move.nano` only through shadow failure. Both self-hosted stages pass
those limited ownership cases. I retain this blocker under
`task_91ae827be4154eaa8f22698aeecc8cf1`; later quick-gate targets did not run.
Gforth differential execution is skipped because Gforth is absent, while its
pin checks pass. My local log is `/tmp/nano-filter-quick.log`.
