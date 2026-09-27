# Owned callback target planning

I compute conservative same-module function target sets before admitting an
indirect owned call. This checkpoint adds the planning API; it does not enable
indirect ownership verification, VM/native invocation or source callbacks.
The four original native resource-callback failures remain unresolved.

## Contract

`nvm_affine_targets_create` accepts a closed module with the existing bounded
owned value signatures, at most eight functions and a zero-argument scalar
entry. I propagate target bits from `FUNCREF` through stack copies, local
loads/stores, control-flow joins, call parameters and function-valued returns.
Local stores replace the previous value on their path; joins conservatively
include both predecessors. I do not prove predicates or discard a branch merely
because its condition is a constant.

Direct and indirect calls contribute parameter and return summaries until no
summary grows. A bounded worklist handles loops within each function; a second
bound limits module summary passes. I check every inferred indirect target's
arity and result count, refuse unresolved indirect sites, and reject cycles in
the inferred call graph. I retain the prior graph rule that even unreachable
direct-call edges participate in cycle detection. An unreachable indirect site
has no inferred targets and refuses.

`nvm_affine_targets_at` returns a target mask for a function-relative instruction
byte offset. Each bit names one same-module function. The plan owns its decoded
instructions and target tables; allocation failure destroys partial state.

This is provenance analysis, not an ownership proof. It does not check argument
tags, nominal resource identities, initialization, consuming transfer, borrowed
origins or cleanup. The executable verifier still refuses owned indirect calls.
The next step must check every planned target against exact ownership signatures
and preserve the same target set in VM/native dispatch before source admission.

## Qualification

I ran the complete affine-bytecode gate on Darwin arm64. Both ordinary and
instrumented runs pass 5,014 checks, plus 5,535 with allocation/visit-failure
injection. The new matrix has 16 cases in six declaration orders: 54 plans
accepted and 42 refused. It covers stack/local joins, loop-carried targets,
strong overwrites, direct forwarding, indirectly returned functions, `DUP` and
`SWAP`, fixed resource parameters/results, direct/indirect cycles, missing and
out-of-range targets, captured functions, and wrong arity/result count. Every
accepted plan is separately checked to remain outside executable admission.
The allocation driver exhausts planner allocation failures for the indirect
forwarding case and checks a subsequent successful plan after each failure.

- `make test-affine-bytecode`: [ordinary terminal](ordinary.log.gz).
- `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 make -j1
  OBJ_DIR=/tmp/pr522-owned-targets-san/obj
  BIN_DIR=/tmp/pr522-owned-targets-san/bin
  FILE_PUBLIC_LIBRARY=/tmp/pr522-owned-targets-san/lib/libnano_file_runtime.a
  'CC=/opt/homebrew/opt/llvm/bin/clang -fsanitize=address,undefined
  -fno-sanitize-recover=all -fno-omit-frame-pointer' test-affine-bytecode`:
  [instrumented terminal](sanitized.log.gz).
- [Undefined instrumentation symbols](instrumentation.txt) confirm ASan and
  UBSan in the private analyzer object. Project objects for this target were
  built in the private directory; external libraries remain ordinary.

The initial private build exposed a hard-coded `obj/` filter in the allocation
link, retaining both the normal and replacement analyzer objects and failing
with duplicate symbols ([original terminal](private-link-before.log.gz)).
I now honor `OBJ_DIR` for both test binaries, the replacement object and its
link filter. MAC `task_0591cf2d2cc040da9f9cc165bd687e46` tracks this repair.
