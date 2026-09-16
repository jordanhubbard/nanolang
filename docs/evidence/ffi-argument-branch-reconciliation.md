# My foreign-argument branch reconciliation

I reconcile `origin/fix/ffi-float-abi-and-arg-limits` at
`606e05a3703cd02eb51cb9e0fab97d3a652c8777` and
`origin/test/extern-call-arg-limit-vm-path` at
`96ed14c6b55dd1df9339243f74bd9f840e8e7057` against integration
`b770cd03`. I retain the current implementation and tests; this merge records
ancestry, not new ABI functionality.

The old float-refusal patch protects a pointer-dispatch implementation that
cannot represent mixed floating-point signatures. My current interpreter and
VM use typed libffi calls. I retain those calls and the shared
`NANO_MAX_FFI_ARGS` ceiling of 16, not the older ten-argument dispatch ladder
or mixed-float ban. The branch's presentation cleanup (`35f14e91`) is already
patch-equivalent according to `git cherry`.

The execution-path regression from `96ed14c6` is already present in
`tests/nanovm/test_vm.c`: an unverified module declares four arguments above
the shared ceiling, executes `OP_CALL_EXTERN`, and must receive
`VM_ERR_OUT_OF_BOUNDS`. This checks execution rejection, not just verifier
rejection. My interpreter, VM and protocol also retain shared-limit checks.

My FFI suite executes an 11-argument alternating integer/float signature and
checks its floating-point result of 66.0. This exercises behavior the older
dispatch implementation could not support.

The serial command completed on this macOS host:

```sh
make test-vm-ffi test-nanovm test-cop-protocol
```

Its log, `/tmp/nanolang-ffi-branches-gates.log`, reports 27 FFI tests passed,
272,379 VM checks passed with zero failures, and 35 protocol tests passed.
Callback allocation/publication/owner-cleanup checks also pass. These are
focused host-local results, not cross-platform or full release acceptance.
The lifetime-safe callback release scope and remaining branches stay open
under MAC release parent `task_cffdafd16e641ac417ccfddb962534b9`.
