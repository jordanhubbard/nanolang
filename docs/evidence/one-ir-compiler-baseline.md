# My One-IR compiler baseline

The remaining `origin/feat/4.6-frontend-contract` head `8299138f` contains
75 commits outside integration ancestry; `git cherry` finds no patch-equivalent
commits. Its work includes compiler bytecode emission, AOT aggregate/list/host
support and a driver that uses bytecode followed by C translation. I do not
treat that branch as an obsolete snapshot or merge it without content review.

Its old compiler acceptance scripts motivated a current-source gate:
`make test-one-ir-compiler`. I build the required tools, emit bytecode for the
current full compiler with normal shadow selection, translate it through
`nvm2c`, compile the resulting C11, run compiler help, then compile and execute
`nl_hello.nano` with exact output. Each command has a timeout; timed-out process
groups are killed and reaped. All artifacts live in a private temporary directory.

The rebuilt gate exits 2 because its Python test fails in 1.470 seconds:
compiler bytecode emission succeeds, then `nvm2c` rejects import 1,
`fs_walkdir`, because it lacks a supported exact builtin host ABI. No AOT
compiler or hello execution is reached. The initial direct run reproduces the
same failure. Logs on this macOS host:

- `/tmp/nanolang-one-ir-compiler-baseline.log`
- `/tmp/nanolang-one-ir-compiler-rebuilt.log`

The gate is intentionally failing until implementation catches up. Its later
checks do not establish full canonical Stage1/Stage2 bytecode equality or
whole-language conformance, which remain separate requirements. Checking that
generated C lacks `nano_vm` is an additional guard, not a proof of its native
implementation. I do not weaken the current exact-host-ABI rejection to make
the compiler pass.

MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71` tracks this execution path,
starting with the unsupported host import. The full 75-commit branch review,
ownership work and release acceptance remain open.
