# My self-hosted semantic rejection gate

I accept a negative self-hosted compiler case only when the compiler exits
normally with a nonzero status, emits the case's required diagnostic, and does
not publish the requested output. I retain the combined compiler output for
each case.

I do not count a timeout, signal, missing executable, unrelated native failure,
successful compile or rejected compile that publishes an artifact as a
semantic refusal. I kill the private process group after a timeout or signal.
I preserve unexpected artifacts for diagnosis instead of deleting them.

The five existing negative cases retain their source and now require these
specific diagnostics:

```text
test_requires_bool.nano                     [E0001] assert condition must be bool
test_function_arg_type_errors.nano          [E0010] Argument 1 to 'add': expected int, got string
test_returned_function_arg_type_error.nano  [E0010] Argument 1 to the function expression: expected int, got string
test_returned_function_arity_error.nano     [E0010] The function expression expects 1 argument(s), but I see 2.
test_opaque_nonzero_argument.nano           [E0010] Argument 1 to 'is_null': expected SDL_Window, got int
```

## Darwin qualification

I qualified a fresh checkout based on canonical main
`4351b43f70bbd938588bac4e425284778c245d15`:

- `make -j8 bootstrap` passed in 297.00 seconds. Both compiler stages, both
  hello smokes, the installed compiler smoke and the no-C-seed smoke passed.
  Native Stage 1 and Stage 2 differed, as the bootstrap report records; I do
  not claim a native fixed point from this gate.
- `make test-selfhost-rejection-gate` passed eight methods. The cases inject an
  exact refusal, unrelated native failure, timeout, signal, missing executable,
  successful artifact publication, rejected artifact publication and a
  pre-existing artifact boundary.
- Each of the five unchanged negative sources passed the exact diagnostic gate
  with `bin/nanoc_stage2`, exited normally with status 1 and published no
  artifact.
- `NANOLANG_SELFHOST_COMPILER=./bin/nanoc_stage2
  ./tests/selfhost/run_selfhost_tests.sh` passed 22 checks in 70.32 seconds,
  including 15 positive programs, five exact negative cases, five import-path
  methods and 20 CLI methods.

The retained logs are under
`/private/tmp/nanolang-selfhost-rejection.2aO1ck/`:

```text
c25b37bb1909252fe8792226924d49cb798cd476b2a875b0c10df77fcf2d53aa  bootstrap.log
db9653f91e8d9dfe0e99ec2c8b96169c132a72511ca36e450516a5f4d9b4e0c1  selfhost-suite.log
b60346daa6007f8e17f3ec88ccb4d01865e0dd6795692877f92752403adf7e93  final/unit.log
34bee04922e04d67111a786631826372ff8365c66c178602457c59ee28eb6f38  final/exact-run.log
b16bbe87d8e58eed90888e3871f5872119e22bc275e796bdd965f4c6fc08efc5  final/before.txt
b16bbe87d8e58eed90888e3871f5872119e22bc275e796bdd965f4c6fc08efc5  final/after.txt
```

The identical before/after inventories cover every implementation and test
file at qualification time, plus the roadmap/evidence version then under test,
the exact Stage 2 compiler, the resolved Apple Clang executable and Python.
This hash appendix is a later documentation-only change. The Stage 2 compiler
SHA-256 was
`ad5521a22fb67551cfe24b8b2602c033bfd04e333f34ebd7576cf255f2b461b5`.

This gate strengthens the self-hosted shell suite. It does not broaden accepted
language behavior, prove the compiler, establish a fixed point, or authorize a
release.
