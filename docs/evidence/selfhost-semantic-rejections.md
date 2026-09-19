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

I qualified a fresh detached checkout at implementation-and-test checkpoint
`c68739ca58412e067d2c8da48e3543bad65157b4`, rebased on canonical main
`d6cacdb94ec4dd0f033f221cc6c71d4489eed0d4`:

- `make -j8 bootstrap` passed in 296.56 seconds. Both compiler stages, both
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
  ./tests/selfhost/run_selfhost_tests.sh` passed 22 checks in 66.45 seconds,
  including 15 positive programs, five exact negative cases, five import-path
  methods and 20 CLI methods.

The retained exact-pin logs are under
`/private/tmp/nanolang-selfhost-rejection-final.MYObhw/`:

```text
57d04e7fc4a96a25b5ceab270455652909690655db693c09ad39e29ab4476b8c  bootstrap.log
6b61213cd52f7582cd0930d185d6f38638ec8f149ff90d77bd73473e7c0ad758  rejection-unit.log
d9cc7411676d560444fdc687bdc626e146be33b7b8e814053a2e082820fe2d7f  selfhost-suite.log
a17ab2f02ae02eb87c68a923848533f3d55bb9a587c369ce1ac26aaf3f436d72  before.log
83fd4fe494175d67f390f9c13ce24c1a7dbf3edf361439f3a397d79f98ed9344  after.log
7d039f6fa72ee305e39ec569e2b6619371bd1da91603e6355e0408b6eb05eda0  source.before
7d039f6fa72ee305e39ec569e2b6619371bd1da91603e6355e0408b6eb05eda0  source.after
```

The identical before/after inventories cover every implementation and test
file at qualification time, plus the roadmap/evidence version then under test,
the exact Stage 2 compiler, the resolved Apple Clang executable and Python.
This hash appendix is a later documentation-only change. The Stage 2 compiler
SHA-256 was
`982c54d8866efd3e007efb601be281af66cf106205dc7917397dfcda83372f61`.
The five retained diagnostic logs have these SHA-256 values:

```text
3a0018af8eaea2287e738b5484c4de3fd2ed7541f9c6f3eedebd3b786c158d6d  test_requires_bool.compile.log
a1071e24bc19bceda7e6696cbb24486d374d391a327c4a4f906f07f4e305c20d  test_function_arg_type_errors.compile.log
88ef6a4f007cd458ad2ff2c36e5ec82bba944a3f8eed3d8b21325dc2739fb107  test_returned_function_arg_type_error.compile.log
bc5747d0a2baf2fadeb22f276b0f47824ad25ffbad77b7843616784b3bb9312c  test_returned_function_arity_error.compile.log
8af0407cdca242272c34de57e30b842c2f7c950c6afb4368bcda9bb9da690ce2  test_opaque_nonzero_argument.compile.log
```

After canonical main advanced through `3ec6d7abcc898089b19ff52f5f0610466cf33ccb`,
I rebased the unchanged production and test patch. At checkpoint
`ce983fbda37fa68b299f2b4aad8bd94cc745dc4a`, `git diff --check`, the shell
syntax check and Python bytecode compilation passed. The eight focused methods
passed again in 0.54 seconds. The retained focused log is
`/tmp/selfhost-rejection-finalhead-unit.log`, SHA-256
`846e900247d381febf9062ddb77dda01ddc28801a255593227085a1e160a960c`.
I do not relabel the earlier bootstrap as a bootstrap of this rebased head.

This gate strengthens the self-hosted shell suite. It does not broaden accepted
language behavior, prove the compiler, establish a fixed point, or authorize a
release.

## Caller artifact preservation correction

Independent review of PR806 at `5bc573ebf3e6a2c580f1b5af68e14947ee633c50`
found that the shell caller removed each fixed negative output immediately
before invoking the checked helper. A later suite invocation could therefore
erase an unexpected artifact from the prior run and bypass the helper's
pre-existing-output refusal.

I now create one fresh negative directory for each shell-suite invocation and
one case directory beneath it. I remove no negative output during setup or
cleanup. Diagnostic logs and unexpected artifacts remain together in that run
directory; only positive test binaries keep their existing cleanup behavior.

The new caller-level regression drives the complete shell script with an
isolated fake compiler. Before the correction it failed because the old fixed
artifact had been deleted. At production checkpoint
`72f7268a189b4f1cc77ef9fd9636f06ca76c8a15`, it proves that the old fixed
artifact and a newly published rejected artifact both retain their exact bytes,
the diagnostic remains in the case log, and the suite rejects the publication.

I qualified that exact checkpoint without a compiler bootstrap:

- `git diff --check`, `/bin/sh -n` and Python bytecode compilation passed;
- all nine `tests.test_selfhost_rejection_gate` methods passed in 1.933 seconds;
- `make test-selfhost-rejection-gate` passed the same nine methods;
- the before/after source and selected-tool inventories are identical.

The retained logs are under
`/private/tmp/nanolang-pr806-caller-final.Ei9naW/`:

```text
48cfe2db73a9ced85f9bfa34d97a5f3fe0d7ea18aaebef744c37144eefb1236d  before.txt
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  static.log
4422a972c89c7ce19baabdfe6348d10d84f8c6b5791aa964e69d0271e036ba2e  unit.log
d84db7fae1e53d33fbf36f5215902f3b0b9d571f55f28f97430f0be520cc4687  make-target.log
48cfe2db73a9ced85f9bfa34d97a5f3fe0d7ea18aaebef744c37144eefb1236d  after.txt
c4643c3ab751f20f3282fb61dab4d4f7c1036cbb55f3b014995f4a2cb612b894  status
```

This correction does not relabel the earlier bootstrap or self-host suite
evidence, and it does not authorize a release.
