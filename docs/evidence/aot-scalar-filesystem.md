# Scalar filesystem artifact adapters

I adapt the known scalar contracts in `modules/std/fs.h`: path normalization,
canonicalization, joining, basename, dirname and relative paths; text reads,
writes and appends; existence, deletion, directory creation and copying; and
file/destination identity comparisons. I require each exact name, argument
count, string parameter tag and return tag in an absolute artifact binding.
I do not accept those names as empty-namespace builtins.

I emit typed native function pointers, preserving C `bool` versus `int64_t`
returns and the order of two string arguments. I resolve each symbol from the
recorded library handle. Missing libraries or symbols abort. String results
must be non-null and remain in their existing native storage, with the library
handle retained. My current filesystem implementations return allocated strings
or process-lifetime literals. I do not guess whether to free either form.

The existing filesystem implementation remains authoritative for behavior,
including its path-size and I/O error-handling limitations. This adapter does
not fix those operations or establish isolation, content authentication,
general memory reclamation, or concurrent lazy initialization.

## Verification

`make test-nvm2c` passes 710 checks on Darwin. Generated C executes against a
fresh library built from my filesystem and array/GC sources. Cases exercise
every added scalar contract, one- and two-argument paths, actual boolean and
negative integer results, and write/append/read/copy ordering. All mutations
stay inside the test's private directory. Each case also rejects an incorrect
final parameter tag. Existing builtin namespace and artifact failure tests
remain passing. `git diff --check` passes.

`make test-one-ir-compiler` still fails, now at import 18 (`file_read`) in the
builtin namespace, after 1.536 seconds. This is a distinct binding from the
artifact-backed function tested here. I have advanced import preflight, not
completed native compiler execution. Remaining host adapters and full compiler
acceptance stay open under MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
