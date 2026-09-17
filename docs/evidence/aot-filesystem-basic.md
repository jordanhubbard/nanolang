# AOT filesystem predicates and entry operations

I adapt builtin `file_exists` and `dir_exists`, including their VM and
generated-native spellings, with one string argument and a declared boolean
result. I use `stat`: `file_exists` accepts any existing target, including a
directory; `dir_exists` additionally requires a directory. I follow symbolic
links and return false for missing or broken targets. These are snapshot
checks, not authority checks or protection against concurrent replacement.

I also adapt builtin `file_delete`/`file_remove` and `file_rename`, including
their generated-native spellings. Removal uses `remove`, not recursive
deletion. Rename preserves source/destination argument order. Both return
0 on success and -1 on failure, without adding atomic-publication or
durability guarantees beyond the native operation.

## Verification

`make test-nvm2c` passes 807 checks on Darwin. Generated executables exercise
all predicate spellings with regular files, directories, missing paths,
file/directory symlinks, broken links and empty paths. They reject an integer
return tag where the import contract requires boolean. Removal/rename tests
exercise all added spellings, successful rename, removal of the new path,
and errors for missing sources. Every mutated path belongs to a private test
directory. `git diff --check` passes.

Compiler preflight now reaches import 22 (`file_compare_identity`) in the
builtin namespace. The artifact-backed adapter for that name does not imply
support for this separate binding. Full compiler acceptance remains unfinished
under MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
