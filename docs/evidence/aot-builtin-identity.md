# AOT builtin identity checks

I adapt the exact builtin `file_compare_identity` and
`file_compare_destinations` signatures: two strings and one integer result.
Their artifact-backed imports remain separate bindings.

For existing identity, I compare device/inode pairs and follow symbolic links.
I return 1 for the same identity, 0 for distinct identities or a missing
candidate, and -1 when the source identity or another required lookup fails.

For destinations, I distinguish absent entries from dangling links. When both
names are absent, I use the existing native contract's exclusive directory
probe to ask the filesystem whether the names identify the same entry. I
remove the probe before success; cleanup failure returns -1. These checks do
not protect against concurrent namespace replacement. Actual cleanup failure
can leave a probe directory.

## Verification

My generated-executable cases cover identical names, distinct files, hard
links, symbolic links, missing source/candidate paths, broken links, empty
input, identical absent destinations and distinct absent destinations. I
check that successful probes leave no directory behind. I also inject a
reported cleanup failure after actually removing the private test probe;
the result must be -1. Wrong candidate parameter tags are rejected.

All test paths are private. `make test-nvm2c` passes 822 checks on Darwin;
`git diff --check` passes. `make test-one-ir-compiler` fails after 1.523 seconds.
Compiler preflight advances to builtin import 24
(`path_normalize`); full compiler execution remains unfinished under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
