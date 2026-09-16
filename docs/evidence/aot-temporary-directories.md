# AOT temporary directories

I adapt the exact `vm_mktemp_dir(string) -> string` builtin. I use `TMPDIR`,
falling back to `/tmp` when unset or empty, construct the complete template
with checked dynamic allocation, and call `mkdtemp` for exclusive creation.
Creation failure returns independently allocated empty text. Allocation
failure aborts before directory creation. Returned paths retain the current
AOT process-lifetime storage convention.

The prefix retains native path interpretation; it is not a validated basename
or a confinement boundary. The created directory belongs to the caller, which
must remove it when finished. I do not add recursive cleanup.

## Verification

`make -j1 test-nvm2c` passes 913 checks on Darwin. Added generated executables
create two distinct paths with the same prefix, check their private-root
prefixes and directory existence, remove them, and confirm the private root
is empty. A missing intermediate prefix directory exercises creation failure.
An incorrect prefix type is rejected. The test restores its `TMPDIR` setting.
My generated source enables the Darwin declaration needed for `mkdtemp`.
`git diff --check` passes.

`make -j1 test-one-ir-compiler` now clears import preflight and fails after
1.517 seconds at function 18: `ARR_NEW` only supports integer or string elements
in the current AOT path. I have not established full compiler translation or
execution. That work remains under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
