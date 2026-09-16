# AOT lexical path normalization

I adapt builtin `path_normalize` and `nl_os_path_normalize` with one string
parameter and a string result. I collapse repeated separators and `.` segments,
cancel ordinary components against `..`, retain unresolved relative parents,
and keep absolute paths at their root. Empty input becomes `.`. This is lexical
normalization, not symlink resolution or a confinement boundary.

I allocate output and component offsets from checked input-derived sizes.
I do not impose the existing native implementation's fixed component or output
limits. Returned AOT strings currently retain process-lifetime storage.

## Verification

`make test-nvm2c` passes 828 checks on Darwin. `git diff --check` passes.
`make test-one-ir-compiler` remains failing at the import reported below.

Generated executables exercise empty paths, roots, repeated slashes, ordinary
and unresolved parent components, 700 ordinary components, 700 leading parents,
700 component cancellations, and a 5,000-byte component. They reject a wrong
input tag. Long constants are inserted directly into the test module's string
pool because the assembly parser rejects such long quoted literals before AOT
translation. This does not test long-literal assembly parsing.

Compiler preflight advances to import 25 (`nl_exec_shell`). Full compiler
acceptance remains open under MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.

I separately track the native/module limits under
`task_82bd388637824cc889b12204d226e75b`: the module truncates output/components,
and generated-native normalization can write beyond `parts[512]` when enough
relative parents are retained. This AOT change does not fix those paths.
