# AOT builtin text reader

I adapt the exact empty-namespace signatures of `file_read`, `vm_file_read`
and `nl_os_file_read` to generated C. Artifact-backed reads retain their
separate native-library binding.

I read in chunks without seeking. I grow checked storage, preserve all non-NUL
bytes, and return empty text for missing input, read errors, close errors or
embedded NUL. Allocation failure aborts, as in my other AOT allocation helpers.
Returned storage currently lives until process exit. This does not establish
general string reclamation or a richer I/O error reporting interface.

## Verification

`make test-nvm2c` passes 752 checks on Darwin. The added generated-executable
cases cover ordinary text, 9,000-byte input, embedded NUL, empty and missing
files, directory input, a FIFO with a bounded owned writer, and injected read
and close failures. Each case also rejects a mismatched result signature.
The FIFO writer is reaped; test paths are private. `git diff --check` passes.

The initial test fixture accidentally changed a shared empty string that also
named the builtin namespace. I fixed the fixture to use a distinct expected
string; I did not relax namespace validation.

Compiler preflight now reaches import 19 (`file_write`). Full compiler
translation and execution remain unfinished under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
