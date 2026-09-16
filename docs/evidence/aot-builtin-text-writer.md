# AOT builtin text writer

I translate the exact empty-namespace `file_write`, `vm_file_write` and
`nl_os_file_write` signatures with two string parameters and an integer result.
I validate both parameter tags and preserve argument order. Artifact-backed
writers remain bound to their recorded libraries.

I return 0 only when `fwrite` reports the full text length and `fclose`
succeeds. I return -1 for null input, failed open, short write or failed close.
I close an opened stream even after a short write. This is not atomic file
replacement or a durability guarantee: an error can leave partial output.

## Verification

`make test-nvm2c` passes 771 checks on Darwin. Added generated executables
exercise exact contents, empty text, failed open, injected failed close and
injected short write. The short-write test counts close calls and requires
exactly one. Every case rejects an incorrect second argument tag. Fixture
contents are themselves a private path so even reversed arguments cannot
write outside the owned fixture. `git diff --check` passes.

Compiler preflight advances to import 20 (`file_exists`), not full compiler
execution. The compiler task remains open:
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.

I found a separate parity defect while inspecting the existing writers:
interpreter and generated-native writes/appends ignore write and close errors;
VM and module writers ignore close errors. I track that required fix under
`task_46cb33d875724130a6f767d998f3014b`. This AOT checkpoint does not resolve it.
