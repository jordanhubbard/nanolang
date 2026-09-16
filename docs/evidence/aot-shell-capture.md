# AOT shell execution and capture

I adapt exact empty-namespace `nl_exec_shell(string) -> int` and
`nl_exec_capture(string) -> string` imports. I do not redirect artifact-backed
imports with those names to these helpers.

Shell execution returns the raw `system` status, matching my native helper;
it does not decode that status into an exit code. Capture retains at most
65,535 stdout bytes, drains the remaining stream before waiting, and ignores
the child status, matching the existing string-only contract. Each AOT result
has independent allocated storage, so another capture cannot overwrite it.
The storage currently lives until process exit.

These are shell operations, not argument-vector execution or a security
sandbox. Caller-provided commands retain shell interpretation. Capture does
not expose a structured exit/I/O result or preserve text after an embedded NUL.
This checkpoint does not claim unbounded capture or general memory reclamation.

## Verification

`make test-nvm2c` passes 842 checks on Darwin. `git diff --check` passes.
`make test-one-ir-compiler` remains failing after 1.646 seconds at the import
reported below.

Generated executables exercise successful shell execution and `exit 7`,
expecting its raw POSIX status of 1792. Capture cases cover empty output,
output followed by nonzero exit, retained results across later calls, and
300,000 bytes of output with a 65,535-byte retained result. Signature and
namespace rejection checks remain in place.

Compiler preflight advances to import 32 (`vm_string_from_char`). Full
compiler acceptance and release remain open under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
