# AOT owned filesystem adapter

I translate artifact-backed `fs_walkdir(string) -> array` calls to a native
adapter for my known string-array contract. I require an absolute, NUL-free
artifact path and the exact function name and signature. I do not bind this
name through my builtin namespace or a library search path.

My generated C loads the recorded path with `RTLD_NOW | RTLD_LOCAL`. Before
calling the function, I require array ABI version 1 and the library's
`fs_walkdir_release` symbol. I validate string-array metadata, allocate AOT
storage, copy every returned string, and release the foreign result. Invalid
metadata, missing symbols, incompatible ABI and loading failure abort execution.
I retain the library handle for subsequent calls.

This is a trusted native interface. A version marker is not proof that a
library implements its contract, and metadata checks cannot validate arbitrary
foreign pointers. An absolute path is not a content hash or protection against
artifact replacement. Symbol resolution follows the loaded library's native
dependency rules. My AOT copies retain process-lifetime storage, matching my
current host-string adapters; I do not claim general leak freedom or concurrent
initialization support.

## Verification

`make test-nvm2c` passes 647 checks and executes generated C with strict C11
warnings. `git diff --check` passes. My added cases
cover an unused adapter, a fixture that overwrites its strings during release
(the AOT copy remains unchanged), invalid array width, a missing artifact,
an incompatible version marker, a missing release symbol, and a real library
built from `modules/std/fs.c` and its array/GC runtime. I also reject malformed
namespace, signature, relative-path and embedded-NUL contracts. Linux test
link commands explicitly include `-ldl`; execution here was on Darwin.

`make test-one-ir-compiler` now advances past import 1 (`fs_walkdir`) and fails
at import 2 (`path_normalize`) in 2.275 seconds. This establishes preflight
progress, not successful compiler translation or execution. Further artifact
adapters and full compiler acceptance remain under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
