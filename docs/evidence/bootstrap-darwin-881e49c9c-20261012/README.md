# Darwin bootstrap checkpoint at 881e49c9c — 2026-10-12

I ran `make bootstrap` from a clean clone (`/Users/jordanh/Src/nanolang-rel`, outside any dot-directory)
at `881e49c9c` on Darwin arm64 with Apple clang and Homebrew OpenSSL 3.6.5.

| Check | Result |
| --- | --- |
| Seed compile through the C seed (`nanoc_v06.nano --emit-nvm`) | Pass after the shared-link buffer repair; it failed before |
| Stage 1 and Stage 2 emit, verify, translate, native build and hello execute | Pass |
| Raw Stage 1 vs Stage 2 module bytes | "byte-identical in this build" (bootstrap driver message) |
| `bin/nanoc` with `bin/nanoc_c` removed | Pass |
| Stage 2 hello binary VM symbols (`nm`) | 0; it does not link `nano_vm` |
| `src_nano/transpiler.nano` | Absent; manifest names only C seed sources `src/transpiler*.c` |

## Scope

This is Darwin-only, one host, one pin. It is evidence toward rows 10684-10688 and does not close them:
Linux, the exact release revision, the pinned VM/C/LLVM/Wasm corpus and hosted CI remain required.
The bootstrap driver itself states that this stamp does not establish release readiness.

## `make test-one-ir-compiler` at the same pin

I ran the compiler-product gate from the same clean clone. It runs 109 tests in 950 seconds and
reports 119 failures. I find 129 occurrences of `AddressSanitizer: detect_leaks is not supported on this
platform` in the log, and the failures I inspected (the sanitizer-driven `-6` exits, the missing
`native invariant` text and the two non-sanitizer-looking `I failed python3.14` cases) all carry that abort.
Apple clang, Homebrew LLVM and Homebrew GCC are installed here, but LeakSanitizer is unsupported
on Darwin arm64, and I do not disable leak detection to make the gate pass. I therefore record this
gate as **not qualified on Darwin**; it requires a leak-capable Linux run. I have not triaged every
one of the 119 failures individually.

## Defect found and repaired

The seed previously failed with "I could not compile shared_c_source ... for file_product" on checkouts
whose path was a few characters longer than my primary checkout. `module_shared_link_command` output
overflowed a 4096-byte buffer and silently cleared `command_ok` before any compiler ran.
Both buffers are now 65537 bytes (`src/module_builder.c`).
