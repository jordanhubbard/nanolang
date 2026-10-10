# I preserve scalar pop values and absence

I add native pop for word and string arrays, retain callable-target flow,
and copy the result before clearing the final element's environment. My raw
VM/native tests preserve empty-array void results, aliases, exact binary64
payloads and closures retained across owner reuse and garbage collection.
The three methods pass under LLVM ASan/UBSan with leak detection retained.

My first raw run selected Apple cc through a shared helper and failed all
eight subcases because that sanitizer rejects leak detection. I retain that
failure and the corrected LLVM run. My new suite now honors CC explicitly.

My full paired eight-method list suite still fails two required subcases:
record pop through C-seed and self-hosted emission. I refuse unsupported
optional aggregate storage rather than silently trap on an empty record
array. Record pop, complete parser qualification, installed compiler routes,
Linux parity and final 5.1 acceptance remain open under #978.

My full `make -j2 test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang`
terminates with exit zero. All 2,431 execution, 3,092 shape and 379 callable
checks pass. The target also runs my three new raw-pop methods successfully
with its explicit CC selector, plus the existing Python controls.
