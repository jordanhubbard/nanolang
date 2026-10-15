# My clean compiler-product checkpoint at 4f56b353c

I ran `make -j2 test-one-ir-compiler` in a clean detached clone at
`4f56b353ced394c13317a3df14fafffe1dfd28ff`. All 86 methods passed in
721.599 seconds; the complete Make command took 732.223 seconds and exited zero.
The source remained clean and its HEAD did not change. My manifest binds the log
hash, source pin, command and terminal result.

I selected Homebrew LLVM through an explicit `cc` symlink directory, so generated
native tests retained their sanitizer and leak checks. The runner records the
compiler and environment selection. The clone remains at
`/private/tmp/nanolang-callables-4f56b353c` because its emitted modules may retain
absolute declared host-artifact paths.

This gate includes both full compiler routes and the listed native storage,
collection and guest-argument regressions. It does not include the three native
returned-call methods, which remain failing at this pin. It predates the function
target shape constraints in `248f046da`, and it is not final release qualification.
