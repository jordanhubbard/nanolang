# I keep unreachable warnings nonfatal

I changed the self-hosted statement checker at source pin `95eb41e5` to create
W0002 through `diag_warning` with the typecheck phase. I continue checking every
statement after a return. My inline shadow verifies warning code, severity,
phase and successful checking, then verifies that a real unreachable type error
still fails checking.

My fresh default `make -j8 bootstrap` passed on Linux ARM64 without changing the
shadow deadline. My three new methods and two adjacent range-bound methods
passed in 26.612 seconds with C seed, Stage1 and Stage2. The warning-only program
compiles and runs; an unreachable wrong return type and a failing mandatory
shadow still reject compilation and preserve the previously accepted output.
The repeatable target is `make test-unreachable-warning`.

This evidence covers warning severity and ordinary native compilation. It does
not resolve the separately retained product compiler abort or establish release
readiness. My bootstrap log is `/tmp/nanolang-unreachable-warning-bootstrap.log`.
