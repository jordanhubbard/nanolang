# Vendored Forth test suites

I vendor Gerry Jackson `forth2012-test-suite` **v0.15.0**
(`9773f84dd12390f342d37195da8848b04e1f4a23`) at `gerryjackson/`.

That tag includes Core tests and optional-word-set files. Optional files
are in the snapshot so the pin is complete. They are not Core evidence.
`make test-forth-jackson` does not run them.

Johns Hopkins and Jackson notices stay in the copied files.

I do not vendor `Forth-Standard/forth200x`. The per-file inventory is
`docs/FORTH_200X_INVENTORY.md`. I do not vendor Gforth.
