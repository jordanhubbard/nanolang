# I qualify checked field reads through fresh raw generations

At source pin 261f67bde I run `NANO_SHADOW_TIMEOUT_SECONDS=300 make -f Makefile.gnu -j1 bootstrap3`. All seventeen steps pass; both raw generations have SHA-256 693587593970b4588910fabb4f3c29a461c89031bb3974e7b61b000c18009389. Stage 1 raw generation takes 619.777s and Stage 2 takes 608.751s. Both native compilers pass hello compilation/execution and the installed compiler works with the C reference compiler removed.

I then run `NANO_SHADOW_TIMEOUT_SECONDS=300 make -f Makefile.gnu -o bootstrap3 -j1 test-checked-borrow-ledger test-checked-borrow-environment test-checked-borrow-metadata` against these exact completed products. The ledger and actual Environment gates pass; all eleven paired metadata/comparator methods pass in 87.978s. The target selects current NanoVirt and both fresh stages, rebuilds the actual raw emitter, and retains VM/native, source-context mutation/reset, failure preservation and advisory-only checks. All eight pinned compiler/runtime providers remain unchanged through this gate.

I retain the complete bootstrap logs/receipt and qualification log. Broader metadata neighbors will run once on the combined field-write candidate, rather than claiming their old results qualify these new shapes. This is Darwin evidence; exact Linux/Darwin release checks and full #998 remain open.
