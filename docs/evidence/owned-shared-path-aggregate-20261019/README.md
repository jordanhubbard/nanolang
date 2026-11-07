# I qualify shared-path reconstruction in the aggregate

I integrate component c76c77b7 as e473681c1 and build `nvm2hl` with the ordinary Make target. I run the new shared-path corpus, the prior nested-path corpus and the old shared-output byte equality control with current NanoVirt and both retained fresh 0d8cd446f stages: 24 methods pass in 352.096s. The command sets `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang` and the space-separated `NANOLANG_OWNED_RECONSTRUCTION_COMPILERS` to both installed stage paths. Original/reconstructed VM and O0/O2 ASan/UBSan native comparisons remain enabled. The component receipt retains the detailed command and artifact evidence; aggregate test logs retain their new evidence directories.

I wire the suite into `test-owned-reconstruction`. These helper changes do not alter compiler source generation. Full #999 and exact Linux/Darwin release qualification remain open; the reference-call pretest follow-up is separate.
