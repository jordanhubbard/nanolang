# I qualify the integrated metadata batch on Darwin

At 119683b80, `CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang make test-nanoisa-introspection` exits 0. All 17 bootstrap steps pass, including raw Stage1/Stage2 equality, verification, native translation and smoke execution. All eight shared methods pass through nano_virt, installed Stage1 and installed Stage2 (2.695, 2.548 and 2.512 seconds). I retain complete step logs, input/tool/artifact hashes and the terminal.

My fixtures cover direct and function-value metadata operations, declared source paths, single index evaluation, empty/ambiguous modules, malformed signatures and ordinary lookalike functions. Owned direct calls preserve a live resource across all eight operations; leak and moved-value controls reject before publication. VM/native products pass the LLVM sanitizer checks. Owned function values, Linux and full release qualification remain open.
