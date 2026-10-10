# Darwin introspection bootstrap at aa0b13713

I ran `CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang make test-nanoisa-introspection` to terminal exit 0. All 17 bootstrap steps pass, with byte-identical raw Stage1/Stage2 modules and native smoke checks. Stage1 generation takes 216.116 seconds; Stage2 takes 222.396 seconds. My six-method introspection suite passes through nano_virt (1.968 seconds), installed Stage1 (1.321 seconds) and installed Stage2 (1.340 seconds).

I retain the complete terminal, manifest, step logs and host-input inventory. I do not retain generated binaries. This qualifies this Darwin checkpoint, not Linux, function-value/owned introspection routes, or the complete 5.1 release. #982/#976 remain open.
