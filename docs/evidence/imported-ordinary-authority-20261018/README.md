# Imported ordinary authority — focused Darwin evidence

I retain a bounded scalar/string/ordinary-record import publication slice for #1001. I do not close array/generic authority, the complete required row, or release qualification.

I built the current standalone emitter and canonical driver with `bin/nanoc_c`; these are C-seed-hosted Nano source binaries, not fresh raw Stage1/Stage2. Their build runs cover binding provenance owner/alias/ambiguity/reset shadows and bound-program/selected-shadow versus raw-emitter isolation. I retain the initial private-table-access build refusal and the corrected terminal build logs.

I ran `NANO_ORDINARY_COMPILERS=nano_virt,/private/tmp/nanolang-1001-canonical CC=/opt/homebrew/opt/llvm/bin/clang python3 -m unittest -v tests.test_ordinary_declaration_provenance`: five methods pass across both producers in 4.026 seconds. Each checks verified VM execution, exact layout/ownership bytes, assembly roundtrip stability and strict C11 ASan/UBSan native execution. The fixtures include local forward ordinals, repeated direct/transitive imports, same-spelled distinct imports, root fields containing same-spelled imported integer/string records, and arrays remaining UNKNOWN. I retain the first additional fixture failure: its expected string tag/order were incorrect; I corrected expectations to tag 5 and checked dependency-first ordinal order without changing producer behavior.

`make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-nanovirt` passes 90 codegen checks. I correct the prerequisite manifest's compiler annotation: that earlier Make invocation used Apple cc because an environment CC did not override the Make assignment. This run explicitly selects Homebrew clang.

I add `test-imported-ordinary-authority`, dependent on bootstrap3, to select nano_virt, nanoc_stage1 and nanoc_stage2 for the aggregate qualification owner. I have not run that fresh-generation gate or Linux here. I require the combined exact candidate before claiming those platforms/generations.
