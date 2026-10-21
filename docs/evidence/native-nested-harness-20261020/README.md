# I retain strict nested-array harness qualification

Hosted Linux job118327388822 at271c88986 fails three actual nested-array C builds under GCC -Werror=misleading-indentation. The offending line belongs to the test-added collection wrapper, not the translator. `gcc-before.log` independently reproduces the unchanged line with GCC16.

I brace the collection abort and put the return on its own line. I preserve the collection-count requirement, every fixture assertion, sanitizers and warnings-as-errors. `nested-gcc.log` passes three actual assembler/verifier/VM/native tests in2.914s with CC=/opt/homebrew/bin/gcc-16. `clang-neighbors.log` passes all nine nested/byte-array tests in4.093s with CC=/opt/homebrew/opt/llvm/bin/clang. Native outputs retain ASan+UBSan and -fno-sanitize-recover=all. These are Darwin local checks; hosted Linux rerun remains required.

I copy only the aggregate nanoisa/nano_vm/nvm2c tools into the isolated271c checkout, with exact hashes in providers.json; I build no compiler generation. Commands and initial hosted failures are retained. The separate SourceSnapshots capture/backend investigation remains open under #976.
