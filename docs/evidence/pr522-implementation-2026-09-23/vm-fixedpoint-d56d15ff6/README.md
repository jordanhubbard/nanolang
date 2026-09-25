# Final-source NanoVM compiler fixed point

At `d56d15ff6934dcd4872aa0f90bfe7ea828cf79fa`, I run the unchanged `tests.test_vm_bytecode_bootstrap` gate in a clean Linux ARM64 checkout. I exclude archived `docs/evidence` from the diagnostic checkout to fit the container disk; compiler source, tests, build inputs and the 1,800-second stage bounds remain unchanged.

Both raw compiler generations are 530,308 bytes with SHA-256 `e6cc1c67ce6c2a8dc2d9ea0729f02c3b157e12a01299d220cb531e80839fd143`. I compare their bytes without normalization. Generation takes 371.129 and 388.151 seconds; the complete gate passes in 776.915 seconds.

Both modules verify. The final generation compiles a source with an arithmetic assertion and shadow; that product verifies and executes. The exact declared host-library closure and assembler helper remain unchanged. My guard records zero NanoLang-generated C compilation calls during VM generations, while preserving native host-artifact compilation and cache checks.

The manifest records all commands, statuses, timings and hashes. Compressed logs retain each stage and the complete terminal. This establishes this route's fixed point on the pinned source and host closure. Standalone-native equality and final hosted platform/sanitizer acceptance remain separate requirements.
