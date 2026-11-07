# I qualify reconstructed tree projections with fresh compilers

I integrate component `3422f6fb2` as `e441aeffa`, rebuild my actual high-level facts extractor, and pass all 25 methods in 259.411 seconds through current NanoVirt and both freshly bootstrapped stages from `0d8cd446f`. Both raw compiler modules retain SHA-256 `6ca8dfd0d83e032f64a3247be6087cbc23ca1b76771dd0aeb7e4796b965d5fa2`. I change no source compiler or runtime admission for this component.

I run `tests.test_owned_nested_path_reconstruction`, `tests.test_owned_multi_reference_reconstruction`, and `tests.test_owned_exclusive_reconstruction.OwnedExclusiveReconstruction.test_shared_output_is_byte_identical`, with `NANOLANG_OWNED_RECONSTRUCTION_COMPILERS` selecting absolute Stage1/Stage2 paths and `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`.

The unchanged accepted tree fixture passes ordinary publication at 127 entry locals and 31 helper locals; its complete two-tree shadow uses 108 locals. I preserve every original assertion, reference mode and checked path. Adjacent sole-use total BOOL assertion folding applies only to the new Nano output; prior profiles and C output stay unchanged. The component receipt retains the original 140-local refusal and measured 124/132 shadow boundary.

I qualify original VM/native execution, reconstructed O0/O2 sanitized C and regenerated Nano, transport, branch/loop paths, later assertion failures, malformed identities and atomic refusals. Shared-byte stability and all twelve previous three-reference methods pass. Original and reconstructed acceptance remain distinct: CALL_REF in a loop pretest and wider shared-alias profiles remain explicit open source compositions. Full #999 and hosted Linux/Darwin release qualification remain open.

Command receipts and generated source/module hashes point to retained work directories; I do not remove those artifacts. Log copies normalize trailing display whitespace and retain original hashes. I attach the new suite to the regular owned-reconstruction gate after this successful terminal.
