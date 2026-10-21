# I retain unchanged wrapper originals and separate shadow inputs

I extract these three sources verbatim from the named methods in tests/test_source_borrow_emission.py. Each shadow NASM comes from the separately pinned existing C-hosted shadow_driver, invoked with selection0/raw. I retain its exact executable/source hashes and all commands. These are independent selected-original-shadow inputs, not three fresh shadow producers or recovered source shadows. The paired reconstruction test also checks that each source still equals its original fixture method.

Expected ordinary and selected-shadow stdout is ABCABCABC for temporary_owner_fixture, ABAB for transitive_wrapper_fixture and ABAB for nested_owner_result_source. Assertions and once-only evaluation remain in the unchanged source. Full execution evidence is retained separately in owned-wrapper-complete-20261020.
