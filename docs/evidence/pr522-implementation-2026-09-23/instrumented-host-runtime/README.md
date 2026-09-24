# Complete instrumented host-runtime qualification

At `c5357ffef`, I freshly compile `dyn_array.c`, `gc.c` and `gc_struct.c` with Homebrew Clang ASan/UBSan and link a private relocatable host-runtime object. Build commands and source/object hashes are retained. My runner substitutes this object into the unchanged One IR suite, selects instrumented native compilation, and enables leak and stack-use-after-return detection. VM, translator and compiler seed executables remain ordinary builds. The existing per-command deadlines and CI shadow limit remain unchanged.

The complete run finishes in 458.990 seconds: 28 methods pass and three fail (`one-ir-terminal.log`). Both complete compiler paths overflow the stack in generated recursive-parser code under ASan; I track that defect under `task_ec9e85c7c9ec4521bde658929f8a5e2f`. The third failure is the real std ABI mismatch below. I do not count the suite as passed.

A targeted diagnostic of the std artifact method reproduces an incorrect-function-type UBSan failure at `fs_walkdir`. The generated adapter casts the symbol to `nh_array_value *(*)(const char *)`; the real library exports `DynArray *(*)(const char *)`. The independently named anonymous structs do not provide an exact function type to the sanitizer. I retain function sanitization, the real artifact and host ABI checks; I track shared compatible C declarations under `task_89d0cef12ac34a1380fb952ff368be77` before implementation.

The retained hosted sanitizer-source failure at `6813cac92` is the two list-constructor bytecode mismatches already corrected locally at `9237c78e5`. It supplies no evidence of final hosted success.
