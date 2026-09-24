# Integer-list constructor bytecode

I reproduce the hosted pinned-subset failure with the unchanged full `make test-nanoisa-src-nano`: 84 checks pass and exactly `blank_l` and `grow_l` fail. The target freshly builds the canonical emitter. A separate direct comparison reproduces the same mismatch; its initial setup lacked the comparison executable, which I subsequently built from the target's compile command.

I align the C-seed bytecode frontend with the canonical emitter: integer-tag list constructors emit `ARR_LITERAL TAG_INT 0`, preserving an exact integer element kind for native aggregate storage. Other element tags retain `ARR_NEW`. Both instructions create an empty integer array of capacity eight in the VM. I do not change either raw opcode contract or weaken bytecode equality.

The pinned comparison now passes all 86 checks. `make test-nanovirt` passes all 90 tests, including the empty struct-list element-tag control. Four focused canonical list methods pass in 1.369 seconds: nested record lists, list mutation, invalid operands and bounds traps across VM/native products.

The broader target runs as `make -o nanoisa_emit test-nanoisa-src-nano`, reusing the emitter freshly rebuilt by the baseline because no NanoLang source changes. Its remaining Python suites are still running; I do not claim that gate passed. Final hosted acceptance and final-source fixed points remain open under `task_99b74dba668b48a18287a989258f7ea8`.
