# My VM source locations

I report a failing instruction from its executing frame. My decoded next instruction is a continuation; a callee return address belongs to its caller. I retain an absolute executing instruction offset in each frame and initialize it at every entry, including tail calls and effect handlers. Existing explicit DEBUG_LINE locations retain precedence.

My fallback lookup selects the greatest debug offset at or before that instruction within the same function. Missing function-local entries produce an unknown line rather than borrowing a preceding function's map. A fused local-load/field-access reports the field instruction after the local load succeeds, including owned-array preflight.

My tests check direct, indirect and tail calls, taken branches, continuation maps, missing function-local maps, and actual fused/unfused field failures. They preserve diagnostic assertions rather than changing program behavior to fit a trace.

## Qualification

Source `4512c513149ee786c4f708eaa78d65337d26cabe` passes 274632 VM assertions and adjacent allocation gates on Linux, plus an explicitly compiled switch-dispatch VM. The same selected source passes ordinary and explicit switch builds and execution on Darwin with before/after source hashes unchanged. An earlier new test incorrectly expected no separator for an unknown location; its failure is retained and the corrected expectation requires `offset.nano:?`.

These results precede integration with the service-classification cache and affine changes in main `e0a5fdd6f`. Fresh integration checks remain required. This correction does not repair mutable capture storage or complete the NanoISA bootstrap release gate.

## Integrated checks

Integrated source `d4f38c14b2b17c38fda6127f4839b71892181386` passes 274632 VM assertions and the adjacent substring, callback, heap and stack allocation checks on Linux and Darwin. Ordinary checks take 10.541s and 13.598s. Corrected explicit switch builds and checks take 29.589s and 13.265s. Every selected source hash remains unchanged before and after each command. I retain the exact commands, raw output, status and selected-source manifest in [my evidence](evidence/vm-source-locations/inventory.json).

My first integration harness used an ordinary `CFLAGS +=` after a Makefile `override`, so its requested switch flag was ignored. Those successful runs are retained as additional ordinary checks, not switch evidence. I corrected the harness to use `override CFLAGS += -DNANO_NO_COMPUTED_GOTO`, used fresh object directories and confirmed the flag on the actual VM compile commands. Production source and assertions stayed unchanged.

Independent review of the call-frame and fused-phase changes passed. My full 5.1 acceptance and mutable capture repair remain open.
