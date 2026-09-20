# My VM source locations

I report a failing instruction from its executing frame. My decoded next instruction is a continuation; a callee return address belongs to its caller. I retain an absolute executing instruction offset in each frame and initialize it at every entry, including tail calls and effect handlers. Existing explicit DEBUG_LINE locations retain precedence.

My fallback lookup selects the greatest debug offset at or before that instruction within the same function. Missing function-local entries produce an unknown line rather than borrowing a preceding function's map. A fused local-load/field-access reports the field instruction after the local load succeeds, including owned-array preflight.

My tests check direct, indirect and tail calls, taken branches, continuation maps, missing function-local maps, and actual fused/unfused field failures. They preserve diagnostic assertions rather than changing program behavior to fit a trace.

## Qualification

Source `4512c513149ee786c4f708eaa78d65337d26cabe` passes 274632 VM assertions and adjacent allocation gates on Linux, plus an explicitly compiled switch-dispatch VM. The same selected source passes ordinary and explicit switch builds and execution on Darwin with before/after source hashes unchanged. An earlier new test incorrectly expected no separator for an unknown location; its failure is retained and the corrected expectation requires `offset.nano:?`.

These results precede integration with the service-classification cache and affine changes in main `e0a5fdd6f`. Fresh integration checks remain required. This correction does not repair mutable capture storage or complete the NanoISA bootstrap release gate.
