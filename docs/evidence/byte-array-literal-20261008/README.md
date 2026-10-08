# Retained byte-array literal mismatch

I execute the previously retained artifacts and record explicit statuses and
SHA-256 hashes in `results.json`. These are artifact-level reproductions, not
a fresh compiler qualification at this worktree's source pin.

The source `let bytes: array<u8> = [300]` followed by an assertion that its first
element is 44 passes in the retained native C executable and fails in NanoVM.
Its retained disassembly contains `ARR_LITERAL 1 1`, the integer element tag.
The separate computed-value push control into an initially empty byte array
passes in NanoVM. Both source probes and the literal disassembly are retained.

At worktree base `fdca4ee9b`, `check_array_literal_annotation` records the expected
element kind, but `check_expression` reinfers a nonempty literal from its first
element. NanoVirt's let/return fallback restores declared kinds only for empty
literals. I must trace the exact annotation lifetime and qualify all contextual
array destinations before selecting a correction. Scalar byte literals have a
separate range-check policy; I do not assume that changing that policy repairs
array storage. Issue #979 remains open, including byte and nested slice parity.
