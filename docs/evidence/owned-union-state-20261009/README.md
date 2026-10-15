# My selected-union ownership state

I implement local-normalized resource-union transitions under #981. Construction
checks the complete selected field range before moving any owned child. Moving
retains the variant. Destructive unpack requires a proven variant, consumes the
source and initializes exact destinations. Nested union payloads become unknown
variants until a new match refines them. Generic record pack/unpack never treats
concatenated union fields as a record. Resource-union destinations cannot be
overwritten, and initialization meets cannot discard live ownership.

The tests cover Some, Empty, two-owner Pair, a nested union and nested Empty;
wrong types and variants; unknown selection; duplicate owned fields; occupied
destinations; repeated moves/unpacks; forbidden flat field access; and branch
joins. Every rejected state transition is compared with its prior clone.

The final Make gate passes 413 affine-state checks and 445 allocation checks,
546 affine-bytecode checks and 856 bytecode allocation checks, ownership
transport/refusal, and the existing sanitized scalar-union VM/native execution.
Both original and expanded test runs are retained. The original transport test
now creates a state successfully and still requires verification/native refusal.

This implements state transitions, not bytecode/CFG or source execution. The
explicit complete-union refusal moves from state creation into bytecode analysis.
Next I must transfer union tokens through the bytecode stack, implement checked
selected unpack and exact call/return/join behavior, and carry these facts into
both frontends and VM/native destruction. No source acceptance fixture is removed
or narrowed, and #981 remains open.
