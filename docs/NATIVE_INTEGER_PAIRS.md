# My native integer result pairs

MAC `task_ebf9bb417d9e4d6aa3b007c3fe868c92`, after verifier PR681.

I lower `I64_ADD_CARRY`, `I64_SUB_BORROW`, `I64_MUL_WIDE_S` and
`I64_MUL_WIDE_U` with the VM's operand order and exact integer tags. The
low word is pushed first, then the integer carry, borrow or high word.
Carry and borrow consume only the low bit of their third integer operand.
Tagged values use my existing exact integer extraction; I do not inherit
the separate binary-operation enum coercion or admit bool/U8 as INT.

I compute wrapping arithmetic in `uint64_t` and convert bits through
`ni64_from_bits`. Wide multiplication uses 32-bit limbs, with unsigned
high-word corrections for signed operands. I require neither signed overflow
nor compiler-specific 128-bit arithmetic in generated C. Each instruction
computes one pair into a scoped temporary before publishing its two stack
results, preserving source operand evaluation and stable join storage.

I retain the verifier prerequisite's ordinary arithmetic fixtures and replace
their recorded native refusal with VM/native parity. I add endpoint and
pattern products, carry chains, ordinary locals/calls/branches/loops and
strict GCC/Clang sanitizer execution. Existing native/shape/verifier gates
remain required. I do not execute malformed inputs or replay the unrelated
held carry-reconstruction compiler failure.
