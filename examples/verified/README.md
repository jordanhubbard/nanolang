# Verified Examples — Formally Proven NanoCore Programs

These examples demonstrate **real-world applications** of NanoLang's formal verification.
Every function in these files falls within the **NanoCore verified subset**, meaning
the Coq proofs in `formal/` guarantee:

- **Type Soundness** — if it typechecks, evaluation never produces a wrong type
- **Determinism** — same inputs always produce the same outputs
- **No stuck states** — well-typed expressions always reduce to a value

Run `nanoc --trust-report <file>` on any example to confirm all functions show `[verified]`.

## Examples

| File | Domain | Industry Lesson |
|------|--------|----------------|
| `medical_dosage.nano` | Healthcare | Therac-25: missing safety interlocks killed patients. Verified dosage calculator with eligibility gates and bounds. |
| `financial_order_validator.nano` | Finance | Knight Capital: $440M loss from inconsistent deployment. Verified pre-trade validation with risk limits and circuit breakers. |
| `sensor_voting.nano` | Aerospace | Ariane 5: integer overflow from unchecked conversion. Triple-redundant sensor voting with range checks. |
| `pid_controller.nano` | Embedded | SPARK/Ada pattern: verified control loop with anti-windup and output clamping. Integer fixed-point arithmetic. |
| `state_machine.nano` | Industrial | Therac-25/Nuclear: unsafe state transitions. Verified state machine where illegal transitions are impossible. |
| `checksum_validator.nano` | Data Integrity | Amazon s2n pattern: verified checksum library called by unverified code. Fletcher, polynomial, and block-wise checksums. |

## Design Patterns Used

These examples follow industry best practices for mixing verified and unverified code:

1. **Verified Critical Path** (SPARK/Ada) — Only the safety-critical logic is formally verified; I/O and UI use conventional testing.
2. **Verified Library** (HACL\*/EverCrypt) — Small verified functions are called by larger unverified applications.
3. **Verified Foundation** (seL4) — The core state machine or decision logic is verified; everything else is built on top.

## What Makes These "Verified"?

Unlike conventional tests that check *specific inputs*, formal verification proves properties hold for *all possible inputs*:

- The **shadow tests** in each file empirically demonstrate the properties
- The **Coq proofs** in `formal/` mathematically guarantee them for all inputs
- The `--trust-report` flag confirms each function uses only NanoCore constructs

The key insight: these programs use **only** NanoCore language features (int, bool, string, arrays, structs, unions, pattern matching, while loops, functions) — no floats, no FFI, no modules, no unsafe blocks.
