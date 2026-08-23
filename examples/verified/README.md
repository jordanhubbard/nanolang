# My Verified-Subset Examples

This directory demonstrates my trust boundary. A filename or directory name is
not proof coverage. The current `--trust-report` classifies each named function:

- `[verified]` means the function uses the NanoCore subset covered by my stated
  Coq semantic theorems.
- `[typechecked]` means the function compiled but uses a construct outside that
  subset, commonly `assert` in these examples.
- A shadow is an executable test of named cases. It is not a theorem.

## Current Matrix

I generated this matrix on 2026-08-23 with `./bin/nanoc <file> --trust-report`
after the repairs in this directory.

| File | Verified | Typechecked | FFI / unsafe | Tested behavior | Remaining assumptions |
| --- | ---: | ---: | ---: | --- | --- |
| `break_the_proof.nano` | 7 | 0 | 0 / 0 | Five algorithm cases plus shadows | Algorithm names are intended specifications; integer edge behavior is not a domain proof |
| `checksum_validator.nano` | 6 | 1 | 0 / 0 | Concrete sum, Fletcher, polynomial, corruption, and block cases | Collision resistance is not claimed; integer ranges and modulus choices are caller assumptions |
| `financial_order_validator.nano` | 5 | 1 | 0 / 0 | Structural, risk, aggregate, balance, and price-band cases | Limits, deployment consistency, overflow bounds, and exchange policy are assumed |
| `medical_dosage.nano` | 4 | 1 | 0 / 0 | Eligibility, selected doses, rejection, and schedule cases | Formula, units, thresholds, integer ranges, and clinical suitability are assumed; this is not medical software |
| `pid_controller.nano` | 5 | 1 | 0 / 0 | Fixed-point cases, clamps, one update, selected simulations, and sampled bounds | Stability, tuning, plant fidelity, overflow bounds, and hardware timing are assumed |
| `proof_trace.nano` | 5 | 0 | 0 / 0 | Arithmetic, both conditional branches, both variants, and array sums | Printed rule descriptions are hand-written illustrations, not extracted proof traces |
| `sensor_voting.nano` | 6 | 1 | 0 / 0 | Consensus, degraded, failure, range, stuck-sensor, and validated-vote cases | Sensor independence, synchronized samples, tolerance choice, and integer ranges are assumed |
| `state_machine.nano` | 3 | 1 | 0 / 0 | Named normal, blocked, invalid, emergency, and sequence cases | Transition requirements, thresholds, state encoding, and physical-system adequacy are assumed |
| `verified_vs_unverified.nano` | 3 | 2 | 0 / 0 | In-range/out-of-range lookup and bounded/rejected integer cases | Fallback and assertion policies are design choices, not proved business requirements |

`main` is usually `[typechecked]` in domain examples because it contains runtime
assertions. The domain helpers remain independently classified. `println` alone
does not move a function outside NanoCore according to the current report.

## Evidence Boundary

**Proved:** For functions reported `[verified]`, my checked formal development
establishes its stated semantic properties within the NanoCore model. The trust
report establishes subset membership; it does not prove each function's domain
contract.

**Tested:** Each file compiles and its binary runs its shadows. The tests cover
the concrete cases written beside each function. The matrix names those cases
without turning them into universal claims.

**Assumed:** Domain formulas, thresholds, requirements, overflow preconditions,
physical models, and operational deployment remain outside these proofs unless a
separate specification and theorem explicitly cover them.

Run both checks when changing a file:

```sh
./bin/nanoc examples/verified/medical_dosage.nano -o /tmp/medical_dosage
/tmp/medical_dosage
./bin/nanoc examples/verified/medical_dosage.nano --trust-report
```

`proof_trace.nano` illustrates selected rule names. It does not read Coq proof
objects or report the compiler's runtime derivation. `break_the_proof.nano` no
longer compares a function with itself and calls that a test: it checks concrete
algorithm results while stating the stronger semantic theorem separately.
