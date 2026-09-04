# What I Measured, and What I Decided

[docs/NANOISA_OPTIMIZATION_POLICY.md](NANOISA_OPTIMIZATION_POLICY.md) says how I
decide. This records what the measurements actually were, including the ones
that led me to decline a change.

Reproduce with `make benchmark-nanoisa`. Set `NANOISA_BENCH_RUNS` for the sample
count and `NANOISA_BENCH_REPEAT` for iterations per sample.

## The harness measured the wrong thing

Until 4.0 the suite timed one `nano_vm workload.nvm` per sample. Those numbers
looked like this:

| workload | median | retired instructions |
|---|---|---|
| `nl_string_operations` | 17.51 ms | 78 |
| `nl_hashmap_word_count` | 17.42 ms | 109 |
| `nl_array_complete` | 18.13 ms | 244 |
| `nl_fibonacci` | 17.41 ms | 32,082 |

A workload retiring 32,082 instructions took the same time as one retiring 78,
because **process startup was the whole measurement**. Execution was under one
percent of it. The suite could not have detected an interpreter change of any
size, which is why every optimization decision it was supposed to inform stayed
open.

Each workload is now measured twice per sample — once with a single iteration
and once with many, both behind one startup — and the per-iteration cost is the
difference divided by the extra iterations. The startup terms cancel. Pairing
the two within a sample also cancels machine drift between them.

## Baseline

Apple clang 21.0.0, arm64, `-O3`, 8 samples, 1000 iterations per sample.

| dimension | median | IQR | IQR/median |
|---|---:|---:|---:|
| cold startup | 17.42 ms | 717.9 µs | 4.1% |
| `nl_forth_interpreter` | 282.7 µs | 4.5 µs | **1.6%** |
| `nl_fibonacci` | 211.2 µs | 3.6 µs | **1.7%** |
| `nl_extern_math` (in-process FFI) | 15.7 µs | 0.3 µs | 1.6% |
| `nl_array_complete` | 15.4 µs | 0.9 µs | 5.6% |
| `nl_function_variables` | 15.6 µs | 1.3 µs | 8.3% |
| `nl_string_operations` | 9.8 µs | 0.2 µs | 2.3% |
| `nl_hashmap_word_count` | 8.3 µs | 0.8 µs | 9.6% |

**The noise band is 1.6–2% on the two workloads long enough to measure well**,
and 5–10% on the ones that finish in under 20 µs. Only the first group can
support a decision; a change smaller than the second group's noise is not
measurable there at all, whatever its true size.

Cold startup is 60–1800× a single execution. For short-lived programs it is the
only number that matters, which is the argument for keeping the loader's work
bounded rather than the interpreter's fast.

### Co-process boundary

| dimension | median | IQR/median |
|---|---:|---:|
| `nl_extern_math` in-process | 15.8 µs | 5.7% |
| the same through the co-process | 764.2 µs | 5.0% |
| co-process launch | 57.9 ms | 14.9% |

Isolation costs about **48×** per crossing. That difference is the entire case
for batching: `vm_ffi_call_cop_batch` pays one signal/ack pair for a whole
batch instead of one pair per call, so its benefit scales with how much of the
764 µs is per-crossing overhead rather than work.

Launch is the floor for recovering from a crashed child. It measures standing
a replacement up; it does not include detecting the death or reissuing the call
that was in flight.

## Decisions

### Computed-goto dispatch — built, measured, accepted

Computed goto replaces the dispatch `switch` with a table of label addresses,
so each handler ends by jumping directly to the next one. The `switch` compiles
to a single indirect branch shared by every opcode, which a branch predictor
has little hope of predicting; a jump per handler gives it a separate site per
opcode with its own history.

Measured against the portable `switch` on the same machine, build flags and
workloads, 20 samples of 2000 iterations:

| workload | switch | goto | change | noise (IQR) |
|---|---:|---:|---:|---:|
| `nl_forth_interpreter` | 421.5 µs | 403.7 µs | **−4.2%** | 1.0–1.6% |
| `nl_fibonacci` | 207.7 µs | 205.0 µs | −1.3% | 1.2–1.9% |
| `nl_hashmap_word_count` | 9.5 µs | 9.4 µs | −0.5% | 4.9% |
| `nl_string_operations` | 11.4 µs | 11.2 µs | −1.4% | 7.6% |
| `nl_array_complete` | 18.8 µs | 18.7 µs | −0.7% | 4.2% |
| `nl_function_variables` | 18.7 µs | 18.5 µs | −1.0% | 2.8% |

Against the policy: the Forth interpreter improves by roughly three times its
noise band, no workload regresses at all, both the NanoLang and Forth groups
are preserved or improved, and nothing moves into allocations or FFI traffic
because the handlers are unchanged. The binary grows 48 bytes. Accepted.

Forth is where it shows because a Forth interpreter written in NanoISA is a
dispatch loop inside a dispatch loop: short handlers, many of them, taken in
data-dependent order. That is the shape computed goto is for, and the flat
results elsewhere are the same fact from the other side -- a workload whose
time goes into instruction bodies has little dispatch to save.

**An earlier revision of this document declined computed goto** on the grounds
that no measurement showed it beating the noise. That was true of the evidence
then and is the right default, but "no measurement exists" is a reason to
measure rather than a conclusion. Building it was cheap enough to settle the
question, and the question came back the other way.

The portable `switch` remains, selected by `-DNANO_NO_COMPUTED_GOTO` or by any
compiler without labels as values. `make test-dispatch-equivalence` runs every
program in `tests/` through both builds and compares output and exit status,
because the two strategies share their handlers and can only differ in control
flow -- a handler that falls into the next one instead of dispatching gives
wrong answers in one build and not the other, which no single-build suite can
see.

### Private superinstructions (fusion) — mechanism shipped, none enabled

`VmDispatchProfile` carries one opt-in flag per candidate fusion and every flag
defaults off, so an unconfigured VM runs the plain verified stream. The bar is
1% of retired baseline instructions in a maintained workload.

No candidate has been measured over that bar, so none is enabled by default.
This is the policy working rather than work left undone: the mechanism exists
precisely so a fusion can be evaluated without shipping it, and shipping one
that has not cleared the bar is what the policy forbids.

### Split payload/tag operand stacks — not accepted on current evidence

Splitting `NanoValue` into parallel payload and tag arrays would shrink the hot
stack's footprint and let untagged code skip the tag entirely. It also touches
every stack accessor, the verifier's depth model, the co-process serializer and
the FFI marshaller.

The measurable prize is bounded by what tag handling costs, and the workloads
that could show it — `nl_hashmap_word_count`, `nl_array_complete` — are the ones
whose noise is 5–10%. At that band a change under 10% is invisible.

Declined for 4.0 for the same reason as computed goto: the evidence does not
exist to accept it, and the policy does not accept changes on the grounds that
they ought to help. Making it measurable means longer-running workloads, not a
different opinion.

### Trap stack ranges instead of copying — not accepted on current evidence

`OP_CALL_EXTERN` copies up to `NANO_MAX_FFI_ARGS` tagged values into the trap.
Passing a range into the operand stack instead would avoid the copy.

The roadmap conditions this on measurement, and the measurement says the copy
is not where the time goes: the same FFI workload costs 15.8 µs in-process and
764.2 µs through the co-process. The boundary crossing is roughly 48× the
entire in-process call including the copy, so removing the copy cannot move the
isolated case, and the in-process case is already at the noise floor.

The copy also has a property the range does not: the trap owns its arguments,
so the operand stack can move underneath it. Giving that up to save something
unmeasurable is a bad trade.

## Reporting changes

Report a change as a distribution against this baseline: same machine,
compiler, flags, workloads and sample count, with the median and the IQR. A
single timing is not evidence — the IQR column above is why. Two of these
workloads have a 1.6% band and two have a 10% band, and a claim of "5% faster"
means something different in each.
