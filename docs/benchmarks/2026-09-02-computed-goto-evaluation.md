# Computed-goto dispatch: measurement and decision (2026-09-02)

Roadmap 4.0, Phase 12, Execution architecture:
*"I will provide computed-goto dispatch where supported and retain a portable
switch fallback."*

This records the measurement that item implies, and the decision it supports.
**Recommendation: do not implement it as specified.** The measured ceiling does
not clear the bar in `docs/NANOISA_OPTIMIZATION_POLICY.md`.

## What the VM costs today

`bench/sum_loop.nano` compiled to `.nvm` and run under `bin/nano_vm`, nine
runs on one machine:

| | |
| --- | ---: |
| Retired instructions | 130,000,032 |
| Median wall time | 694.1 ms |
| p25 / p75 | 688.5 / 698.0 ms |
| **Cost per retired instruction** | **5.34 ns** |
| **Run-to-run spread** (p75-p25)/median | **1.4%** |

5.34 ns is roughly 17 cycles at 3.2 GHz, covering the whole per-instruction
path: dispatch-module resolution, validity checks, cursor advance, the profile
counter, the trace check, the superinstruction check, the opcode dispatch, and
the handler body.

## What computed-goto could save

Switch dispatch and computed-goto dispatch were measured directly, in a loop
that reproduces the VM's per-instruction shape: an identical prologue in both
variants, differing only in how the opcode is dispatched. 130 million
instructions, three runs:

| Variant | ns/instruction |
| --- | ---: |
| `switch` | 0.79 - 0.84 |
| computed-goto | 0.67 - 0.69 |
| **Saving** | **0.12 - 0.15 ns** |

Computed-goto is 14-18% faster *at dispatch*. That is a real effect and matches
the published range for threaded interpreters.

But dispatch is not what the VM spends its time on:

```
saving        0.13 ns
VM cost       5.34 ns
ceiling       0.13 / 5.34 = 2.4% of total execution time
```

**2.4% is the ceiling, and it is optimistic.** The measured prologue is lighter
than the real one -- it has no module resolution, no cursor seek, no
`byte_offset` validation and no trace branch -- so the real denominator is
larger and the real share smaller.

## Against the acceptance policy

`docs/NANOISA_OPTIMIZATION_POLICY.md` accepts a general execution optimization
when, among other things:

- *"its median improvement is larger than the observed run-to-run noise"* --
  the ceiling is 2.4% against 1.4% noise. Less than a factor of two, before
  accounting for the optimism above.
- *"the simpler implementation wins when performance is statistically tied"* --
  a 2.4% ceiling against 1.4% noise is close to tied.

## Against the cost

The dispatch switch is 2,291 lines with 173 `case` labels and 146 `break`
statements. Threading requires every handler-level `break` to become a jump to
the dispatch point.

The `break` statements are not uniform: some terminate the handler, others
belong to nested `switch`, `for` and `while` constructs inside handlers. A
mechanical conversion would rewrite both kinds, and converting an inner one
silently changes control flow rather than failing to compile. The failure mode
is a handler falling through into its neighbour -- wrong results, in the VM hot
path, with no diagnostic.

That is a poor trade for a ceiling of 2.4%.

## What this suggests instead

The measurement says dispatch is roughly a seventh of the per-instruction cost
and the rest is prologue plus handler work. If execution speed is worth
pursuing, the evidence points at the prologue and the handlers, not at how the
opcode is selected. `dispatch_module_for` is already a single pointer compare
on the common path, so the remaining candidates are the per-instruction
validity checks and the profile and trace branches -- all of which can be
measured the same way before anything is rewritten.

## Reproducing

```sh
make -f Makefile.gnu nano_virt nano_vm
./bin/nano_virt bench/sum_loop.nano --emit-nvm -o /tmp/sum.nvm
./bin/nano_vm /tmp/sum.nvm --profile-isa /tmp/prof.json   # retired count
# time nine runs of ./bin/nano_vm /tmp/sum.nvm and take the median
```

The dispatch comparison is a standalone C program; its source is quoted in the
pull request that added this document, and it depends on nothing in the tree.

## Status

The roadmap item stays open and unticked. It has now been *evaluated* rather
than merely deferred, and the evaluation argues against implementing it in its
current form. Reopening the question needs either a workload where dispatch is
a larger share, or a safe mechanical conversion of the handler bodies.
