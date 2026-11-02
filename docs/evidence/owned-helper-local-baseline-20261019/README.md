# My unchanged helper-owner baseline

I retain this design-only baseline at aggregate3446ec677 plus rebased design
6225c7a46/3df0982f9. I run no compiler build and change no production admission.
`probe.py` records exact commands, process-group timeouts/cleanup and byte outputs.
`pins.json` pins copied current NanoVirt/Stage1/Stage2, VM/assembler/facts and
reconstruction sources. Every listed hash is checked again after execution.

All three ordinary producers compile the unchanged retained fixture; original
verification and VM execution pass (exit0, empty stdout). Each ordinary module
contains main()->INT with12 locals and combine(&mut Leaf,&Leaf)->INT with41
locals. `module-summary.json` retains canonical hashes, declarations and opcode
counts; assembly retains exact ownership/path descriptors and instruction order.

The separately selected original-shadow executable is produced by the retained
C-hosted driver from `/private/tmp/owned-string-baseline-20261019/shadow_driver`,
not by three newly built shadow producers. Its original provenance is retained
in `../owned-string-reconstruction-baseline-20261019/`. It verifies and executes
(exit0, empty stdout), with exactly two functions: synthetic entry12 locals and
combine41 locals. This establishes original-shadow cardinality rather than
assuming it from the ordinary source.

Current facts extraction refuses every input with its retained exact diagnostic;
both C and Nano CLI reconstructions refuse and preserve prior output bytes.
No bypass, generated program, native qualification or reconstructed local count
is claimed. Existing source limits remain unchanged. The128-local generated
entry/helper/shadow feasibility and new numeric witness implementation remain
pending review. Historical alternate/else-zero/former-refusal evidence stays
separate; this fresh baseline executes the unchanged anchor only.

I checked worker capacity before serial execution (CPU idle15–22%,232GiB free).
No binaries are duplicated in this receipt. Full #999 and platform gates remain
open. The known beads precommit warning is unrelated to this baseline.
