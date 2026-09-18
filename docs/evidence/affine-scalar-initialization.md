# My scalar initialization meet acceptance

I track MAC `task_714b48782ed04de19976c790820227b4`. Contract `ae47d250`
precedes production `b3fc3e87`, based on main `8123c745`. I change only the
[declared scalar initialization analysis](../NANOISA_AFFINE_SCALAR_INITIALIZATION.md)
at owned joins. Source declarations and path-specific fallthrough remain
unchecked tasks `task_d3fdac5a43784608851272407743977b` and
`task_b077ad608868462da669b5a7a427567d`.

My meet first checks exact shared declaration facts, every non-scalar or
reference-mode local, stack provenance, region, caller origins and live
reference paths. Only declared mode-zero `int`, `bool`, `u8` and `float`
initialization bits can decrease. Loads retain the existing live-local check.
Exact state equality remains available unchanged. A rejected meet changes
neither authority nor scalar initialization in its destination.

A deduplicated circular worklist holds at most one entry per instruction.
When a stored scalar fact decreases, I recheck the affected instruction and
its successors. Each instruction has one initial visit plus at most one visit
per decreasing scalar bit. I enforce `instruction_count * (local_count + 1)`;
my existing bounds cap this at `4096 * 257`, within `uint32_t`. I retain distinct
reachable-instruction counts and separately report visits. A test-only reduced
ceiling proves refusal and subsequent successful analysis without changing the
production limit or adding a runtime option.

I measured these checks:

- State tests: 314 ordinary and 343 allocation-instrumented checks.
- Bytecode tests: 441 ordinary and 751 allocation-instrumented checks,
  including both predecessor orders, repeated descendants, initialized loop
  carries, unused loop locals, missing initialization at zero-iteration exits,
  forced visit ceilings and allocation failures during reprocessing.
- Both changed analysis files compiled with ASan/UBSan; their 314/441 normal
  checks passed with leak detection and halt-on-error enabled. Supporting
  unchanged objects were not rebuilt with instrumentation.
- New ordinary VM/native cases retain an owned root while a Boolean scalar is
  initialized in one branch, and while an integer temporary is initialized
  only in a zero- or multiple-iteration loop body. All twelve owned-runtime
  cases passed 1,521 checks, twenty VM invocations per case, and generated
  ASan/UBSan/LSan native execution with allocation-failure cleanup checks.

Logs are `/tmp/nanolang-affine-init-unit-final.log`,
`/tmp/nanolang-affine-init-sanitizers-r2.log` and
`/tmp/nanolang-affine-init-runtime-positive.log`. The owned-runtime Python
method passed in 34.853 seconds. Existing same-frame, nested, caller and
multi-caller reference gates, their allocation controls, caller analysis and
owned assertions also passed in `/tmp/nanolang-affine-init-authority.log`.
Those gates retain independent ownership and reference refusals.

That initial combined run stopped the owned-transfer target on an existing
stale diagnostic assertion. I recorded companion MAC
`task_baa6c6aa57ac4054b03f63890c8da46a` before repairing the harness. Successful
affine analysis with unsupported entry parameters or record results reaches
the exact entry/helper signature guard, rather than the old generic
"execution semantics" text. Failed affine analysis still takes precedence.
My first companion edit applied the signature expectation too broadly; its
unit assertion caught that error. I preserve both failed logs and require the
original acceptance/refusal decisions plus their appropriate exact reasons.
The corrected full gate passes 184 ordinary and 275 allocation checks, with
canonical roundtrip and execution passing in
`/tmp/nanolang-affine-init-transfer-r3.log`.

The first private sanitizer link command also failed: it included a CLI main
object and omitted decoder objects. I corrected only the harness object list;
`/tmp/nanolang-affine-init-sanitizers.log` preserves that link failure. Neither
incident is attributed to a runtime correctness defect or a held compiler
product failure.

I rebased cleanly onto main `58e0353d` after PR #622, at `dde5dced`. The two
production analysis files are unchanged from reviewed `b3fc3e87`. My runtime,
wire format, opcode inventory and source producers remain unchanged throughout.

The complete integrated gate passed at `dde5dced`:
`/tmp/nanolang-affine-init-integrated.log`. It reruns state and bytecode ordinary/
allocation checks, owned transfers, the twelve-case owned runtime, same-frame
references, nested references, caller analysis, caller references, multi-caller
references and owned assertions. The final owned-runtime method took 34.950
seconds. Its independent authority and allocation checks retain all refusals.
Only evidence and roadmap status changed afterward.
