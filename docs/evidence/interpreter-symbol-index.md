# My interpreted shadow lookup cost

I track my default shadow deadline in MAC
`task_628759a2daf743b9bf13c9a7fea2ced0`. I keep the ten-second default and every
selected dependency/root shadow. I do not treat a passing retry as an explanation
for an earlier timeout.

## Measured boundary

The supervised C-seed interval starts before callback-policy selection and runs
all selected shadows. Module builds and FFI loading precede it. One diagnostic
run at `a681a65d` uses the existing explicit 60-second override and a PTY for line
buffering. All 823 shadows complete in 9.953435 seconds between the runner's
start/completion messages. This interval excludes callback selection and is
already close to the default deadline. Resource-prefixed shadows account for
4.134794 seconds; NanoISA-prefixed shadows account for 2.720589 seconds.

A bounded debugger sample of normal shadow execution captures 24 stacks. Twenty
one include the variable-lookup family (`env_get_var` or same-file lookup); only
one includes function lookup. Calls such as `str_length` first search for a
lexical function-variable binding. A missing binding previously scanned every
symbol. The samples select an optimization target; they are not whole-run CPU
percentages or proof of every timeout's cause. Host load was nonzero.

I leave the host profiling policy unchanged: `perf` is unavailable at
`perf_event_paranoid=4`. An initial `stdbuf` timing attempt stops before shadows
because its `LD_PRELOAD` conflicts with the existing capture contract. PTY
buffering avoids that injected-library conflict.

On unchanged NanoLang source pinned to `28644fc3`, the baseline compiler hits
the default ten-second deadline. With the index implementation subsequently
committed as `6910c066`, all 826 shadows complete in 2.392369 seconds at the same
default deadline; the full native compiler build succeeds. These are measured
runs on this host, not a universal timing guarantee. I preserve both results:
`/tmp/nanolang-symbol-index-{baseline,after}-{timing.jsonl,metadata.json}`.
The earlier diagnostic and sample evidence is
`/tmp/nanolang-shadow-timing-pty-{lines.jsonl,metadata.json}` and
`/tmp/nanolang-shadow-cpu-samples.log`.

## Index contract

I retain the symbol vector as authoritative storage. Hash buckets hold slot
indices; each slot stores its hash and the prior bucket entry. I insert in slot
order, so a chain yields the most recent matching definition first. Same-file
lookup filters those candidates by the existing pointer-or-text file equality.
Source-position visibility and function/module resolution are unchanged.

I synchronize scope rollback using saved hashes and links. A popped name may
already be freed, so I never dereference it while removing a link. Normal
insertion synchronizes before it reuses a slot. Symbol-array relocation cannot
invalidate index entries because they contain no Symbol pointers. The separate
C-header-constant insertion path explicitly invalidates the index. In-place
value, type and source metadata updates remain visible through the live slot.
Allocation failure discards the optional index and uses the original reverse
linear lookup.

## Validation

My dedicated fixture compares indexed results with reverse scans through repeated
shadowing, freed-slot reuse, full resets, imported raw slots, null names, file
identity and all three initial index-allocation failure points. A deterministic
4096-symbol workload performs 2000 lookups with 1000 name comparisons. I keep the
same fixture in `test-units` through `test-env-symbol-index`.

The ordinary fixture passes. The fixture's env/index translation unit also
passes ASan, UBSan and leak detection, linked to the existing runtime objects.
The first leak check exposed a pre-existing 208-byte import-tracker container
leak, recorded as `task_73162d300ee44cb4a16d723bd49beeec`. I release the owned
tracker array and container; I do not invent ownership for future entry payloads.
The initial failure remains in `/tmp/nanolang-symbol-index-sanitizer.log`, and
the corrected check is `/tmp/nanolang-symbol-index-sanitizer-r2.log`.

All 39 environment scoping checks, parser/typechecker/interpreter unit tests,
ten lexical-boundary methods and two interpreter/VM callee-snapshot methods
pass. Full bootstrap and imported callback metadata checks are running; I record
their final results before completing this slice.
