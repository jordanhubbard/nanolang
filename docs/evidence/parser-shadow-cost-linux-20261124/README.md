# My bounded parser shadow cost observation

I investigate #1229/#1241 using compiler/provider source at
`3331bdfaaf66176e7b1dc704285768a4692bf873`, with no production source edit.
I do not repeat the failed full 1,306-shadow compiler gate.

```sh
python3 docs/evidence/parser-shadow-cost-linux-20261124/read_archive.py
```

My independent reader verifies 3,857 records, 3,518 Git inputs and 43 retained
ARM64 native images. I check compressed/decompressed hashes and sizes, exact Git
source, native/provider/tool identities, raw PC/LR samples, mappings, actual
addr2line commands/results, allocation counters and every selected shadow event.
I retain the actual libc/loader debug files used by the recorded symbolization.
I reuse identical compressed inputs from the preceding source-C receipt.

## My observation boundary

I copy the entire original `src_nano/parser.nano` byte for byte into a fresh
retained root, then append only:

```nano
fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
```

I preserve all 276 original shadow declarations and assertions. I select
`--root-shadows-only` explicitly. This is a diagnostic subset: it has fewer
loaded declarations and a different execution context from the full compiler.
I retain the unchanged 600-second outer limit and original default 10-second
supervised shadow deadline, including SIGALRM. I set no timeout override.

My first ordinary/private pair adds the existing private 1ms PC sampler through
link-time `dup2` wrapping. Both runs complete all 277 shadows with zero failures,
write completion receipts, exit 0 and produce native artifacts. The sampler
records 690 ordinary and 1,047 private PC/LR pairs. Ordinary sampled PCs include
310 in libc allocation and 82 in GC registration; private samples concentrate in
carrier walking, child traversal, visited keys and field lookup. These are
interrupted-PC observations, not complete stacks or exclusive-time profiles.

I then predeclare one new pair with fixed-counter allocation/resource observation
on identical parser bytes and unchanged production providers. It wraps direct
linked `malloc`, `calloc`, `realloc`, `free` references and `_exit`, without an
allocation ledger, allocator-policy change or ownership change. Both runs again
complete all 277 shadows with zero failures, receipts and exit 0.

| Shadow-span observation | Ordinary | Private collector |
| --- | ---: | ---: |
| Direct wrapped malloc calls | 1,386,245 | 1,572,507 |
| Direct wrapped malloc requested bytes | 475,246,027 | 513,936,379 |
| Direct wrapped calloc calls | 1,850 | 183,890 |
| Direct wrapped realloc calls | 477 | 501 |
| Direct wrapped free calls | 1,155,357 | 2,706,541 |
| User CPU | 0.353397s | 1.005581s |
| System CPU | 0.310633s | 0.043671s |
| Minor faults | 108,788 | 2,671 |
| Major faults | 0 | 0 |
| GC objects before → after | 99 → 223,306 | 99 → 99 |
| GC payload/header accounting before → after | 4,059 → 11,118,995 bytes | 4,059 → 4,059 bytes |

The observer spans the shadow child after stdout redirection through its normal
exit. CPU and fault counters use that child's `getrusage(RUSAGE_SELF)` deltas;
whole-command CPU records are separate. Request-byte totals are cumulative,
include realloc requests without subtracting old storage, and do **not** measure
live heap. Link-time wrappers do not count allocator calls made internally by
shared libraries, including internal `strdup` allocation. Free-call totals cannot
be subtracted from wrapped allocation counts to infer live objects.

GC statistics measure their existing registry, not all native allocations or
physical resident memory. I observe 223,207 extra managed objects /11,114,936
accounted bytes in ordinary execution; I do not classify their object types from
these aggregate counters. The private run returns to its starting GC counts.
I preserve these differences without claiming a historical timeout cause,
controlled speedup, complete ownership, bounded process RSS or default admission.

All four generated native entry points execute and return 0. Their added `main`
is a link/entry smoke test; the parser assertions above execute in the C seed's
interpreter child. I do not claim native parser functional coverage from that
entry point.

## My guards and retained setup failures

Each successful pair retains all 3,500 selected tracked-input hashes before and
after. Compiler/test inputs match the source commit. The only tracked difference
during preparation is my predeclared roadmap row; I retain its exact version for
each pair separately. All link provider and 24 actual compiler/loader identities
match. I retain every observer source, command, image, native product and stream.

I preserve two failed setups with zero shadow starts:

- My initial `--emit-nvm` invocation refuses because that C-seed path accepts only
  checked File graphs. I retain the option refusal and correct the observer to
  the ordinary native path.
- I initially link the diagnostic executable into its temporary work directory.
  This development C seed derives its source root from the executable's parent
  layout, so imported native compilation cannot locate `runtime/nl_string.h`.
  I retain that generated C and diagnostic, then link the observer under the
  same repository `bin/` layout as the source-qualified image. This temporary
  uninstalled executable is distinct from the installed deployment gate #1167;
  I do not claim an installed-product regression or change its accepted scope.

The sampler-only and counter-observed pairs have separate fresh roots. I do not
mutate either setup failure or an earlier successful root to obtain a pass.

## My remaining work

I have localized this bounded parser workload to allocation/registration traffic
in ordinary mode and carrier-graph traversal in private mode. I must measure and
qualify any subsequent repair before full-source admission. The existing ordinary
ownership and fresh-STRING/caller/aggregate contracts under #1116/#1213 remain
required; I will not enable collection or discard borrowed/escaped values merely
to reduce these counters. An optimization must preserve complete typed-list,
record/union/tuple/array, malformed-input, alias and fault guards.

No product compiler, allocator policy, collector cadence, assertion, deadline,
compiler pin or admission table changes here. I leave both full-source gates,
worker admission, default ownership, paired platforms, raw/installed bootstrap
and all 69 release requirements open. I establish no historical timeout cause.
Puck capture/regex/merge/CI, frozen roots/audits/stashes and NAS/TCC/migration
pauses remain unchanged. All my native jobs are terminal; I hold no Darwin job.
