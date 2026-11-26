# My bounded Darwin compiler phase observation

I retain a diagnostic after my
[published original1800s Stage1 timeout](../darwin-stage1-generation-timeout-20261126/README.md),
under [#1265](https://github.com/jordanhubbard/nanolang/issues/1265) and #976.
Manifest SHA-256
`7b7df15d26a933d75769af9116f321a245254b9e5dc4ff326d44c32fbe0ab9ec`
binds35 new records. My reader also checks the parent68 records and all4683
pinned Git source blobs. My diagnostic and original factory acceptance are
separate; compiler_accepted and whole_release remain false.

I use a new APFS root at source d907f264d16d348b4589d3760bd213803e1c6f35.
I copy exact guarded tracked files and seven cold tools without hard links;
I rebuild seed/host preparation through my unchanged original bootstrap helper,
stopped before Stage1 generation. I copy no old built object/archive cache.
The original helper's seed and verification pass276.424826s and0.662000s under
stricter diagnostic600/60 limits; the whole prefix command takes280.725478s.
This preparation differs from my original factory's cache state, so these times
are not a controlled performance comparison. Compiler shadows remain30 and
ordinary invocations retain default10. I change no original acceptance budget.

I then run the original Stage1 VM argv with existing phase-recorder hooks,
original no-generated-C guard and a300s outer diagnostic cutoff. The VM is
terminated by that cutoff: exit-15, elapsed300.072803s including cleanup.
Marker times below are relative to the observer's VM start, not CPU durations:

| Marker | Seconds |
| --- | ---: |
| inputs-start | 0.885 |
| imports-done | 37.433 |
| lex-done | 47.444 |
| parse-done | 61.992 |
| checked-emit-start | 69.398 |
| typecheck-start | 69.447 |
| typecheck-done | 294.712 |
| program-lowering-start | 294.828 |

The typechecking marker interval is225.264956s. Program lowering begins before
the cutoff, but I record no lowering completion. This observation establishes
neither the cause of the original1800s failure nor a lowering defect. I make no
unchanged full bootstrap retry or speedup claim.

I retain the actual new seed, six new host dylibs, prefix receipt/guard/logs,
observer source/request/environment, all phase events and full command streams.
Every non-constant section of the old and new raw seeds is byte-identical,
including478639 code bytes (SHA-256
`3e29df75749f3c9995a6852137e074b5d7bff8e7dd9a4b5fbd4e05ff49a442d9`).
Exactly eight of16264 constants change: six receipt-bound host paths and two
tracked File provider source paths with identical source hashes. My comparison
permits only those eight exact substitutions. Whole modules are different:
1110640 versus1110644 bytes. This comparison explains diagnostic input identity;
it does not substitute normalized equality for my required raw Stage1/Stage2
fixed point or paired-platform acceptance.

Both admissions pass APFS, space, CPU and active-job checks. All4683 tracked
source/seven image/nine tool/new seed/six new host/working-input guards match.
I also verify the preserved failed factory's tracked files, cold images, terminal
result and ten extra seed/index/receipt/host inputs against its sealed evidence.
No owned process remains, the controller is absent and its reservation releases.
No C product source/candidate, owner checkout, global package/toolchain/SDK or
NAS/TCC/migration state changes. The #1262 handoff remains pending.

I still require a demonstrated cause, measured repair and fresh original17-step
bootstrap/raw equality/installed/no-C-seed/native36/emitter12 qualification.
Pure translator/native body/runtime/profile/ownership/installed cutover, current
compiler-source/private, exact-candidate CI and all69 release gates remain open.
I preserve existing first-name/duplicate/owner/body/indirect-call and empty-print
contracts while selecting the next bounded measurement inside lowering.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/darwin-stage1-phase-diagnostic-20261126/read_terminal.py --repo .
```

The reader executes only Python evidence readers/comparisons. It launches no
compiler, VM or product and makes no current process-liveness claim. Without
--repo it explicitly reports zero pinned Git source-blob checks.
