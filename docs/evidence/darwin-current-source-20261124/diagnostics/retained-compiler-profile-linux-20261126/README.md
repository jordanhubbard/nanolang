# My retained-input compiler profile

I continue #1265/#976 on sparky at source
`d907f264d16d348b4589d3760bd213803e1c6f35`, after publication
`cdbb2dfafb9f8131abfb5f332814fab82a755edf`. My product source tree is exactly the
pinned baseline. My observer and bytecode witness live outside that tree.

## What I measured

I select a test-only Python compiler wrapper in a fresh guarded host context.
It forwards the original arguments and standard I/O to `/usr/bin/cc`, preserves
heritable descriptors and exit status, and records wall and child CPU time.
Synchronous shell markers delimit each artifact lookup. I use no ptrace or
preload. All four lookups record successful cache reuse and return the exact
paths and bytes from this context's pinned prefix libraries.

| Lookup | End-to-end seconds | Compiler invocations | Recorded compiler wall / child CPU seconds | Assembly emissions / wall seconds |
| --- | ---: | ---: | ---: | ---: |
| `file_product` first | 43.549767 | 331 | 25.691181 / 25.253730 | 85 / 21.422708 |
| `file_product` repeat | 43.730018 | 331 | 25.713004 / 25.078889 | 85 / 21.178067 |
| `nanoisa` first | 14.407527 | 172 | 4.787277 / 4.688695 | 43 / 2.531446 |
| `nanoisa` repeat | 15.174651 | 172 | 5.469832 / 4.850662 | 43 / 2.610803 |
| Empty input | 0.024225 | 0 | 0 / 0 | 0 / 0 |

The wrapper changes compiler identity and adds interpreter/process overhead.
Recorded spans begin after observer initialization, and child CPU excludes the
observer. I do not equate these end-to-end times with the earlier production-tool
25s/7s observations. Within each lookup, recorded compiler spans do not overlap
and lie wholly inside its synchronous enter/leave markers.

Each `file_product` lookup performs81 preprocessing calls,85 C-to-assembly calls,
162 object-emission driver calls, one shared-link validation and two tool queries.
The metadata contains81 ordinary/shared source units. Each `nanoisa` lookup
performs42/43/84/1/2 corresponding calls across42 source units. All recorded
lookup compiler invocations exit0. Four source inputs are emitted to assembly
twice within each `file_product` validation snapshot; one is repeated for
`nanoisa`. Their argument vectors differ only in output paths. I have not proved
output-byte equivalence or that reusing those outputs would preserve authority.

The repeated `file_product` calls together take1.954899s in the first lookup and
1.899437s in the second. That is only part of the21s assembly-emission work.
I do not substitute removing this small duplicate subset for a complete repair.
The largest individual first-lookup assembly emissions map to
`managed_array_shapes.c`1.533362s, `file_flow.c`1.478046s,
`services_flow.c`1.384521s, `socket_flow.c`1.243962s and
`websocket_flow.c`1.202408s. The exact arguments and source-index mapping remain
in my replayed report.

My source reading places the work in `hashes_match` →
`module_preprocess_fingerprint` → `module_gcc_validation`, with retained snapshot,
assembly capture/read capture and object replay beneath it. The invocation data
measures those work categories; it does not prove that any correctness check can
be removed or establish the cause of the original Darwin1800s failure.

## My execution and controls

My fresh prefix completes in150.809625s. Witness compilation, dump and execution
exit0 in2.372364/0.019324/117.116767s. I record4486 start/finish events for2243
compiler invocations:1213 during seed preparation,24 during witness compilation,
and1006 during the four positive lookups. Every start has a matching finish.
The empty-input refusal launches no compiler.

My separate observer controls compare the exact wrapper bytes with direct
compiler execution. Version output, invalid-option refusal, stdin preprocessing
and an inherited input descriptor produce identical stdout, stderr and exit
status. These four cases test the observer; they do not qualify a compiler
product or prove equivalence for every possible invocation.

My44-record base manifest is
`c9ddffbb5bb4d8672f4c0eef06fa7e7f260b12d5c3e404d823eab1762cdd47e1`.
My separately sealed22-record post-terminal observer supplement is
`96428d4884e1e2d4a0bcc19f77cc1f8174898d0fde16182a37802b045bf4e5ee`.
The reader verifies all66 records,4683 pinned Git blobs, the prior151-record
helper/boundary evidence and its original prerequisite chain. It independently
replays compiler events, synchronous intervals, source-unit mapping, grouped
wall/CPU totals and all four observer controls.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/retained-compiler-profile-linux-20261126/read_terminal.py --repo .
```

All baseline source, nine copied images,22 selected tool/library files, Python
observer tool, observer/witness sources, six host libraries, seed, working inputs
and original accepted-root guards match. The controller is absent, no owned
process remains and the reservation is released.

I retain seed600/verify60, compiler-shadow30, ordinary10 and witness180/60/180
budgets. I change no original qualification1800 limit, assertion, C product
source, compiler/bootstrap/CI selection or source/tool/environment/link/ownership
authority. This is an observer-specific diagnostic. A reviewed repair or scoped
C-owner handoff, focused stale/malformed/changed-input controls, original native/
bootstrap/candidate/paired acceptance, #1262, installed pure-translator cutover
and all69 release requirements remain open. I start no Puck build or unattended
loop. The profile was terminal before the new shared-volume storage instruction;
I preserve its original root and place subsequent publication work on that volume.
