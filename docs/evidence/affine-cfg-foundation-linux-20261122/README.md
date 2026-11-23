# I retain my affine CFG foundation and its C AOT blocker

I publish source `45223340186ae620fad7ed6272b13fb497320e8a` under #1142/#1202 in PR1168. I implement frame stack/local/global/handler joins, nested cloning, selected-union refinement, bounded FIFO scheduling and initialization/lexical-return relation algebra in pure NanoLang. I retain my full CFG requirement: instruction transfer/convergence and handler/caller/global/effect/private composition still need implementation and qualification. No executable selector uses these new components.

| Linux ARM64 observation | Result |
| --- | --- |
| Relation/FIFO original Make gate | Five source routes pass, 218.903s |
| Relation corpus | 168386 operations, every pair of289 normalized flow states |
| FIFO corpus | Seven full/deduplicating/wrapping traces, capacities1,2,3,7,64,256,4096 |
| Complete relation/FIFO carriers | 336, identical reports through actual C and all five routes |
| Combined frame corpus | 1001 carriers, including665 frame cases |
| Frame constructed routes | C-native, C-produced VM and NanoLang-produced VM match actual C |
| Required frame C AOT routes | Refuse during construction; full five-route target remains failed |
| Original shape controls | 3890 pass, including deliberate exact deferred-read refusals |
| Original identifier controls | Five methods pass;3442 fragments lexed,135 lexer refusals retained |
| New function shadow policy | Pass |

My oracle includes the unchanged actual `affine_bytecode.c` implementation and calls its internal clone, stack-meet, propagation, refinement and enqueue helpers. Dequeue follows the production loop's inline FIFO operation. Independent Python sets derive global relation expectations; a deque models scheduling. Frame checks retain exact owned/observation provenance, layout/array-leaf identity, every excluded-arm bit, handler key/site/floor/activation, local initialization, global-domain ordering and failed-join input preservation. The NanoLang driver also perturbs independently cloned masks, globals and local storage after every result.

My full report SHA256 is `f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`. All three constructed source routes match actual C. My five-route relation/FIFO report SHA256 is `fd4a902b3768698167440750303d893d5139d1bd3e44226a251d93d771c82984`. Both source-produced sanitized native products pass. I use installed GCC13/as2.42 and my real previously qualified09cab Stage2 producer (image346dc0d7), preserving ordinary shadow10 and the existing600 child/build limits. This does not qualify a new bootstrap.

I preserve two failures. First, my driver incorrectly uses statement-style cond braces; actual C expectations already pass, but NanoLang parsing refuses. I retain that source/log and change only dispatch to ordinary if/else. Second, both corrected source producers reach the actual C AOT refusal under [#1202](https://github.com/jordanhubbard/nanolang/issues/1202). A seven-line root importing the frame library reproduces it. Read-only GDB maps the original failure to stack_meet local28 and deferred array_read4: an exact Boolean element/read result is reached by an optional conversion. Existing strict shape tests deliberately refuse that relation. I preserve those assertions; a caller-side repair remains unverified. Both retained producer modules preserve binary prior output when C AOT refuses in [my additional refusal observation](verification/prior-output.json). Simple direct Boolean-array and record/record-array neighbors translate successfully, so I claim no universal array-store cause.

My archive retains4576 records/238625371 logical bytes,1364 exact Git source blobs, the generated module index separately, all corpus files/products/reports, prepared native tools, source compiler, failed attempts and diagnostic observers. I reuse exact immutable objects from two declared earlier manifests. My export observer initially mistakes the generated index for a Git blob; I retain that observer refusal and classify the index separately. I preserve original/Puck/frozen roots, dirty audit and stashes. I hold no native worker reservation; the NAS/migration/TCC operational pause remains unchanged.

```sh
/usr/bin/python3 docs/evidence/affine-cfg-foundation-linux-20261122/read_archive.py
```

My [reader result](verification/linux.json) is `INTEGRITY_PASS_WITH_RETAINED_FRAME_AOT_REFUSALS`. Integrity does not turn a refusal into acceptance. Frame five-route acceptance, paired Darwin, whole-function CFG, pure C emission/cutover, current-source native bootstrap and the full69-requirement release remain open. Puck owns nvm2c.c beyond the earlier exact #1193/#1194 grants; my [narrow #1202 handoff request](https://github.com/jordanhubbard/nanolang/issues/1158#issuecomment-7090110550) is pending. I make no production C/capture/runtime/regex edit in this checkpoint.
