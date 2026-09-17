# My native aggregate retention audit

I inspected the unmodified translator at code pin `512c2ec6` while its separately pinned positive compiler acceptance was running. This audit executes no fixture, changes no linked binary and does not attribute the compiler's RSS to a particular pool. MAC `task_7ff98f74f0ee40afba708605c01fad52` records the next bounded design/repair obligation.

## My audited lifetime rules before repair

| Pool | Allocation and identity | Release boundary |
| --- | --- | --- |
| `nrec_owned_head` | `nrec_snapshot` copies a whole by-value `nrec_t` into a linked owner and returns its stable value address. Nested record fields use these snapshots. | `nrec_release_snapshots`, after entry returns. |
| `narr_owners` | `narr_new` owns an integer/Boolean array handle; reserve grows a separately owned buffer. Foreign handles receive buffer ownership with no owned handle. | `narr_release_owned`, after entry returns. |
| `nsarr_owners` | String-array handles and buffers follow the same stable-handle model. | `nsarr_release_owned`, after entry returns. |
| `nsarr_strings` | Host directory-walk adaptation separately copies returned text so releasing the foreign result does not invalidate it. | The same string-array exit cleanup. |
| `nrarr_owners` | Record-array handles and checked growing buffers retain copied record values. | `nrarr_release_owned`, after entry returns. |

The relevant emitter helpers are `emit_narr_storage`, `emit_nsarr_storage`, `emit_nrarr_helpers`, and the `nrec_snapshot` definition emitted inside `nvm2c_emit` in `src/nanoisa/nvm2c.c`. All release calls are in generated main's post-entry cleanup. Ordinary function epilogues free invocation `r`/`rl` slabs and root-list metadata; they do not release these process pools. Previous checked growth and exit-leak fixes remain valid bounded work.

My current `nvm2c_map_roots.inc` traverses nested records, string arrays and record arrays to find reachable map/string edges, but does not mark aggregate allocation owners. `emit_map_roots` and `nroot_add` intentionally omit integer/Boolean-array handles because their elements contain no map/string pointers. That omission must change before aggregate reclamation is safe. Clearing or overwriting the last source-level reference currently does not free these pool allocations early.

## My implementation boundary

I first establish owner identity and byte accounting for record snapshots and all three array buffer/handle types. I retain stable handles through checked growth and distinguish a borrowed handle from an owned backing buffer; a null owned-handle field is not evidence that the buffer is unreachable. Marking should follow the currently published handle's actual owner rather than dereferencing a possibly expired foreign stack handle saved in an owner record.

I publish primitive-array roots as well as aggregate/string roots, and close the entire current graph before freeing any owner. By-value record copies can keep child arrays or nested snapshots alive even when an older containing snapshot becomes unreachable. Array mutation and escaped element values must preserve these edges. Copied directory-walk strings need independent reachability so an escaped string can outlive its array. Collection must include aggregate-only modules without map or string-allocation opcodes.

I account for new owners and positive buffer-growth bytes in the shared safe-point scheduling design. I preserve caller frames, globals, live operands, return handoffs, simultaneous self-tail staging, allocation-failure behavior and forced cleanup. I do not collect from an allocator before its result becomes a published root. Memory retained by live program data and between safe points remains distinct from unreachable pool debt.

Acceptance requires small ordinary churn workloads with measured peak/live owner bytes, nested and shared aliases, returned records/arrays/element strings, cross-call mutation, borrowed-handle growth, self/non-self tail calls, and final cleanup under sanitizers. Only after those checks can a new explicitly bounded full compiler run assess practicality. I do not repeat or minimize a historical fault to establish this static lifetime fact.

## My task boundaries

Completed record-array exit cleanup (`task_0916ab0afb014b5984d69fbb11b0432d`), checked growth (`task_617006c46f9746da9694c6a4e0a0ceaf`), and invocation-local frames (`task_4c5aeade9fa944dd80eebfbd1ae91072`) do not claim early process-pool reclamation. General temporary-string reclamation is `task_4d3105329e73454083846ad41473500d`; exact host-result adoption remains `task_d5f899966241452a900422938fff3265`. The C-runtime copied-array-string contract (`task_93bb44374587a757753418fc28c2095d`) remains independent. Full native compiler acceptance (`task_fc43d8d1923b40ebb343ae56da535dfc`) stays open until its actual gates pass.

## My bounded reclamation results

I now track owner identity and live/peak bytes for record snapshots, integer/Boolean arrays, string arrays, record arrays, and copied directory-walk strings. I publish primitive-array roots and close the full graph before sweeping. Borrowed handles retain backing owners through the current handle's owner identity; I never free their foreign handle or original stack buffer. Positive buffer growth contributes collection debt. Collection remains at published-root safe points, with a 64 KiB minimum budget adjusted to surviving bytes, and final cleanup remains unconditional.

My unchanged translator retained 2,784,000 bytes in the ordinary 3,000-iteration mixed-array churn workload before exit cleanup. The repaired translator peaked at 65,928 bytes on the same workload and ended with zero tracked live bytes. I measure tracked allocation sizes, not process RSS or allocator overhead. These figures establish the bounded fixture's retention improvement, not a bound on reachable application data.

I passed 40 regression methods in 116.991 seconds across aggregate/string reclamation, allocation debt, root scaling, map lifetimes/globals, string joins, record growth/frames, floats, branches and returned-allocation cleanup. The new aggregate suite compiles ordinary bytecode to strict C11 and runs with AddressSanitizer, UndefinedBehaviorSanitizer and LeakSanitizer. Its controls include caller/global/operand aliases, nested records, escaped by-value children, returned arrays, cross-call mutation, 10,000 simultaneous self-tail swaps, borrowed-handle growth and escaped copied strings. A separate rerun of all four aggregate methods, including two added nested-copy cases, passed in 4.316 seconds. Native translator checks passed 2,414 cases and shape checks passed 1,092; opcode and sanitizer-driver checks also passed.

I preserved the initial strict-C failure for an unused retained record-array getter as separate MAC `task_7aaeef5823f24054a37af998ad29a01f`. An unchanged translator produces the same warning for a mixed-array module. I explicitly reference that retained helper, preserving strict warnings. I also retain an initial test invocation containing a misspelled method name; the corrected existing allocation-cleanup method passed.

Local evidence is retained in `/tmp/nanolang-aggregate-retention-{regressions,nested,nvm2c}.log`, `/tmp/nanolang-aggregate-retention-baseline/`, and `/tmp/nanolang-unused-record-getter/`. I have not run a third full native compiler self-compilation. Its resource acceptance task remains open; host-result buffer adoption remains a separate obligation. Conservative local roots can retain still-published values, and this repair does not claim lexical liveness analysis or arbitrary foreign handle-copy support.

After restacking onto `f1b727ed`, I rebuilt the translator, assembler and VM and passed all ten aggregate/string/debt methods in 14.057 seconds. The source merge was clean; I preserved both roadmap additions.
