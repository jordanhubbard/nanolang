# My native aggregate retention audit

I inspected the unmodified translator at code pin `512c2ec6` while its separately pinned positive compiler acceptance was running. This audit executes no fixture, changes no linked binary and does not attribute the compiler's RSS to a particular pool. MAC `task_7ff98f74f0ee40afba708605c01fad52` records the next bounded design/repair obligation.

## My current lifetime rules

| Pool | Allocation and identity | Release boundary |
| --- | --- | --- |
| `nrec_owned_head` | `nrec_snapshot` copies a whole by-value `nrec_t` into a linked owner and returns its stable value address. Nested record fields use these snapshots. | `nrec_release_snapshots`, after entry returns. |
| `narr_owners` | `narr_new` owns an integer/Boolean array handle; reserve grows a separately owned buffer. Foreign handles receive buffer ownership with no owned handle. | `narr_release_owned`, after entry returns. |
| `nsarr_owners` | String-array handles and buffers follow the same stable-handle model. | `nsarr_release_owned`, after entry returns. |
| `nsarr_strings` | Host directory-walk adaptation separately copies returned text so releasing the foreign result does not invalidate it. | The same string-array exit cleanup. |
| `nrarr_owners` | Record-array handles and checked growing buffers retain copied record values. | `nrarr_release_owned`, after entry returns. |

The relevant emitter helpers are `emit_narr_storage`, `emit_nsarr_storage`, `emit_nrarr_helpers`, and the `nrec_snapshot` definition emitted inside `nvm2c_emit` in `src/nanoisa/nvm2c.c`. All release calls are in generated main's post-entry cleanup. Ordinary function epilogues free invocation `r`/`rl` slabs and root-list metadata; they do not release these process pools. Previous checked growth and exit-leak fixes remain valid bounded work.

My current `nvm2c_map_roots.inc` traverses nested records, string arrays and record arrays to find reachable map/string edges, but does not mark aggregate allocation owners. `emit_map_roots` and `nroot_add` intentionally omit integer/Boolean-array handles because their elements contain no map/string pointers. That omission must change before aggregate reclamation is safe. Clearing or overwriting the last source-level reference currently does not free these pool allocations early.

## My proposed implementation boundary

I first establish owner identity and byte accounting for record snapshots and all three array buffer/handle types. I retain stable handles through checked growth and distinguish a borrowed handle from an owned backing buffer; a null owned-handle field is not evidence that the buffer is unreachable. Marking should follow the currently published handle's actual owner rather than dereferencing a possibly expired foreign stack handle saved in an owner record.

I publish primitive-array roots as well as aggregate/string roots, and close the entire current graph before freeing any owner. By-value record copies can keep child arrays or nested snapshots alive even when an older containing snapshot becomes unreachable. Array mutation and escaped element values must preserve these edges. Copied directory-walk strings need independent reachability so an escaped string can outlive its array. Collection must include aggregate-only modules without map or string-allocation opcodes.

I account for new owners and positive buffer-growth bytes in the shared safe-point scheduling design. I preserve caller frames, globals, live operands, return handoffs, simultaneous self-tail staging, allocation-failure behavior and forced cleanup. I do not collect from an allocator before its result becomes a published root. Memory retained by live program data and between safe points remains distinct from unreachable pool debt.

Acceptance requires small ordinary churn workloads with measured peak/live owner bytes, nested and shared aliases, returned records/arrays/element strings, cross-call mutation, borrowed-handle growth, self/non-self tail calls, and final cleanup under sanitizers. Only after those checks can a new explicitly bounded full compiler run assess practicality. I do not repeat or minimize a historical fault to establish this static lifetime fact.

## My task boundaries

Completed record-array exit cleanup (`task_0916ab0afb014b5984d69fbb11b0432d`), checked growth (`task_617006c46f9746da9694c6a4e0a0ceaf`), and invocation-local frames (`task_4c5aeade9fa944dd80eebfbd1ae91072`) do not claim early process-pool reclamation. General temporary-string reclamation is `task_4d3105329e73454083846ad41473500d`; exact host-result adoption remains `task_d5f899966241452a900422938fff3265`. The C-runtime copied-array-string contract (`task_93bb44374587a757753418fc28c2095d`) remains independent. Full native compiler acceptance (`task_fc43d8d1923b40ebb343ae56da535dfc`) stays open until its actual gates pass.
