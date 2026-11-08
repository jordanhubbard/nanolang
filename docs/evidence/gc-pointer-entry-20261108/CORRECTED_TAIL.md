# My corrected pointer-entry candidate

I keep three changes separate: the #1119 shutdown registry correction, the
#1118 STRING terminator guard, and this #1065 allocation change. My original
tail-storage candidate remains rejected with its actual sanitizer failure.

After the registry correction, I repeat the paired experiment with the same
actual allocation/lookup/retain/release/finalization workload. Both variants
include the correction. Each execution handles 100,000 objects across fifteen
sizes, checks checksum 82246700 and finalizes another 1,000 live owners at shutdown.

| Mode | Corrected separate entries (CPU seconds) | Corrected private tail entries |
| --- | --- | --- |
| Plain O2 | 0.011995, 0.003187, 0.003288 | 0.002628, 0.002668, 0.002651 |
| ASan/UBSan | 0.036877, 0.036784, 0.033781 | 0.027932, 0.028663, 0.024647 |

I retain every sample, including the slower first baseline sample. These are
component measurements with baseline executions preceding candidate executions;
they do not establish whole-compiler timing or a universal speedup. Exact source,
commands and stable input/product hashes are in `corrected-paired-*.json.gz`.
Both measured sources include 10b960482's registry correction; the later STRING
terminator guard is not exercised by this direct-object workload.

My implementation reserves aligned private hash-entry storage after the payload
within the same allocation, reducing direct allocation calls from two to one.
I leave `GCHeader` unchanged and retain its existing logical size/accounting,
exact managed-pointer identity, reference counts and foreign-pointer refusals.
I check header, padding and private-entry arithmetic before allocation. Registry
entries retire before their containing allocations are freed; shutdown keeps an
owner visible during its finalizer, then removes it before freeing storage.

I execute the original hash-collision assertion successfully against both
corrected private variants and again against the actual final production source.
My permanent three-owner registry control also checks already-freed pointers,
own/still-live visibility, order/counts, accounting, repeated shutdown and
reinitialization. Missing retirement would make these lookups touch freed entry
storage. All thirteen GC controls pass under ASan/UBSan, including header,
terminator and private-entry overflow refusals and unchanged statistics.

`final-tail-unit.json.gz` records actual final-source entry alignment, logical
size, identity, reference counts, finalization and varied-size controls.
`shutdown-collision-final.json.gz` records the actual final-source collision
control. `tail-corrected-environment-unit.json.gz` and
`interpreted-bstring.json.gz` retain the complete environment and actual binary
payload/owner/finalizer sanitizer controls. Every recorded command succeeds and
its source/product hashes remain unchanged. I do not infer NativeSDK custody,
full source/platform acceptance or a release from these local controls. Those
unchanged gates and deadlines remain required.
