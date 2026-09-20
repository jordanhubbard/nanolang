# I reconcile only my matched private File runtime

I audit `task_82ffef0affec2b23c4390931370f6774` against its original
[contract](NANOISA_FILE_PRIVATE_RUNTIME.md) and current MAC description.
My canonical audit base is PR888 merge
`f1606e2c84e67491e9652a5bf71944d235216d95`. I propose bounded completion for
review; I have not changed its MAC state. Historical checkpoint text saying
that the next dispatcher is pending remains accurate for that earlier pin.

| Original obligation | Merged acceptance and precise scope |
| --- | --- |
| Fresh serialized-v2 identity, immutable five-method/eight-type catalog, one invocation across initializer and entry | [Carrier858](evidence/file-runtime-carrier.md), [frames879](evidence/file-runtime-frames.md), [VM885](evidence/file-private-vm.md), and [native888](evidence/file-private-native.md) retain fresh plans, startup identity, exact nominal maps and no-acquisition refusal. Native invocation additionally compares all generated facts and semantic ABI before begin. |
| Complete scalar/passive/owner/Result carriers, moves, borrowed formals and roots | Carrier checks cover exact Result fields and unique File/OpenResult ownership. Frames cover pending argument prefixes, caller suffixes, return staging, nested formal forwarding and all six transfer-generation failures. Actual VM and generated native replay the complete original corpus through those same helpers. |
| Bounded preallocation and matched storage layout | Carrier accounts project-owned storage before acquisition; frames distinguish VM overlap from native disjoint placement, including equal-size mode controls and depth64. This is the stated64MiB project allocation scope, not libc FILE storage or process RSS. |
| First execution error plus secondary cleanup; scalar publication only after cleanup | All four stages check root drain, close/report failures, initializer failure preventing entry, fresh reentry and output sentinels. The injected fclose-report fault really closes then reports failure; I do not claim arbitrary libc close-failure reproduction. |
| Private VM execution | PR885 executes copied checked instructions with the VM frame model; corrected fixtures report182,075 instrumented/17,163 linked checks. Its256-report seal includes initial failures and bounded later provider integration. |
| Genuine generated native execution | PR888 emits C functions, labels, operators and direct calls, not a bytecode interpreter. O0/O2 replay567 instrumented/73 linked calls per configuration, plus isolated no-VM binaries and ABI/fact mismatch refusals. Explicit linked runtime objects are required; installed packaging remains separate. |
| Actual Linux/Darwin lifecycle, fault and sanitizer acceptance | Carrier277 reports, frames341, VM256 and native238 reports retain platform pins, actual compiler/provider maps, actual temporary streams and allocation/host/cleanup failures. Native seal contains6,199 unique artifacts and18 equal source/tool pairs. Sanitizer coverage is explicitly partial where common objects were not rebuilt; linked hook counters do not measure absence of I/O. |
| Unchanged old corpus and public refusal | Each stage retains relevant hosted/body/flow/carrier/frame/opcode/wrapper neighbors. Native generation failure preserves prior output. No default public File route was activated by these private stages. |

My actual merges are858 `21ed843950d0bd9a397d663cf9dfd3184f5d156a`,
879 `87ae702e18a8c0bd14742b3d0c5f138dc4e2795b`,
885 `d416b24560c9bd77ea4470e3c63f436d0eafe34f`, and888 above.
They are ancestors of this audit base. The sealed evidence retains first
compile, fixture, setup, launcher and path-normalization terminals rather than
relabeling them as passing. This reconciliation performs no new execution and
makes no fresh whole-product or whole-provider qualification claim.

The original82ff scope explicitly excludes public selection, source bindings,
mandatory source shadows, installed routes, loops, indirect calls and richer
borrowed calls. I therefore recommend closing82ff after review and actual-merge
ledger reconciliation, while keeping6fc,72556,6931,d03c and release parents open.
Public activation belongs the independently coordinated dfa149 child. My
[next control/call design](NANOISA_FILE_CONTROL_CALL_EXTENSION.md) retains the
other exclusions as required work rather than deleting them from the parents.
