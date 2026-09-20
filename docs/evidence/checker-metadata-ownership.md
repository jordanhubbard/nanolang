# I retain checker-created metadata until environment destruction

I qualify production and fixture f7c58a0de2ecdeca9ce30d25d33c83d6c45372d8
from canonical union8825485b5e9d in my preserved checker-metadata-ownership tree.
I change only explicit checker allocation ownership. The registry shallowly
releases blocks after existing environment cleanup; borrowed AST/manual/builtin
signatures and caller-owned runtime arrays remain outside it.

My fresh Linux GCC gates all pass:

| Gate | Seconds | Scope |
| --- | ---: | --- |
| Ordinary focused plus union lifecycle |12.897| Original unchanged parsed callback lifecycle, eight checker/order/slot controls, borrowed controls, merged882 union fixture. |
| O0 ASan/UBSan/LSan focused |9.116| All COMMON_OBJECTS and RUNTIME_OBJECTS rebuilt in fresh obj-checker-gcc, exact focused harness, detect_leaks=1, no LSAN suppression, halt on errors. |
| Complete metadata and environment scoping |27.705| Existing Make targets, including metadata C tests, generated-list poisoned initialization, imported signature/cache and scoping controls. |

These are first terminals at the corrected ownership production, not retries of
historical executables. The original595-byte parsed failure remains in its
historical report; the current focused harness explicitly invokes that parsed
function, unlike the subsequently narrowed old temporary harness. No claim is
made about every compiler allocation, allocation-failure recovery, Darwin or
complete release readiness.

The [report manifest](checker-metadata-ownership/report-sha256.json) seals the
commands, logs, source endpoints and per-phase compiler/provider maps. Seven
actual tool identities remain equal; all528 final provider files match current
qualified storage. An external GCC wrapper copies every successful explicit-o
output before a Make recipe can remove it. The artifact store retains826 files
(compiled outputs/provider snapshots and command records), under
/tmp/nanolang-checker-owner-f7c-gates/artifacts. This is the recorded artifact
scope, not a claim that every transitive program or temporary source is archived.
The qualified source tree and tools remain untouched; this ready tree adds only
documentation. Parent00c47 remains open pending its original-clause audit.
