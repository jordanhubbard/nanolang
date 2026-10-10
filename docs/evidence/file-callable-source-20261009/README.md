# My paired callable source qualification

I extend #989 from `030b12862` with independent C/Nano callable source checking,
ownership and lowering. My [source contract](../../NANOISA_FILE_CALLABLE_SOURCE.md)
states the admitted forms and remaining boundaries.

I retain these terminal local results:

| Check | Result |
| --- | --- |
| C seed/bytecode source drivers | 15 methods pass, 97.225 seconds |
| Fresh Nano compiler VM/native source drivers | 15 methods pass, 356.981 seconds |
| Instrumented source drivers | 15 methods pass, 104.220 seconds |
| Instrumented lowerer with allocation-prefix checks | 13 methods pass, 132.314 seconds |
| Imported callable, direct C/Nano byte comparison | Pass, 20.873 seconds |
| Paired body checks, C parser/checker instrumented | 5 methods pass, 51.049 seconds |
| Combined callable fixture, instrumented C parser/checker | Pass with leak detection and allocation-prefix checks |
| GCC16 changed service sources | Syntax/warnings check passes |

The source drivers exercise callable local reassignment, returned callees,
higher-order parameters/results, imported helper values, shared aliases,
exclusive writes and File moves. Negative cases cover purity, signature/mode,
argument count/type, owner duplication, overlapping exclusive borrows, recursion
and failing shadows. Existing destinations remain unchanged on refusal. Native
products execute with the runtime grant and refuse its absence; symbol checks
exclude VM dispatch and public VM execution.

I retain the initial25 body-harness assertion failures caused by its fixed three-allocation assumption, and the corrected all-prefix recovery test.

I retain the initial nested-signature shadow failure and corrected checker run.
Punctuation token values are empty, so my Nano annotation reconstruction now
spells punctuation explicitly. I retain callable parameter signatures in both
parsers, including borrowed and nested callable parameters.

Cross-frontend comparison additionally exposed function/constant/signature
ordering differences that the old within-frontend comparisons did not cover.
I align the C File wire with independent Nano emission. `cross-frontend.json`
records exact equality of all10,408 bytes of the combined fixture. The imported
fixture has a permanent direct C/Nano comparison in the driver test, as does
the combined fixture. My lowerer allocation-prefix loop verifies unchanged
output on each injected failure and identical bytes after recovery.

The sanitizer driver instruments service publication/lowering and links existing
common/runtime objects. The lowerer test instruments its included implementation
and explicit File runtime sources; this is not a claim that every linked object
was instrumented. I record changed source hashes in `source-sha256.json`.

My fresh bootstrap compiled and verified Stage1 but refused its host closure: three modules resolved to different immutable cache generations than the seed while body-checker tests built against the shared cache. I retain the manifest and exact path/hash differences under `bootstrap-closure-failure`. A fresh run uses a dedicated module cache, with exact closure checks unchanged. These logs do not establish Stage1/Stage2 equality or exact Linux/Darwin
release acceptance. Full5.1, mixed profiles and remaining service/backend gates
stay open. My user's untracked guide fixture retains SHA256
`c739aeb158c5b3e94c15d8de1232e4e1415a3f20b2e80fcf96fba39f1dedb976`.
