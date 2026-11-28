# My isolated symbol-name projection

I qualify source `8806139ca8bf3f60a25d88d241f67747c6ef9e8a` against baseline `189578c89cf1c7b84e3e83f0ccde72e30c784045`. My only production change projects `(at symbols i).name` directly instead of retaining a whole temporary Symbol. Reversing that two-line change restores the exact old lookup function. I preserve its signature, reverse order, count and index checks, missing return and complete empty-print branch. I add no lookup cache or record-shape, ABI, type-admission, C/profile, assertion or deadline change.

My new shadows and fixture controls check empty and missing names, case and prefix distinctions, nearest duplicates,133-entry boundaries, same-length replacement through aliases and independence of copied tables. All833 existing typechecker assertion lines, every old fixture byte outside added controls,14 copy sizes and24 complete diagnostic inputs/expectations remain unchanged.

In one fresh isolated root I build actual baseline/candidate C-native and verified VM products, run both versions' controls and compare all24 complete original diagnostic records through all four combinations. Both actual C translations pass. All108 commands exit zero at original600 build/60 runtime/default10, and selected source/image/tool/producer-host/input guards match. I retain all products and terminal native-library files. This quick cohort does not establish full five-route sanitizer or newly built compiler/factory acceptance.

The retained C translations show the intended materialization boundary: record locals1->0, record temporaries2->1 and retained local-record GC roots1->0. Both versions keep one nrarr_get, one empty fputs and one fflush in the same miss path. I recompute these facts from the archived complete C files and preserve the exact function excerpts. This is generated-code evidence; I claim no measured whole-compiler speed gain or historical timeout cause.

The original full five-route cohort, both generated sanitizer natives, genuinely updated compiler/native36/factory300+emitter12 and fresh/current/paired/full69/pure-translator/cutover acceptance remain separate requirements. #1255 remains open; no release acceptance follows from this component.

My independent reader verifies4939 records, exact source commits and overlays, unchanged original expectations, all command streams/products, selected guard maps and actual materialization. Manifest SHA256: `8daef2c428d6701398da9829b3d5fb016cc6b3e2e0c8ef9676e050a5ed4b59b0`.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/lookup-projection-quick-linux-20261125/read_archive.py
```
