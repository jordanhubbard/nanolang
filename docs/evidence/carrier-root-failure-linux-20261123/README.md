# I retain typed-list roots before reclaiming carriers

I continue #1116/#1140 at pushed1d2e922c48 and file #1211 after reproducing
an actual existing typed-list use-after-free. Source checkpoint
`1097c4c6a0209da1634d6ccbd3ffaff4290a7876` changes only my roadmap and
[carrier ownership boundary](../../EVAL_CARRIER_OWNERSHIP.md). My production
C, tests, Make recipe and retained compiler/VM/AOT images remain unchanged.
The actual complete GCC13 Linux ARM64 ASan/UBSan providers come from qualified
source761f9407ff; I record all108 provider object/archive identities and retain
their bytes in the unchanged private root. This is a bounded native observation,
not a fresh full-provider build or full-suite rerun.

My C/API control proves that an ordinary record binding copies its caller's
record, while typed-list push/get preserves the binding's raw StructValue
pointer. Without replacement, all assertions and explicit caller/list/env
teardown pass with leak detection enabled. With replacement, the same list
pointer still refers to the old header, which env_set_binding_impl freed.
The next field read fails ASan in eval_test_record_field.

My actual parsed/checked source shadow pushes a binding, replaces it, then
gets the retained record and checks its old integer/STRING fields. It fails
ASan inside env_define_var_impl while copying the already-freed list result.
Its shadow starts and aborts at that failure; I do not claim either source
shadow completes. My first source fixture omits the required unsafe block
around generated extern list calls and is correctly rejected before execution.
I retain that refusal and add only the existing required block, preserving
all value assertions and the actual replacement before measuring the UAF.

The initial observer reused API stream filenames during that fixture correction.
I retain its first product/root and command results, the initial complete source
refusal and the corrected complete API/source reports. I do not claim to have
separately retained the overwritten first API streams. The retained current
streams belong to baseline.json and its identified private product.

I verify my source/products/archive with:

```sh
python3 docs/evidence/carrier-root-failure-linux-20261123/read_archive.py
```

My reader passes1,424 records/1,394 exact Git inputs and two actual native
products. It verifies raw/compressed hashes, object coverage, the prior archive
digest, unchanged production/test/Make/tool inputs, exact pointer/refusal/failure
observations and the sole fixture transformation. Shared source/image objects
remain in the hash-bound preceding archives. Provider identities are recorded;
their separate object/archive bytes are not all recopied into this capsule.

I do not repair #1211 here or enable any new cleanup. A copy on list push would
change retained pointer identity, and a copy on get cannot read a freed header.
General reclamation needs accounted allocation provenance and roots for active
expressions/arguments, bindings, aggregate descendants, typed lists, returned
C/API values and async/handler lifetimes, plus every existing early-discard path.
My boundary document records the GC shutdown ordering constraint before adding
callbacks; it does not claim an uninstalled callback currently fails.

#1211/#1116/#1140 remain open. The previous complete source LSan1937bytes/82
allocations and record/union3720/159 failures remain preserved in the
[preceding receipt](../scalar-projection-ownership-linux-20261123/README.md);
I do not rerun or reclassify them. Full source/platform and69-requirement release
acceptance remain open. Puck keeps #1204/#1190 merge/representation/regex/
capture/checkpoint/CI ownership. All frozen roots/audits/stashes and NAS/TCC/
migration pauses remain intact. No native job remains mine at publication,
no Darwin reservation or NAS action.
