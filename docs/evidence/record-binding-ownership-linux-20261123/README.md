# I retire copied record-binding storage

I qualify source `fead5bf2d9953eb30a7ef9afb3601743b3c04936` under #1140/#1116.
I retire an explicit record-literal temporary after a normal binding has
independently copied it and immediate tracing has finished. I separately
retire copied direct callable fields during owned environment-record
replacement and teardown. I preserve ordinary copying; this is no binding
adoption optimization. I change no public layout or original test/deadline.

My helper checks the AST literal kind, actual record result, non-borrowed
binding and different destination carrier. Returned/projected records,
field-store inputs, nested environment-record lifetime and generic raw-array
cleanup retain their existing rules. Literal-tree cleanup retires its copied
callable metadata before my existing string/nested-carrier cleanup, preserving
shared arrays. Borrowed environment slots remain non-owning under #1207.

My unchanged complete expression/globals fixture reports572 leaked bytes in26
allocations with retained baseline providers, then passes fresh complete
ASan/UBSan providers with leak detection enabled. My new repeated global
initialization/replacement test balances991 callable allocations and991
releases; the identical test linked to retained old providers refuses at
991/394. Snapshot strings/records remain independent while array identity is
shared. Legacy C/API input, raw-array callable fields and nested record
lifetimes survive. The nested control explicitly cleans its caller-held
carriers after checking survival; I do not present that test cleanup as new
production reclamation.

All original134 interpreter and88 typechecker methods pass normally and with
complete ASan/UBSan providers, with leak detection disabled for those broad
suites. My original strict nine-route/64-call audit retains zero names with
leak detection enabled. The borrowed-value and environment controls also pass
instrumented and ordinary builds. No original fixture is narrowed or changed.

My complete original record/union fixture still fails LeakSanitizer with the
same3720 bytes in159 allocations. Its assertions/shadows finish; I retain the
complete failure and products. General local record/escape/field/container
reclamation, #1116/#1140 and paired current-source release qualification remain
open. I assign no cause to historical source-factory deadlines.

I also retain a separate preexisting build defect under #1208: exact9b604958e0
eval.c with NANO_INTERPRETER fails strict syntax compilation because AST_SET
references undeclared old_value. Ordinary units compile tracing out. A repair
must serialize live old/new values or establish snapshot ownership before
replacement retires old storage. I claim no tracing-enabled acceptance here.

I verify my complete archive with:

```sh
python3 docs/evidence/record-binding-ownership-linux-20261123/read_archive.py
```

My reader passes1476 records and1387 exact Git inputs. It verifies compressed
and raw hashes, actual source/provider/tool identities, every original fixture
hash, baseline/candidate/old-provider results, original failure reports and
native test products. My manifest reuses identical objects from the prior
borrowed-value archive; its manifest is hash-bound and every referenced object
is verified independently. Full source, observers, commands, streams and
original private paths remain lossless. Puck-owned nvm2c/merge/regex/capture/CI
inputs and retained tool images stay unchanged. No NAS/TCC/migration operation
or frozen-root/audit/stash mutation is part of this work.

My first archive reader refuses a missing source entry for the unchanged
environment symbol-index test. I retain the failed reader/archiver and full
stderr, add that exact Git source to the final archive, and check its blob
against both the baseline and candidate pins. My final reader passes with
the same native results; this is an archive-input repair, not a product retry.
