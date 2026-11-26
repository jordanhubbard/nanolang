# My callable observer allocation-bound follow-up

I fix the test-only C observer's operation-name pointer-table allocation bound.
I preserve the preceding diagnostic and every old packet. C product source and
pure NanoLang collector source are unchanged; full native acceptance remains
withheld under #1262.

Manifest: `83e3c9eb5b6c51c935524d6b56494268b7ea9426f4a00551caf36e58352c4e59`.
My reader verifies5,258 new records and the preceding22,538 records, including
all five preceding journal replays, then replays this complete new diagnostic
without executing products.

My source review found that the original test observer allocated its operation-
name pointer table before checking whether that count could fit the packet.
The earlier978-case receipt contained no oversized operation-name-count packet.
I do not claim to have executed the old observer's oversized allocation. I move
its necessary count bound before calloc and add eleven malformed controls:
huge operation/import counts, a huge operation-name length, function arity/local/
capture/result widths, and import parameter/result/presence/tag widths.

The new full diagnostic passes all3 methods/100 commands in76.300037s at the
original600/60/default10 limits:989 matrix packets,435 explicit expectations,
imported shadows, all505 unchanged original C assertions and the same26 captured
original module inputs. Ordinary and ASan/UBSan actual C agree with C-seed native
and both producer VM routes. All4002-source/49-image/22-tool/genuine-producer/
six-host/receipt guards and owned cleanup pass. The old978 packets form an exact
prefix; the only C-observer source change is the preallocation count condition.

The permanent full five-route builder remains unchanged. The original native
refusal and both read-only producer traces remain in the
[preceding receipt](../callables-three-route-diagnostic-linux-20261126/README.md).
The NanoLang collector, driver, C product and bootstrap selection are byte-
identical across this observer fix. The diagnostic continues to record
accepted=false, full_five_routes=false and acceptance_withheld=true. Three
execution routes do not establish five-route/native acceptance.

```sh
/usr/bin/python3 docs/evidence/darwin-current-source-20261124/diagnostics/callables-three-route-bounds-linux-20261126/read_archive.py
```

I retain every command, budget, stream, artifact, selected guard, observer/
collector/replay source, and both collector rejection controls. The reader pins
and runs the preceding reader, verifies the exact C-observer edit, regenerates
all packet bytes and criteria, preserves the original C corpus bytes, and
replays this journal with product creation forbidden.

C-owner #1262 repair/handoff, full five-route/private/public qualification,
representation/module/body/runtime/profile/ownership composition, paired
platforms, installed cutover and all69 release requirements remain open. No C
product repair, assertion/deadline change, full native CI, tag or publication
is claimed.
