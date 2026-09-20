# My layout fixture and canonical source comparison

I retain the full scope in [my correction contract](../LAYOUT_FAULT_FIXTURE_PRIVATE_SYMBOLS.md).
My only fixture change is four symmetric private-symbol aliases around the
embedded allocator-instrumented decoder. Production and old assertions do not
change. Root reviewed the exact e761 checkpoint before execution.

My fresh ca377 builds and bootstraps pass on both hosts. Complete Make then stops
at the old fixture's duplicate symbols. After correction, my layout target
passes49 checks on each host, and the unchanged original90-method phase passes
on Linux in160.989 seconds and puck in201.715 seconds. Both complete Make runs
later stop at the same verifier corpus:174/176 verified, two failures, zero skips.
The U8 and LexerToken-insert tasks stay open. This is not whole-Make acceptance.

[The seal](layout-private-symbols/seal-sha256.json) covers both original and
corrected reports, actual commands/logs, endpoint source/tool maps, phase products,
provider-copy proofs and retained verifier-corpus files. My first copy helper's
symlink-order failure exists as an explicitly observed tool terminal, not a
fabricated historical log. Both partial copies and the original helper remain.
The corrected copy runs have actual file-backed logs.

Initial providers were byte-checked and copied, then Make rebuilt components in
the relocated roots. My product maps reflect that; only selected source/tool
endpoint pairs claim equality. This is a named-tool inventory, not the complete
transitive toolchain. Unchanged inner runners' deleted temporary files are not
reconstructed; the failed verifier corpus actually retained its524 files per
host. The original10-second compiler shadow policy and native120-second limits
remain unchanged.

My ready branch integrates canonical example and CI selector changes after the
e761 gates. The fixture and all qualified compiler/runtime source remain equal;
I do not relabel those executions as a rerun at the ready head.
