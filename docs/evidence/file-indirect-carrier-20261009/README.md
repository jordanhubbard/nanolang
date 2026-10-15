# I qualify my indirect File carrier

Under #989 I implement the distinct owning carrier described in
[my runtime contract](../../NANOISA_FILE_INDIRECT_RUNTIME.md), over `d8c4c495c`.
I retain exact plan/function identity, selected-target membership before owner
transfer, waiting-call identity through return, shared fuel and first-error
cleanup. Old acyclic/cyclic APIs retain their kind guards and public layouts.

My first fixture encoded the indirect result count as one byte instead of the
required two; preparation refused it before execution. I retain [that terminal](first-run-stderr.log)
and [the first run](first.log). I correct the fixture and keep indirect value
arguments independent of the direct-call reference map. The initial corrected
carrier corpus passes in 11.052 seconds. I then extend coverage to real File
owners, exact fuel boundaries, allocation-free carrier execution and cleanup
errors; those additions have their own final result.

My final Darwin command is:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-indirect-runtime-sanitizers test-file-cyclic-dispatch
```

Both carrier methods pass in 24.070 seconds and both existing matched cyclic
dispatch methods pass in 39.092 seconds. I retain [the final log](final.log),
[command terminals](terminals.json) and [changed source identities](sources.json).
All recorded children exit zero, are reaped and leave no process group.

My carrier fixture covers both VM and native arena layouts; both selected
candidates; scalar, File and OpenResult arguments/results; initializer plus
entry; permuted nominal catalogs; exact successful fuel and every smaller fuel
limit. I check invalid options, old-kind refusal, copied-input lifetime,
out-of-set and out-of-range targets, null identity, selected-child
disagreement, refusal before owner transfer, and cleanup failure preserving the
original error. My identity refusal fixture uses a null plan pointer; there is
no public cross-context callable import API.

I test all 205 measured create/begin allocation sites with persistent and single
failure, unchanged output on refusal, balanced allocations and fresh recovery.
I prohibit tracked allocation during carrier execution. The new corpus also
runs the complete older acyclic/frame/cyclic carrier fixtures. The carrier
runner rebuilds its named providers with ASan/UBSan and enables leak detection;
common linked objects outside that provider list are not newly instrumented.
The separate matched cyclic dispatch run uses its ordinary allocator controls.

I wire the ordinary indirect carrier gate into platform CI. Remote/Linux
qualification remains open. Manual carrier operations in two arena layouts do
not establish actual indirect VM or generated-native dispatch. Those adapters,
callable arguments/results, indirect borrows, paired source/full shadows,
installed grants and the full 5.1 release requirements remain open.
