# My standalone ownership-test output parent

I repair #1175 in `91bfdc3280e5cfaa10d785912cb0d3ad868ad135`. I create my generated
.test_output parent before requesting a child with tempfile.mkdtemp. I preserve
existing products and make no compiler/runtime or production-library change.
Only this test file differs from the paired ad2 conversion source snapshot.

My actual standalone Make target atad2 fails in the fresh Darwin clone before
comparison: Make exits2 and its one method errors in0.030s with FileNotFoundError
for .test_output/ownership-module-products-*. I preserve the normal Make terminal,
its dry-run and the original test source.

My corrected actual Make target starts in a separate fresh detached Linux clone
with .test_output absent and no manual parent preparation. It creates the parent
and passes the original2563-case/five-route ownership gate, preserving eight
private deferrals and both actual producers. The full cold Make invocation,
including fresh prerequisites, exits0 in433.858s; its method takes383.108s.
An external Python observer retains temporary products outside the clone and
does not create the generated parent. I retain that observer and environment.
All source/provider guards match and the qualification clone remains clean.

My archive contains8578 logical files,214924041 raw bytes and8298 unique gzip
artifacts totalling50478580 bytes. Separate verification checks every compressed
and raw hash and all3560 guarded files against the exact91bfdc328 Git blobs.
I retain the failed Darwin terminal, corrected Linux Make terminal, actual
products, source, tools and qualified bdffa4ccc source-provider receipt; build
caches are omitted. This is a Linux cold-root gate, not paired new-pin raw
bootstrap or full release acceptance. The production-library pair remains
[ad2f8fbeb](../conversion-projection-darwin-20261121/README.md).
