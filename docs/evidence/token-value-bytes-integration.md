# I qualify token ABI integration with canonical PR908

I freeze `51d4bb83f9032d630bb2e04e7b4aedcf4552e29b` after actual PR908
merge `0061feed573ad7e4cac2a9e61d5ed38f0fb62440`. My
[identity comparison](token-value-bytes-integration-identity.json) retains all17
token source/fixture inputs and checks25 incoming source/fixture paths against
that main. My original token seal and independent review remain unchanged.

PR910 changes the actual C-seed typechecker. I therefore start empty Linux and
puck trees, with no copied compiler objects, stage binaries or module caches.
I retain a hash-equal source archive and24,085 tracked inputs on each host.

| Host | Fresh bootstrap | Full ordinary token and paired schema | Parser/module/wrapper adjacency |
| --- | --- | --- | --- |
| Linux GCC | PASS285.223s | PASS27.630s | PASS267.777s |
| puck Apple Clang | PASS279.826s | PASS27.726s | PASS276.247s |

Both complete ordinary methods pass. Each C seed, Stage1 and Stage2 selects
exactly76 token shadows and63 generator shadows, emits the exact nine token
rows, then runs the actual generator main in an isolated directory. All four
generated files equal the committed outputs, as do the independent Python
outputs. I retain12 exact shadow-selection reports and8 four-file comparisons.
No assertion, deadline or schema normalization changed.

The final adjacency target invokes another actual bootstrap through Make,
just as in my original qualification. I retain its separately hashed products;
I do not relabel the final compilers as those used by the preceding paired
phase, or claim binary fixed-point equality. Expected negative wrapper/module
controls retain their diagnostics inside successful outer phases.

[My supplemental manifest](token-value-bytes-integration/report-sha256.json)
records252 JSON report blobs and1,349 retained artifact objects totaling
443,166,985 bytes under `/tmp/nanolang-token-integration-artifacts`. Twelve
source/tool before/after comparisons match. Fresh current checks find no
mismatch in either24,085-source/12-tool map,824 Linux endpoint products or631
puck endpoint products. The source maps match across hosts. The downloaded
puck report archive is22,267,075 bytes and matches its retained remote SHA256.

My preserved roots are `/home/jkh/Src/nanolang-token-integration-51d` and
`puck.local:/tmp/nanolang-token-integration-51d`; raw reports are
`/tmp/nanolang-token-integration-linux` and the corresponding puck report root.
The seal retains command/environment/SDK/compiler identities, raw output,
selected products, generated C and executables. I do not claim every deleted
transient executable or a complete transitive system toolchain inventory.

This supplemental run is ordinary integration acceptance only. Original seven
compiler/sanitizer configurations keep their original pins and selected-provider
scope. Subsequent canonical PR893 changes are outside51d and require their own
compatibility assessment. Full paired File source execution, broader compiler
policy follow-ups and task8bbc remain open. Root review and actual PR911 merge
remain pending; no hosted-CI result is inferred.
