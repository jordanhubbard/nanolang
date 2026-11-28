# My fresh guarded Linux bootstrap after collection scheduling repair

I qualify the frozen Linux ARM64 checkout c908523c3 at
~/.worktrees/nanolang/bootstrap-allocation-policy-20261121 under #1161/#1164.
I build original tools from source with system GCC13, run the original VM/heap
controls, then execute the real Make bootstrap3 with1800 seconds per command
and30 seconds for complete compiler shadows. I preserve the failed original
74644c29f receipt and its later cached-link refusal marker separately.

Every fresh step succeeds. My measured generation/native stages are:

| Stage | Raw generation | Verification | C translation | Independent native build |
| --- | ---: | ---: | ---: | ---: |
| Stage1 |767.051s |0.217s |60.625s |271.767s |
| Stage2 |765.257s |0.216s |62.677s |247.560s |

The C-produced seed takes161.295 seconds and verifies. Both derived native
compilers independently compile and execute hello. Final verification checks
seed, Stage1 and Stage2 again. Both raw derived modules contain1033184 bytes
with SHA-256 b7e92e258e35f42bfde2045aefa0ff06ba6e132ebb155379fbe0d88d697296a9.
Exact byte comparison passes; I rewrite no module and alias no generation.
My six imported host paths and their library hashes match the seed exactly.
The version2 receipt retains the canonical module-cache context. All original
source/tool/host/artifact guards pass, with no unexpected-native-code marker.

Make installs bin/nanoc as the Stage2 symlink, compiles/runs its installed
smoke and passes the original no-nanoc_c backstop with the C reference compiler
temporarily removed. Its complete bootstrap3 invocation takes2358.423 seconds;
that aggregate does not replace the unchanged per-command deadline. My source
checkout remains clean. This is installation inside the build checkout;
fresh-PREFIX packaging and external installed-client acceptance remain separate.

The original VM driver passes569591 assertions. Original bstring/substring,
callback, heap/stack allocation-failure, heap sanitizer and record/map heap
controls pass. The successful original Make recipe deletes its VM executable;
I separately derive, execute and retain an additional complete driver with the
same source/providers/flags and only its output path changed. It also passes
569591 assertions. I retain original and additional terminals distinctly.
Eighteen original effect/record VM/native methods also pass in29.079 seconds,
including sanitized generated products; their temporary products are removed
by their original fixtures, while exact source-built tool images are retained.

My manifest covers1497 logical inputs,436851068 raw bytes and90362705 unique
compressed bytes with checked raw/compressed SHA-256 identities. It retains
actual raw/native compilers, original tools, host generations, emitted C,
complete phase clocks, command logs, source identities, receipt and regression
images. I recheck all original guards after the additional retained-driver run.
I preserve the original source root; this archive is not permission to resume
its root-bound receipt from a different path.

This closes the demonstrated Linux deadline/cache-context failure at this pin.
It does not establish paired Darwin, exact final-release-head qualification,
full sanitizer ownership, pure NanoLang translation/cutover, complete native
profile/lifecycle or Literate AI execution authority. Those independent gates
and the complete original #976 release scope remain open.
