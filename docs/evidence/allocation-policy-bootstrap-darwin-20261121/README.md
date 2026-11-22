# My repaired-pin Darwin bootstrap deadline

I qualify exact source `c908523c3dd88b5776058f61fcc1a38f3fc32e92` on Darwin
ARM64 with Apple Clang21.0.0. I own the isolated worktree
`~/.worktrees/nanolang/darwin-bootstrap-20261121` under #1161/#1164.
I coordinate through WebMux with Sparky and record our distinct ownership in
[#1161](https://github.com/jordanhubbard/nanolang/issues/1161#issuecomment-7071914300).
Sparky owns the separate newer-pin `bdffa4ccc` qualification and implementation.
I change no compiler source, tool selection, assertion, selected shadow or deadline.

| Original operation | Exit status | Elapsed time |
| --- | ---: | ---: |
| Fresh strict tools, two build jobs |0 |109.975s |
| Complete `make test-nanovm` |0 |48.352s |
| Seed generation |0 |243.343s |
| Seed verification |0 |1.584s |
| Raw Stage1 generation |Timed out and killed |1800.014s |
| Complete `make bootstrap3` invocation |2 |2057.366s |

My original VM target reports569591 passed assertions and zero failures.
Its original bstring/substring/callback prerequisites and heap/stack allocation
failure and recovery controls also pass. The original recipe removes its main
VM test executable; I retain the actual tools, provider objects, test sources
and full command logs, without claiming that removed image is retained.

My raw Stage1 runs `bin/nano_vm bin/nanoc_seed.nvm --
src_nano/nanoc_v06.nano --emit-nvm -o .../stage1.nvm` under the original
1800-second command limit and30-second complete compiler-shadow setting.
The parent kills its process group at the deadline. Its output log is empty;
I have no phase recorder for this generation and establish no assertion failure
or precise runtime cause. No raw Stage1 output, published bootstrap receipt,
Stage1 sentinel or Stage3 sentinel exists. Stage2, native translation/build,
installation and no-C-seed acceptance do not run. The diagnostic collector's
normal exit records completed collection; Make's exit2 and the explicit
`qualified: false` status determine the failed qualification.

I retain process CPU/wall samples and host resource observations. The last
sample records7m28.89s of NanoVM CPU over29m25s wall time. Shared-host load is
high, disk free space remains above508GiB and the native memory-pressure query
reports93% system-wide free percentage. These observations do not prove a cause, establish an
infrastructure-only incident or justify changing the deadline. I perform no
unchanged retry for green.

All1328 local source-input guards, five tool guards, six imported host-library
path/hash guards and the published seed-artifact guard still match after the
failure. No unexpected-native-compiler marker exists. Before/after hashes of
the selected compiler/linker tools also match. My source comparison with the
retained Linux receipt matches1327 inputs; the remaining input is each
platform's independently generated `modules/index.json`. I retain the initial
strict comparison refusal, my generated input's bytes and both receipt hashes rather
than claim complete cross-platform input-byte equality. I did not rewrite a
module, reuse another root's receipt or alias a compiler generation.

My [manifest](manifest.json) retains1786 logical inputs,125509610 raw bytes
and28735144 unique compressed bytes. Every compressed/raw SHA-256 identity
and decompressed payload is checked. I retain the exact failed manifest,
commands, stdout/stderr, source inputs, actual tools and seed, host libraries,
local provider objects, collector code and qualification status. My retained
source root remains in place; its root-bound manifest is not portable.

This result leaves paired repaired-pin acceptance open under #1161/#1164.
It does not replace the successful Linux result, Sparky's distinct newer run,
full SDK custody, fresh-PREFIX/root-independent host closure, pure NanoLang
translation, full sanitizer/ownership gates, Literate AI execution authority
or any original #976 release requirement. I tag no release from this result.
