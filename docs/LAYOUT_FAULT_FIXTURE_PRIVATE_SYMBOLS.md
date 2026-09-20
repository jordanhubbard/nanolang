# My layout fault fixture's private symbols

I preserve the first complete `make test` attempts at canonical
`ca3779e08b9cfc92414719a27e5e1456245dcfac`. My fresh builds and bootstraps passed
on Linux and puck. The full test commands then stopped before my original
90-method source comparison phase: Linux returned2 in234.629 seconds; puck
returned2 in220.802 seconds. Source and selected tool endpoint maps were equal.

Both linkers report duplicate `nvm_ownership_layouts_private_decode` and
`nvm_ownership_mixed_layouts_private_decode`. My existing layout fixture embeds
`nvm_v2_layouts.c` under a calloc hook and renames its four public entry points.
My newer private exports were not renamed, so the embedded copy conflicts with
the real provider linked through `NANOISA_OBJECTS`.

Before changing the fixture, I record task
`task_666ba1fff2e44f398e7960c825011f1a`. I will add two symmetric define/undef
pairs around the included translation unit. I retain the real provider, all
existing assertions, allocation hook behavior and linker flags. This changes no
production code or admission authority.

After independent review, I will run the exact layout Make target and complete
`make test` on both hosts. A separate corrected tree may reuse the freshly built
canonical providers only after exact source and product hashes are checked. I
will retain original roots and reports, label the reused bootstrap explicitly,
and preserve each first terminal. The complete suite remains pending until it
actually passes; no earlier partial run establishes the 90-method result.

Original reports: `/tmp/nanolang-canonical-full-ca377-linux` and puck
`/tmp/nanolang-canonical-full-ca377-puck`. Their manifests retain commands,
file-backed logs, bounded process-group cleanup and source/tool/product maps.
Deleted temporary files from unchanged inner runners are not reconstructed.

My first external product-copy helper failed before tests on both hosts: it
created `bin/nanoc` before its relative symlink target, then immediately tried to
hash that unresolved copy. Both observed tool terminals returned1. I retain the
first helper and partial destinations; the original output is in the agent tool
transcript, not a separately captured historical log. I do not fabricate one.
The corrected helper copies regular files before symlinks and archives itself
before checks. Fresh corrected destinations and file-backed preflight logs keep
the original roots unchanged. This does not change the fixture checkpoint.
