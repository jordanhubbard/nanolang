# My string-search branch reconciliation

I reconcile worker head `07d38702ae4e052e300489636a213c69b611c749`
against integration `fe3532ff`. My existing ancestor `fdb5efe7` already
implements its shared search header, registry entry, interpreter and VM
dispatch, generated-C inclusion, and private typechecker helper rename.
I retain the current production source rather than replace it with the older
snapshot. In particular, last-search retains its `INT64_MAX` length guard.

My existing `tests/string_search.nano` covers first and last byte offsets,
overlapping matches, empty strings and needles, missing and longer needles,
and UTF-8 byte offsets. This subsumes the branch's added source cases.
I add its first-search null-needle assertion to the existing VM builtin test;
the remaining null and overlap assertions are already present.

I ran these focused gates on this macOS host:

```sh
make test-nl-string test-vm-builtins
python3 -m unittest \
  tests.test_selfhost_cli.SelfhostCliTests.test_string_search_native_stages \
  tests.test_selfhost_cli.SelfhostCliTests.test_string_search_emitted_c
python3 -m unittest \
  tests.test_bytecode_shadows.BytecodeShadows.test_string_search_shadows_and_production
```

All commands exit zero. The native test compiles and executes the fixture
through C-seed, Stage1 and Stage2; the emitted-C test independently compiles
and runs its output; the bytecode test checks shadows and production output.
The log is `/tmp/nanolang-search-reconciliation.log` on this host.
These tests use existing compiler binaries, not a new bootstrap or a clean
cross-platform release build.

MAC `task_a9b152cf5299491694d96a2385527e98` is stopped and unowned when
inspected. I attach evidence without claiming a successful ledger closure.
Full branch reconciliation and release acceptance remain open.
