# My owned binding-state static review

I reviewed PR937 production head
`f2bac81a812d85a15bc0e9c4e27adb50d34e67fd` without changing its branch. The
production delta adds an activation-owned sidecar and one-slot tuple cell
edges; it stores no pointer into the movable VM stack. Reads retain ordinary
values before publication, initialization and assignment move independently
owned inputs after validation, and clear detaches the old edge before release.
The sidecar is accounted storage rather than a traced heap object. This
checkpoint does not attach cells to frames, admit capture opcodes, construct an
environment, or qualify VM/C/LLVM/Wasm capture execution.

The four reviewed inputs have these SHA-256 identities:

```text
f5f7818e150ae539f10f4b5d419aed52a8687b627d17ef23cb53d517d44b8bbc  Makefile.gnu
dd46dbd3903d3c3b95e1ef70c48b319b3cc2292b4f16df8153ffdeb637c0d6b8  src/nanovm/binding_state.c
231b33ff29b3c360151f59d8d08eb37e453802d9fd898475d152f1f8a3c56371  src/nanovm/binding_state.h
00f2071bafa25e774122996142cbcbd240b809d6cb1e8513e578e7efef269e55  tests/nanovm/test_binding_state.c
```

My first command selected the product worktree's older Makefile and stopped
with `No rule to make target test-binding-state`; that was a setup error and is
not a product result. The corrected exact-head command was:

```text
make -C /private/tmp/nanolang-binding-review-f2 -f Makefile.gnu \
  test-binding-state CC=/usr/bin/clang \
  CFLAGS='-std=c11 -Wall -Wextra -Werror -O1 -g'
```

It compiled the new fixture and stopped at link time before executing a test:

```text
Undefined symbols for architecture arm64:
  "_isa_tag_name", referenced from:
      _val_to_cstring in value-af65ec.o
```

`src/nanovm/value.c:246` calls `isa_tag_name`; its definition is
`src/nanoisa/isa.c:31`, while the new target links only the fixture, heap,
cycle collector and value implementation. The retained corrected log has
SHA-256
`d0108df4c08a89ee4645710ea1768bd0dd9b3b608fc8518bfe9d53ccb28b48eb`.
I require an exact provider-link correction and a fresh strict result before
clearing this bounded review. PR522 and full 5.1 publication remain held.
