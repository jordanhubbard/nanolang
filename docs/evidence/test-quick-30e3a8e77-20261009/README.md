# I pass the Darwin broad gate on 30e3a8e77

I ran from the unchanged development worktree at
`30e3a8e771424f8334444d108183f5202bc15efc`:

```sh
CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_VM_EXAMPLE_SHADOW_TIMEOUT_SECONDS=60 make test-quick TEST_TIMEOUT=3600
```

The tool session 78700 completed with exit 0. I retain its complete output in
`test-quick.log`. This is the defined test-quick gate, not every release gate.
The final Forth graphical initialization is explicitly skipped because
xvfb-run and timeout are unavailable. This source pin predates the qualified
native-constant correction and does not establish Linux or final-release parity.
