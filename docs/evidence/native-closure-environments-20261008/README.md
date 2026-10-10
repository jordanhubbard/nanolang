# My native closure environments

I translate `CLOSURE_NEW`, `LOAD_UPVALUE` and `STORE_UPVALUE` to native owned
record environments. A tagged callable retains its module-local target and its
own environment pointer. Aliases share mutations; separate instances retain
separate environments. Capture fact slots follow bytecode locals internally and
preserve their existing scalar and aggregate shapes.

I pass the environment explicitly to captured functions. Dispatch checks the
runtime callable tag, the analyzed target set, environment presence, capture
count and stored target identity. Function references remain tag 11; closures
remain tag 15. Equality uses closure identity, and printing preserves my VM's
`closure(index)` form. My record fields retain both tag and environment pointer.

I trace callable locals, operand slots, record fields, globals and the active
environment. The popped indirect callee remains a root while call-boundary
collection runs. My existing record owner family collects unreachable
environments and releases remaining owners at shutdown. I do not add an
interpreter or embedded bytecode runtime.

My six new VM/native execution methods, one implicit-root refusal method and
nine existing native callable methods pass with
strict C11 and ASan/UBSan/LSan. They cover the unchanged source-produced returned
chain; zero and multiple captures; instance identity and alias mutation; closures
inside records and record-array globals; managed string, array and record
captures; captured callables; float and bool tags; at least two observed
collections in lifetime fixtures; and five malformed-callable controls.
My final prepared-producer parity/CLI/product run passes all 65 methods in
22.242 seconds, including implicit-root refusal with prior-output preservation.
My final complete native gate passes all 2,431 checks. I also retain the earlier
64-method and native passes before adding the implicit-root guard.

```sh
PATH=/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:$PATH \
  python3 -m unittest -v tests.test_native_closures tests.test_native_callables
PATH=/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:$PATH \
  make -j2 test-nvm2c
```

I retain my first missing projection-shape terminal and unused float-helper
strict-warning failure. The first lifetime fixtures also contained a wrong
struct-array tag and forward string constants; moving the constants initially
duplicated them. Those fixture corrections preserve the original assertions.

My self-hosted producer still rejects captured lexical names. My function arrays
still store non-owning named target IDs and cannot yet carry closure environments.
Those repairs, fresh compiler-product qualification, raw fixed-point and the
other complete 5.1 gates remain required. These native environment tests do not
close the parent task or authorize a release claim.
