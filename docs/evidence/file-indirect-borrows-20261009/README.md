# I qualify explicit indirect File borrows

Under #989, over `646ed691c`, I add the private counted-reference indirect call
and its complete query/runtime/dispatch path. [My contract](../../NANOISA_FILE_INDIRECT_BORROWS.md)
records the wire layout, candidate/mode checks, ownership transitions, frame
bounds, lease lifetime and remaining source/public requirements.

My first batch passes33 Python schema checks, all3,018 C NanoISA checks, both
File opcode methods (1,890 encoding/transport/generic-refusal checks each), all
six indirect query methods, and both initial expanded sanitizer dispatch methods.
I retain its complete log. I then extend the corpus with the maximum256 borrowed
formals and an independent generated-reference-map mutation control.

That boundary fixture executes and emits all86 modules but fails its final
zero-live-allocation assertion. Its metadata came from the tracked builders, but
its resize used raw `realloc` after the dispatch fixture removed allocator macros.
I retain the failing wrapper, exact capture command/status, diagnostic and tail.
I use the matching tracked resize in instrumented mode; the zero-live assertion
remains unchanged. The subsequent complete run passes.

## My final results

Both corrected sanitizer dispatch methods pass in157.174 seconds on Darwin.
Each mode captures86 exact modules, emits77 native C products and refuses nine
malformed/unsupported modules. Actual VM/native O0/O2 traces agree byte for byte:

| Mode | Traces per execution | Result |
| --- | ---: | --- |
| Allocator/host instrumented | 1,160 | VM = native O0 = native O2 |
| Linked providers | 1,154 | VM = native O0 = native O2 |

The new cases cover both selected targets, catalog permutation, shared aliases,
repeated shared reference slots, exclusive and mixed disjoint borrows, reordered
maps, indirect forwarding, real writes through exclusive references, all lower
fuel budgets, denied open and assertion/cleanup failures. Malformed lengths,
indices and slots, scalar-slot mappings, candidate mode disagreement and exclusive
overlap refuse. The256-formal case executes in both layouts and checks the258-slot
staging bound. Altering a copied native reference-map comparison refuses before
host acquisition at O0/O2. Existing callable, owned, scalar, loop and cleanup
cases remain in the same corpus.

The existing richer-plan preparation sweep retains436 allocation refusals with
fresh recovery in VM and each native replay; emitter controls retain354 refusals.
Tracked execution allocates no project heap. The listed providers and generated C
use ASan/UBSan with leak detection; common linked objects retain their prior
builds. I do not claim that every linked object was sanitizer-rebuilt.

Both preserved cyclic dispatch methods pass in34.682 seconds; both indirect
carrier methods pass in9.128 seconds. GCC16.2.0 accepts the changed runtime/query/
adapter/fixture sources and all77 generated C products under `-Wall -Wextra -Werror`.
All five workflow checks and `git diff --check` pass. The earlier Linux/Darwin
CI observations apply to646ed691c, not this changed revision.

I retain exact inputs, serialized manifests, generated-source hashes, all final
command statuses and both reference traces. The command endpoints are successful,
reaped and process-group cleaned. I run with Homebrew LLVM Clang on PATH and
`OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5`:

```sh
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-nanoisa test-file-opcodes test-file-indirect-queries \
  test-file-indirect-dispatch-sanitize
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang \
  OPENSSL_PREFIX=/opt/homebrew/Cellar/openssl@3/3.6.5 \
  test-file-indirect-dispatch-sanitize test-file-cyclic-dispatch \
  test-file-indirect-runtime
```

Paired C/Nano source and full shadows, public grants, installed products, Linux
qualification, mixed profiles and all remaining5.1 release requirements stay open.
