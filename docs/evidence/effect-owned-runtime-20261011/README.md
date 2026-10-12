# My affine effect execution checkpoint

I extend #976 from `21131d2e7` with matched standalone VM/native effect execution.
This checkpoint does not complete C/Nano producer wiring or the full 5.1 release.

I emit direct native handler activations with bounded heap storage, lexical
scalar aliases, moved resource arguments/results and explicit return propagation.
I release each activation's roots on resume, lexical return, trap and allocation
failure. My VM now creates fresh ownership reference contexts for handler
activations, including dynamic contexts beyond the eight distinct-function bound.
It releases them on resume/unwind/error/destruction. I retain the independent
runtime allowlist, exact operation metadata and affine verification; I do not
admit global/effect combinations, array signatures or borrowed continuations.

## My qualification

- `reuse.log`: `make -f Makefile.gnu test-affine-effect-runtime` passes 20
  serialized executable cases. These cover resource records and unions, strings
  and string-bearing owners, float and void results, mutable scalar captures,
  nested and recursive handlers, contained and escaping direct helper calls,
  lexical returns to main and helper owners, handler-local borrows, unhandled
  effects, assertions and the 1,024-activation limit. Native CLI translation
  matches the direct emitter byte for byte.
- I inject every executed VM allocation position in `vm.c` and `heap.c`: 3,139
  positions across these cases, including 3,065 in the depth-limit case. After
  every outcome I reuse the same VM without an injected fault and check empty
  stack/frame/handler/reference state and the original heap object count.
- I inject every native allocation position in the 19 non-depth cases: 108
  positions. For the 2,048-allocation depth case I test seven selected positions
  (1, 2, 10, 100, 1,023, 1,024 and 2,047), plus the uninjected terminal. Native
  failure preserves caller output and leaves zero tracked allocations. This is
  not an exhaustive native depth-case fault sweep.
- Generated native C passes `-Wall -Wextra -Werror -O1`, AddressSanitizer,
  UndefinedBehaviorSanitizer and leak detection with Homebrew LLVM.
  `sanitizer-reuse.log` repeats the final matrix with the VM, heap and VM test
  harness also rebuilt under those sanitizers. Other linked objects retain their
  existing instrumentation; I do not claim a fully instrumented toolchain.
  `sanitize-vm.py` preserves the host-specific compile and execution recipe.
- `qualified.log` passes affine bytecode (1,557 checks; 2,253 with faults), owned
  runtime (1,526 checks), caller references (1,548 checks), selected union runtime
  (2,380 checks), mixed runtime (2,006 lifecycle checks), their existing allocation
  controls and native sanitizer cases. `contracts.log` passes the ownership
  contract gate. `scalar-projection.log` also records passing scalar-union and
  ownership declaration-projection neighbors before the final VM-context change.
- `sources.sha256` identifies the final implementation/test inputs. The protected
  user fixture remains unchanged at
  `c739aeb158c5b3e94c15d8de1232e4e1415a3f20b2e80fcf96fba39f1dedb976`.

I reproduce the primary gate with:

```sh
make -f Makefile.gnu test-affine-effect-runtime
make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-owned-runtime test-caller-references test-owned-union-runtime test-mixed-samples-runtime
```

## My retained failures

`native-effects-final.log` records the actual VM failure: a borrow created inside
an effect handler had no live originating reference context. I corrected runtime
activation setup/teardown and added recursive handler borrows and helper calls
past depth eight. The assertions remain in the passing matrix.

`native-effects-neighbors.log` records a stale linked-module test: it removed all
resource flags while retaining OWN_PACK, then expected admission. My replacement
is a valid copy-only scalar-union module, executed standalone before the same
linked-module refusal assertions. The same log records Apple's leak-detector
refusal. Setting CC only in the environment did not override this Makefile;
`native-effects-neighbors-corrected.log` preserves that failed setup. Explicit
Make command-line CC selects Homebrew LLVM without weakening leak checks.

My first new union fixtures lacked a required field name and then placed
extension kind 1 after kind 4. I corrected the fixtures to the existing wire
contract; `native-effects-neighbors-final.log` and `sanitizer-union-refusal.log`
retain their rejection. `stale-generator.log` records an invocation of the old
fixture binary after requesting an object target with no rebuild recipe; the
phony affine gate rebuilds the actual executable. `sanitizer-first.log` records
the diagnostic recipe's initial command-selection error. None establishes a
compiler/runtime failure or a passing qualification. I retain these terminals
instead of silently replacing them.

My next implementation is C/Nano producer emission of exact operation signatures
and handler transfers. Source checking already has separate evidence; these
hand-authored executable fixtures do not establish source-producer completeness.
