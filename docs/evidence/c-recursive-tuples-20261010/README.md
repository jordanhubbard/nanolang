# My recursive C tuple annotations

I retain complete tuple element annotations through parsing, owned copies,
comparison, generic function binding/substitution, projection and module
metadata serialization. I keep flat tags as compatibility views. My purity
and ownership traversal also visits the retained nested annotations.

My C source producer now executes generic tuple identities, destructuring,
nested tuple/array results, structural tuple parameters and result substitution,
record/enum/callback elements and imported tuple results. I reject repeated
bindings with incompatible nested shapes and preserve prior output. I also
check declared tuple initialization, assignment, ordinary call arguments and
returns, plus projection bounds. My retained baseline shows three invalid
shape programs publishing output; a later call control exposed a diagnostic
that was printed without recording a fatal checker error. All corrected
refusal controls preserve their prior output.

I pass all 30 methods in tests.test_cseed_tuple_generics: 21 existing generic
regressions and nine tuple methods. Positive products execute mandatory shadows,
module verification, NanoVM and sanitized generated C AOT. The same 30 methods
pass with separately instrumented parser, checker, environment, nominal binder,
module loader/metadata, resource-flow, evaluator and NanoISA codegen objects
(ASan/UBSan). Other dependencies remain ordinary and compiler leak detection
is disabled; this is not whole-toolchain sanitizer qualification.

I also pass all 10 shared enum methods, parser tests (including recursive
annotation ownership/equality), generated module-metadata round trips and the
resource-flow allocation gate (300 owners, 1,832 allocation-failure positions).
My source compiler builds and verifies a fresh self-hosted compiler module:

```
bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug -o /private/tmp/nl51-tuple-current-seed.nvm
bin/nano_vm --verify-only /private/tmp/nl51-tuple-current-seed.nvm
```

I run the shared suite with NANO_GENERIC_CSEED_COMPILER selecting the C producer
or a wrapper executing that new compiler module through bin/nano_vm, and
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang. My unit command is
make -f Makefile.gnu -j4 nano_virt test-parser test-module-metadata
test-resource-flow-allocations. inputs.json pins the changed compiler inputs
and the generated module.

The freshly built self-hosted compiler passes 25 of the 30 shared methods.
Five positive cases remain rejected by its scalar-only tuple admission:
imported record results, nested array results, structural tuple parameters,
structural tuple results containing arrays, and nominal/callback elements.
Its invalid-shape and bounds controls pass. I retain these failures instead
of treating C success as producer equivalence. Self-hosted tuple admission,
remaining generic coverage and final installed/platform release gates remain
open under #976; I have not rerun the Stage 1/Stage 2 fixed point here.
