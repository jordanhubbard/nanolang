# My self-hosted rich tuple execution

I admit supported arrays, ordinary records, enums, callable signatures, nested
tuples and byte values as tuple elements. I retain callable recursion context
and reject direct or nested resource storage. I distinguish function arrows
from generic closers when splitting tuple spellings.

I build my current source compiler with mandatory shadows and verify its module:

```
bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug -o /private/tmp/nl51-rich-tuple-seed.nvm
bin/nano_vm --verify-only /private/tmp/nl51-rich-tuple-seed.nvm
```

I pass all 34 methods of tests.test_cseed_tuple_generics through that compiler
executed by bin/nano_vm. My C producer passes the original 33 methods and the
additional byte method in separate runs. Positive products execute shadows,
module verification, NanoVM and generated C with LLVM AddressSanitizer and
UndefinedBehaviorSanitizer. My generated C checks link the existing ordinary
runtime object; this is not whole-toolchain sanitizer qualification.

The five previously rejected tuple cases now execute: imported record results,
nested array results, structural tuple parameters/results, and nominal/callback
elements. Additional controls execute shared array alias mutation, recursive
record/callback signatures and byte tuples. Invalid shapes, bounds and resource
tuples refuse publication while preserving existing output.

I retain failed development shadows for function-arrow splitting and an outdated
admission assertion, plus the byte-tuple baseline. My corrected source adds
positive admission assertions alongside unsupported-element refusal controls.
inputs.json pins the changed source, test file and generated compiler module.

I run the Python suite with NANO_GENERIC_CSEED_COMPILER selecting a wrapper
that invokes the new module through bin/nano_vm, and
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang. The default producer is
bin/nano_virt. My final self-hosted suite takes 21.559 seconds.

I have not rerun the Stage 1/Stage 2 fixed point or exact-candidate Linux/Darwin
qualification for these changes. Remaining generic/aggregate coverage, ownership
and service integration, compiler product cleanup and the full 5.1 contract
remain open under #976.
