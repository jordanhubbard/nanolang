# My higher-order generic inference

I bind ordinary arguments before specializing generic callbacks. I keep callback
variables separate from the enclosing function variables, infer outer results
from the concrete callback, and repeat deferred callbacks until their constraints
resolve or no progress remains. Both my C checker and source checker/emitter use
this order-independent behavior. My source instance registry compares variable
names and types instead of their discovery order.

I also resolve a generic array literal's nominal template element in the concrete
C NanoISA parameter scope, without overwriting the shared template annotation.
This fixes the retained wrap(T)->array<T> module that executed in NanoVM but
failed C AOT admission as a record-array literal.

My seven higher-order methods cover both argument orders, inferred results,
dependent callbacks, imported aliases, nested arrays and records, multiple array
result element kinds, and conflicting or unanchored bindings. Refusals preserve
an existing output. Positive source routes run mandatory shadows, module
verification, NanoVM and generated C with LLVM ASan/UBSan.

Validation commands use python3 -m unittest with these modules:

- tests.test_higher_order_generic_functions
- tests.test_contextual_generic_functions
- tests.test_cseed_tuple_generics (including its inherited generic tests)

My C source route and freshly rebuilt self-hosted source route each pass all 52 methods. My compiler-component sanitizer route
passes the 18 higher-order/contextual methods, instrumenting parser, checker,
environment, NanoISA emission, nominal binding, evaluator, module loading and
metadata, and resource flow. Compiler leak detection is disabled and other
compiler objects are uninstrumented. My existing native contextual suite passes
all 11 methods.

I build my current source compiler with bin/nano_virt src_nano/nanoc_v06.nano
--emit-nvm, including mandatory shadows, and verify its module. inputs.json pins
the implementation and module. The source-route suite selects this module with
a NANO_GENERIC_CSEED_COMPILER wrapper running bin/nano_vm MODULE -- "$@".
NANO_NATIVE_TEST_CC selects /opt/homebrew/opt/llvm/bin/clang for generated C.

## My remaining native work

My legacy native higher-order route still loses direct generic callable and
aggregate metadata. A separate, unlanded complete-signature experiment passes
17 of 18 contextual/higher-order methods; expanded array results still choose
integer accessors. An additional native execution probe of the 21 C generic
methods exposes 11 failures in typed locals, transitive/imported calls and array
metadata. These diagnostics are retained as prototype evidence, not results of
the landed native emitter or proof that all 11 defects are new regressions.
I do not include that incomplete native experiment in this change.

The native rows remain open. My latest installed-stage fixed point predates this
change; exact-candidate Linux/Darwin qualification and the full #976 release
contract also remain open.
