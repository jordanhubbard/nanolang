# My contextual generic function values

I bind a declared generic function's complete signature to a concrete callable
context in both NanoISA source producers. I reuse direct-call specialization
queues and their limits instead of emitting unresolved templates. My covered
contexts are typed locals, reassignment, returned callbacks, direct arguments,
tail-call arguments and indirect-call arguments. I retain lexical bindings and
module aliases, and I mark emitted callable references in self-hosted metadata.

I execute separate int/bool instances, structural arrays, tuples containing
records, returned callbacks, imported aliases, reassignment and forwarding.
I reject inconsistent generic bindings and incompatible ordinary callback
returns, including unused function bodies, while preserving existing output.
My self-hosted return hints formerly erased callable signatures. My C checker
needed the function's separate return signature as well as its type-info view.
I retain both failures and the tail/indirect emission failures.

Validation:

- I pass 43 methods through bin/nano_virt: nine contextual methods plus the
  existing 34 generic/tuple methods.
- I build the current source compiler with mandatory shadows, verify its module,
  and pass the same 43 methods through that fresh self-hosted module.
- I pass all nine contextual methods with ASan/UBSan instrumentation in parser,
  checker, environment, nominal binding, evaluator, module/metadata, ownership
  flow and NanoISA codegen. Other compiler dependencies remain ordinary and
  compiler leak detection is disabled.
- I pass all 18 existing callback methods with NANO_CALLBACK_COMPILERS=nanoc_c.
  That suite includes its separate NanoVirt callback method.
- Positive shared-suite products execute source shadows, module verification,
  NanoVM and generated C with LLVM ASan/UBSan. The generated products link the
  ordinary runtime object; this is not whole-toolchain sanitizer qualification.

I run python3 -m unittest tests.test_contextual_generic_functions
 tests.test_cseed_tuple_generics -v with NANO_NATIVE_TEST_CC selecting Homebrew
LLVM clang. NANO_GENERIC_CSEED_COMPILER selects bin/nano_virt or a wrapper running
/private/tmp/nl51-contextual-final-seed.nvm through bin/nano_vm. I build that
module with bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug, then
run bin/nano_vm --verify-only. inputs.json pins changed source and module bytes.

I do not close all contextual or generic coverage. My legacy native AST-to-C
route now checks the callback but still emits undeclared nl_identity; I retain
that failure for implementation. Other storage contexts and context inference
inside generic higher-order calls remain outside this tested corpus. The full
installed Stage 1/Stage 2 fixed point and exact Linux/Darwin release gates have
not been rerun for this checkpoint. The complete #976 scope remains open.
