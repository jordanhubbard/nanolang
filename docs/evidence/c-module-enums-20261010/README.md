# My C module enum identities

I bind every internal enum to a module-owned canonical name while retaining
its source spelling for namespace lookup. Renaming only later collisions was
insufficient: my retained baseline accepted one generic binding mixing a root
Shade with an imported Shade. I now reject that program before replacing its
prior output. I retain foreign enum ABI spellings.

I preserve enum metadata through generic arguments, arrays, substituted
signatures and returned values. Selective enum aliases reference the canonical
declaration and are scoped to their importing module. I reject conflicting
aliases in one importer and allow the same alias in different importers.
My checker, NanoISA emitter, interpreter and native C emitter respect visible
local values that shadow enum names. Native declarations and generic instances
emit named enum parameter/result types after placeholder resolution.

My validation passes:

- 10 shared enum methods through the C producer and through a freshly built
  self-hosted compiler module. They exercise nested/local/returned enums,
  record fields, arrays, qualified result annotations, selective aliases,
  same-named declarations, separate importers, one-letter enum names,
  root/imported ownership and prior-output-preserving identity refusals.
- All 21 existing C generic regression methods.
- The same 10 enum and 21 generic methods with separately instrumented parser,
  checker, environment, nominal binder, module loader, evaluator and NanoISA
  codegen objects (ASan/UBSan). Other linked compiler objects remain ordinary;
  compiler leak detection is disabled. Positive products execute mandatory
  shadows, verification, NanoVM and sanitized generated C AOT.
- 60 environment assertions and 10 lexical-scope methods, including alias
  owner separation, qualified original-name lookup and conflict refusal.
- Two native C methods: the comprehensive existing enum fixture and a generic
  enum result with a later local record shadow. The retained baseline exposed
  missing native enum type spellings, then a future local symbol incorrectly
  hiding an earlier enum receiver. Both corrected cases compile and execute.

I build the self-hosted compiler with:

```
bin/nano_virt src_nano/nanoc_v06.nano --emit-nvm --strip-debug -o /private/tmp/nl51-c-enum-seed.nvm
bin/nano_vm --verify-only /private/tmp/nl51-c-enum-seed.nvm
```

I execute tests/test_module_enum_generics.py with NANO_ENUM_COMPILER set to
bin/nano_virt or a wrapper running that new compiler module with bin/nano_vm.
I set NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang for generated C.
I run tests.test_cseed_generic_functions, make -f Makefile.gnu test-env-scoping,
and tests.test_native_enum_identity. The native compiler tests use NANO_CC
and NANO_LDFLAGS to select LLVM and the installed OpenSSL library.

The fresh compiler module and sanitizer runs precede the final legacy-native
signature/visibility corrections; those corrections are covered by the final
native tests. I do not claim a full-toolchain sanitizer run or a fresh installed
Stage 1/Stage 2 fixed point. Direct qualified constructors and remaining enum
boundaries, tuple metadata, full generic acceptance and exact release gates
remain in the parent scope.
