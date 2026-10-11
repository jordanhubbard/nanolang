# My native contextual generic functions

I retain an owned specialization name on each checked generic function-value
expression and an owned complete signature in my native instance registry.
I emit concrete declarations and bodies from that signature, including imported
aliases, while retaining the original generic declaration. My parser clones and
frees expression annotations, and my environment frees instance signatures.
Generated names avoid declared functions and existing instance names.

My native tuple registry now owns recursive annotations and compares complete
identities. Literal selection uses its checked tuple metadata. I collect tuple
and callable typedef dependencies before emission, then emit prerequisites first.
This preserves distinct record tuples, nested tuple elements and function-valued
parameters instead of erasing their C prototypes.

Validation:

- My ordinary native route passes the nine contextual methods plus two added
  methods for distinct nominal tuples and nested tuple/callable dependencies.
- All 11 contextual methods pass with compiler-component ASan/UBSan in parser,
  checker, environment, native transpiler, nominal binding, evaluator, module
  loader/metadata and resource-flow code. Compiler leak detection is disabled;
  other compiler dependencies and the generated native products remain ordinary.
- Both NanoISA source producers pass all 45 shared methods: 11 contextual and
  34 existing generic/tuple methods. Positive products execute mandatory shadows,
  module verification, NanoVM and generated C with LLVM ASan/UBSan.
- All 21 adjacent native callback, imported typedef and enum methods pass.
- Parser, environment-scoping and transpiler unit targets pass, including their
  lexical-boundary and assertion-literal Python controls.
- My current compiler source builds with mandatory shadows into a fresh verified
  module. inputs.json pins the source, tests and resulting module.

I select native execution with NANO_CONTEXTUAL_NATIVE_COMPILER=bin/nanoc_c
(using an absolute path) and run python3 -m unittest
 tests.test_contextual_generic_functions -v. Without that variable, the suite
uses the existing NANO_GENERIC_CSEED_COMPILER producer selection and
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang for generated C.

My first instrumented-native run failed because its relocated executable derived
/private/tmp as the runtime root. I reran with the executable under an isolated
bin directory and links to the repository runtime inputs; the corrected run
passes all 11 methods. This harness repair does not qualify arbitrary relocation.

I close the reproduced legacy native callback and typedef defects. The broader
contextual-generic row remains open for additional storage contexts and inference
inside generic higher-order calls. The installed Stage 1/Stage 2 fixed point,
exact Linux/Darwin release gates and the full #976 release scope remain open.
