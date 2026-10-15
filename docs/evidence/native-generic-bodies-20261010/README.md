# My native generic body specialization

I register direct and contextual generic functions by their complete checked
signatures. I substitute local annotations in each concrete body, retain their
owned metadata until callable typedef emission, and recheck transitive calls
inside the concrete parameter scope. Emitted parameters carry source locations
so retained template bindings cannot override them. Imported calls select their
concrete instances, and array access consumes checked result metadata.

I buffer global initializers and function bodies before declaring their discovered
specializations and dependent typedefs. Effect dispatch initializes its emission
context before body generation; its declarations remain after the typedefs.
Instance snapshots survive queue reallocation. I bound recursive specialization
by per-declaration ancestry while accepting 70 independent record instances.

The finite-instance control also exposed a source-producer metadata defect.
I now emit layout and ownership hexadecimal strings in even 2048-character
chunks, below the assembler's 4096-byte quoted-string buffer. I retain the same
metadata bytes and verifier checks. My helper shadows cover empty, small and
multi-chunk payloads, and the full 70-instance program executes through both
NanoISA producers and native C.

My packaged wrapper now links lib/libnano_compiler_inputs.a instead of a partial
copied list of File input objects. This retains the Socket/WebSocket nominal
providers required by its verifier and service snapshot code.

## My validation

- All 41 native methods pass: 23 generic, seven higher-order and 11 contextual.
  NANO_CONTEXTUAL_NATIVE_COMPILER selects the compiler for every positive and
  refusal invocation in tests.test_cseed_generic_functions; the other suites
  already support this selection. Native refusals preserve prior output.
- The same 41 methods pass with ASan/UBSan in parser, checker, environment,
  native transpiler, nominal binding, evaluator, module loader/metadata and
  resource flow. Other compiler objects and native output binaries are ordinary;
  compiler leak detection is disabled.
- Both C NanoISA and freshly rebuilt source routes pass all 54 methods from
  tests.test_cseed_tuple_generics (including inherited generic tests),
  tests.test_contextual_generic_functions and tests.test_higher_order_generic_functions.
  Their products run mandatory shadows, module verification, NanoVM and C AOT
  compiled with LLVM ASan/UBSan.
- All 22 adjacent lexical-scope, imported-callback and native-effect tests pass.
- make -f Makefile.gnu test-env-scoping test-transpiler passes its units and
  lexical/assertion-literal controls.
- make -f Makefile.gnu test-wrapper-gen test-wrapper-packaged-interpreter passes
  five wrapper units, nine publication methods and three packaging methods.
- My current src_nano/nanoc_v06.nano compiles with mandatory shadows and the
  resulting module verifies. inputs.json records the implementation and module.

I run the native suite with NANO_CONTEXTUAL_NATIVE_COMPILER set to an absolute
compiler path and python3 -m unittest tests.test_cseed_generic_functions
 tests.test_higher_order_generic_functions tests.test_contextual_generic_functions -v.
For NanoISA routes I omit that variable and select the fresh module through a
NANO_GENERIC_CSEED_COMPILER wrapper invoking bin/nano_vm MODULE -- "$@".
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang selects generated-C sanitizers.

My instrumented evaluator's unchanged 5000-call shadow failed under the default
8176 KiB stack, immediately rather than at its timeout. Raising the timeout did
not help. With ulimit -s 65520 the same test and full 41-method suite pass. This
establishes stack-limit sensitivity; the original child failure did not provide
a precise sanitizer fault trace. I retain both observations and do not reduce
the recursion assertion or disable shadows.

My initial buffering order caused two native effect regressions: foreign-callback
escape and owned-local cleanup. Both passed through the prior instrumented
compiler and now pass after effect context initialization precedes body generation.
The wrapper link and metadata chunk failures are also corrected, with original
logs retained.

I close these reproduced native/body, finite-metadata and dependency gaps. I do
not claim full generic/closure completeness or release readiness. The installed
Stage 1/Stage 2 fixed point predates this change; full #976 scope and exact-candidate
Linux/Darwin gates remain open.
