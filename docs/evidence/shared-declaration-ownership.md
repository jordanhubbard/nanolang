# My shared declaration ownership classifier

I extract my existing typechecker local-let classifier into
`src_nano/compiler/declaration_ownership.nano`. Typechecking retains its
wrapper; NanoISA global classification inverts the same flags. Native-shadow
C generation computes flags once for each of its three global-declaration
passes instead of rescanning every block and function for each declaration.
The transpiler does not import the full typechecker.

I preserve classification for Parser-produced declaration IDs: function
parameters and lets in ordinary or unsafe block statement lists are local.
Declaration order, module identity, initializer order and generated names
remain unchanged. I retain the old scalar queries as independent shadow
oracles, and test globals, parameters, nested blocks, unsafe scopes,
shadow-local declarations and an empty parser. I do not claim equivalence
for manually constructed malformed Parser indices.

Validation:

- `make -j8 test-nanoisa-src-nano`: 86 comparison checks and all 64 Python
  methods pass. These include the bytecode-executed emitter's exact assembly
  comparison and initializer-order assertions.
- `make -j8 bootstrap`: a fresh three-stage native bootstrap passes its
  configured smoke gates. Its native binaries differ; this establishes no
  canonical bytecode fixed point.
- Before and after canonical compilers emit identical C for a 40-worker
  fixture with global/local name overlap, parameters, nested and unsafe
  scopes, shadow declarations and a mutable primitive global. Both outputs
  are 37,116 bytes, SHA-256
  `75a42c681bdc37c0b6e4ad612f1d49dc7eca56bf1b5093c64f877daf9970d407`.
  The new compiler also builds and runs that fixture, checking initial
  values, mutation and a worker result.

Retained logs are `/tmp/nanolang-shared-ownership-gate.log`,
`/tmp/nanolang-shared-ownership-bootstrap.log`,
`/tmp/nanolang-shared-ownership-{before,after}.log`, and
`/tmp/nanolang-shared-ownership-native-build.log`. The comparison fixture is
`/tmp/nanolang-shared-ownership-native.nano`; its generated C is retained as
`/tmp/nanolang-shared-ownership-{before,after}.c`.

This closes the shared-classifier substep of
`task_36ceaa830d7d46ba8a5471326f525aac`. Full VM compiler-source execution,
canonical bytecode convergence and the NanoISA-only compiler product remain
open. I keep the pinned bootstrap source and running native experiments
separate from this source change.
