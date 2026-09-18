# My literal-string translator checkpoint

I implement `NANOISA_LLVM_LITERAL_STRINGS.md` under MAC
`task_e63298a462a24967b57e6ecce3c9223d`. My reviewed production checkpoint is
`d0bc758f`, restacked on main `d66f9e73`; final test correction is `beb6a8f5`.
I tested on Linux ARM64. This is not Darwin or full-language acceptance.

I store literals in immutable module-owned byte arrays, reached through
one-based constant handles and explicit-length descriptors. I retain empty
truthiness, embedded NUL, UTF-8 byte lengths, unsigned byte ordering and content
equality. My direct module-API test creates equal bytes at distinct pool indices
because normal assembly/loading deduplicates them; VM, LLVM and Wasm all check
that their identities do not replace content comparisons.

My final integrated gates pass:

- Nine literal-string methods in 30.229 seconds, including explicit/implicit
  results, calls, branch-selected locals/globals, initializer result discard,
  same-instance versus fresh-instance execution, numeric-consumer errors and
  prior-output preservation. `/tmp/nanolang-llvm-literals-integrated-final.log`.
- Eleven scalar-global methods in 8.497 seconds; twelve profile decisions in one
  method (0.256 seconds), including separate unchanged CLOSED_SCALAR refusal;
  ordinary verifier cleanup checks and all 96 verifier assertions.
  `/tmp/nanolang-llvm-literals-integrated.log`.

Before the additive main restack I also pass sixteen LLVM methods (37.723s),
seven generic numeric methods (130.700s), and the affected forty-seven methods
covering eight then-current string methods plus the thirty-nine existing Wasm,
truthiness, implicit-return, U8 and generic comparison methods (67.918s).
I retain `/tmp/nanolang-llvm-literals-regressions-r3.log` and
`/tmp/nanolang-llvm-literals-wasm-r4.log`. Production source is unchanged by the
restack. The ninth string method was added on the integrated tree.

Positive programs execute in NanoVM, LLVM before/after optimization, native
Clang -O2 with ASan/UBSan options, and Wasmtime. Reentry additionally compares a
single Node Wasm instance with a new instance. These compiler options do not
claim instrumentation of linked LLVM libraries or a managed-allocation test;
this slice allocates no runtime string objects.

I preserved initial logs for obsolete tests that expected literal signatures
and results to be refused. I replaced those negative cases with unsupported
array signatures/results and retained paired string positives. I also retained
the first numeric-consumer test: typed instructions correctly refused known
string operands during ordinary verification, while generic arithmetic failed
at runtime. The final test distinguishes those boundaries. No verifier guard
was removed to admit those invalid operands.

My literal profile conservatively refuses ADD/CAST_INT/CAST_FLOAT throughout
string-bearing modules, including numeric uses in such modules. My old scalar
profile remains separate. Computed strings, managed allocation, aggregate/heap
identity and declared host linkage remain required roadmap tasks, respectively
`task_51da49b39230468784da3481b893563b`,
`task_488a05eb5e2a417caf83a8353363a30d` and
`task_2d2e9eb552394f6e84e90f5aa08484e2`. Full LLVM/Wasm coverage and the release
publication hold remain open.
