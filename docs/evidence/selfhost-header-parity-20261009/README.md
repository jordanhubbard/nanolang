# My self-hosted header parity checkpoint

My unchanged private-header baseline succeeds through nano_virt but both e1fa92e03-built self-hosted stages report E0011 for HEADER_VALUE and preserve prior output. I retain the source, diagnostics and binary hashes.

I now share header path discovery and the exact literal scanner through one declared compiler-support snapshot query. I copy its transient string result before another call, retain constants in invocation-scoped bindings with their source owner, preserve first-definition and explicit-function priority, and emit integer immediates after ordinary lexical resolution. Reset clears the table. The checker rejects integer-constant calls before lowering; the initial component test exposed that missing refusal and remains retained.

The source compiler compiles with all shadows. My corrected self-hosted component executes the header value/snapshot/refusal suites through checked shadows, verified VM and strict native ASan/UBSan/LSan products: three methods pass in6.756s and the real SQLite method honestly skips absent headers. C-seed methods also pass. I reuse the same suite for installed Stage1 and Stage2 through `make test-selfhost-header-constants`.

My full Darwin gate returned 0. All 17 fresh bootstrap steps pass with unchanged source inventory; the raw Stage1 and Stage2 modules are byte-identical. Both native compiler smoke tests and installed execution without nanoc_c pass. The installed-stage suite runs eight methods in 6.481 seconds: six pass and two real SQLite methods skip because headers are absent. I retain all bootstrap logs, the manifest, receipt and artifact hashes in `qualified-bootstrap/`.

My adjacent native gate initially selects AppleClang in some harnesses and fails because leak detection is unsupported. I retain that failure. Selecting HomebrewClang through both CC and NANO_NATIVE_TEST_CC passes all 2,435 checks with leak detection enabled. Linux qualification and real SQLite header execution remain open under #985; this checkpoint does not establish release readiness.
