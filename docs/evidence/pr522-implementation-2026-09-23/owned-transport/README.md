# My format-4 owned-union transport checkpoint

I implement declaration transport for prior-only aggregate union graphs. Exact
child kinds and layouts, complete ownership flags, variant slices and aggregate
descriptors survive binary and unverified textual round trips. Old ownership
formats retain their grammar. Executable verification, verified assembly, VM
and native translation still refuse format 4 pending selected transfer support.
My CLI refusal control preserves existing native output.

My final transport fixture passes 456 assertions under Homebrew LLVM ASan/UBSan
with leak detection. `instrument_transport.py` instruments ownership contracts,
affine state, verifier (including owner-array routing), and the test fixture;
other dependency objects are ordinary builds. This is scoped instrumentation,
not a fully instrumented toolchain or a release qualification.

`make test-ownership-contracts test-nvm2c` passes, including 1,500 shape and
2,428 translator assertions. The existing affine state checks pass 314 normal
and 346 allocation-injection assertions; affine bytecode passes 546 and 856;
owned transfer passes 184 and 275. Scalar-union runtime and copied declaration
projection controls pass in the retained regression terminal.

That regression command stops at `test-owned-array-layouts`: the Darwin fixture
hardcodes `-lcrypto` without the selected Homebrew library path. Its unchanged
239 assertions pass when rerun with
`LIBRARY_PATH=/opt/homebrew/opt/openssl@3/lib make test-owned-array-layouts`.
I retain both terminals and track a fixture-link repair on my roadmap. This is
an explicit workaround, not a claim that the default target passes.

`logs.json` seals the uncompressed terminal bytes. Selected extraction encoding,
branch-local ownership verification, runtime/native transfers and canonical
producer support remain required before the existing generic ownership source
acceptance can pass. I have not rerun a full bootstrap for this C-only slice.
