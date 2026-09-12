# My executable language contract, version 1

I reuse seven source programs and their byte-exact expected stdout from
`tests/cross-backend`. My versioned manifest fixes their membership; changes
to these programs or outputs are changes to this contract and need review.

Run `make test-language-contract` for fresh tools and the complete matrix.
Run `python3 tests/run_language_contract.py` to check already-built tools.
`NANOLANG_SELFHOST_COMPILER` selects a different self-hosted compiler;
`CC` selects the host compiler for AOT C. I execute cases serially because
my native compiler's shared temporary-file isolation remains unfinished.

Each case must execute successfully with exact stdout through four paths:

- My C seed produces a native executable.
- My Stage 2 compiler produces a native executable.
- My C frontend emits bytecode and my VM executes it.
- My `nvm2c` translates that same bytecode to C11; the host compiler builds
  an executable that runs without my VM.

Missing tools, translation refusals, timeouts, failed execution, and output
differences fail the matrix. I do not consult legacy `.xfail` files. Compiler
and host-build steps have a 60-second limit; executions have a 10-second limit.
Each case uses a private temporary directory that I remove after its run.

This is a small tested contract, not a full specification or a proof. It does
not yet cover rejected programs, ownership, FFI, effects, all match-expression
semantics, or self-hosted NanoISA lowering. My bytecode frontend is C-based.
All 28 execution rows pass with the checked tools after the AOT aggregate
repairs (2026-09-12). Broader contract coverage remains in `docs/ROADMAP.md`;
these seven programs do not establish release readiness.

`make test-language-claims` uses the existing compiler tools to characterize
inferred locals, selected type rejections and shadow handling. It also compiles
and executes the specification's unsafe math wrapper through the C seed.
Its shadow test requires C-seed and bytecode-CLI rejection but records the
remaining Stage 2 native omission; passing that characterization is not
universal shadow conformance. `make test-bytecode-shadows` checks the NanoVM
test-module path and publication guards. Build `bootstrap3`,
`nano_virt` and `nano_vm` first when those tools are missing or stale.

`make test-native-shadow-emitter` exercises my self-hosted transpiler's separate
C test entry. I build the emitter with the C seed and Stage 2, then compile and
execute its generated C. The suite checks selected shadows, independent local
scopes, calls to NanoLang main, assertion failures even with `NDEBUG`, and the
unchanged production entry. It also checks record-string lifetime through
callee locals. This target uses existing compiler tools; it is not evidence
that my native driver executes shadows before publication. That integration,
shadow typechecking, and root/import selection remain roadmap work.
