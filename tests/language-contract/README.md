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
My current AOT record-return and variant failures remain required failing rows,
tracked in `docs/ROADMAP.md`; I do not call this matrix release-ready.
