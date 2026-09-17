# My freestanding Wasm translator

I translate verified NanoISA through the same scalar lowering as `nvm2llvm`,
then LLVM's wasm32 backend and `wasm-ld`. I produce a standalone core Wasm
module, without an AST backend, Emscripten, NanoVM or host imports.

```
make nvm2wasm
bin/nvm2wasm program.nvm -o program.wasm
wasmtime run --invoke nano_entry program.wasm
```

My exported `nano_entry` takes no parameters and returns an `i32`, matching
my LLVM executable entry result before a host interprets it as an exit status.
Wasmtime prints that invocation result; it is not language stdout. LLVM gives
`main` a target-specific C startup convention on wasm32, so I use the explicit
entry-name API instead. My LLVM CLI also accepts `--entry-name nano_NAME`;
its default remains `main`. Other entry names are refused to avoid collisions
with internal lowering helpers.

I require Python 3, my built `nvm2llvm`, LLVM `llc` with wasm32 support and
`wasm-ld`. `NANO_NVM2LLVM`, `NANO_LLC` and `NANO_WASM_LD` each select one
executable path; I execute them without a shell. A missing or failing tool
fails publication. With `-o`, I stage work beside the destination and replace
it only after successful translation and linking. I preserve source files,
hard-link aliases and prior output on failures. Without `-o`, I write the
completed binary module to stdout. I do not permit unresolved linker imports.

My initial tested profile is integer/bool scalar storage, calls, recursion,
loops, branches and assertions. I inherit the LLVM profile's explicit
refusals, including unsupported imports, layouts, initializers and implicit
returns. Heap values, host calls, linked modules and reference/passive
contracts remain work; this foundation does not complete my Wasm release gate.

## My execution evidence

`make test-nvm2wasm` requires Wasmtime and Node in addition to the LLVM test
tools. Missing tools fail this explicit gate. Four shared fixtures execute
the same modules on VM, C, LLVM interpreter, optimized LLVM, linked LLVM and
Wasm. Node checks the Wasm import list and instantiates it without imports.
I exercise arithmetic boundaries, division/remainder, bool tags, argument
order, recursion, stack joins and loop state. Three additional methods check
entry results/names, publication/source preservation and profile refusals.

Seven Wasm methods pass in 1.725 seconds. The preceding combined gate passes
six Wasm and eleven LLVM methods in 6.720 seconds. Logs are
`/tmp/nanolang-wasm-entry-contract-tests.log` and
`/tmp/nanolang-wasm-named-entry-tests.log`; the initial C-main ABI failure is
retained in `/tmp/nanolang-wasm-foundation-tests.log`. I used installed LLVM
23.0.0git and Wasmtime 43.0.0 on Linux ARM64. VM/C comparison tools came from
the separately built product checkout; no full compiler rebuild was needed
for this translator slice. Darwin execution remains untested.

After integrating main through PR #559, code `f0d4cba8` passes all seven
Wasm and eleven LLVM methods together in 6.779 seconds. I preserve the final
log at `/tmp/nanolang-wasm-final-gates.log`.

My shared typed F64 continuation adds arithmetic, comparisons, exact constants, float helper calls/storage and checked scalar numeric casts. Eleven Wasm methods and seventeen LLVM methods pass together in 42.767 seconds (`/tmp/nanolang-llvm-wasm-floats-final.log`). Positive float fixtures execute through `ScalarWasm.compare`, including Wasmtime and import-free Node; invalid float-to-int cases trap in Wasmtime. Unsupported string values still fail without replacing prior output. The executable entry remains integer/bool, CAST_BOOL remains refused, and heap/import/full-language coverage remains open.
