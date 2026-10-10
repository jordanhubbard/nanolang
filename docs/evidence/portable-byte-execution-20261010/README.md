# My portable byte-read execution evidence

I extend exact portable file-read admission with copied `array<u8>` results,
separate byte authority, packed origins and bounded native/Wasm adapters. I also
repair two source-product gaps: C AOT rejected the byte builtin, and my
self-hosted emitter did not lower the builtin already accepted by its checker.

I pass 2,443 C AOT checks (`nvm2c.log`), 27 ordinary/sanitized array-origin
checks (`array-origins.log`), the existing native read adapters and 698 Node
adapter checks. `byte-c-parity.log` covers binary alias mutation, NUL/255,
independent reads, empty/missing files and the exact portable size bound through
C AOT, LLVM and Node. Native allocation-prefix recovery passes in
`prior-full-run.log`, together with text and packed/mutable runtime controls.

My prior full run has one failure: the then-installed self-hosted compiler
reports an undefined byte-read builtin. I preserve that failure. After my
emitter fix, `seed-source.log` and `fresh-native-source.log` pass source byte
contents, mutation through aliases and independent reads from the fresh seed,
C frontend and native translation of the fresh compiler. Each source output
executes through NanoVM, C AOT, LLVM and Node/Wasm. `mixed-read.log` exercises
text and byte imports together with separate grants and shared scratch.

I built the fresh native compiler by translating `bin/nanoc_seed.nvm` with
`bin/nvm2c`, then compiling at `-O1` with LLVM Clang and
`bin/nano_aot_runtime.o`, `-lm -lcrypto -lffi`. Omitting the AOT runtime does not
produce a usable compiler artifact loader.

I installed and relocated the runtime package to
`/private/tmp/nl51-byte-relocated`. Its installed native archive and Node adapter
execute a three-byte read, including embedded NUL and 255. `install.log`
retains installation output. I verified Python syntax and the pure envelope
parser on generated modules; I have not executed these modules in Wasmtime43.

My final focused run passes all eight byte methods independent of the installed
compiler (`qualified-byte-tests.log`). I explicitly exclude
`test_source_products` from that run while its installed compiler is rebuilding;
I retain that test unchanged as an outstanding gate.

My generated Wasm allocation gate now passes both text and byte results
(`wasm-allocation-prefixes.log`). I instrument the actual emitted freestanding
allocator boundary, exercise each of the two observed allocation prefixes per
fixture, require MEMORY without a trap, then recover and execute successfully.
Managed live objects and bytes return to zero before and after disposal.
These are generated Node/Wasm executions. They do not establish Wasmtime or
all allocations in more complex programs. Both existing portable execution
targets include the new gate.

My full bootstrap run `obj/bootstrap-nanoisa/run-mi9we72o` remains in progress
at Stage 2. `stage1-progress.json` records successful seed and Stage 1 steps,
including native translation and hello execution, with unchanged source inputs.
`stage1-source.log` additionally passes the stronger byte-source fixture from
the newly bootstrapped Stage 1 native compiler through VM/C/LLVM/Node.
I do not treat the older installed compiler receipt as evidence for this change.
The installed self-hosted source gate, generated Wasmtime execution and exact
release-platform gates remain open. Byte reads also do not close aggregate,
linked-module or remaining compiler host capabilities in my 5.1 scope.
