# My execution and standalone-host continuation

I retain an exact unused read declaration in Wasm with a `used, retain` adapter
symbol. Linker `--undefined` alone did not preserve it; my
[first attempt](unused-import-first.log) retains that failure. The corrected
six-method [execution gate](producer-final.log) passes with exactly one declared
Wasm import and no attempted host read for the unused-call program.

I test all successful allocation prefixes of a generated native LLVM read call.
I redirect allocation in the emitted managed runtime and the linked scratch
adapter, leaving the harness allocator uncontrolled. Every failure returns a
managed error, clears live managed objects/bytes, permits subsequent successful
entry, and leaves zero tracked allocations after disposal. The loop includes
failures before and after the host callback. I instrument emitted LLVM functions
with ASan explicitly and build the adapter/harness with ASan/UBSan; this is native
execution coverage, not instrumented Wasm or a proof of all allocation paths.

I compile the source fixture, including its main shadow, through NanoVirt's C
frontend, installed self-hosted `nanoc`, a fresh self-hosted compiler module in
NanoVM, and that module's native C translation. Each output runs through NanoVM,
C AOT, native LLVM with an explicit allowlist, and Node/Wasm. The fresh compiler
module is built from current `src_nano/nanoc_v06.nano`; its native translation
links `bin/nano_aot_runtime.o`. My [initial producer attempt](producer-first.log)
used `nanoc_c --emit-nvm` outside that driver's documented File-only scope and
omitted the native AOT runtime object. These are corrected harness selections,
not relaxed source assertions. I do not claim a new bootstrap fixed point.

During native producer checks, isolated loading exposed an actual missing
`nl_websocket_catalog_interface` provider in the NanoISA facade. I add
`nsi_websocket_plan.c` to both affected NanoISA/Forth manifests and derive the
examples Forth library sources from its manifest. Three [closure tests](host-closure.log)
pass, including independently compiled libraries loaded in fresh Python
processes without prior provider loads and a native Forth program.

I could not obtain the pinned Wasmtime43 wheel: files.pythonhosted.org failed DNS
resolution, and no local wheel was found. Actual generated Wasmtime execution,
Wasm allocation-failure coverage, the full read boundary corpus on all producers,
and exact candidate platform qualification remain open. Byte/aggregate results
and remaining compiler capabilities remain requirements under #976.
