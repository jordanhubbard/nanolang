# My canonical source-emitter setup deadline

I preserve the original timeout in `/tmp/nanolang-product-e46c425b-focused.log`: the scalar-match class setup exceeds its 180-second deadline while building `src_nano/nanoisa_emit.nano` with the product Stage2 compiler.

A separate qualification of that same command at compiler production `e46c425b6634315a76ef69d1bde45be705b8f400` completes successfully in 440.639 seconds within a declared 900-second measurement budget. The [manifest](canonical-emitter-setup-measurement.json) retains argv, compiler hash, elapsed time and successful exit. The compiler hash remains unchanged. The generated emitter then emits a normal hello module; assembly, verification and VM execution pass.

I set only this compiler-building setup command to 900 seconds. The shared command helper still defaults to 180 seconds, and existing explicitly shorter controls remain unchanged. I preserve all raw-producer, selected-shadow, VM, native and output-preservation assertions. This is a setup deadline correction, not a performance improvement or an infrastructure attribution.

My complete affected scalar-match suite is currently running against the canonical product route with the corrected setup policy. I keep task `task_3d46381f10bd4dfcb313c2697024191f` open until that acceptance and canonical integration complete.
