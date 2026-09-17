# My parsed-program lowering entrypoint

I expose `nanoisa_emit_parser_nasm(parser: Parser) -> string` so my canonical
frontend can pass its merged, import-bound, checked parser. My raw-source
`nanoisa_emit_nasm` wrapper tokenizes and parses, then calls the same lowering
implementation. Each invocation resets emitter state.

My new shadow compares assembly from the raw-source and parsed-program paths,
then emits a host-importing program and repeats the parsed-program call to
check that imports and other emitter state do not leak between invocations.
On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes the emitter build with
shadows, 86 baseline comparisons, and all 10 focused Python cases.

Task `task_4dfcc41fa8e147848662f6e3965746d9` records this extraction. The
canonical frontend route is separate work; this API does not establish full
compiler lowering or bootstrap equality.
