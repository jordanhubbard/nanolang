# My byte-array focused qualification

I qualify production220383f26 with fixture-only87995e4c1. Both ordinary compiler builds and shapes pass; I independently copy and hash-verify those products for corrected discovery. Both fresh sanitizer compiler/provider builds and shape controls pass. My five controls pass in all four host/configuration combinations; generated native outputs run at O0 and O2 with matching instrumentation. ASan/UBSan and leak detection stay enabled.

I then compile the unchanged original mutation source through the actual C-bytecode producer, verify/run NanoVM, and reassemble its exact output for nvm2c O0/O2. All four configurations pass. Source SHA256 is d28507135b8fe8638d16599aa24bf2c67014cc6caf435a04c2152174b77f2b78, byte-equal to the retained facc first failing input. The actual constructor is now ARR_LITERAL 2 1, with CAST_U8 retained.

My full four-producer LLVM/Wasm source method remains open. These scoped results do not establish existing scalar-array destination refusal or full release readiness.

I retain complete products, source/tool/provider maps and raw child output under /home/jkh/nanolang-qualification/byte-array-879-linux and /Users/jkh/nanolang-qualification/byte-array-879-puck. Darwin reports and generated evidence are copied locally under byte-array-879-puck-report-copy. The companion retained-map hashes identify the full maps. Earlier unavailable-gmake, invalid nanoisa target and imported-TestCase discovery failures remain under byte-array-220-* qualification roots. No assertion, warning policy, leak detection or deadline was weakened.
