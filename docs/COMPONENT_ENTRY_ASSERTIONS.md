# My component entry assertions

I compile parser, checker and transpiler library drivers during stage2 and execute them during stage3. Returning zero or printing a library banner does not establish component testing.

I require each driver entry point to execute deterministic assertions: parser token consumption and actual function/return structure; checker acceptance plus a rejected return type with recorded diagnostics; transpiler generation of the selected function body and explicit shadow selection. Each driver retains its mandatory shadow, but my claim concerns executed main assertions, independent of imported-shadow policy.

I change no parser/checker/transpiler semantics and do not enable every imported shadow. I retain the broader component audit. My runner reports component entry assertions rather than implying complete component or compiler correctness. Before landing, I compile and execute all three changed drivers with a recorded fresh compiler identity and retain command/log evidence. A failure remains a failure; I do not disable assertions to obtain a successful gate.

My first fresh stage3 compile stops before entry execution because the new drivers call `tokenize_string` without a direct lexer import. The parser compiler reports an implicit declaration and undefined reference. I preserve `/tmp/nanolang-component-entry-missing-lexer-import.log`, SHA-256 `a8d4af2d092c0f3609286d444923e58d1e8af12eab1269e078916d2781999f63`. I correct the driver dependency explicitly; this is not successful assertion evidence.

After the lexer correction, the parser driver compiles. The checker driver then stops before execution on missing direct parser bindings (`nl_parse_program` and parser query calls). I preserve `/tmp/nanolang-component-entry-missing-parser-import.log`, SHA-256 `36c807448e16ff435affb9e642d7608dc36b7269eff4240a5147fed39a6a7e56`. I audit all three entry-point dependencies and add direct parser imports to checker and transpiler drivers; transitive imports do not supply these root bindings.
