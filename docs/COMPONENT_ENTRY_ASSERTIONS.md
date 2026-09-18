# My component entry assertions

I compile parser, checker and transpiler library drivers during stage2 and execute them during stage3. Returning zero or printing a library banner does not establish component testing.

I require each driver entry point to execute deterministic assertions: parser token consumption and actual function/return structure; checker acceptance plus a rejected return type with recorded diagnostics; transpiler generation of the selected function body and explicit shadow selection. Each driver retains its mandatory shadow, but my claim concerns executed main assertions, independent of imported-shadow policy.

I change no parser/checker/transpiler semantics and do not enable every imported shadow. I retain the broader component audit. My runner reports component entry assertions rather than implying complete component or compiler correctness. Before landing, I compile and execute all three changed drivers with a recorded fresh compiler identity and retain command/log evidence. A failure remains a failure; I do not disable assertions to obtain a successful gate.
