# Self-hosted introspection declaration validation

I validate all reserved introspection extern declarations before reachability and ownership routing in program and shadow emission. I preserve ordinary functions with similar names. My full compiler bytecode build passes its shadows. The fresh compiler component passes nine malformed-declaration controls covering all eight operations, prior-output preservation, and the ordinary-function control. My nano_virt contract passes all four methods with the extended refusal cases and LLVM sanitizers.

I retain the broader component result: the two positive metadata-operation methods still fail because metadata calls are not lowered. Installed stages, a fresh bootstrap, hosted qualification, and #982/#976 remain open. This change does not implement source-module facts or metadata call execution.
