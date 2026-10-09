# Isolated self-hosted introspection function values

I prepare ordinary NanoISA wrapper functions for introspection declarations so function references, returned functions and higher-order calls have real bodies. Direct calls retain the same value emitter. My candidate is copied outside the checkout; the broad aa0b13713 gate keeps unchanged sources.

The candidate full compiler builds with shadows. My all-eight-operation fixture passes verified VM execution, C11 translation, strict native compilation and ASan/UBSan/leak execution. It checks returned function values, higher-order invocation, negative/out-of-range indices, and single evaluation. The six-method direct introspection suite also passes. I retain the exact source patch, fixture, build/test drivers and logs.

This patch is not integrated. C-producer callable support, resource-owning routes, permanent test wiring, fresh bootstrap and cross-host qualification remain open under #982/#976.
