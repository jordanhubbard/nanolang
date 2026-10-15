# Isolated C-producer introspection function values

I register checked metadata declarations as ordinary NanoISA functions and emit their bodies after registration. The isolated build reuses unchanged checkout objects except for separate candidate codegen/module objects, and publishes its compiler outside the checkout.

My first all-eight-operation callable run fails the physical-path assertion when callable_probe is declared in probe.nano. The checker tracks exported facts under the declared name but only attaches the source path to the filename-derived entry. I retain the failed run and diagnostic probe. The prepared module-loader correction registers the same source path under the declared identity before checking.

With both changes, the original callable fixture passes verified VM execution, strict C11 native compilation and ASan/UBSan/leak execution. All six existing introspection methods pass (1.192 seconds). Function-value, returned-function, higher-order, boundary-index and single-evaluation assertions are unchanged. I retain exact build commands, scripts, patches and logs.

These patches remain unintegrated while the aa0b13713 broad gate runs. Full integration, permanent tests, fresh bootstrap and ownership-route qualification remain open under #982/#976.
