# Isolated C introspection in owned programs

I extend the earlier C callable/declared-path candidate to recognize exact metadata extern signatures in the owned source profile and lower their direct calls to source facts. Indexed names use stack branches without temporary locals or array allocation, preserving owner slots and single evaluation. I use the existing generic equality instruction admitted by the ownership verifier. I do not change the verifier.

The complete prepared.patch is a superset of the earlier C callable patch: apply this one instead of both. It remains outside production while the aa0b13713 broad gate runs.

The original owned count probe and a resource-owning all-eight-operation fixture pass verified VM execution, native translation, strict C11 and ASan/UBSan/leak execution. The all-operation fixture keeps a live resource across metadata calls and consumes it afterward, checking index bounds and single evaluation. Leak, use-after-move and unused malformed-signature controls fail while preserving prior output. The seven-method direct/callable suite also passes.

I retain the initial fixture-generation error, missing compiler environment and unsupported typed-equality emission failures as well as corrected logs. Self-hosted owned lowering, owned callable values, permanent tests, integration and fresh platform qualification remain open under #982/#976.
