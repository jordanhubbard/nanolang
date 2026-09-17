# My concrete callback identity

I preserve the concrete union spelling already stored in `NSType.element_type_name` when rendering a union type or an array containing it. My named-function values therefore retain signatures such as `fn(Box<array<int>>)->Box<array<int>>` instead of flattening both positions to `Box`.

I resolve callback result spellings against the current parser declarations. Without that step, a returned `Box<int>` was classified as a record while its annotated destination was classified as a union, producing an error with identical expected and actual spellings.

My shadows check nested array spelling, non-generic union fallback and actual named-function signature construction. My driver regression checks both selfhost stages: an ordinary `fn()->Box<int>` callback passes typechecking and source-only C publication; substituting `fn()->Box<bool>` rejects the passed function and preserves an existing output file. Source-only publication does not establish executable native callback lowering.

I keep MAC parent `task_e05a42e2e09b47cc9c53fa6923eeeaef` open for complete signature comparison, serialization, native/VM execution and ownership boundaries. This repair is child `task_8ef982b488d8421fb7e5b5ea280e7d2c`.

I passed `make bootstrap` through both native stages with default shadow checks, then all 15 methods in `test_selfhost_generic_contexts`, `test_native_nested_generics` and `test_generic_selected_patterns` (63.391 seconds). The existing Stage 3 native comparison is a smoke gate, not a canonical bytecode fixed point. Logs remain in `/tmp/nanolang-callback-identity-bootstrap-final.log` and `/tmp/nanolang-callback-identity-drivers-final.log`. Independent source review found no blocker within this scope.
