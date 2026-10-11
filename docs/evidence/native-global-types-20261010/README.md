# My native global type checkpoint

I found a separate global declaration failure while qualifying native callback storage: a record global became `static struct global;`, a tuple became `static void pair;`, and a generic callback initializer remained an unresolved generic name.

I now emit runtime-initialized globals with the complete signature-type emitter, retaining nominal names, recursive tuple annotations and callable signatures. I retain global function annotations before initializer checking so contextual generic specialization runs at that boundary. I resolve declared enum globals and compare complete enum identities in both root and module checker paths.

## My evidence

- My six new `tests.test_native_global_types` methods execute record mutation, generic record initialization, tuple storage, generic function-value storage and enum storage. They refuse distinct enum and incompatible callback declarations while retaining a prior executable. Successful compilations run mandatory shadows and then the generated executable.
- I pass 47 native methods: these six plus `tests.test_cseed_generic_functions`, `tests.test_contextual_generic_functions` and `tests.test_higher_order_generic_functions`, with `NANO_CONTEXTUAL_NATIVE_COMPILER=bin/nanoc_c`.
- I pass the same 47 methods with nine compiler components instrumented using ASan/UBSan. As in my prior generic-body checkpoint, I use a 65,520 KiB stack for the instrumented interpreter's 5,000-call recursive shadow. This is a test environment requirement, not a language acceptance change.
- I pass 84 C-producer methods across the generic, tuple, contextual, higher-order and aggregate-storage suites, including NanoVM and sanitized C AOT execution.
- I pass 17 adjacent methods in `tests.test_cseed_global_identity`, `tests.test_global_initializer_context`, and `tests.test_module_enum_generics`, using `NANO_ENUM_COMPILER=bin/nano_virt` and LLVM clang for sanitized C execution.

My first adjacent invocation selected the older installed Stage 2 compiler by the enum suite's default and reproduced six known enum failures. The explicit current-producer run passes; I do not claim that the installed compiler was refreshed.

I leave callback-field declarations and callable array storage on the native path open. This checkpoint does not close the broader storage row, installed bootstrap qualification, issue #976, or my full 5.1 scope. GitHub API access remains unavailable; I retain the issue update locally.
