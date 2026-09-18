# My exact optional float call storage

I record task_0a49d074fc174478a57d76a698a7f3eb before implementation. Fresh canonical functional-array lowering executes the unchanged filter example in the VM, while native translation refuses mixed plain FLOAT and OPTIONAL FLOAT arguments to positive_float. I retain the source module and dump in /tmp/nanolang-functional-filter-evidence; no refused native binary was produced.

I extend the existing directed scalar storage contract to float. Concrete float producers may enter optional float parameter/local storage, and present optional float consumers retain exact tag checks. I keep optional absence distinct from a float, preserve exact array payloads and do not infer int-to-float promotion. I check both call orders, local copies, ordinary missing reads without forcing a float consumer, and existing wrong-tag refusals. Existing primitive and numeric-union constraints remain separate.

I require paired VM/native GCC and Clang sanitizer controls, shape tests, and unchanged canonical filter acceptance before closing this dependency. This does not complete generic indirect callbacks or heap array elements.

## My measured gates

At production checkpoint `d75aefe6` and test checkpoint `cc5d1406`, my full native gate passes 2,422 checks and the shape solver passes 1,269 checks. Thirteen focused ordinary methods pass with GCC in 7.574 seconds and Clang in 9.513 seconds, using ASan/UBSan/LSan. They cover both call orders, parameter/local copies, float transport and growth, adjacent record/array storage, absence remaining void, and exact float-to-bool/string shape refusal while preserving previous output.

With source callback checkpoint `13c7e74b`, eight functional-array methods pass GCC O2 sanitizers in 39.248 seconds and Clang in 29.083 seconds. The unchanged filter example executes in both VM and native code, including selected shadows. This integrated evidence uses the separate source branch explicitly; I do not claim its source lowering is part of this native patch.

I retain `/tmp/nanolang-optional-float-focused.log`: its first refusal checks expected the wrong diagnostic word, although both exact shape refusals occurred. An extra source-producer test lacked nano_virt in this isolated native checkout; I retained that setup error and ran the source acceptance through the explicitly identified functional tree instead. Final direct checks use the precise diagnostic and pass. Full and focused logs are `/tmp/nanolang-optional-float-full.log`, `/tmp/nanolang-optional-float-focused-final.log` and `/tmp/nanolang-optional-float-clang.log`.
