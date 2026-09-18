# My exact optional float call storage

I record task_0a49d074fc174478a57d76a698a7f3eb before implementation. Fresh canonical functional-array lowering executes the unchanged filter example in the VM, while native translation refuses mixed plain FLOAT and OPTIONAL FLOAT arguments to positive_float. I retain the source module and dump in /tmp/nanolang-functional-filter-evidence; no refused native binary was produced.

I extend the existing directed scalar storage contract to float. Concrete float producers may enter optional float parameter/local storage, and present optional float consumers retain exact tag checks. I keep optional absence distinct from a float, preserve exact array payloads and do not infer int-to-float promotion. I check both call orders, local copies, ordinary missing reads without forcing a float consumer, and existing wrong-tag refusals. Existing primitive and numeric-union constraints remain separate.

I require paired VM/native GCC and Clang sanitizer controls, shape tests, and unchanged canonical filter acceptance before closing this dependency. This does not complete generic indirect callbacks or heap array elements.
