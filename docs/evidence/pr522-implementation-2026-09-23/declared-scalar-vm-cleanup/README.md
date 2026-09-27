# My typed VM provider string cleanup

I use the ordinary typed libffi call for provider-owned strings and invoke the
same-image release companion after copying the result. I retain null-result
refusal and cleanup after copy allocation failure. A new 16-argument provider
checks integer, floating, bool, byte and string positions, then overwrites and
frees its result in cleanup; the VM must retain the original text.

`make nano_vm` passes. All 11 methods of `tests.test_artifact_string_release`
pass on Darwin. The VM is an ordinary build; the existing generated native
products use ASan/UBSan and leak detection. This is not whole-VM sanitizer or
Linux qualification. I select `/opt/homebrew/opt/llvm/bin/clang` with a temporary
`cc` symlink in PATH because the test invokes `cc` explicitly.

I retain both initial failures. Apple Clang's runtime explicitly refuses
`detect_leaks=1`; adding LLVM's bin directory alone does not replace `cc`.
The VM allocation-failure harness also omitted file CLI/runtime dependencies
from its link command. I add the same objects/library as the production VM
link, tracked by task_b5990d1ac7fc440fb41ee735215ad96b. Assertions remain intact.

Kind-4 modules remain refused. This advances the VM lifetime foundation; native
ABI support, producer emission and the original module-linking tests remain
required under task_b8838417bbc54fb98a4c49eea1b0885a.
