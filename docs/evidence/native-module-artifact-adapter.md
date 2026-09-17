# Native module artifact facade adapter

I accept `nlc_module_artifact(string) -> string` only as an artifact import
with an absolute library path. I snapshot its borrowed string through my
existing owned-string adapter before a subsequent facade call can replace it.
I do not infer arbitrary foreign ABIs or substitute a built-in resolver.

My exact-contract matrix covers the new adapter and five existing NanoISA
facades: six signatures, 47 variants. Wrong symbols, result types, arities,
parameter types and import kinds remain rejected with prior output preserved.
The assembler rejects empty and relative artifact paths before translation.
A strict C fixture changes one thread-local result buffer across successful
and missing-input calls; the earlier returned path remains intact.

I also executed generated native C against the real compiler-support facade
from commit `15849af749f493291428ee8a85f85636020a5178`, with its required
self-capture helper. It built a temporary foreign module, returned an immutable
absolute generation path, preserved that result across a missing-input call,
and exposed the expected result 37 from the produced library. No VM or compiler
execution replaces the module-builder call.

Logs: `/tmp/nanolang-module-artifact-native-tests.log` and
`/tmp/nanolang-native-module-artifact/real.log`. This adapter is the native ABI
companion to `task_81e682a57a3d431e845b0f41f140352e`; emitter/driver binding and
full bootstrap acceptance remain separately required.
