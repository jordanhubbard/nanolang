# My ordinary producer validation count

I measure the actual verifier and private v2 producer, using unchanged D18 verifier/producer source and the actual current local nanoisa DSO. I add one counter only at verify_module_ranges entry in a temporary verifier copy. I compile the original producer source with it and bind the actual library, binary, commands and streams. The original source tree and library remain unchanged.

For1,8,64,256 and1,024 functions, the actual producer performs the same number of whole-module structural visits. Each visit checks the module's function range table. The1,024-function case therefore repeats that table1,024 times. Every independently queried and produced maximum matches the fixture's known1-through7 push/pop stack depths. This is a local AArch64 call-count measurement, not hosted x86 timing, a product optimization or qualification.

I retain the instrumented/original sources, fixture, actual probe/DSO bytes and identity metadata. The instrumentation adds only the counter declaration/increment; it changes no branch or return. Run `python3 verify.py --workspace /path/on/large/volume` to validate retained bytes and counts without executing a product. An implementation still must preserve refused-depth zeros, private routes, between-call mutations, wire bytes, allocation/publication assertions and every original platform/release gate.
