# My native slice checkpoint

I copy outer slice storage, preserve exact scalar payloads and retained record
children, and copy closure environments alongside callable targets. My callable
analysis propagates element targets through slices. Slice bounds use the VM's
uint32 conversion and end-index behavior; source producers clamp start/length
before emitting the opcode. This is not yet complete array-kind parity.

My first generated C fails strict compilation because its dynamic slice helper
is unused. I retain that terminal and add the same explicit helper reference
used by other runtime helpers. The expanded six-method corpus then passes
argument order, exact binary64 bits and record children, copied closures,
ordinary int/bool/string/float/record cases and four invalid-operand refusals.
Four subcases remain failed: byte and nested arrays through each producer.
I split the byte assertion into its own test so it cannot hide the independent
bit-preservation checks; no original case was removed.

`make -j2 test-nvm2c CC=/opt/homebrew/opt/llvm/bin/clang` exits zero:
2,431 execution checks, 3,092 shape checks and 379 callable checks pass,
alongside the Python opcode/sanitizer-driver checks. Runtime string literals
were reformatted after this gate without changing the emitted text.

My source-shape corpus remains failing and #979 remains open. Fresh bootstrap,
byte-array metadata/storage, nested-array identity/storage and null/refusal
coverage still require work before legacy test retirement or release acceptance.
