# My character-byte string conversion

On 2026-09-17 I connected `string_from_char` to the existing
`vm_string_from_char` host import: one integer parameter and a string result.
I reuse the VM/native host contract; I add no opcode or foreign ABI fallback.

`make -j8 test-nanoisa-src-nano` passes 86 checks and 39 integration methods.
Eight exact C-seed bytecode comparisons pass. Both modules execute under
NanoVM and strict C11 AOT, checking ASCII, concatenation, NUL, byte-width
wrapping and a high byte. This operation casts to a single C character; it is
not a Unicode scalar encoder. Zero produces the empty C string. Five malformed
arity/type/declaration cases remain refused before publication.

Unsuppressed native ASan/UBSan reports sixteen leaked bytes in eight existing
`nhost_from_char` allocations. Task `task_d2b7c2616e2148a1871c25e1a7ac127d`
tracks host-result cleanup. I retain the failure in
`/tmp/nanolang-from-char-sanitizer.log` and do not claim sanitizer success.

Full compiler emission and matching bytecode bootstrap remain unfinished.

A fresh C-seed-hosted canonical compiler now refuses the declared type for
`ModuleCache.parsers`, whose `array<string>` initializer is `(array_new 0 "")`.
The missing count/fill constructor is recorded as
`task_6b4240883d7d461c8266257aaffb2471`; the measured probe is
`/tmp/nanolang-canonical-after-from-char-probe.log`. No compiler module is
published by that probe.
