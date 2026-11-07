# My ordinary/resource authority reconciliation

I retain the failure tracked under #1054: `owned_order` marked both layouts
COMPLETE-only but expected `OWN_PACK` to execute for prior-order ordinary
records. Current main refuses both ordinary orders, following its explicit
ordinary/resource distinction. I preserve that refusal and give the positive
resource control the RESOURCE flags its opcode requires.

My final `make test-ordinary-record-authority` passes in 6.048 seconds. Both
normal and ASan/UBSan fixture builds retain every original allocation-failure
and decoder assertion. They check ordinary aggregate forward execution in VM,
C, LLVM and Wasm; explicit prior-order resource execution returns 7 in VM and
C. Resource-forward, ordinary-forward OWN_PACK and ordinary-prior OWN_PACK
modules refuse VM verification, VM execution and native translation, preserving
previous native output. Signals and checksum errors cannot satisfy refusal.
The changed C fixture also passes GCC 16 at `-O3 -Wall -Wextra -Werror`.

My first corrective run reaches a producer serialization refusal for resource
forward edges. I retain that result: the producer must refuse that declaration.
To test public consumers independently, I change only the resource flags in a
valid emitted ordinary-forward module and recompute its checksum. This is an
explicit malformed-input control, not a producer admission or a relaxed codec.
I do not change the verifier, runtime, codec, opcode semantics or guest assertions.

I use the installed Homebrew LLVM directory on PATH, OpenSSL's library directory
in LIBRARY_PATH, and Wasmtime 50.0.1. `gate.log` retains the final command/result;
`receipt.json` records fixture and tool hashes. This closes a local fixture
mismatch only. Integration, hosted sanitizer qualification and the complete
retained suite stay open under #1054.
