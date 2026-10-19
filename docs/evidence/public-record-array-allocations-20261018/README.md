# My public VM allocation-prefix qualification

I extend the public consumer corpus after `072b2775d`. My final Darwin gate passes in 26.842 seconds. I rebuild the actual public proof, verifier, VM, decoder, heap and callback providers listed by the test with counted allocation hooks and ASan/UBSan. I retain the ordinary ABI, without the private-runtime definition.

For each of the five flat array tags I inject at all 472 allocations observed in successful public root execution, in transient and persistent modes: 4,720 injected prefixes. Injection begins after successful VM initialization. Each prefix must drain activation state; failed execution has no stack result; successful execution still returns the expected integer. After disabling injection and destroying the VM, every tracked allocation and byte must be gone. A newly initialized VM then executes the same module successfully and again releases all tracked ownership.

I require runtime allocation failure to report MEMORY, except the existing boolean ownership revalidation API's exact reference-lifetime refusal, which maps proof allocation failure to TYPE_ERROR. My first stricter diagnostic assumption discovered that existing mapping at prefix93; I preserved the log and constrained the exception by both status and exact message. I did not change production behavior or allow arbitrary guest errors.

The complete gate also reruns the unwrapped and instrumented public consumer corpus and 20 O0/O2 sanitized native executions. This receipt does not inject into VM setup, uninstrumented common providers, v2/native translation, or generated-native allocation sites. Those remain separate qualification obligations, as do source publication, native field replacement and final combined Linux/Darwin checks. I do not close #1001.

An independent read-only review found no blocking issue in injection bounds, cleanup/recovery assertions, provider replacement or the constrained refusal mapping.
