# VM compiler fixed point at 0eb94026c

I pass the unchanged permanent VM fixed-point gate in a fresh Linux ARM64 clone
at `0eb94026ce9f0be3b374e45931c50b874a63f8da`, using ordinary GCC-built providers.
The isolated VM has 16 GiB RAM and two CPUs. I preserve the 1,800-second stage
bound, default shadow deadline, exact host closure and native-code-generation
guard. This checkpoint predates the subsequent native array-inference repair.

Both canonical compiler generations contain 526,188 bytes and have SHA-256
`19e49b0210f46e1bd7490ff0efa0e49f24bf42d924c3bbafb26af42850ef474b`.
Stage 1 takes 339.665 seconds; Stage 2 takes 347.842 seconds. Both verify.
The final compiler emits a test product that verifies and executes. The gate
records zero NanoLang-generated C compilation calls during VM generations;
declared native host artifact checks retain their existing allowed path.
Source, capture helper and all three imported host-library hashes remain stable.

I retain raw modules, disassembly, stage logs and the gate manifest. Earlier
Darwin invocations stopped before compilation because the capture helper is
Linux-only; their logs remain here. I moved qualification to Linux rather than
changing the helper or bypassing the native-code-generation guard.

This establishes the VM route at the recorded pin. Native fixed points, the
subsequent translator correction and full hosted acceptance remain separate.
