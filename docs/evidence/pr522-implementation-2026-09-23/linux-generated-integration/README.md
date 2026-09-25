# My Linux generated-consumer integration checkpoint

I tested exact commit e18e8bd5a in /qualification-generated-e18e8bd5a within
Ubuntu 24.04 ARM64 (container nanolang-pr522-debug, Colima context
colima-nanolang-pr522). Both original generated-C methods pass with GCC 13.3.
Their artifacts remain at /tmp/nano-record-array-generated-a5900arh.

My complete LLVM/Wasm command uses LLVM 18.1.3, Node 18.19.1 and Wasmtime 49.0.0.
The Wasmtime release archive SHA-256 is
211255e48fcf3107e6ea357b6214938c4f45e29ea1b295398e7ab6b269d51294,
checked against its GitHub release asset digest before installation.
I retain exact selection and the full owning command log. Factored C bytes,
emission and native LLVM pass. Wasm startup linking fails on two compiler-generated
memset references. I do not claim complete Linux LLVM/Wasm qualification.

The failure and correction obligation are task_efde11810990406ba36311702a0746d6.
The full intermediate corpus remains at /tmp/nano-record-array-llvm-ec3dfes7 in
the container. This checkpoint preserves the failed link and owning results;
it does not replace a complete product/report archive after correction.
