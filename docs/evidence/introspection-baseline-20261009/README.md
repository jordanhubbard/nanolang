# Introspection producer baseline

I extend the existing four-method NanoISA contract with NANOLANG_INTROSPECTION_COMPILER. Both installed self-hosted stages reject the required metadata operations; both accept unused malformed intrinsic declarations that nano_virt rejects. The ordinary lookalike function control passes. I retain per-stage logs rather than marking parity complete.

My initial nano_virt run exposed a fixture mismatch between Darwin /var paths and canonical /private/var metadata. After canonicalizing the fixture path, VM assertions pass, but the system compiler sanitizer rejects detect_leaks=1. I retain both failures and honor NANO_NATIVE_TEST_CC; the LLVM run retains full address, undefined-behavior and leak checks. The terminal logs define the observed results.

My self-hosted driver currently does not import module_introspection.nano, and merge_with_imports strips module declarations/public markers before lowering. Its NanoISA extern registration uses the host ABI allowlist. I must preserve original source module facts, validate every intrinsic declaration before publication, and lower those facts without host FFI. #982/#976 remain open.
