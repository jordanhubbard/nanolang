# My Documentation Index

I provide this page as a stable entrypoint for older links.

My canonical documentation map is located at **[docs/README.md](README.md)**.

## Quick Links

### Getting Started
- [Getting Started Guide](GETTING_STARTED.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Examples](../examples/README.md)

### Reference
- [Language Specification](SPECIFICATION.md)
- [Standard Library](STDLIB.md)
- [Shadow Tests](SHADOW_TESTS.md)
- [Async Primitives](ASYNC_PRIMITIVES.md)
- [Performance Monitoring and LLM Optimization](PERFORMANCE_MONITORING.md)

### Virtual Machine
- [NanoISA Architecture](NANOISA.md) - I document my complete VM backend here, including my ISA, bytecode format, co-process FFI, and daemon.
- [Portable NanoISA vs. Runtime Representations](NANOISA_PORTABLE_ISA.md) - I document the portable ISA contract separately from the verified and optimized runtime representations my VM builds from it.
- [How I Optimize NanoISA](NANOISA_OPTIMIZATION_POLICY.md) - I record the evidence and acceptance rules that govern changes to the optimized dispatch representation.
- [Forth 2012 Pins](FORTH_2012.md) - I pin the standard revision, test suites, Gforth differential, licensing, and the environmental contract. The session runtime compiles colon definitions to verified NanoISA. The precise label is [FORTH_STANDARD_SYSTEM.md](FORTH_STANDARD_SYSTEM.md). I do not claim a Standard System.

### Formal Verification
- [NanoCore Proofs](../formal/README.md) - I use Coq to mechanize my metatheory. These proofs cover my preservation, progress, determinism, and semantic equivalence.

### Modules / FFI
- [Module System](MODULE_SYSTEM.md)
- [NSI.md](NSI.md) - v0 contracts, generated stubs, POSIX fabric (`NSI_FABRIC.md`), and 4.5 policy/replay (`NSI_EFFECTS.md`). I do not claim a kernel.
- [NanoLang 4.5](RELEASE_4.5.md) - Current public cut covering 4.1–4.5. Last public GitHub Release was `v4.0.0`.
- [NanoLang 4.4](RELEASE_4.4.md) - 4.4 product on `main` before Phase 19; not a public tag.
- [LinkedIn post for 4.5](LINKEDIN_4.5.md) - Draft covering changes since `v4.0.0`.
- [Extern FFI](EXTERN_FFI.md)

### Contributing / Maintainers
- [Contributing](../CONTRIBUTING.md)
- [Planning / design notes](../planning/README.md)
