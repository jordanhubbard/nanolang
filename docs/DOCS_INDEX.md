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
- [Forth 2012 Pins](FORTH_2012.md) - I pin the standard revision, test suites, Gforth differential, licensing, and the environmental contract. I do not claim conformance yet.

### Formal Verification
- [NanoCore Proofs](../formal/README.md) - I use Coq to mechanize my metatheory. These proofs cover my preservation, progress, determinism, and semantic equivalence.

### Modules / FFI
- [Module System](MODULE_SYSTEM.md)
- [Extern FFI](EXTERN_FFI.md)

### Contributing / Maintainers
- [Contributing](../CONTRIBUTING.md)
- [Planning / design notes](../planning/README.md)
