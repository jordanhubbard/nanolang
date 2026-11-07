# My frontend integration capability 0.1

I retain parsing, ownership analysis, lowering and diagnostics in both frontends. Equivalent source must retain verified NanoISA behavior; refusals must not become admission. My broad src prefix excludes explicitly owned ISA, VM and backend units when the native graph is derived.

I expose the existing native interfaces in `src/`, `src_nano/`; I do not introduce
a new wrapper executable or change the existing ABI. Consumers retain the original source/build boundary.
This contract has no newly generated public artifact. Typed native exports and producer ownership remain
required under LITAI-001.3 before source custody or native execution admission.
