# My native integration capability 0.1

I preserve verified NanoISA-to-C translation, ABI, native runtime ownership and VM/C-AOT equivalence. My future native graph must identify actual backend translation units within src rather than treating all src as one producer.

I expose the existing native interfaces in `src/`; I do not introduce
a new wrapper executable or change the existing ABI. Consumers retain the original source/build boundary.
This contract has no newly generated public artifact. Typed native exports and producer ownership remain
required under LITAI-001.3 before source custody or native execution admission.
