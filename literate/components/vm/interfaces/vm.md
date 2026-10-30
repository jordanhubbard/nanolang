# My vm integration capability 0.1

I preserve verified execution, service-call provenance, stack behavior, string ownership, garbage collection and safety limits. Invalid or unsupported bytecode must refuse before execution.

I expose the existing native interfaces in `src/nanovm/`; I do not introduce
a new wrapper executable or change the existing ABI. Consumers retain the original source/build boundary.
This contract has no newly generated public artifact. Typed native exports and producer ownership remain
required under LITAI-001.3 before source custody or native execution admission.
