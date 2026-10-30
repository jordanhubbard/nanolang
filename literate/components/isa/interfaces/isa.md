# My isa integration capability 0.1

I retain versioned encoding, verifier authority, assembler/disassembler round trips and rejection of malformed instructions, proof facts and unsupported profiles. I preserve old profile bytes and bounds.

I expose the existing native interfaces in `src/nanoisa/`; I do not introduce
a new wrapper executable or change the existing ABI. Consumers retain the original source/build boundary.
This contract has no newly generated public artifact. Typed native exports and producer ownership remain
required under LITAI-001.3 before source custody or native execution admission.
