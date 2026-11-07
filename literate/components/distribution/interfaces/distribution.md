# My distribution integration capability 0.1

I preserve installed native compiler, VM, assembler, translator, resolver, libraries and headers. I use a fresh PREFIX to test installation, plus clean native package consumption; VSCode npm metadata is not compiler package acceptance. I neither invoke scripts/release.sh nor tag or publish from this contract.

I expose the existing native interfaces in `editors/`, `.github/`; I do not introduce
a new wrapper executable or change the existing ABI. Consumers retain the original source/build boundary.
This contract has no newly generated public artifact. Typed native exports and producer ownership remain
required under LITAI-001.3 before source custody or native execution admission.
