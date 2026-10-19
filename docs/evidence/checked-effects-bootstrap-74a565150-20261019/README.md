# I retain my checked-effects raw bootstrap

At `74a565150` I pass all 17 guarded bootstrap steps, including both raw generations, verification, native translation/build and hello compilation/execution. The raw Stage 1 and Stage 2 modules are byte-identical with SHA-256 `b790c67874db5d44b4f8f21b4f9ead65797443d08ebf081961a0d1fb32701c31`. I install the verified Stage 2 product.

This qualifies the new checked-global compiler record shapes through actual raw generation and native translation on Darwin. It does not establish clean-environment reproducibility, complete compiler correctness or Linux qualification. I retain the exact source/tool manifest and all step terminals. Log copies normalize trailing display whitespace; `retained-original-hashes.json` pins both original retained files and copies.

Fresh-stage metadata and generic regression qualification follows this bootstrap; I do not close #998 or declare 5.1 ready from the bootstrap alone.

## Fresh metadata qualification

I pass all 50 methods in 195.891 seconds across effects, declared purity metadata, generic binding metadata, match coverage, affine completion, affine use and the declared-purity contract. The first six suites run through NanoVirt and both freshly generated stages; the contract suite compares the C native compiler and fresh Stage 2. The raw-checker helper fixtures use the actual C-hosted Nano checker, with no mocked facts. All pinned tool/runtime/generation hashes remain unchanged after the suite. `fresh-metadata.log` retains the terminal and method names.

I set `NANOLANG_EFFECTS_COMPILERS`, `NANOLANG_PURITY_COMPILERS`, `NANOLANG_GENERIC_METADATA_COMPILERS`, `NANOLANG_MATCH_COMPILERS`, `NANOLANG_AFFINE_COMPILERS`, and `NANOLANG_AFFINE_USE_COMPILERS` to absolute NanoVirt/Stage1/Stage2 paths separated by colons; `NANOLANG_SELFHOST_COMPILER` selects Stage2. I run `tests.test_function_effects_metadata`, `tests.test_function_purity_metadata`, `tests.test_generic_binding_metadata`, `tests.test_match_coverage_metadata`, `tests.test_function_affine_metadata`, `tests.test_affine_use_metadata`, and `tests.test_purity_contract`.

I add the missing purity/generic selectors to the retirement workflow after confirming both suites pass the fresh products. Their previous defaults exercised only NanoVirt. YAML parsing and diff checks pass. These are Darwin results; hosted main CI and release qualification remain queued. Imported/branch affine qualification and full #998 remain open.
