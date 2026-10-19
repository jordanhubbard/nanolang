# Integrated eight-shared reconstruction qualification

I integrate the reviewed component and focused assertion-test correction at `1e2951ef8`, preserving both roadmap histories. I rebuild only the facts executable and launcher with `make -f Makefile.gnu -j1 CC=/opt/homebrew/opt/llvm/bin/clang nvm2hl`.

All seven `tests.test_eight_shared_reconstruction` methods pass in 219.085 seconds through the current seed and both fresh 3446 stages. I set `NANOLANG_OWNED_RECONSTRUCTION_COMPILERS` to the two stage paths and `NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang`; the harness adds the seed. The original-shadow fallback is the independently emitted retained assembly, not a newly synthesized original input. All nine provider hashes captured before the run match at terminal.

I retain ordinary 92/103 and generated consistency-shadow 98/103 local counts, original and reconstructed VM/native execution, signed minimum/maximum and negative/zero multiplication, exact 128/129 local and 4096/4097 proof boundaries, typed nominal/ownership refusals and immutable IR. The later-call assertion test isolates the helper failure from caller assertions, checks its diagnostic, rejects sanitizer errors, uses `-DNDEBUG` in reconstructed C, and proves that removing only the helper assertion permits success.

The proof traversal counts are distinct from multiplication runtime work. No frontend, runtime, or verifier changes are made, and I claim no new raw generation for this facts/Python component. Exact hosted platform gates and finite remaining scope under #999 remain open.
