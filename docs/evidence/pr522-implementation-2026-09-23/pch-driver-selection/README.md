# PCH negative-control compiler selection

I reproduce the hosted marker-control failure on Ubuntu 24.04 ARM64 with `NANO_CC=clang`: three self-capture methods pass, but the GCC-specific literal-marker refusal receives success. The test detects GCC through `cc --version` while the module builder obeys the inherited `NANO_CC`.

I let the fixture select a compiler explicitly and pass the same driver inspected by the GCC-specific test. Its original return-code, diagnostic and no-publication assertions remain intact. The scanner's self-compilation test still inherits the selected compiler. Production compiler precedence is unchanged.

All four methods pass with parent `NANO_CC=clang` (0.664 seconds), and all four pass with parent `NANO_CC=gcc` (1.159 seconds). Commands are `docker --context colima-nanolang-pr522 exec -w /qualification -e NANO_CC=clang nanolang-pr522-debug python3 -m unittest -v tests.test_module_builder_self_capture` and its GCC equivalent. Linux probe/provider hashes and local source hashes identify the tested inputs. The local macOS owning target also passes, with its three existing GCC-only skips; Linux executes all four without skips.

Final hosted sanitizer acceptance remains open under `task_c07358e3ce754fc6bcfe4e6f99a9fe76`. These runs use ordinary probes, not a complete instrumented host runtime.
