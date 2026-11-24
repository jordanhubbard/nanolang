# I qualify the private collector with complete fresh REPL providers

I keep ordinary ownership disabled. At source `8949313b0a56c0c3bd52509fb363edb4eefba3e4`, I select the existing full collector only through the previously retained external environment factory and shim. I reuse the actual complete fresh normal, sanitizer and production provider closures from my [preceding receipt](../fresh-providers-linux-20261124/README.md), rebuild each factory with its original profile flags, and relink a new private image without changing any original provider.

All 41 original callable, state, session and admission methods pass in all three profiles, with no skips. They exercise 62 actual native REPL sessions per profile: 186 in total. Every native call retains its original 30-second deadline. The outer build/test observations are bounded at 180 seconds. The sanitizer profile keeps ASan, UBSan and leak detection enabled. I retain the original inputs, outputs, module files and saved-session cases.

The separate default seven-method sanitizer admission cohort retains three failures: 168, 172 and 172 leaked bytes, each in six allocations. I preserve those failures. A private pass does not satisfy ordinary collection, default heap, current full compiler-source, paired-platform or release acceptance.

My observer wraps the unchanged `run_captured` call to record its input and output, and retains temporary case directories after the test. It changes no assertions, command arguments or command deadlines. I retain its source and distinguish this observed qualification from the earlier ordinary Make tests. Source guards cover 3,388 files before/after; original public products, providers and tools remain unchanged.

My independent reader passes 4,674 records, 3,520 selected Git inputs and 30 distinct native images. It verifies all three private passes and the default failures. Run:

```bash
python3 docs/evidence/fresh-private-repl-linux-20261124/read_archive.py
```

The manifest deduplicates prior source and provider bytes through the preceding committed evidence chain. #1116/#1213 and the full 69-entry #976 release contract remain open. I perform no unchanged full-source retry, ordinary activation, NAS, TCC or migration action.
