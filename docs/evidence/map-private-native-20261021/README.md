# Private native map consumer qualification

I qualify the private module-taking STRING→INT map translator on base c71f22d10,
following design 4ede28aff. This does not admit maps through the public translator,
VM, loader, serializer or publisher. I retain the exact executable STRING producer
whitelist: PUSH_STR and immutable transport; dynamic producers, CAST, and embedded
NUL constants refuse this native target. Private VM counted-byte behavior is separate.

## Final gates

- `python3 -m unittest -v tests.test_record_map_native`: 2 methods, 74.209s, PASS.
  `environment.json` pins the exact invocation variables and four original modules
  (seed, retained C-hosted Nano emitter, retained Stage1 and Stage2). No frontend rebuild.
- Compiler allocation oracle: 195 measured positions, each denied transiently and
  persistently, independent success retry, prior buffer/source unchanged; 1,961 checks.
  Returned buffer is one 65,536-byte allocation; final tracked balance is zero.
- Private publication/source destruction and target controls: 534 checks. Both the
  original module and independent copied-image lifetime paths are exercised.
- Generated native: 48 actual executions, raw and forced collection, O0/O2,
  ASan/UBSan. Four originals plus alias/replacement, missing GET (VOID), assertion
  failure, unused declarations, initializer residue, 17 duplicate tail arguments,
  nonzero layout/map-field index with scalar/array siblings, and map growth.
- Generated allocation oracle: each original reaches 26 positions, alias replacement
  13, growth 47, at both O0/O2. All 656 transient/persistent denial processes fail
  with exact SIGABRT/native-invariant diagnostics; 12 independent retries succeed
  with zero tracked live allocations. Aborted processes are not in-process rollback
  evidence. Compiler constructor recovery above is a separate in-process test.
- Old public outputs from current macro-enabled translator versus c71f22d10 source
  are byte-identical for all four original unannotated modules.
- GCC16 and Clang strict syntax: ordinary/private translator and both C harnesses,
  eight cases PASS. Six legacy allocation/work budget outputs byte-identical.
- Old execution-plan and copied-image neighbors: 2 methods, 17.635s, PASS, with the
  actual Make-defined NANOISA_OBJECTS + NANOISA_UTF8 closure. No old test edits.

The permanent `test-record-map-native` target is included in `test-units` and runs
synthetic controls without optional originals. `RECORD_MAP_MODULES` accepts exactly
zero or four paths; aggregate qualification can pass the same four retained inputs.
The test uses inherited provider flags, separate generated-code sanitizer flags,
and existing process-group/build timeout helpers. Private translation remains opt-in.

## Retained failures and corrections

1. The first added alias fixture encoded AGG_PACK ordinal as u16 rather than actual
   u32 and failed structural verification. The fixture and new target operand reads
   now use exact ISA u32; no layout limit changed.
2. The valid unused-declaration/no-HM/no-AGG case translated but generated C lacked
   nagg accounting helpers. The reviewed correction enables aggregate helpers only
   for the private canonical map route, leaving old routes unchanged. Final four
   originals and the unused case were rebuilt/executed after this correction.
3. The temporary public byte-comparison driver initially named nonexistent/incomplete
   v2 headers. Corrected to existing nvm_v2_sections.h; production was unchanged.
4. Neighbor invocation first lacked RECORD_ARRAY_* variables, then supplied the VM
   object closure (unresolved g_argc/g_argv), then an incomplete pathname-filtered
   query closure (missing nominal catalogs). The final invocation obtains the exact
   Make variable expansion instead of guessing provider membership. Assertions and
   production were unchanged; all failed terminals are retained.

`terminal-summary.json` retains complete status-file hashes and local artifact paths;
`native-allocation-counts.json` retains every measured executable count. Full command
logs remain at those paths. `checks.json`, budget/neighbor environments, provider and
source pins retain reproducibility without redundant binary bundles. Hosted platform
qualification and future public lifecycle integration remain open.
