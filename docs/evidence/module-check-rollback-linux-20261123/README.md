# My rejected-module lifetime repair

I pin production repair `2a0879a142e1dd1e4746e59568eb8d93240a77a4` and
final test extension `bea5f4997ff4e23315e4a37e9581fba5827a3275`. My reader
verifies 2836 records, 1433 exact Git inputs and 100 distinct native ELF images.

Under #1223 I withdraw rejected ordinary module declarations before retiring
their AST. I retain the original C/API ASan use-after-free through my base
[REPL receipt](../repl-restore-linux-20261123/README.md). My new recovery control
also fails against those retained providers because the rejected `broken`
declaration remains visible. I preserve that refusal, its executable and the
actual failed module bytes.

I capture the importer's alias boundaries before imports and checkpoint its
declaration tables after successful dependencies load. On checker failure I
restore preexisting functions, symbol metadata and module export counts; retire
new declarations and checker-owned payloads through existing owner cleanup;
and withdraw only this importer's newly inserted namespace, function, enum,
effect and global aliases. Successfully loaded dependencies retain their ASTs,
source identities and callable slots. The loader which inserted a failed cache
marker removes it when that load is terminal. An inner circular-import refusal
cannot remove its caller's live marker. I cache no rejected AST as a success.

My native control checks 32 failures on one source path without growth in checked
declaration counts, every ordinary declaration kind, restored prior shadows, live
STRING/static-array survival, and a corrected same-path load. It preserves an
older root callable alias while replacing failed importer aliases with a new
dependency. It checks circular-import refusal followed by corrected recovery,
plus missing-file, parser and lexer failures followed by a valid same-path load.
Successful owners and all rejected metadata reach explicit teardown.
My final test-only extension first admits a real list-bearing record and checks
its generic/helper registration, then includes that shape in all 32 rejected
loads. It passes normal, sanitizer and production-profile links with unchanged
product providers. I retain the earlier test source for the complete matrices;
no product source, original assertion or deadline changes for this extension.

The first complete sanitizer control finishes those assertions, then reports
534 leaked bytes in 68 allocations. I track that separate defect under #1226:
module enum checking allocates its value vector twice, and duplicated enum owner
names have no recorded owner. I reuse the checked allocation and put copied
owner strings on my checker ownership list in both checker entries. I preserve
explicit negative/gap values and implicit values. My first unconditional enum
owner free passes focused imports but aborts the unchanged borrowed-name C/API
environment suite. I preserve that failed executable/source/output and keep
borrowed API names non-owning in the corrected source.

Final normal and ASan/UBSan runs pass the complete new recovery control,
the original three callable and fifteen state REPL methods, and the existing
tuple, record/list, STRING, root, loop, collection, native-byte, engine/fault and
adapter controls. These focused sanitizer runs keep leak detection enabled.
Original evaluator134 and checker88 pass normally and under sanitizers with
their historical broad leak detection boundary disabled. Original environment60,
nominal-owner and enum-ownership controls pass in both profiles with leak
detection enabled. I link their original sources directly against the current
providers to preserve the five frozen public tools.

Separately named ordinary and explicit STRING-candidate C seeds each pass the
original seven-method payload suite and compile the complete original compiler
source with 1306 matching successful shadows, the default shadow deadline and
original600-second parent bound. Their resulting --help checks pass. Original
lexical-scope, module-introspection and embedded-fixture controls pass. The
unchanged production optimization/warning flags also pass the complete new
recovery control plus all eighteen REPL methods. Tool-image and host-library
closure guards are retained.

My final production :load then :save observation still writes zero loaded
fragments. #1225 and complete #1205/#1140/#1116, default STRING ownership,
current-source raw/bootstrap, paired-platform and whole-release acceptance
remain open. Service graph acceptance remains in its existing dedicated lane.
I make no historical timeout-cause or performance-gain claim.

Run `python3 docs/evidence/module-check-rollback-linux-20261123/read_archive.py`
to verify exact committed Git inputs, raw/compressed archive hashes, retained
negative controls, command-time source guards and qualification scopes. The
archive includes final providers and separately retained failed executables.
Mutable provider paths were rebuilt between observations; I do not claim all
intermediate passing executables or objects survived. Identical older objects
are referenced by verified relative paths. I preserve all frozen roots,
audits/stashes and Puck-owned source/capture/regex/CI; I perform no Darwin or
NAS/TCC/migration action.
