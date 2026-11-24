# I localize five original native-shadow failures

At unchanged source `cbbfdcb70cb192abaf3f043fe07b7c8e7fa99db2`, I run only the five original methods that failed after [my successful original bootstrap](../core-supervisor-bootstrap-linux-20261124/README.md). I use a NEW local root and copy its actual qualified Stage2 image with unchanged SHA; I derive no new compiler or stage alias. Its immutable host closure remains guarded. I change no test body, assertion, command option, original90-second invocation budget or ordinary10-second deadline.

| Original method | Actual status | stdout | stderr |
| --- | ---: | --- | --- |
| foreign alarm cancellation | 1 | no-main diagnostic | empty |
| foreign early-zero exit | 1 | no-main diagnostic | empty |
| separate shadow locals | 1 | no-main diagnostic | empty |
| shadow-only source | 1 | no-main diagnostic | empty |
| source-only without native compiler | 0 | empty | empty |

The exact first-four diagnostic is `I cannot lower this checked program: no main`. Each exits in0.020–0.026 seconds, before reaching the native shadow execution requested by the foreign controls. The fifth publishes C in0.036 seconds with NANO_CC=/bin/false but omits its original required `without executing shadows` notice. All five original methods fail, zero errors/skips. This localizes current failures; it establishes no historical regression origin or new supervisor clock defect.

My external Popen observer preserves actual argv, selected environment, original timeout, full stdout/stderr/status and test-owned directory bytes before/after each original invocation. I keep normal TemporaryDirectory cleanup. The early-exit case's prior output remains byte-identical; other refused cases publish no output. Both foreign helper C sources survive unchanged, and source-only C contains no original forbidden shadow entry prefix. All3524 selected Git inputs, qualified producer, immutable host libraries and system-tool guards match. This is diagnostic consumer reuse, not fresh frontend/bootstrap or admission of a repair.

The archive contains5137 records, manifest SHA256 `8ce328185858ed974c60905df750b98f563cc1f9818dbba299a9d4a563b980db`. I retain actual source, tools, original fixtures/helpers/prior outputs, published C, streams, setup/provenance and original-method results, reusing verified parent chunks. I verify them and the borrowed qualified producer against the parent bootstrap archive with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/native-shadow-localization-linux-20261124/read_archive.py
```

I own the focused pure entry/driver repair under #1253. All original main/module, shadow-local, type/selection, foreign ABI, deadline and prior-output guards remain required. The full unchanged18-method target and12-method emitter companion must pass after repair against genuinely current products. Complete SAN, ordinary full-source, paired Darwin, pure C emission/cutover and all69 release requirements remain open. No production source edit, second full bootstrap, Darwin reservation or NAS/TCC/migration action is part of this diagnostic. Both old roots remain frozen; the separate service-clock plan stays unrun.
