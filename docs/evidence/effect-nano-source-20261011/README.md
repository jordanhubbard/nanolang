# My independent Nano ownership effect checkpoint

I implement exact retained operation signatures and source lowering for resource effects in my Nano-written ownership producer. I collect used operations, sort canonical retained keys, describe positional scalar/STRING/record/union transfers, and mark operation union layouts complete. I lower performed arguments in source order, isolate handler resource parameters, preserve scalar captures, move inferred resource results, retire consumed destructuring shells before calls/suspension, and propagate void returns and lexical exits. I retain existing verification and finite ownership-profile limits.

My fresh C-built Nano compiler passes all 1,182 mandatory shadows and the complete shared resource-effect source matrix (`seed-build3.log`, `seed-source-matrix.log`). The source matrix includes selected shadows, imported same-named resource/helper identities, VM verification/execution, strict generated native C with ASan/UBSan/leak checks, and ownership/ambiguous-handler refusals preserving prior output. `NANO_OWNED_EFFECT_DRIVER` selects this compiler in the same test used for my C producer. I accept each frontend’s exact existing ambiguity diagnostic, preserving the refusal assertion.

My focused component build passes 507 shadows. A small producer-only driver separately executes fourteen handler cases and six transfer cases in VM and sanitized native C (`producer-execution2.log`, `producer-transfer.log`). This driver intentionally bypasses frontend/module/shadow-source processing; these component results do not replace the full source matrix. Its source is retained as `producer.nano`.

## My measured deadline correction

The first full C compiler build reached the mandatory ten-second shadow deadline after 955 completed shadows while another compiler job ran. An isolated rebuild also failed, after 1,053 completed shadows; concurrency was not the sole cause. I retain both complete traces. The unchanged expanding-layout shadow took 3.328 seconds in the isolated run. My type normalizer rebuilt already space-free strings one character at a time on each generic descent.

I return unchanged strings immediately when no ASCII space exists. I preserve all cycle/expansion assertions, the 128-substitution boundary, and the ten-second deadline. The next complete build passes 1,182 shadows; its recorded shadow durations total 7.352 seconds and the layout check takes 1.204 seconds. This is a measured test result, not a guarantee for every host or load.

## My retained failures and remaining work

- `baseline.log` records twenty-one positive source failures in the older compiler and one diagnostic-spelling mismatch for a correctly refused ambiguous handler.
- `producer-execution.log` records the new emitter’s initial assembly defect: symbolic `s0` operands did not name its positional string entries. I use numeric retained indexes; the corrected twenty-case component execution passes.
- `seed-selected.log` includes two harness-key errors from a temporary compiler name. Under the declared `nanoc_stage1` identity, both my fresh seed and the published compiler fail the same four ordinary selected-pattern methods and pass the other 45. Choice.Some/Box.Some generated destructuring bindings remain an ordinary-emitter gap. I do not claim that entire neighboring gate passes.
- An earlier NanoISA rebuild was interrupted after the assembly correction changed its inputs; it establishes no product qualification.
- My corrected full compiler-product audit spends 327.019 seconds emitting the compiler and then reaches the separate ten-second VM shadow deadline. `product-manifest.json` confirms unchanged inputs. No compiler module is published by that failed run. I keep this required gate open; the successful C-built compiler and component execution do not establish the NanoISA-built compiler product.

I reproduce the normal source gate with `NANO_OWNED_EFFECT_DRIVER=<fresh-compiler> python3 -m unittest -v tests.test_owned_effect_c_source`. `audit.py` retains the attempted fresh-product recipe; its host-specific paths identify this run. A diagnostic rerun retains the generated shadow module and uses the same VM deadline with tracing, so I can attribute the remaining cost.

Global/effect combinations, collection signatures, borrowed continuations, the ordinary selected-pattern gap, fresh-product qualification and the broader 5.1 contract remain open. This checkpoint does not complete #976 or release 5.1.
