# My retained Stage2 emitter observation

I retain three failed native/emitter jobs and one passing cache job. These
selected logs establish the terminal states below. I preserve the complete ZIP
inputs separately and pin the diagnostic's bootstrap ZIP by SHA256 before
restoration; this selected-log receipt does not claim to replay those ZIPs.

| Job | Source | Original result |
| --- | --- | --- |
| [Older Linux bootstrap](https://github.com/jordanhubbard/nanolang/actions/runs/46042142609/job/140652282080) | d18fed86140b6a4cf983898ac32a079002f6b784 | 17 bootstrap steps and 36 native methods pass; Stage2 emitter-fixture compilation times out at 300 seconds; zero emitter methods execute. |
| [Corrected Linux bootstrap](https://github.com/jordanhubbard/nanolang/actions/runs/46043406093/job/140656541879) | cd6bf9fbfff108f3dffb541ae4c266e8d6b83334 | The same Stage2 setup timeout follows passing bootstrap/native checks. |
| [Corrected Darwin bootstrap](https://github.com/jordanhubbard/nanolang/actions/runs/46043406093/job/140656542299) | cd6bf9fbfff108f3dffb541ae4c266e8d6b83334 | 17 bootstrap steps and 36 native methods pass; C-seed shadows stop at ten seconds before any emitter method executes. |
| [Corrected Linux cache](https://github.com/jordanhubbard/nanolang/actions/runs/46043406093/job/140656542401) | cd6bf9fbfff108f3dffb541ae4c266e8d6b83334 | The original cache/worker job passes. |

I qualify neither C repair nor my release with these results. I keep #1253 and
#1265 open. I do not infer infrastructure failure from a separate passing
control, and I do not change the original tests, assertions or deadlines.

My new observer restores the older Linux artifact 15222823364, SHA256
`f80edbf52d706501319cdefa9d5dcdce3ea933b5b3723f2621cf1fffaa258dc5`,
with its original source checkout. I reuse the already tested restoration code
at 51c82debb858145371cd9f28005d683e0e79184f, copied outside the product tree and
checked against its exact source hash. I require unique, complete, byte-equal
cache generation evidence, original tool/source/product identities and all six
receipt-bound host libraries. I refuse mismatched inputs before execution.

I recover the exact Stage2 command from the retained failure log, recreate its
private destination and TMPDIR, preserve the original pipes and 300-second
limit, and sample only the new owned process group. I retain complete stdout,
stderr, child commands/CPU times, terminal status and regular intermediate
files. A bounded post-kill drain reports incomplete output as an observation
failure. My sampler does not change the configured product deadline. Native process controls
precede the product; before/after source and tool identities remain mandatory.

I restore the artifact's terminal cache state. Its original state immediately
before the failed command is unavailable. This is a targeted diagnostic,
including if the command later succeeds. It is never the original emitter
suite or a replacement qualification result.

My four new tests and twelve pinned restoration tests pass locally (16 methods).
I also accept the exact command from the actual retained ZIP log. My selected
receipt manifest is
`e32cd7415edb0c99576c31756e4723500b1a007966cc4f940c76535422155395`.
My verifier checks its 17 source/log/control records without executing a product;
a changed manifest refuses before opening selected inputs. Hosted observation
remains pending in this receipt.

```sh
python3 docs/evidence/c-repairs-linux-20261126/stage2-emitter-observer-20261127/verify.py
python3 -m unittest -v tests.test_stage2_emitter_diagnostic
```

I launch the hosted observer through
`.github/workflows/stage2-emitter-diagnostic.yml`. I preserve the product at the
original d18 revision and load observer code independently from the workflow
revision. The full 17/36/12 platform gates, C repairs, paired translator,
current-candidate/cutover, frozen CAST_STRING assertions and all 69 release
requirements remain open.

My [first hosted setup refusal and correction](hosted-setup-refusal-46051728142/README.md) retain the missing sparse-checkout fixtures, all original failure bytes, and the passing eight-method fixture admission check. My source snapshot above records the first observer; the child receipt binds the corrected workflow. Neither receipt claims hosted product qualification.

My [namespace-admission correction](namespace-admission-46052578737/README.md) preserves the next observation’s early library-publication refusal and the omitted recovery call. My current observer requires restored-link postconditions and a genuine native Stage2 readiness compile/run before the large fixture. Its 18 local protocol/restoration methods pass; hosted readiness and original qualification remain open.
