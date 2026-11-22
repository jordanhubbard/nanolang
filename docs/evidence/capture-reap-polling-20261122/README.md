# I observe child exit after output closes

I retain two full hosted observations at `5bb78f2f43ac78d08134cf81e38aee83c6ee593f` and a subsequent focused supervisor correction. My [manifest](manifest.json) records raw and compressed hashes and lengths; my [archive audit](archive-audit.json) verifies every payload. Shared payloads refer to prior retained evidence without changing it.

## My hosted observations

[Fixed GNU run45005584925](https://github.com/jordanhubbard/nanolang/actions/runs/45005584925) passes my original128 source methods in775.873 seconds, with24 conditional skips and no failures/errors; my separate original response method and600-case matrix, actual cold capture, and40 ordinary guide snippets also pass. My ordinary guide inherits no shadow override and uses its original ten-second default. Both checkout sources are restored. The selected private source-fixed GNU assembler image is `e1becd22bd1b7804e99d46ba0bd9dc0ad5273d289b6c9bf35bae1bb9a411786a`. Its source archive, source patch, producer tests, native images and outputs remain in this archive. This result qualifies that private cohort; I have not promoted the profile into production admission.

[Clang run45005587292](https://github.com/jordanhubbard/nanolang/actions/runs/45005587292) reaches the unchanged1500-second source-suite parent limit. My streaming log retains91 completed successes and25 conditional skips; `test_standalone_preprocessed_assembler_search_recovery` is active when the parent terminates the group. I retain no completed `tests.json` and claim no complete pass. Later response/debug/cold cohorts do not execute. Both checkout sources are restored. I use the already-supported private GNU2.42 backend image `9ba35517aa067b7eebac20e16c1ef59fee7351667f741ac4fab0018fad54d55f` with actual Ubuntu Clang21.1.8 image `78e9dd2ccc9e88859b3a29109a44bfab00e3d6cb70f00f3daae063eed3d05251`.

## My controlled polling observation

My existing supervisor polls a disabled descriptor for up to25 milliseconds after EOF while awaiting child exit. That descriptor cannot wake the poll. I first retain an unsuitable20-millisecond control with an external `sleep` executable: its standalone command itself takes35.266 milliseconds. That observation does not establish a false timeout.

I then use the same shell-only command and100-millisecond capture budget before and after the correction:

```sh
printf ready; exec 1>&- 2>&-; capture_tick=0; while [ "$capture_tick" -lt 1000 ]; do capture_tick=$((capture_tick+1)); done
```

My original probe observes EOF at5 milliseconds and reap at31; my corrected probe observes EOF at7 and reap at9. Both accept exactly `ready`. These are individual native Darwin observations, not a timing guarantee or evidence that this delay explains the complete hosted Clang timeout.

I change only the polling interval after EOF from25 to1 millisecond. I preserve every deadline and fresh completion-clock check, output/exit refusal, process-group termination and child reap. My source before the change is `6a11f68bdddd2c4c3902d88ebbacedd021f2ae1d1f15066f15cae309eca5156c`; afterward it is `386969772f51564da962bba36b425dcf3cd5d7087d1276571c837e0cc06a1945`. My retained native probes are respectively `d0a56b45fe322c5bf07ece487330bdc1e1780ddddec1cd6363ec978d04c25a1c` and `ab2c45591288b39160210ee314335fa2ab3016a8372a78dce7ce100a22daa145`.

My strict native build passes. My existing late-completion controls, timing milestones and13 diagnostic controls pass; the three GNU-query methods explicitly skip on Darwin. My four actual native response refusal/lazy calls also pass with their original50-millisecond query and five-second parent limits. Full fresh hosted Clang/GCC and paired final-platform acceptance remain required.

## My independent peer receipt

I retain the README fetched from Sparky's [native Linux evidence commit77100354](https://github.com/jordanhubbard/nanolang/blob/77100354c501c2ac9ee105e6ae2e6ad5abde5cb4/docs/evidence/clang18-gnu-query-linux-20261122/README.md). It reports native red/green GNU-query controls and passing external/slow-capture cases at5bb. My initial receipt fetch did not include its complete archive. I subsequently fetch the evidence directory into a separate sparse worktree and independently verify all11410 raw/compressed records and lengths,3843 unique payloads and all declared Git source blobs (3569/3573/3573 inputs), including the sole red builder override and all four native image hashes per set. My [independent reader result](peer-query-archive-verification.json) and [observer](verify-peer-query-archive.py) retain this verification without repeating native tests. The peer's original archive remains on its integration branch.

My [independent hosted reader](audit-hosted-fixed.py) also checks the full successful5bb fixed-GCC reports, all600 native/replay streams, all2095 exact upstream outcome lines, eight witnesses, actual probe/helper/assembler hashes and the40-snippet ordinary guide terminal; its [result](hosted-fixed-45005584925-independent-audit.json) keeps this older source pin explicit.

Sparky subsequently reviews exact3ef and reports GNU query3 plus existing late-completion/timing controls passing natively on Linux, with a2ms EOF/reap observation in [comment7077384165](https://github.com/jordanhubbard/nanolang/issues/1159#issuecomment-7077384165). I retain the comment as a peer report pending its archive. Under the explicit one-line Make handoff in [comment7077434483](https://github.com/jordanhubbard/nanolang/issues/1159#issuecomment-7077434483), I append tests.test_gnu_tool_queries to the existing test-bytecode-shadows invocation. My [dry-run recipe](query-gate-enrollment.dry-run) preserves both old modules, all prerequisite declarations and every limit. This enrollment does not claim a completed full Make target or new whole-platform acceptance.

## My independent PR concurrency

Under #1178 I replace the constant user-guide workflow concurrency group with a PR-number group for validation and one shared `deploy` group for non-PR publishing. I preserve cancel-in-progress and every other parsed YAML field. My [static review](userguide-concurrency-review.json) records two different PR identities and both publishing event identities. I retain the unavailable Python YAML parser setup separately and use installed Ruby YAML successfully. I claim no live scheduler cancellation experiment or cause for an old cancelled run.
