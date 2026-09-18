# My owned value-call source graph evidence

I qualify the bounded contract in [NANOISA_SOURCE_OWNED_VALUE_GRAPHS.md](../NANOISA_SOURCE_OWNED_VALUE_GRAPHS.md) for tasks `task_46122ab40a234b6787d0de2927fd5bf2`, `task_9c2b62a5450c4fedace6b068c7f190d5` and the positional source acceptance of `task_d54033e921f24e279e53e8ed4cf50d17`. I change paired source producers and tests, not runtime authority or wire formats.

My final production/test source is `79a386f86b7f2864d524c9ea74db048c58a1d243`, including canonical `2ed1dc57`. I retain the complete log at `/tmp/nanolang-source-owned-graph-final-integrated.log` and its separate status file. My final command is:

```sh
make -j2 test-source-borrow-emission test-owned-value-results \
  test-owned-value-graphs test-multiple-consuming-calls \
  test-caller-references test-multi-caller-references test-owned-assertions
```

My command exits0. Fresh bootstrap, stage comparison and installed-compiler smoke checks pass. All38 paired source methods pass in337.385s. My runtime gates pass with these measured counts:

| Gate | Passed checks |
| --- | ---: |
| Owned/void results | 1,552 |
| Result allocation lifecycle | 3,485 (312 budgets;272 injected failures) |
| Return preflight | 117 |
| Owned graph lifecycle / preflight | 1,847 /338 |
| Invocation proof / verifier reuse | 529 /69 |
| Multiple consuming calls | 4,542 |
| Multiple consuming allocation / preflight / publication | 210 /216 /65 |
| Caller references / allocation / parameter allocation | 1,548 /43 /55 |
| Multiple caller references / binding allocation / owner allocation | 2,004 /93 /89 |
| Owned assertion lifecycle | 959 |

The source/tools in `/tmp/nanolang-source-owned-graph-final-sha256.txt` match before and after the paired suite. Retained stage emitters and shadow drivers are in `/tmp/nanolang-source-owned-graph-79a-tools/sha256.json`; Stage1/Stage2 emitters have identical SHA256 `21d708dd6c75af4faee9abedf7066f1de58433297dd2e9ce8c0609b1835f33f7`, and their shadow drivers have identical SHA256 `dde2d3312bad23589088a342ecea8f67bf70b6c18eb6bde914f0310ef311090f`. I claim no Darwin qualification or full release readiness from these Linux checks.

I compare C-seed, both selfhost stages and their raw emitters for ordinary modules and selected shadows. My source controls include factories, scalar wrappers, owned forwarding, VOID cleanup, eight-frame and eight-positional-argument bounds, distinct nominal owners, finite nested parameter/local owners, lexical ownership routing and synthetic shadow entry calling ordinary main. I compare canonical contracts and local-name records, roundtrip text/binary metadata, strip advisory names and execute the admitted artifacts on VM and sanitized native targets. I preserve exact semantic/output-retention refusals; refused artifacts are not executed.

## My retained earlier outcomes

I keep the superseded 8c3 qualification as interrupted (status143) in `/tmp/nanolang-source-owned-graph-paired.log`. Its bootstrap completed, but I corrected incomplete lexical wrapper scanning before substantive paired acceptance. I do not count this run as passing.

At `eda6e6cc`, my fresh bootstrap passed and the 37-method suite completed in333.237s with five failed subcases in two methods. Two canonical record-first tuple controls stopped at parsing rather than the intended semantic ownership boundary. Three raw emitters correctly refused a wrong nominal return with an exact-constructor diagnostic absent from the test matcher. I retain the complete log/status in `/tmp/nanolang-source-owned-graph-corrected-paired.log` and keep the parser defect separately open as `task_d9730c3ab71e45b283e3c38094638d47`; no parser repair is included here.

My tests-only `4a6adfa8` correction uses the scalar-first tuple spelling to reach the intended unsupported-owner semantic boundary, recognizes the existing exact-constructor refusal and adds a separate nested-owner matrix. Three affected methods passed in2.316s using unchanged hashed production/tools; `/tmp/nanolang-source-owned-graph-focused.log` records that result. The final integrated gate above rebuilds tools and repeats all38 methods after canonical compiler/runtime changes, so this focused result is not substituted for integrated acceptance.

I also retain two corrected command-selection errors (`bin/nano_virt` and `test-multi-consuming-calls` were not Make targets), without treating them as compiler defects. The corrected initial runtime gate passed separately at eda6. Final qualification above supersedes that source pin for integrated claims, not its historical evidence.

## My remaining limits

I leave the unchanged affine example taskc435 and full normative ownership open. String literals/parameters and PRINT require the peer's separate runtime prerequisite and a later paired source checkpoint. Borrowed CALL_REF graphs remain separate; imported/indirect/recursive graphs, reference/value mixtures, resource entry results, nested resource result fields and implicit source drops remain refused. The complete graph remains capped at eight emitted functions, including the synthetic shadow entry. The record-first tuple parser defect remains open.
