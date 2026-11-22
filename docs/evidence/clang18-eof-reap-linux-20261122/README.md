# My independent post-EOF supervisor control

I review exact candidate `3ef446575d2b4c818c96cf83d46c1048f0edf441` from the capture owner under #1159 in a new detached Linux ARM64 root. Relative to5bb, the only production change is module_builder.c's poll interval:1ms after output EOF,25ms otherwise, still bounded by remaining deadline. All deadline/fresh-completion-clock, output/NUL/size/exit refusal, process-group termination and reap code remains unchanged. Original GNU-query/source test files are byte-identical.

My installed supported Clang18/GNU2.42 builds the actual probe/helper. All three original GNU-query tests pass0.073s; both original late-completion and timing-milestone methods pass0.722s, with zero errors/skips. These retain late-success/clock-error rejection with trace on/off, output preservation, complete output/actual query timing,50ms/0.3s supervision,254accepted/255refused,NUL and exit7 guards. Both original delayed-first-output and delayed-reap timing cases remain unchanged.

I execute the owner's exact shell-only command once through capture-configured at100ms:

```sh
printf ready; exec 1>&- 2>&-; capture_tick=0; while [ "$capture_tick" -lt 1000 ]; do capture_tick=$((capture_tick+1)); done
```

It returns0 with exactly ready. Actual milestones are spawn0ms/first-output1ms/EOF1ms/reap3ms: a2ms EOF-to-reap observation. I do not infer a latency bound, a Linux before/after gain, or a cause for the hosted Clang timeout. I repeat no complete source/response or earlier original-root cohort.

All3573 source/test/spec/Make inputs, installed Clang18 SHA256 `6d99f6207153b8b2063ace5d2a6177debaf6a6b6f31649e03afac07b98ba0b4d`, GNU2.42 SHA256 `1ffda50efb6d91b6b05ef933aced099595c36577594c0296c91161f2d13db374` and actual probe/helper image guards hold. Ambient helper remains unset, ordinary shadow timeout10s, no global capture/sanitizer override is supplied, and only the declared single EOF observation sets100ms. Every method imports the intended source in a fresh process, all command streams/fixtures remain retained, and the detached new root stays clean. Old roots and the translator implementation checkout remain separate.

My lossless archive retains3631 logical records,32419693 raw bytes,3599 unique gzip payloads totalling8218051 bytes and two symlink targets. An independent reader checks all raw/compressed hashes and3573 exact source Git blobs. Publication audits are separate from those counts. Capture implementation, complete unchanged-budget hosted cohorts and production admission remain with Puck. I hand off the exact one-line Make recipe enrollment separately in #1159 comment7077434483; I do not edit that line. No full-suite or release acceptance is claimed.
