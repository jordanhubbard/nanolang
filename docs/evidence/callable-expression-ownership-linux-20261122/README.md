# My owned callable expression foundation

I publish source `315d9375c4` for #1140 on PR1168, after my
[constructor prerequisite](../callable-container-ownership-linux-20261122/README.md).
My original strict nine-route/64-call audit now retains **zero callable names**
normally and in a fresh complete ASan/UBSan provider build with leak detection
enabled. Its source and every assertion remain byte-identical. My retained
constructor-only pin still reports576 names; I preserve that earlier failure.

I give callable variable/field/aggregate reads independent names and complete
signatures. Returned callables transfer ownership without a lost intermediate
copy. I retire direct callable scope, source argument, computed-callee,
constructor/tuple input and handler temporaries. Private async argument bundles
distinguish borrowed synchronous inputs from owned queued expressions. I copy
borrowed cached task results and preserve return targets/control-flow flags.
Borrowed C/API arguments survive. I preserve existing nested-record copy
behavior and add no field or carrier reclamation. My public Value/container
layout is unchanged; my header adds only a callable-result ownership comment.
Complete spawned-result/handle lifecycle acceptance remains open.

Five raw expression reads each own one independent result. Four additional
64-call routes balance actual name allocations/releases: discarded320/320,
reassignment192/192, effect parameters64/64 and async forwarding192/192. The
new read-ownership control fails against retained constructor-only providers.
An external diagnostic executes those same four source methods verbatim and
passes with the fresh ASan/UBSan provider closure and leak detection enabled.
It does not replace the complete default fixture.

My final ordinary interpreter/typechecker suites and escaped record/union
fixture pass. Every fixture shadow runs. An additional C/API control retains
its caller's original name/signature, receives an independent escaped record
callback, releases that record and successfully invokes the original caller.

I keep **both complete sanitizer gates unsuccessful**. Their assertions and
shadows finish, then LeakSanitizer reports572bytes/26allocations for the
expression/globals fixture and3720bytes/159allocations for record/union. I
suppress no finding. #1140/#1116 stay open; zero retained names in the original
strict audit does not establish complete container/lifecycle reclamation.

The new REPL display/load/reload target cannot build. I reproduce its strict
native errors with exact original30ac8a4f67 `src/repl.c`: removed
`Environment.parent`, undefined `g_mod_count` and missing `cmd_modules`.
I file #1205 and retain both original/current failures. My REPL callable-result
retirement changes are not runtime-qualified. I retain the new explicit target
and defer its ordinary-unit wiring until that independent build prerequisite
passes. Existing gates, warning flags, assertions and deadlines stay unchanged.
The original strict audit and the two qualified focused C targets are wired
into ordinary units.

I preserve the first invalid observer fixture (bare function references used as
statements, rejected with E034), its source/image and the corrected fixture's
full results. The corrected test discards actual function-call results. I also
retain private prototype and intermediate observations separately. Their
source/provider identities are revision-specific; I claim current acceptance
only for the final source and stated checks above.

My independent reader verifies1528 archive records,1382 exact Git source
inputs and376663954 logical bytes: final source/tool guards, original strict
assertion identity, public-header layout preservation, complete streams and
selected linked products. Individual provider identities are observed hashes;
I retain their code in linked images, not every separate object file. The
reader separately checks the full unsuccessful gates and the external control.

```sh
python3 docs/evidence/callable-expression-ownership-linux-20261122/read_archive.py
```

Puck-owned `nvm2c.c` is unchanged at SHA256
`12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67`,
with consumer
`c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea`.
My retained09cab native Stage2 producer remains
`346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83`.
Tools remain native Linux GCC13.3/as2.42. No new full bootstrap, Darwin or
whole-release acceptance follows. I preserve all frozen roots, primary dirty
audit/stashes and the NAS/migration/TCC pause. The later narrow #1202 grant is
separate; no caller hunk is included in this receipt.
