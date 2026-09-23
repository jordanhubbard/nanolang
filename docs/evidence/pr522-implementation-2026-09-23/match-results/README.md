# My canonical match result evidence

I retain first failures separately from corrected checks. My final local frontend and compatibility gates pass; these results do not
clear the complete hosted or release gates for #522.

`paired-first.log` retains the two product defects exposed after the original
three match cases passed: a C nested empty-array result loses its element kind,
and both producers hit poisoned native join provenance after an unresolved
inference kind joins an integer. The exact failing sources are retained here.
`paired-corrected.log` passes all six initial paired methods after both fixes.

`paired-expanded.log` records two incorrect test expectations: a tagged heap
carrier was expected to be refused statically. `prior-heap-boundary.log` checks
the previous translator and confirms its existing acceptance with runtime
failure at the declared integer boundary. I correct the fixture to require VM
and native failure, the native invariant diagnostic, and no ASan/UBSan failure.
This is a fixture correction, not permission to interpret heap data as integers.

`instrumented-expanded.log` passes eight paired methods with scoped checker and
translator ASan/UBSan instrumentation and leak detection. Other linked compiler
objects are ordinary objects. Generated native consumers are also instrumented.
The scoped-build script initially failed to parse a multiline Make recipe;
`instrument-build.log` retains that harness error and the corrected build log
is separate. No instrumented success is inferred from the failed build.

`translator-full.log` passes all 2,428 translator assertions and its prerequisite
shape/driver controls. `bootstrap.log` records successful fresh native bootstrap
and normal tool rebuilds. The nested ordinary-union refusal fixture had become
obsolete in earlier implementation; I retain its original source and direct
assertion failure, then preserve it as a positive case and use a wrong nested
declaration for the negative test.

Compressed logs retain exact uncompressed bytes; `logs.json` records their sizes
and SHA-256 hashes. Final-source fixed points, complete platform/sanitizer gates
and release documentation remain separate requirements.

The first fresh-stage matrix passes all 72 compatibility, generic identity,
selected ownership/pattern and scope methods. The full scalar-match gate passes
all 12 methods in 419.784 seconds, including its independently compiled Stage 2
producer. The standalone obsolete-union refusal run retains its two expected
fixture failures; the corrected full scalar-match gate passes both refusals.

Adding the original nested-union source to the paired producer gate exposes a
separate C checker mismatch (union versus struct). `paired-final.log` and
`instrumented-final.log` retain that failure; `instrumented-nested-corrected.log`
passes nine methods after nominal checking is corrected, and
`nested-refusals-instrumented.log` passes wrong-declaration constructor/variable
controls without replacing prior output.

Hosted run `35927493387` at `8e15a3a99` fails x64/source jobs on two obsolete
nested-array record refusal fixtures. The units-01 terminal separately records
a 60-second compiler-shadow deadline during bootstrap3. Those job logs remain
separate from local source qualification; the sanitizer schedule is not passed.

Positive nested-array record execution reveals a native record-return shape
conflict in both producers, retained in `flat-record-fixtures.log`. The targeted
correction preserves known array shapes in tagged record fields;
`flat-record-corrected.log` passes both the positive and negative fixture methods
with the updated translator translation unit instrumented. The final normal rebuild, translator regression and paired source checks
pass after these C fixes.

After both later C corrections, `bootstrap-final.log` passes fresh native
bootstrap. `translator-final.log` again passes all 2,428 translator assertions.
`paired-current.log` passes all 17 paired match/record-source and flat-record
fixture methods; `callback-final.log` passes all 31 C callback/function-value
methods. `fresh-matrix-final.log` passes all 72 methods, including all 23 original
compatibility methods. `frontend-full.log` passes all 91 methods in the complete
`make test-nanoisa-src-nano` target, including generated native ASan/UBSan/leak
checks for the restored nested-array fields. The 12-method scalar-match run
preceded the final C nominal and record-return corrections; I do not present it
as a complete scalar-match run against those final binaries.
