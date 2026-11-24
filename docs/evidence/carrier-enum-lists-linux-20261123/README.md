# My enum-list interpreter qualification

I qualify source `7c6a9c1a14e5a4f7a1a0be624f6abea1dddf8a97` for #1235 on Linux ARM64. I preserve scalar enum
get/pop results and explicit `List_int` scalar provenance through parameters,
results, record fields, sibling allocation pressure, immediate free and caller
storage. I leave native enum-list C generation open under #1236.

| My original command | Exit | Original shadow starts/completions | Wall seconds |
| --- | ---: | ---: | ---: |
| Full private collector C seed | 0 | 1306/1306 | 151.029845 |
| Ordinary C seed | 0 | 1306/1306 | 142.770218 |

I keep the original 600-second outer limit, default ten-second shadow deadline,
source selection and assertions. Both generated native seeds run `--help`.
My normal/O1, ASan+UBSan+LSan/O1 and production/O3/Werror consumers pass the new
scalar API/source controls, existing record/caller/refusal/pressure guards and
real Make targets. My source control exercises 128 parameter/result, sibling
pressure and field rounds, including enum values 0, 17 and -3. Scalar cells
whose bits equal a registered record address do not root that record.

My broad evaluator134/checker88 retain their original explicit LSan exclusions.
Borrowed values, active roots, record lists, STRING scopes, native bounds,
original allocation-failure controls, default REPL34 and private REPL41 pass.
The active-root peak remains 16 under the unchanged <=32 assertion; the
second native allocation batch grows by zero bytes across 128 calls. My five
frozen public images and nine library identities remain unchanged.

I retain the original enum fixture's assertion failure in both old images and
the independent old private pressure refusal. The repaired ordinary/private
images complete all six original enum shadows and all seven expanded pressure
shadows with zero failures. All four commands then fail native compilation
and preserve prior output. These are interpreter successes and native failures;
I do not call the source-to-native enum pipeline accepted.

The broad neighbor controls precede removal of two trailing spaces in eval.c.
I retain that earlier source and passing compiler image. The reader verifies
that formatting is their only source difference; I rebuild all six affected
consumers, rerun the focused controls/Make gates and both complete original
compiler-source commands on the final source bytes.

I use retained, hash-checked unaffected provider objects. This is not a fresh
shared-provider build, normal product aggregate enrollment, paired Darwin
qualification, raw/bootstrap/installed acceptance or whole-release acceptance.
I preserve Puck source ownership, all frozen roots/audits/stashes and the
NAS/TCC/migration pause. #1116/#1211/#976 and #1236 remain open.

I verify this archive without rerunning a native job:

```sh
python3 docs/evidence/carrier-enum-lists-linux-20261123/read_archive.py
```

My manifest records exact Git inputs, retained journals and provider/products.
I reuse content-addressed objects from the linked source-cost archive. My
reader checks hashes, actual compiler traces and unchanged cohort selection,
new/neighbor controls, retained failures, prior-output preservation and the
stated acceptance boundaries. `verification.json` records its terminal result.

My first archive reader stopped because the inherited Git input list omitted
the original enum fixture. I retain that reader/manifest failure and add the
unchanged tracked fixture explicitly. Compiler source, tools, assertions and
qualification products do not change for this archive coverage correction.
