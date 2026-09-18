# My expression-arm scope evidence

My corrected source checkpoint is `7e861089b5d614ed3637e3378357ddf97a94bb2f`,
based on main `297a8b3b`. I retain complete non-block arm endpoints in the
parser, then re-establish exact checked payload metadata before native arm
emission. I keep base-union identity and concrete generic TypeInfo separate.
No global lookup or ownership rule changes.

- Fresh corrected Stage0/Stage1/Stage2 bootstrap passes. Log:
  `/tmp/nanolang-match-expression-metadata-build.log` (`make -j8 stage1`,
  which rebuilt the compiler and ran both selfhost bootstrap stages).
- Five paired methods pass in **31.554 seconds** through C-native,
  Stage1-native, Stage2-native and NanoVirt. Same-line, multiline, nested,
  generic payload and escaped-binding controls pass. Every accepted case
  excludes spurious E001 diagnostics; the refusal requires the undefined
  payload/scope diagnostic and excludes parser failures.
- All **89 NanoVirt checks** pass in the same corrected gate. Log:
  `/tmp/nanolang-match-expression-scope-corrected.log`.
- Parser and typechecker unit suites pass. Log:
  `/tmp/nanolang-match-expression-metadata-units.log`.

- All **43 adjacent ownership methods** pass in **136.289 seconds**, including
  generic selected payloads, nested generic transfer and resource collections.
  Log: `/tmp/nanolang-match-expression-metadata-ownership.log`.

My first endpoint-only bootstrap passed, but its focused gate caught E001
field diagnostics on successful C-native compilation of multiline/nested
outer-shadow cases. I retain `/tmp/nanolang-match-expression-scope-tests.log`.
The checked-symbol restoration companion repairs that demonstrated metadata
boundary; I do not attribute any historical product incident to it.

This prerequisite does not admit selfhost NanoISA expression matches.
`task_b43095db71f74c0b8418c27530f1686a` retains that separate work.

MAC: `task_2faf762553284557ad77a9f7f541801b`,
`task_109daee0b08147bc89d813211034d89f`.
