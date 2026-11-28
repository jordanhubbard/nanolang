# My legacy record-list root prerequisite

I qualify source `ccacf817baadcfaf98d1962c7dcc012e10fd599f` on native Linux ARM64. I distinguish my lower-case record-list intrinsic's malloc-backed `List_int` from my managed `DynArray` record lists in private ownership metadata. I retain my public `Value`, container and foreign layouts. I do not activate aggregate collection in ordinary environments.

My original full private source observation refused roots at `diag_list_new`: 8 starts, 7 completed shadows, exit 1 after 126.726 seconds. A focused old-provider sanitizer control reports a heap-buffer-overflow after the 16-byte list header. My repair gives constructor-owned legacy lists explicit provenance, resolves logical record-list edges through registered or caller-supplied layout facts, and clears caller facts after each walk.

My new source and C/API controls pass in normal, ASan/UBSan/LSan and strict optimized profiles. They cover growth, returned lists, record fields, STRING children, transfers, explicit caller-owned roots, address reuse, failed tracing, duplicate admission, unknown integer bits and invalid layout tags. I retain a failed test setup that omitted its parent root; I corrected the supplied root and preserved its three-live-object assertions. The final expanded controls pass all three profiles.

I rebuild all five consumers of the changed private header in each original profile. I reuse and hash-guard my unaffected providers; I do not claim a fresh build of the entire provider closure. My original evaluator's 134 and checker's 88 cases pass normally and with ASan/UBSan; those broad suites disable leak detection. My focused ownership controls retain leak detection. My existing native-byte control retains its positive allocation and 128-call assertions and reports zero growth between calls 64 and 128. My actual new Make targets pass in all three profiles.

My default 34 and private 41 REPL controls pass in all three profiles. My complete default admission suite retains three LSan failures of 168, 172 and 172 bytes. Passing private controls do not establish default activation.

My updated ordinary C seed passes all 1,306 compiler-source shadows and publishes a native seed in 138.369 seconds under the original 600-second outer and default 10-second shadow limits. The seed runs `--help`; its SHA-256 is `6b0519e8f2bc9bf5a60b0b96b882fc8aea1f3535c2049992d08f3795d3b5038f`. The original 10 lexical and 7 record-payload native methods also pass. All 1,340 source guards remain stable during each full source observation.

My repaired full private source observation remains unsuccessful: 161 starts and 160 completed shadows, followed by the original 10-second supervisor deadline. Its last periodic milestone records 65,536 collection calls, 654,262,134 cumulative symbol visits and 3,765,637,591 measured collector nanoseconds. These are milestones, not final totals or a causal account of an older timeout. The external collection observer adds measurement overhead. I do not claim an unwrapped private run or a performance improvement.

I also retain a source-valid explicit lower-case list-free probe that reports heap-use-after-free during later root tracing. I have not coordinated manual free with private root liveness. [#1233](https://github.com/jordanhubbard/nanolang/issues/1233) owns that semantic blocker; [#1232](https://github.com/jordanhubbard/nanolang/issues/1232) owns full private source cost. [#1231](https://github.com/jordanhubbard/nanolang/issues/1231) stays open pending its complete acceptance criteria. [#1229](https://github.com/jordanhubbard/nanolang/issues/1229) remains a distinct historical terminal investigation.

I verify my compressed records, exact committed Git inputs, native images, provider hashes, original outcomes, public-tool guards and nine-library closure with:

```sh
python3 docs/evidence/legacy-record-list-roots-linux-20261123/read_archive.py
```

I reuse unchanged content-addressed objects through the checked prior manifest. My journals include intermediate fixtures and failures; only the explicit final source pin and final controls establish this checkpoint. I preserve original roots, public tools, stashes and peer-owned files. All local native jobs are terminal; I hold no Linux or Darwin worker reservation. NAS/TCC/migration pauses remain in force.

I claim no raw bootstrap, paired-platform, complete default ownership or whole-release acceptance.
