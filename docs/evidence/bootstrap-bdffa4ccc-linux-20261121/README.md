# My Linux bootstrap at bdffa4ccc

I qualify exact source `bdffa4cccff93ed5b6870aa2f74289f94f2643ee` in an isolated Linux checkout. My complete original `make CC=/usr/bin/cc BOOTSTRAP_NANOISA_TIMEOUT=1800 bootstrap3` exits0 in2295.580s. Each raw generation retains1800 seconds; complete compiler shadows retain30 seconds.

My original five-target VM/heap command passes in88.664s, including569591 VM assertions. My raw Stage1 and Stage2 modules are byte-identical within this host: 1033296 bytes, SHA-256 `92b1c7a7a91fa58b2f0d06c34e5afb3efe46844f3a73d998f97a3a4267f59671`. Platform-specific raw imports differ between hosts; I do not require cross-host byte equality.

My final source/tool/artifact guards and all six host-library closures match. I retain receipt v2, SHA-256 `c76d723871bd3f935b1ea40afb677ffd0e0f159f5db043416b3673c769900ebc`, its fixed module-cache context, all17 successful receipt commands, the final `.bootstrap3.built` sentinel, `bin/nanoc -> nanoc_stage2`, and the original installed-checkout/no-C-seed checks. No unexpected-native marker exists.

| Receipt command | Seconds | Exit |
| --- | ---: | ---: |
| seed | 135.781 | 0 |
| seed-verify | 0.266 | 0 |
| stage1 | 757.117 | 0 |
| stage1-verify | 0.217 | 0 |
| stage1-translate | 60.632 | 0 |
| stage1-native | 274.657 | 0 |
| stage1-hello-compile | 1.119 | 0 |
| stage1-hello-execute | 0.005 | 0 |
| stage2 | 721.722 | 0 |
| stage2-verify | 0.215 | 0 |
| stage2-translate | 59.569 | 0 |
| stage2-native | 264.586 | 0 |
| stage2-hello-compile | 1.020 | 0 |
| stage2-hello-execute | 0.007 | 0 |
| seed-final-verify | 0.267 | 0 |
| stage1-final-verify | 0.216 | 0 |
| stage2-final-verify | 0.216 | 0 |

I preserve 2795 logical files in [manifest.json](manifest.json), with exact source/tool inputs, raw/native products, emitted C, build logs, host generations and controller inputs. I verify every compressed and decompressed SHA-256 after archival and after copying into this evidence directory. [analysis.json](analysis.json) records original commands, receipt steps, guard results and host-library hashes. Shared compressed content is stored once; manifest paths retain every logical input.

I keep the independent failed c908523c3 Darwin qualification in [draft PR #1172](https://github.com/jordanhubbard/nanolang/pull/1172) separate and make no causal claim about its timeout. I do not qualify later parser/checker/workflow changes with this older pin. Fresh PREFIX operation with source/build roots unavailable, full pure-translator cutover, exact final-head platform acceptance and release acceptance remain open under #1167/#1142/#976. Additional component checks use separate outputs after this bootstrap archive is preserved.
