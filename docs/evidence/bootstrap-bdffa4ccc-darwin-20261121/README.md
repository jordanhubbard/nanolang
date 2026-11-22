# My Darwin bootstrap at bdffa4ccc

I qualify exact source `bdffa4cccff93ed5b6870aa2f74289f94f2643ee` in an isolated Darwin checkout. My complete original `make CC=/usr/bin/cc BOOTSTRAP_NANOISA_TIMEOUT=1800 bootstrap3` exits0 in1277.802s. Each raw generation retains1800 seconds; complete compiler shadows retain30 seconds.

My original five-target VM/heap command passes in38.476s, including569591 VM assertions. My raw Stage1 and Stage2 modules are byte-identical within this host: 1033548 bytes, SHA-256 `bcb9f3d52abf53eea08835ac9d0c04cb9a360b03b62bec38509cfb2212ff94d9`. Platform-specific raw imports differ between hosts; I do not require cross-host byte equality.

My final source/tool/artifact guards and all six host-library closures match. I retain receipt v2, SHA-256 `67bb7749a5382bab73eeb9e2607a5f872d944fec3edc8e04d973a7171c29fecc`, its fixed module-cache context, all17 successful receipt commands, the final `.bootstrap3.built` sentinel, `bin/nanoc -> nanoc_stage2`, and the original installed-checkout/no-C-seed checks. No unexpected-native marker exists.

| Receipt command | Seconds | Exit |
| --- | ---: | ---: |
| seed | 52.603 | 0 |
| seed-verify | 0.448 | 0 |
| stage1 | 539.304 | 0 |
| stage1-verify | 0.188 | 0 |
| stage1-translate | 29.350 | 0 |
| stage1-native | 31.986 | 0 |
| stage1-hello-compile | 0.596 | 0 |
| stage1-hello-execute | 0.252 | 0 |
| stage2 | 542.230 | 0 |
| stage2-verify | 0.210 | 0 |
| stage2-translate | 29.078 | 0 |
| stage2-native | 31.787 | 0 |
| stage2-hello-compile | 0.485 | 0 |
| stage2-hello-execute | 0.210 | 0 |
| seed-final-verify | 0.204 | 0 |
| stage1-final-verify | 0.144 | 0 |
| stage2-final-verify | 0.143 | 0 |

I preserve 2086 logical files in [manifest.json](manifest.json), with exact source/tool inputs, raw/native products, emitted C, build logs, host generations and controller inputs. I verify every compressed and decompressed SHA-256 after archival and after copying into this evidence directory. [analysis.json](analysis.json) records original commands, receipt steps, guard results and host-library hashes. Shared compressed content is stored once; manifest paths retain every logical input.

I keep the independent failed c908523c3 Darwin qualification in [draft PR #1172](https://github.com/jordanhubbard/nanolang/pull/1172) separate and make no causal claim about its timeout. I do not qualify later parser/checker/workflow changes with this older pin. Fresh PREFIX operation with source/build roots unavailable, full pure-translator cutover, exact final-head platform acceptance and release acceptance remain open under #1167/#1142/#976. Additional component checks use separate outputs after this bootstrap archive is preserved.
