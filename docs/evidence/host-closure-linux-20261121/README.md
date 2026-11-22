# My Linux native host-deployment checkpoint

I repair the installed compiler's absolute-cache dependency under #1167. My native emitter records each artifact adapter's exact library SHA-256 and source path in a dedicated ELF/Mach-O section. My installer packages those declared bytes; my loader verifies executable-relative content-addressed entries. Raw module imports remain unchanged. I retain the [deployment contract](../../NATIVE_HOST_DEPLOYMENT.md), missing-import refusal, original signature/ownership checks and ordinary ten-second shadows.

My development compiler has SHA-256 `9ec68e892463feb0b0dd4bae1736bf5ef87ccee755401854bea5838ffab1a01f`. I translate the previously retained, verified development raw compiler; its NanoLang source hash still matches current source. This is a new native deployment product, not a newly generated raw Stage1/Stage2 fixed point or final-head bootstrap. I retain the actual raw input, translator, generated C, runtime object and selected six libraries.

My actual Make install succeeds, then I rename the complete prefix before running it. An independently successful bwrap setup hides /home/jkh/.worktrees/nanolang, /home/jkh/Src/nanolang and /home/jkh/Src/nanolang-callable-ownership. I leave the fresh prefix, external client inputs/cache and system dependencies available. NANOLANG_ROOT, NANO_MODULE_PATH and compiler-tool/helper overrides are unset. Builtin module emission sets NANO_CC=/bin/false; a fresh external C-backed provider uses the qualified system compiler and installed capture helper.

| Actual isolated command | Seconds | Exit |
| --- | ---: | ---: |
| actual-make-install | 12.527 | 0 |
| namespace-setup | 0.017 | 0 |
| builtin-compile-module | 0.115 | 0 |
| builtin-verify-module | 0.016 | 0 |
| builtin-execute-module | 0.032 | 0 |
| builtin-compile-native | 0.266 | 0 |
| builtin-execute-native | 0.018 | 0 |
| foreign-compile-module | 0.266 | 0 |
| foreign-verify-module | 0.017 | 0 |
| foreign-execute-module | 0.033 | 0 |
| foreign-compile-native | 0.467 | 0 |
| foreign-execute-native | 0.017 | 0 |
| corrupt-installed-library | 0.165 | 134 |
| missing-installed-library | 0.166 | 134 |
| restored-installed-library | 0.114 | 0 |

Both builtin and foreign programs compile, verify and execute as raw modules and native programs. Their outputs are host-closure-prefix-ok and external-provider-ok. I cover absolute-path and PATH compiler invocation. A changed packaged libfile_source_inputs and a missing hash entry both refuse with134 and preserve an existing output. Restoring the original bytes restores compilation. All six original development library hashes and the separate frozen c908523c3 bootstrap guards match afterward. I do not modify a frozen qualification checkout.

My unchanged original structured-C driver reports2458 passing assertions and zero failures. My complete adapter/deployment Python cohort passes26 methods in144.462s, including original provider cleanup and ABI/owner controls; my actual Make deployment target passes its five methods in6.815s. Digest controls match independent hashlib vectors, including one million bytes and block/padding boundaries. My original provider tests retain their sanitizer and allocation-failure checks. [uninstall.json](uninstall.json) records an actual Make uninstall from a copied prefix; it removes its compiler/owned blobs and preserves the original qualification prefix. The shared-receipt regression retains a blob until its last owner is removed.

My first test run has bytes/text harness mismatches, the first combined original cohort has two setup errors for a missing original comparison helper, and my first Make target names an absent alias. I preserve these failed observations and correct only the harness/setup/target. I build the missing helper from its original Make recipe and retain the actual comparison image. Original temporary test products removed by their own tests are not represented as retained products. The first string-scanning deployment candidate is superseded by the dedicated native section, which also passes an ordinary-string collision control; its generated/native candidate remains archived separately.

My manifest retains5282 logical files,398888093 raw bytes and75106294 unique compressed bytes, with every packed/raw hash verified. It includes complete available source/test inputs, exact tools/products/logs, selected host generations and the actual fresh installation/client/cache. The source base is381001751 plus the retained diff and explicit added-source hashes; the following source commit identifies the final implementation. I keep paired Darwin qualification, a new complete final-head bootstrap, full pure-translator cutover and release acceptance open. Linux success alone does not close #1167 or #976.
