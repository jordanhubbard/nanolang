# My bootstrap source traversal checkpoint

I repair [#1201](https://github.com/jordanhubbard/nanolang/issues/1201) after source/evidence7d59803dd3. I prune hidden cache directories before descent, skip hidden and unsupported filenames before file checks, and retain sorted source paths, file aliases, directory-alias exclusion, absent roots and all five suffixes. I change only the collector and its existing test module; Make, compilers, assertions, budgets and native admission stay unchanged.

My three traversal controls retain two failures against the original collector on both Python3.12 and3.15. All three pass after the repair; the original BootstrapBoundaries class is AST-identical. Actual-root selection matches all1342 published paths/hashes except the intentional bootstrap script digest.

My unchanged `make -f Makefile.gnu -j1 CC=/usr/bin/cc test-bootstrap-dependencies` passes five component methods and27 dependency/guard methods with no skips/failures, in541.556s. This includes the actual VM/sanitized native emitter. The ordinary prerequisites rebuild NanoVM, nvm2c and the AOT runtime. I retain both tool identities and my first external observer's consequent guard failure; I do not repeat the passed gate.

With those rebuilt tools recorded, my original guarded seed generation and verification pass in117.633/0.268s at the unchanged1800/30 limits. My second external observer incorrectly expected identical bytes across separate seed receipts. I preserve its failure and both raw seeds: all six host libraries retain identical bytes at new immutable generation paths, and decoded seed differences are limited to those six paths. This is a separately labeled diagnostic comparison. I do not normalize either seed, change the product's raw-stage comparison or claim a raw fixed point. Source/tests/tools, old host artifacts and prior receipts/native-stage products remain unchanged; the new seed and closure are verified and retained.

My original `test-nanovm` target passes569591 assertions, checked heap allocation recovery, and stack overflow/failure/entry-atomicity/recovery controls in38.348s, with unchanged source/tool/seed/product guards. My [previous archive](../builder-source-routes-linux-20261122/README.md) preserves the earlier tool images and wider source-route qualification. Neither receipt qualifies a fresh raw/native stage, paired current-source Darwin, Builder C-adapter admission or the full69-requirement release.

My independent reader verifies1400 records/65821132 logical bytes. I retain all1342 selected inputs:1341 match source commita5fd6696d2bfef515fa8c00a8344330301899ae0, and generated modules/index.json matches its frozen source hash. My first Git-only reader refusal is retained; the corrected reader checks that generated input separately. I check every retained file and rebuilt tool image, both seed hashes and the bounded decoded comparison:

```sh
python3 docs/evidence/bootstrap-source-inputs-linux-20261122/read_archive.py
```

I hold no live native job or Darwin reservation. I leave Puck/frozen roots, capture/regex/runtime/Make ownership, common Git, existing stashes and the migration/TCC operational pause unchanged. I make no NAS cutover, copy, deletion, mount, export, service or recovery action. I claim no historical timeout cause or controlled end-to-end speed gain.
