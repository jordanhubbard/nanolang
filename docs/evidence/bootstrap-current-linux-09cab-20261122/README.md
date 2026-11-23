# My current-source Linux bootstrap

I qualify exact09cab2184f6c889ea8059cf7c5ec29baf456301e under [#1142](https://github.com/jordanhubbard/nanolang/issues/1142) and [#976](https://github.com/jordanhubbard/nanolang/issues/976). I run the original `make -f Makefile.gnu -j1 CC=/usr/bin/cc bootstrap3` in my owned Sparky worktree, with supported TMPDIR scratch isolation and the existing1800-second stage/30-second complete-compiler shadow limits. I begin with no Stage1/2 products, bootstrap receipt or Stage1/2/3 bootstrap sentinels. I change no source while the command runs.

I finish with Make status0 in2059.715 seconds. My raw Stage1 and Stage2 are literally byte-identical:1049504 bytes, SHA256 `33de2785c7a66379ba75212020d78eb80fe1269f388985d06860cc80b9b9f964`. I retain both independently generated files and all17 successful harness commands.

| My original phase | Seconds |
| --- | ---: |
| seed | 122.933 |
| stage1 | 718.394 |
| stage1-translate | 61.542 |
| stage1-native | 176.294 |
| stage2 | 712.870 |
| stage2-translate | 55.307 |
| stage2-native | 182.625 |

I verify both native ELF64/AArch64 compiler products, their original hello compilation/execution and exact `Hello from NanoLang!` output. My final three raw-module verification commands pass. Make installs bin/nanoc as the Stage2 symlink, compiles/runs the installed hello, hides bin/nanoc_c for the original no-C-seed gate, then restores the same C-seed bytes. All four bootstrap sentinels exist at terminal.

I retain1342 unchanged source inputs, six prepared tools, six immutable host-library paths/hashes and five recorded bootstrap products. The original prerequisites relink nanoisa before the harness freezes its prepared tool identities; my pre-Make and prepared identities are both retained. My guard, receipt and raw-module import observations pass after terminal, and no unexpected-native marker exists. Source/native files are actual independent products; I introduce no stage wrapper or artifact alias. Native Stage1 and Stage2 image hashes differ; my required equality applies to raw modules.

My independent reader verifies1562 logical records/266178081 bytes and reuses exact hashed objects from my two declared previous archives. I match1341 selected inputs plus the extra hello/no-C-seed script against the source commit, and retain generated modules/index.json with its frozen hash. Of592 permitted host-cache filenames,151 are present and captured;441 are explicitly absent. That guard allowlist is distinct from an inventory of consumed files. I retain all declared sources, tools, host libraries, products, logs, process observations, guard configuration, present cache files and native sources in this receipt.

```sh
python3 docs/evidence/bootstrap-current-linux-09cab-20261122/read_archive.py
```

I also retain the preceding [collector/VM qualification](../bootstrap-source-inputs-linux-20261122/README.md): original32-method dependency gate and569591 NanoVM assertions plus heap/stack recovery. Its source/tool identities remain revision-specific. I retain raw process observations without assigning a historical timeout cause or claiming a controlled performance gain. My reader also refuses three duplicate-object descriptor corruptions (size, compressed hash and object path); I retain the controls in verification/check-descriptors.py and descriptor-controls.json.

This is my reference-C bootstrap: the translator used here is still my C implementation. My required NanoLang translator, complete pinned source/product census, current-source Darwin and installed-prefix relocation, callable/sanitizer obligations and full69-requirement release acceptance remain open. Older paired receipts establish their own pins. I keep this result separate from pending Builder adapter admission and Puck's capture/CI integration.

I release my Linux native worker at terminal and hold no Darwin reservation. I leave Puck/frozen roots, existing stashes and the primary dirty audit untouched. I perform no migration transport/cutover/source deletion, TCC, mount/export/service or recovery action.
