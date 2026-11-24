# I qualify complete current reference compiler-source runs on Linux

I qualify source `8949313b0a56c0c3bd52509fb363edb4eefba3e4` on Linux AArch64 after my complete fresh normal/sanitizer/production [provider builds](../fresh-providers-linux-20261124/README.md) and current [private REPL qualification](../fresh-private-repl-linux-20261124/README.md). I preserve the original full `src_nano/nanoc_v06.nano`, dependency selection, all 1,306 shadows, the 600-second outer limit and default ten-second shadow supervisor.

| Reference environment | Full command | Shadows | Native product |
| --- | ---: | --- | --- |
| Ordinary | 159.663s, exit 0 | 1,306 complete, zero failures | Published; `--help` passes |
| Existing explicit private collector | 126.327s, exit 0 | 1,306 complete, zero failures | Published; `--help` passes |

Both children report successful dispatch, report, completion and flush phases. Both supervisors reap exit 0, receive the required completion byte and report no timeout or clock failure. Each run begins after three bounded scheduling rounds with at least one CPU-second per round and at least 75% measured CPU share. I retain actual timings, images, child identities, streams, providers, before/after source/tool guards and complete tracked-source diff checks. These are observations, not a historical timeout explanation or a performance claim.

## I preserve the first missing-tool failure

My first separate root copies the hash-verified complete compiler providers but lacks `bin/nano_vm` and `bin/nvm2c`. Its full command exits 1 after 155.128s. All 1,306 shadows complete; `compiler_tool_root` reports two failed assertions for those missing companions. Every other completed shadow reports zero failures. The supervisor records exit 1 without a timeout or signal, and I publish no seed. I initially misread completion count as assertion success and corrected that report before publishing evidence.

I leave that root unchanged. In a different detached local root I build the actual same-source VM, assembler and C translator through serial original Make recipes. The existing void-local/tail-restart tool test passes all four cases through 20 actual assembler/VM/C/native commands. An extracted fixture retains eight original functions and seven original shadows byte for byte, adds useful root-discovery and entry shadows, and passes all nine before either corrected full-source run. The complete runs retain all original shadows; the extracted fixture does not replace them.

The original copied ordinary reference image remains identified by its qualified hash. Companion Make builds may rebuild timestamp-dependent copied objects. My private reference image links the recorded post-build providers with the previously qualified external factory/shim. I distinguish those actual link inputs rather than claiming the two images differ only in one object. No ordinary environment or production ownership selection changes.

## I check the current repair boundaries

- Both current reference images pass all 18 original native-publication methods, through 32 recorded commands each. I preserve real linker/compiler refusals, prior bytes and permissions, source aliases, symlinks, cleanup, successful replacement and real coverage controls.
- All 17 original public-C modules pass on the current ordinary reference image: 66 methods and 685 recorded commands. I preserve the original source, sanitizer/compiler flags, diagnostic and prior-output assertions, and deadlines.
- Both images emit C for the unchanged signed64 enum fixture and a lexical value that shadows an enum namespace. All 16 strict GCC/supported-Clang18 C99/C11 builds and their 16 native executions pass. Two unknown-member refusals retain prior C output and permissions: 38 commands in total. Existing unsupported-profile and declaration controls remain in the complete public-C suite.
- My reader verifies 25 profile/helper/test components remain byte-identical to the frozen complete source-C checkpoint `c93263c64f6d5dd1db0e856284d76f1c8e0fa131`. The sole public-header difference is the module-cursor ownership comment. This comparison supplements current execution; it does not replace it.

These results complete the scoped #1237, #1240 and #1241 repairs. Earlier failed receipts remain unchanged at their own pins. I retain #1229's older unexplained terminal/deadline records and assign no cause to them. #1121 still needs its broader platform boundary; #1234 retains 104 unrecovered historical provider paths. Those historical gaps do not become current source acceptance.

## I retain an independently checked archive

My reader passes **18,963 records, 3,520 Git inputs and 220 distinct native images**. The manifest deduplicates immutable bytes through the preceding evidence chain. I retain both local roots, actual terminal files, native products, source excerpts, compiler objects, archives, generated files still present at terminal, commands, observers and superseded observer content. Original compiler/test cleanup removes some intermediate generated C, snapshots and intentionally replaced products; I do not claim those deleted bytes are archived. My bounded private-parent library map matches retained library guards, but is not a trace of every dynamic load or a hermetic copy of the operating-system SDK.

```sh
python3 docs/evidence/fresh-compiler-source-linux-20261124/read_archive.py
```

The reader needs this repository and the named Git commits. It can run from any current directory. I preserve the first setup failure, all older roots, the primary dirty audit and stashes. I make no NAS, TCC, migration, cleanup, cutover or service-recovery change.

My ordinary ownership policy remains disabled; the separate current default admission receipt still retains its 168/172/172-byte failures. I claim no current Darwin acceptance, raw Stage1/Stage2 equality, installed no-C-seed acceptance, complete pure-translator acceptance, exact-head CI success or whole-release acceptance. All 69 entries in the 5.1 contract remain in scope.
