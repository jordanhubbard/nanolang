# I retain three supervisor controls and owned codegen cleanup

I qualify source `cbbfdcb70cb192abaf3f043fe07b7c8e7fa99db2` against base `43deef317e7db6e62c70e9159eef81dcbabf694d` on native Linux ARM64. My four changed inputs are `src/nanovirt/codegen.c`, `src/nanovirt/shadow_runner.c`, `Makefile.gnu`, and `tests/test_shadow_reap_deadline.py`. The separate C-seed/runtime clock repairs are already in this base. I retain 44736 records; manifest SHA256 is `23f16688cb4493c92a62b62f709b34d0d992ca09bf4f8611f653d05478e4d0f9`.

| Cold profile | Native tools | Completion/import/VM methods | Original bytecode methods | Result |
| --- | --- | --- | --- | --- |
| Normal O1 | All five pass | 13/10/6 pass | 42 pass | Component cohorts pass |
| ASan/UBSan/LSan | All five pass | 13/10/6 pass | 42 run, one failure | Complete component acceptance remains open under #1252 |
| Production O3 | All five pass | 13/10/6 pass | 42 pass | Component cohorts pass |

I build all five native images from cold objects in each profile, using the original strict flags and schema generation. I retain their images, selected 3524 Git inputs, source/tool guards, commands, full output streams, and the generated inputs/products that the original tests leave behind. Original fixtures clean some temporary per-case directories; this archive does not recreate those removed files. My new deadline controls retain their inputs, native helper, prior outputs, process observations and full streams.

I reject completion after the parent deadline in all three core supervisors, including when waitpid returns a clean child exit before the next clock check. My Linux regression controller stops only the parent after observing its actual sleep, then releases the child through a file gate. The child's native monotonic completion timestamp is more than the original one-second deadline beyond that stop. I use an explicit int64 foreign result. All three controls pass in each profile; C-seed and NanoVirt prior output bytes survive refusal. The C seed records clean completion/reaping with parent_timeout=true. I retain the original source386 NanoVirt accepting the late child, publishing an NVM module which independently verifies and executes, and the three-method red regression with exactly one NanoVirt failure. These observations establish this defect; they do not establish the cause of #1229 or #1247.

I free my private duplicated extern name/module lookup strings after emission. I also mark only synthesized qualified function aliases as owned in private CG state and free them once. Borrowed AST/module/generic names, closure snapshots and the emitted NVM string pool retain their ownership. I change no public NVM/GC layout, default, serialized metadata or ABI. I preserve the first full SAN bytecode run with43 failures and the extern-only repair with18 failures. After both ownership repairs, the full original cohort reports no LeakSanitizer or use-after-free finding.

My remaining sanitizer failure is the unchanged `test_foreign_cache_requires_readable_hash_record`, damage=directory: the existing loader requests0x8000000000000000 bytes and ASan aborts before the expected foreign-support refusal. The retained libc witness shows both fseek calls succeed, ftell returns LONG_MAX and the descriptor is not a regular file; it does not allocate that size. `src/module_builder.c` remains exact `971b242622730d2581bd5d576bec8dc7ce48a05c67e4917765d4f347f346bb09`, owned by Puck. I use no allocator suppression, relaxed assertion, skipped case or increased deadline. #1252 remains open pending a reviewed owner repair or narrow handoff.

I preserve the prepared source386 bootstrap root without starting any bootstrap stage or publishing a bootstrap receipt. Original stage1800/compiler-suite30/ordinary10 budgets remain. Current Stage2/native-emitter, full-source, paired Darwin and all69 release acceptance remain open. The ordinary full-source600/default10 refusal is not retried here. #1250 remains a separate static service-clock lead, with no service source change or demonstrated service failure. No NAS/TCC/migration operation or Darwin job is part of this archive.

From a repository checkout containing these commits, I verify the chunks, all records, recorded Git source, tool/image identities, negative cases and declared partial result with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/core-supervisor-owned-names-linux-20261124/read_archive.py
```

The reader requires the parent manifests/chunks referenced by this manifest. Integrity PASS is distinct from full sanitizer or release acceptance. I keep #1249, #1251 and #1252 open.
