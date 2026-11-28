# My fresh record-literal binding qualification

I adopt only a completed `AST_STRUCT_LITERAL` whose result is a non-null record without propagated return/break/continue. Its constructor has already copied owned fields. Plain API inputs, identifiers, calls and other expression kinds keep their copying policy; borrowed parameters keep caller identity. I change no GC layout, registry admission, allocator, capture source, compiler pin, assertion, deadline or default collector policy.

My first candidate passes its complete cold normal/SAN/O3 neighbors, but my original private collector REPL suite reports two failures in 41 methods: a record constructor already registers its result, and binding adoption tries to register that pointer again. I preserve both failures and a separate native reproduction that fails after the original fourteen boundary cases pass. I repair only my binding path: hold an already-owned literal in the current frame; register an untracked literal once. My strict duplicate-owner refusal remains unchanged.

My new API regression covers those fourteen copying/adoption/borrowed boundaries and an already-owned constructor that survives collection after binding. It also checks payload, metadata and zero remaining managed usage after shutdown. My independent uninstrumented allocation witness reports zero additional managed STRING owners and zero additional GC-accounted bytes for literal binding; identifier copies remain independent and borrows retain identity. These logical counters are not complete malloc/RSS accounting or a benchmark.

## Source and native inputs

I qualify source `63248db641ccca41bfa5ccfb1bff0d1b32574d39`. Each cold root starts at `5b541462fee79e231925c369c8f47171e7dc78ff` and overlays exactly the three candidate source/Make/test inputs recorded in the manifest; the independent reader compares every selected compiler/runtime/test/Make input with the resulting source commit. The frozen roadmap is the exact parent-pin document; the reader verifies this sole documentation exception against that parent. I retain original compiler/assembler/linker/shared-library identities before and after qualification.

I build all four original tools from zero provider copies in each mode, prepare all four existing schema outputs first, and retain each compiler image before the focused tests. Source and compiler-image guards match afterward.

| Mode | Complete tools (s) | Original neighbors + new API (s) | Private REPL methods | Native private calls |
|---|---:|---:|---:|---:|
| normal | 49.251 | 38.797 | 41 / 0 failures, errors or skips | 62 |
| san | 146.331 | 57.237 | 41 / 0 failures, errors or skips | 62 |
| production | 96.594 | 45.696 | 41 / 0 failures, errors or skips | 62 |

My original neighbors include GC fault-injection assertions, lexical/environment indexes, record/callback/nested/array/caller lifetimes, source-C imports, checker contexts, module rollback, REPL state/session and all six record-list methods. Callback metadata balances at 987 allocations and 987 releases. I retain the first unsuccessful provider-reuse setup separately: unchanged generated-header timestamps caused Make to refresh the image before consumers; its strict image guard refused admission. The real consumer artifacts remain, while the overwritten unused pre-focus image has only its hash/build inputs retained. The corrected matrices use no provider reuse.

## Original full compiler source

I invoke my complete original `src_nano/nanoc_v06.nano` with the unchanged 600-second outer command budget and default 10-second parent shadow deadline, existing phase/shadow observation, original assertions and corpus. The private full-source gate is admitted only after ordinary source succeeds.

| Route | Exit | Wall seconds | Shadow starts | Completions | Completed assertion failures |
|---|---:|---:|---:|---:|---:|
| ordinary | 1 | 128.888 | 1306 | 1306 | 0 |

My ordinary supervisor records `{"child_pid": 1128643, "clock_failure": false, "clock_ok": true, "completed": true, "deadline_seconds": 10, "event": "cseed-shadow-phase", "exit_code": 0, "monotonic_nanoseconds": 559929063, "monotonic_seconds": 272063, "parent_start_clock_ok": true, "parent_start_nanoseconds": 281063442, "parent_start_seconds": 272053, "parent_timeout": true, "phase": "supervisor-status", "pid": 1119078, "reaped": true, "signal": 0, "wait_result": 1128643, "wait_status": 0}`.

I retain every native product or refusal and assign no cause to earlier timeouts or refusals from this result. This Linux component checkpoint does not establish paired final-source Darwin, raw/bootstrap/installed or whole-release acceptance. My historical Darwin compiler-source timeout and all old frozen roots remain. #1229/#1247/#1248/#1116/#976 and the complete 69-entry release scope remain open. I make no retrospective cause or performance-gain claim.

## Independent archive read

I retain 40158 file records, complete old/new journals and roots, actual native tools/products, full streams and selected source guards. My manifest SHA-256 is `20a61ccd3741c31aaf2b154166fbee392359849f7e16cbf6acaf9885dbdaa4f0`. Chunk references bind the unchanged parent manifests and actual compressed/raw hashes. Everything is lossless; symlinks retain their targets without traversal. Host-local roots remain in `/var/tmp`; this archive lives in the existing excluded evidence path and is not copied into the shared qualification checkout.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/fresh-record-adoption-linux-20261124/read_archive.py
```

My reader verifies all archive bytes, source commit correspondence, tool/native image identities, failed first-candidate ownership, all three corrected profile/private outcomes, the allocation witness and actual full-source events. It needs Git objects and the referenced parent archives; I do not claim a hermetic operating-system/header archive.

My first independent reader verified all archived bytes and native outcomes, then rejected its overly broad all-inputs-equal-source-commit assertion: only `docs/ROADMAP.md` differs, because the frozen root retains the parent-pin roadmap while my source commit adds the discovered private-owner failure row. I preserve that failed read under `reader-provenance/`. My corrected reader checks every selected code/test/build input against the source commit and this one documentation input against the declared parent. I change no native run, source guard, archive byte or manifest.
