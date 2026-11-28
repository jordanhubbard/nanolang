# My native emitter-helper phase observation

I retain one #1255 diagnostic for exact3021c729bdb5a204b23de28a3baa48777c0ae199. The original [cold bootstrap/native consumer terminal](../shadow-helper-cold-bootstrap-linux-20261124/README.md) remains unchanged: bootstrap/native36 pass, original emitter setup times out300 seconds before all12 methods.

I copy the actual same native Stage2 image into a new isolated root under its own subject name, verify source/tool/host identities and local CPU/process/storage admission, then invoke the unchanged emitter fixture with the existing phase recorder. I create no Stage2 alias, sentinel or receipt and use no C-seed fallback. Source-root/cwd/TMPDIR/cache differences are explicit; this observation cannot establish historical timing equivalence or cause.

The unchanged300-second deadline is observed at300.068s. Kill/reap finishes at301.680s, exit-9, no product; all selected guards match. Marker offsets are inputs0.025s, imports-done98.253, lex-done209.794, parse-done231.978, checked-emit-start235.919 and typecheck-start235.938. Thus the observed import interval is98.228s, lexer111.541s and parse22.184s. No typecheck-done or shadow-execution marker is recorded. Ordinary shadows still default to10 seconds.

My independent reader verifies4635 records, original-source Git bindings, the actual native producer against its qualified cold receipt, exact argv/recorder path, unchanged300/default10, ordered raw markers, no product and source/image/system-tool/host/recorder guards. Manifest SHA256: `29c6f69ad2cc0cdd71467c00fd49f5e9a45cd97296dc9171b8e8e6bbbb79dece`.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/emitter-native-phase-linux-20261124/read_archive.py
```

Static reading finds a strlen call in every current C char_at helper access. I retain that only as a sampling lead. A separately bounded parent-GDB diagnostic may sample the measured lexer interval for8 seconds before an owned source repair or explicit C-side handoff. I change no source/flags/assertions/deadlines, perf/ptrace settings or product selection here, repeat no full bootstrap/target, and claim no performance gain, full native/current paired/pure-translator/cutover/release acceptance.
