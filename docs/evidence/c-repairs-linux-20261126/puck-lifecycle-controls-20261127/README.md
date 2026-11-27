# My bounded Puck native-control compatibility check

I execute only the exact observer controls from
5ca56494a9dc8eb84c66da61402343a2a34994f2 while the hosted worker waits.
One tiny C compile passes1.521894s; all three controls and collection finish
3.098422s. I use a private temporary directory, leave every repository alone,
and remove that directory after copying/verifying all18 received files.
My mac control-plane process is running and WebMux HTTP health returns ok
before the check. This is bounded light-duty validation, with no unattended
coding loop or NanoLang compiler/emitter invocation.

My exact C and Python source hashes are
`f91ed11f33e7cd886d31c6c89d7b47b8ed62f167d3f2fe6d40e42cfa7c21cd12`
and `10211a25dbacb409dd06061d55ef4a2b2fdd64d5a3fa5d898f466d1a0a36f9fc`.
Source guards match before/after. I retain the exact workflow source as
provenance, but do not execute that workflow on Puck. The native ARM64 Mach-O
observer SHA-256 is
`b6e550be03005aeb202ff9fb70e9ef94d5b788085fc7b72b42941839a7e58477`.
The recorded build uses clang with std=c11, Wall/Wextra/Werror, O2 and lproc.
Its120s build and5s per-control command bounds remain unchanged.

I observe macOS27.2.1/build26B131, Darwin27.2.0 ARM64, Apple Clang21.0.0 and
`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk`. I retain these observed
identities; this is a separate toolchain context from the pinned hosted gate.

| Control | Marker after origin | Event after marker | Samples/events | Outcome |
| --- | ---: | ---: | ---: | --- |
| Success |25.126ms|1.662ms|4/1|Clean exit, below100ms|
| Late |26.176ms|153.051ms|120/1|Clean exit, at least150ms|
| Blocked |25.321ms|none|78/0|Blocked100.971ms after trigger, owned child killed/reaped|

All original control predicates pass, including armed/marker/trigger order,
identity stability and the full blocked100ms window. I make one attempt.
These are owned-control children. Observation of the real compiler's shadow
PID, actual source/tool/host closure and the original emitter deadline remain
separate. Cached exit_abstime still does not establish completed teardown.

I seal24 files/156541 raw bytes, manifest SHA-256
`a5b89954d8f58634a217bdd4b10223d5196291eeb493548603ae5263d24fd5da`.
My reader verifies every byte, actual Git source, ARM64 binary identity,
transfer/source/service/build records and cleanup receipt. It replays the
original Python controls/assertions from captured native streams with product
execution forbidden; the summaries match byte for byte. An altered manifest
refuses before reading records. The compiler_product_invoked=false field
refers to NanoLang compiler products; the one Clang build is retained.

```sh
python3 docs/evidence/c-repairs-linux-20261126/puck-lifecycle-controls-20261127/read_puck_controls.py \
  --archive docs/evidence/c-repairs-linux-20261126/puck-lifecycle-controls-20261127/archive \
  --repo . --workspace /path/to/mounted/volume
```

My [hosted diagnostic45991837855](https://github.com/jordanhubbard/nanolang/actions/runs/45991837855)
remains authoritative for the restored original C product context. This
receipt accepts native control compatibility only. It neither explains nor
waives the original Darwin emitter failure; original/native/paired/frozen-
assertion/pure-translator/cutover/exact-candidate and all69 release gates
remain required.
