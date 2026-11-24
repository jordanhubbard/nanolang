# I retain an exact-source bootstrap PASS and a failed native gate

I run my original `make bootstrap3` once at source `cbbfdcb70cb192abaf3f043fe07b7c8e7fa99db2` on Linux ARM64. Make exits0 in1858.686 seconds. I keep the original1800-second per-command stage budget,30-second compiler suite and ordinary10-second default; the total Make wall time is not a per-stage budget. All17 original steps pass, both native compilers build and run hello, installation passes and the installed compiler executes with bin/nanoc_c temporarily removed. All3524 selected Git inputs, qualified images and system-tool guards remain unchanged.

| Original step | Wall seconds | Result |
| --- | ---: | --- |
| Seed / verification | 102.086 / 0.265 | PASS |
| Raw Stage1 / verification | 649.853 / 0.215 | PASS |
| Stage1 C translation / native build | 54.622 / 166.112 | PASS |
| Stage1 hello compile / execute | 0.215 / 0.002 | PASS |
| Raw Stage2 / verification | 657.331 / 0.215 | PASS |
| Stage2 C translation / native build | 55.311 / 160.248 | PASS |
| Stage2 hello and final seed/Stage1/Stage2 verification | See original receipt | PASS |

My raw Stage1 and Stage2 modules are byte-identical1049600 bytes, SHA256 `964957dfa356d2b630798c85599e2a83aac514040c6d91434157bd7706563900`. The original receipt pins1353 bootstrap source inputs, six tools and six immutable host libraries, with no unexpected-native marker. I preserve the original receipt, native/raw/C products, actual host/cache inputs, guard/probe logs, commands and streams. Native product translation here uses the C reference translator; the pure NanoLang translator requirement stays open.

After the bootstrap controller exits, my dependent controller runs the original `make test-native-shadows` against this root's own Stage2. All18 methods run in19.222 seconds; five FAIL and Make exits2. Two valid shadow-only cases report missing main, the source-only case lacks its required notice, and two foreign-process cases lack the expected stderr diagnostic. I do not infer their individual status or runtime cause from the original assertion text. I track them under #1253 and retain [separate bounded localization](../native-shadow-localization-linux-20261124/README.md). The original twelve-method emitter companion is NOT RUN after that first failure. The dependent controller returns failure even though its inner Python test controller exits0 after recording the actual Make failure.

The original fixtures remove some temporary per-case data. I retain their exact source and full output, not reconstructed deleted files. The separate localization observer preserves the five methods' actual test-owned inputs and statuses before cleanup. I also preserve a stale intermediate observer template that incorrectly described progress as live after the run had ended, and its correction. The original bootstrap, native test journals and all native bytes remain unchanged.

This archive contains6490 records. Manifest SHA256 is `dc9080dac206655d64267f2e3755fcd463376d83ffdc35d359225593c36421db`. I reuse verified chunks from earlier receipts; referenced parent manifests/chunks are required. I verify integrity, recorded Git source, all17 actual step logs, raw equality, host/tool/product identities, installation, original18/FAIL5 and the unrun companion with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/core-supervisor-bootstrap-linux-20261124/read_archive.py
```

Bootstrap PASS does not close the failed native gate, #1252 sanitizer cache failure, ordinary600/default10 compiler-source refusal, exact paired Darwin, source-root-independent installation, pure translator cutover or whole69-entry release acceptance. I make no unchanged full retry, historical timeout-cause or performance-gain claim. All jobs in this root are terminal; no Darwin or NAS/TCC/migration action occurs here.
