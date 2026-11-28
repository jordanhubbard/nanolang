# My lookup-kernel bootstrap and native consumer terminal

I build exact source `95a4062efebe0e97d8cde271117eac1deab21595` in a fresh local Linux ARM64 root. All original tools are rebuilt with strict C99/O3/PIC flags in 103.638990 seconds. I copy no provider, stage, receipt or sentinel. Two scheduling observations and local storage/process checks pass before the original Make bootstrap.

My original `bootstrap3` passes all 17 steps in 1777.004031 seconds with unchanged 1800-second stage and 30-second compiler-suite limits. Ordinary shadow invocations retain 10 seconds. Both native hello routes, final verification, installed Stage2 and execution with the C seed removed pass. Stage1 and Stage2 modules are literally identical: 1060172 bytes, SHA256 `4b3a3a1cb72fb2b22df28f84666f7596a8f5648246bc8ea956e1dd77e2bda703`. My actual native Stage2 is SHA256 `78228e7dbd4485d1b243088a2d25ac5578816e08e1a48d4e0792cd47bf086739`. These observations establish no historical timeout cause or performance gain.

My original native effect/record target passes all 36 methods in 32.044 seconds (Make 32.402636 seconds). The original emitter target then stops during setup: this Stage2 compiling unchanged `tests/selfhost_shadow_emitter.nano` exceeds the original 300-second factory deadline. Make exits 2 in 344.934580 seconds, with zero of 12 methods executed and one setup error. The controller propagates failure. Final source/tool/image guards match; a terminal read-only process scan finds no process whose argv names this root. I preserve complete successful and unsuccessful streams and perform no unchanged full retry.

My reader verifies 6992 records, all 3910 recorded source inputs against exact Git blobs, original tool flags, all 17 step logs, both raw modules, native products and host images, installed/no-C-seed messages, the native36 success and emitter setup failure. Original native test and recipe bytes match the earlier retained baseline. Manifest SHA256: `667a97fd742fd62558ed12eefa3578906acc55f8364131bd0b1f0171feb883d1`.

This C-reference bootstrap does not complete pure translator/cutover #1142, paired current-source acceptance, emitter12 acceptance, whole-runtime sanitizer or all 69 release requirements. #1255 remains open. Puck retains #1252 by explicit user assignment.

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/lookup-kernel-cold-bootstrap-linux-20261125/read_archive.py
```
