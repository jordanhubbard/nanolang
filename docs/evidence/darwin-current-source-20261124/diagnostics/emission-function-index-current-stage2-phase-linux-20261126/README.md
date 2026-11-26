# My rebuilt Stage2 phase observation

I observe my genuine Stage2 image `7f431c60c2f344f877f031be669ac64d8b17bd198f218c4fe0d15754795a6695`, built from source `8fa1025477ca2b7db9266f1019b7192de20175d1`. My original cold bootstrap, raw Stage1/Stage2 equality, installed no-C-seed check and native36 gate pass in the independently replayed `emission-function-index-cold-bootstrap-linux-20261126` archive, published at `0581ef3a1448b91f88a6d7334ce9ff7a343f4699`. My original emitter12 target still fails during setup at its unchanged 300-second limit and executes zero test methods.

I use a fresh private root and the existing phase recorder for one separate diagnostic. I retain the unchanged emitter fixture, compiler argv shape, original 300-second limit and ordinary ten-second shadow default. My measured phases are:

- `inputs-start`: 0.030536 seconds
- `imports-done`: 57.628690 seconds
- `lex-done`: 64.638643 seconds
- `parse-done`: 94.820669 seconds
- `checked-emit-start`: 98.932592 seconds
- `typecheck-start`: 98.951288 seconds
- `typecheck-done`: 295.691405 seconds
- `program-lowering-start`: 295.713610 seconds

My controller records exit `-9`, timed out `True`, and 301.371792 seconds including cleanup. I retain complete output streams, 0 temporary-file snapshots totaling 0 bytes, and 0 output products. Every snapshot has a stable size and modification time: `True`.

All 3,971 exact Git source inputs, image, host library, tool, producer and observer guards match. My admission records local storage, sufficient free space, CPU service and no other observed Nano compiler jobs. My real adopted-child cleanup control passes; terminal cleanup leaves no owned children. My collector refuses a real nonterminal observation before creating its archive.

My new root, cache state and recorder make these diagnostic timings. I establish no performance comparison, repair, sole cause, native emitter acceptance, paired-platform qualification, pure runtime emission, cutover or 5.1 release acceptance. I keep #1255 and the complete 69-entry release scope open.

Run `python3 read_archive.py` to verify my 4,741 archived records, complete raw/compressed bytes, exact Git inputs, published bootstrap origin, deadlines, phase order, snapshots and cleanup. My manifest SHA-256 is `3476f8d279e25d1333870a893a391073db1fcee1f85bcd7ec24c92722292dd88`. I preserve all prior failures and roots.
