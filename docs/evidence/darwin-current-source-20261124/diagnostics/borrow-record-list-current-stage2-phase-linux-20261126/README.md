# My rebuilt Stage2 phase observation

I observe my genuine Stage2 image `8c2c8660e05fb97a4bf0017c0de2a9b7f3a7b5fc114128583b62f5cb01dd7fa4`, built from source `cfd62a78d5f8afaa6bbd7cc835d75ca97e3caf00`. My original cold bootstrap, raw Stage1/Stage2 equality, installed no-C-seed check and native36 gate pass in the independently replayed `borrow-record-list-cold-bootstrap-linux-20261126` archive, published at `cdcdd3956f2ea632899d964d85383be261a1ea42`. My original emitter12 target still fails during setup at its unchanged 300-second limit and executes zero test methods.

I use a fresh private root and the existing phase recorder for one separate diagnostic. I retain the unchanged emitter fixture, compiler argv shape, original 300-second limit and ordinary ten-second shadow default. My measured phases are:

- `inputs-start`: 0.038631 seconds
- `imports-done`: 56.465766 seconds
- `lex-done`: 63.108881 seconds
- `parse-done`: 91.247704 seconds
- `checked-emit-start`: 95.386018 seconds
- `typecheck-start`: 95.419656 seconds
- `typecheck-done`: 277.944437 seconds
- `program-lowering-start`: 277.972276 seconds

My controller records exit `-9`, timed out `True`, and 301.324745 seconds including cleanup. I retain complete output streams, 272 temporary-file snapshots totaling 18,897,812 bytes, and 0 output products. Every snapshot has a stable size and modification time: `True`.

All 3,971 exact Git source inputs, image, host library, tool, producer and observer guards match. My admission records local storage, sufficient free space, CPU service and no other observed Nano compiler jobs. My real adopted-child cleanup control passes; terminal cleanup leaves no owned children. My collector refuses a real nonterminal observation before creating its archive.

My new root, cache state and recorder make these diagnostic timings. I establish no performance comparison, repair, sole cause, native emitter acceptance, paired-platform qualification, pure runtime emission, cutover or 5.1 release acceptance. I keep #1255 and the complete 69-entry release scope open.

Run `python3 read_archive.py` to verify my 5,656 archived records, complete raw/compressed bytes, exact Git inputs, published bootstrap origin, deadlines, phase order, snapshots and cleanup. My manifest SHA-256 is `e3eda5f75ec6ab26d002ec040e143357d2edcca8323db51cbf37f47cff2e053a`. I preserve all prior failures and roots.
