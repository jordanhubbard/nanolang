# My rebuilt Stage2 phase observation

I observe my genuine Stage2 image `8d7c240099edbe11cde499a268705f34660c2d940da4895378fc2a2961c886a5`, built from source `771a14330475bcd784d7cb19a215d6a4f15728d4`. My original cold bootstrap, raw Stage1/Stage2 equality, installed no-C-seed check and native36 gate pass in the independently replayed `emitter-record-list-cold-bootstrap-linux-20261126` archive, published at `08c223f945474f459364c24d5e8767e4ede1b0ab`. My original emitter12 target still fails during setup at its unchanged 300-second limit and executes zero test methods.

I use a fresh private root and the existing phase recorder for one separate diagnostic. I retain the unchanged emitter fixture, compiler argv shape, original 300-second limit and ordinary ten-second shadow default. My measured phases are:

- `inputs-start`: 0.026364 seconds
- `imports-done`: 40.222733 seconds
- `lex-done`: 45.045145 seconds
- `parse-done`: 62.937999 seconds
- `checked-emit-start`: 65.199585 seconds
- `typecheck-start`: 65.214924 seconds
- `typecheck-done`: 124.426143 seconds
- `program-lowering-start`: 124.443512 seconds
- `program-lowering-done`: 162.593262 seconds
- `shadow-lowering-start`: 162.614060 seconds
- `shadow-lowering-done`: 229.691808 seconds
- `checked-emit-done`: 229.718383 seconds
- `shadow-assembly-start`: 229.744671 seconds
- `shadow-assembly-done`: 272.196886 seconds
- `shadow-execution-start`: 272.213875 seconds
- `typecheck-start`: 273.437040 seconds
- `typecheck-done`: 273.453669 seconds
- `program-lowering-start`: 273.467717 seconds
- `program-lowering-done`: 273.481209 seconds
- `shadow-lowering-start`: 273.507460 seconds
- `typecheck-start`: 273.523642 seconds
- `typecheck-done`: 273.537438 seconds
- `program-lowering-start`: 273.551669 seconds
- `program-lowering-done`: 273.566533 seconds
- `shadow-lowering-start`: 273.580780 seconds
- `typecheck-start`: 273.595743 seconds
- `typecheck-done`: 273.620935 seconds
- `program-lowering-start`: 273.636497 seconds
- `program-lowering-done`: 273.660815 seconds
- `shadow-lowering-start`: 273.688208 seconds
- `shadow-lowering-done`: 273.715446 seconds
- `typecheck-start`: 273.732521 seconds
- `typecheck-done`: 273.747305 seconds
- `program-lowering-start`: 273.761567 seconds
- `program-lowering-done`: 273.782393 seconds
- `shadow-lowering-start`: 273.796783 seconds
- `shadow-lowering-done`: 273.816493 seconds
- `typecheck-start`: 273.831840 seconds
- `typecheck-done`: 273.846534 seconds
- `program-lowering-start`: 273.860822 seconds
- `program-lowering-done`: 273.874686 seconds
- `shadow-lowering-start`: 273.899015 seconds
- `shadow-lowering-done`: 273.915218 seconds
- `typecheck-start`: 273.930705 seconds
- `typecheck-done`: 273.955658 seconds
- `typecheck-start`: 273.971549 seconds
- `typecheck-done`: 273.992188 seconds
- `program-lowering-start`: 274.006528 seconds
- `program-lowering-done`: 274.031286 seconds
- `shadow-lowering-start`: 274.056786 seconds
- `shadow-lowering-done`: 274.072816 seconds
- `shadow-execution-done`: 274.122998 seconds
- `product-assembly-start`: 274.137517 seconds
- `product-assembly-done`: 279.517145 seconds
- `native-publication-start`: 279.531243 seconds

My controller records exit `-9`, timed out `True`, and 300.419070 seconds including cleanup. I retain complete output streams, 1 temporary-file snapshots totaling 625,720 bytes, and 0 output products. Every snapshot has a stable size and modification time: `True`.

All 3,971 exact Git source inputs, image, host library, tool, producer and observer guards match. My admission records local storage, sufficient free space, CPU service and no other observed Nano compiler jobs. My real adopted-child cleanup control passes; terminal cleanup leaves no owned children. My collector refuses a real nonterminal observation before creating its archive.

My new root, cache state and recorder make these diagnostic timings. I establish no performance comparison, repair, sole cause, native emitter acceptance, paired-platform qualification, pure runtime emission, cutover or 5.1 release acceptance. I keep #1255 and the complete 69-entry release scope open.

Run `python3 read_archive.py` to verify my 5,198 archived records, complete raw/compressed bytes, exact Git inputs, published bootstrap origin, deadlines, phase order, snapshots and cleanup. My manifest SHA-256 is `4bf05eea5d3dc4d046557a9be010e173beba9a0ebf919060eb33f5ba6ff27558`. I preserve all prior failures and roots.
