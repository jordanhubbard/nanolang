# My inferred scalar copy repair

I publish source `0962d3c9dbf4376b2fac82b8f04f0c24e0ccd8b3` for #1202
on PR1168 under [my exact narrow handoff](https://github.com/jordanhubbard/nanolang/issues/1158#issuecomment-7093099516).
I extend only `sim_push_slot`'s existing inferred-scalar `LOAD_LOCAL` admission
condition to `DUP`/`PICK`. My reader reconstructs the complete new C file by
replacing that one old line. I change no merge function, shape API, fixture,
assertion, corpus selection, warning flag, compiler version or deadline.

All64 original source-copy cases and valid raw PICK0 translate. PICK1 remains
invalid: assembler, verifier and C consumer refuse it, preserving prior output
at their original assertions. The original test deliberately repairs the
container checksum after mutating PICK0's depth, so downstream refusal tests
actual stack depth rather than a corrupt checksum. My reader checks that exact
mutation and the retained binary prior C output. The unchanged inferred-copy
graphs pass312 checks and the original shape unit passes3890 checks, both
normally and under ASan/UBSan. Exact targets, wrong payloads and poisoned-graph
refusals stay strict.

My complete original frame gate now passes every1001 carrier in actual C and
all five source routes:

- C-seed native;
- C-produced NanoISA/VM and sanitized native AOT;
- NanoLang-produced NanoISA/VM and sanitized native AOT.

All six complete reports have SHA256
`f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`.
I retain168386 relation operations,289 normalized flow states,7 FIFO traces
and665 frame cases, including every original invalid join/refinement case.
Both original unittest methods pass with zero failures/errors/skips in422.750s.
My observer retains898 terminal command observations with exactly the three
expected invalid-PICK refusals. It records original arguments/budgets and
complete streams; it changes no test result. Python temporary outputs use one
new private retained directory. Native build helpers keep600 seconds, original
shadows keep their default ten seconds, and route batches retain their original
600-second budgets.

I build the actual C reference from its unchanged source and fresh consumer
providers, excluding the separate affine-bytecode object because the reference
already includes that actual implementation. I substitute no model oracle.
The original route helper instruments generated C with ASan/UBSan and links
the retained `nano_aot_runtime.o`; I do not claim its entire runtime closure
was newly instrumented. I retain the original C and native09cab NanoLang source
producers, rather than claiming a fresh frontend/bootstrap qualification.

My [matched original-consumer control](matched-baseline.json) separately runs
that retained image against the exact newly accepted AND/PICK0 modules. Both
still refuse the exactly constrained Boolean destination and preserve prior C
bytes. I keep these two bounded diagnostics separate from the898-command gate.

My old C source remains SHA256
`12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67`;
my candidate is
`994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0`.
My retained original consumer is
`c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea`;
my candidate consumer is
`007dab36cbffea6c151072155d99d220c2c1aaa3b2ea1847932bba5b00b93d8c`.
Other guarded source/test/producer/VM/runtime images match before/after.
[My post-qualification local tool profile](tool-profile.json) records native
GCC13.3/as2.42 identities; I grant no new generic toolchain or CI admission.

My archive contains4711 records,1384 exact Git inputs and234999896 logical
bytes, reusing hashed objects from my earlier receipts. I retain original and
candidate consumers, both producer modules, every input/report, emitted C,
native products, all command streams and strict negative controls. My reader
checks compressed/raw hashes, Git identities, the exact one-line lease, every
terminal result, all64 cases, the depth/checksum/prior-output guards and every
complete1001-line report against actual C and its independent expectations.

```sh
python3 docs/evidence/scalar-copy-admission-linux-20261122/read_archive.py
```

This qualifies my internal frame/FIFO/relation foundation. I grant no
instruction/function/private/executable authority, whole-CFG convergence,
paired current-source Darwin, pure C emission or69-requirement release
acceptance. #1142 and the separate #1204 query/return-merge blocker remain open;
Puck retains those merge functions and #1190/checkpoint/CI/capture ownership.
I change no Puck checkout, frozen root, dirty audit or stash, and take no
NAS/TCC/migration action. All native observations are terminal.
