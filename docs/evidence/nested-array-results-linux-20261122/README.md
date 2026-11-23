# My nested-array result localization

I publish source `57696a367262ab67d56fa212cd9e4bc724ae0c8f` for #1204 on
PR1168. My original 3,080-case ownership-query checkpoint remains in
[my preceding receipt](../ownership-queries-linux-20261122/README.md).
I qualify no compiler repair here.

I reduce the refusal to a nested-array read returned in a record field after
another branch returns an empty-record factory result. My eight-source matrix
varies flat/nested input, direct/sliced output and absent/present factory return.
All eight sources and their shadows pass through C-seed native and both
producer/VM paths. Six cases translate with each producer; their twelve
ASan/UBSan native executions pass alias/copy assertions. The two nested/factory
cases refuse C AOT with both producers: four valid translation failures.
Four wrong-element frontend refusals preserve binary prior module output.

My strengthened matrix also executes the empty-return branch in native main.
I retain the earlier matrix, which exercised that branch only in shadows,
without rewriting its sources or results. Both complete runs retain four
failures. Each run records 92 commands: 84 exit zero, four correctly refuse
wrong types and four refuse valid AOT translations. The unmodified build
helper selects its original 600-second budget; ordinary shadows retain their
default ten seconds. My first coordination claim called a supplied timeout
argument sixty seconds; I corrected that claim because build=True uses the
helper's existing 600-second budget. I increase no deadline.

My read-only debugger traces reach `merge_record_results` → `merge_parameter`
→ `merge_fact` before shape tracking. The existing destination is kind3/ARR;
the incoming field is kind8/VALUE. The reduced direct/sliced cases fail at
function1/offset59 and208 with discover_globals0. The full query fails at
function8/offset545 with discover_globals1. Both have track_shapes0/final0
and canonical_recordsfalse. Slicing is not necessary to reproduce this
coarse return-field conflict. My earlier ARR_SLICE-only lead is incomplete.
The NanoLang producer can encounter the reverse VALUE-versus-ARR order.
These observations locate the refusal; they do not establish a safe widening
or replacement for the later payload and storage checks.

I preserve every failure's prior output and retain inherited debugger ncurses
version warnings separately. I assign no product or timing cause to those
warnings. My five identifier controls pass; the actual lexer checks 3,451
fragments with zero reserved names and retains 135 existing lexer refusals.
All twenty named witness shadows are present and their successful producer
builds execute the ordinary shadow selection.

I leave Puck-owned `nvm2c.c` at SHA256
`12d43399e8b3fcda44964f90830d8aa3b24c637df31f0ce2f151527756372d67`
and the consumer image at SHA256
`c94adddd92eefc081ee5cb28df971a36fe2f7cbe8e9730ec68ca6b1c3d24a7ea`.
I apply and compile no copy/slice caller candidate. My source producer is the
retained native09cab Stage2 image, SHA256
`346dc0d73af5960dea863fa724a0b1845f5613dff10b8661180ed74bf4eb3d83`.
The Linux native toolchain remains GCC13.3/as2.42.

My reader checks 1,992 archive records, 506,461,980 logical bytes, 1,372 Git
source identities, every recorded command stream and product, both matrix
versions, the generated compiler index, input/output guards and all three
traces. The original direct witness is [reproduce.nano](reproduce.nano).

```sh
python3 docs/evidence/nested-array-results-linux-20261122/read_archive.py
```

My integrity status retains four required failures. #1202, the full #1204 query
gate, frame/whole-function CFG, pure C emission, later-source paired bootstrap
and the full 69-requirement release remain open. The strict payload/shape and
original complete corpus requirements remain mandatory after any caller repair.
I preserve frozen roots, the dirty audit/stashes and the NAS/migration/TCC pause.
