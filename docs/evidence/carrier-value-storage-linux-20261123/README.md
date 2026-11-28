# I qualify actual carrier storage before source activation

I qualify source `6e8054c87a7b8fe7be9c240eb825c8c3005b730f` under
#1211/#1116/#1140. My new private carrier_values.c/.h adapts actual StructValue,
UnionValue, TupleValue, Array and explicitly typed record-list storage to my
[accounted root engine](../carrier-root-trace-abort-linux-20261123/README.md).
The engine exposes borrowed adapter metadata; its existing ownership policy
and all earlier checks remain. I add the actual storage fixture to ordinary
units. I do not yet link this adapter into evaluator/environment/GC cleanup.

Adoption owns one header and its owned STRING/FUNCTION scalar storage.
Referenced carriers remain separately owned. An iterative walk follows both
registered and valid borrowed legacy containers, including shared/cyclic graphs.
Complete graph transfer gathers first, then removes registered ownership without
copying payloads or adopting caller storage. Failure before transfer or during
marking preserves every owned allocation. Typed lists require explicit producer
facts; ordinary INT bits never become roots. Extra edge facts represent typed
fields whose public storage cannot carry that provenance; replacement preserves
the old facts if allocation fails. Owned lists must be managed GC_TYPE_ARRAY/
ELEM_INT containers. A descriptor does not grant new source/foreign permissions.

My normal and ASan/UBSan full focused Make targets pass against fresh complete
source-provider builds. Both also pass the complete original private engine
and allocator fixtures, including its new known/unknown context checks. My
final corrected adapter is rebuilt separately and linked to those unchanged,
hash-recorded providers in normal, adapter-only-O2 and ASan/UBSan profiles.
All twelve final compile/link/full-run/matched-witness commands pass. All three
full storage runs and matched runs have empty stderr; leak detection stays on.
The optimized profile changes only the adapter object's optimization level;
it is not a full optimized interpreter build or a performance measurement.

My actual-constructor controls check nested record/union/tuple/STRING/callable/
array payloads after caller-source retirement, shared arrays, borrowed cyclic
containers, duplicate/multiple real list roots, typed field facts, ordinary
integer refusal, borrowed lists, complete transfer and caller teardown,
packed/legacy/Unicode/empty names, array STRING/FUNCTION storage and an actual
self-referential array. Five real creation/adoption allocation failures retain
caller payloads, one fact-replacement failure preserves old roots, and eighteen
collection/transfer walk failures each preserve all129 carriers before recovery.
All adapter/engine metadata and runtime object counts balance at teardown.

I retain two development failures without changing existing product behavior:

- My first standalone fixture omits runtime CLI globals g_argc/g_argv and fails
  linking; no storage test executes. I add only the fixture-owned definitions
  used by existing evaluator tests. Complete first source/streams remain.
- My draft adapter delegates tuple cleanup to free_tuple, whose standard
  constructor uses unmanaged strdup STRINGs. An explicitly caller-created
  owned tuple containing a managed STRING produces an actual ASan bad-free.
  I retire its scalars through the adapter's managed/unmanaged policy instead,
  leaving env.c/free_tuple unchanged. The byte-identical witness passes all
  final profiles. I also remove pointer-to-pointer type punning in scratch
  growth before the final normal/O2/sanitizer runs; I assert no historical
  failure or performance cause from that review change.

I verify my source/products/archive with:

```sh
python3 docs/evidence/carrier-value-storage-linux-20261123/read_archive.py
```

My reader passes1,529 records/1,402 exact Git inputs and eight archived native
ARM64 executables: three final storage products, the actual bad-free product,
and both profiles of the complete engine/fault controls. It verifies raw/
compressed hashes, object coverage, exact source scope and Make-only addition,
unchanged existing source/test/runtime/tool inputs, commands, complete streams,
matched-witness preservation, GCC13.3/as2.42 identities and the hash-bound prior
archive. I record all15 development executable identities and227 object/archive
identities; other development executables and provider bytes remain in their
unchanged private roots rather than being recopied. Shared objects refer to
preceding archives by verified paths. This is not generic toolchain admission.

Source constructor/clone registration and binding/evaluation/argument/typed-
field/host/async roots, early-discard coordination and automatic GC/external-list
lifetime handling remain next. The original #1211 checked source UAF and full
source1937/82 and record/union3720/159 LSan failures remain preserved, open and
un-retried. I make no complete source, paired current-head bootstrap, Darwin
or69-requirement release claim. Puck retains #1204/#1190/nvm2c/merge/regex/
capture/checkpoint/CI. All frozen roots/audits/stashes and NAS/TCC/migration
pauses remain intact. Native qualification is terminal; no Darwin reservation
or NAS action.
