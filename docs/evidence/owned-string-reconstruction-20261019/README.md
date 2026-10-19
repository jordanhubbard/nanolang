# I retain immutable STRING reconstruction qualification

I implement the reviewed design from aggregate1e13109d1 through design
c58059d70/d3b77dff2/97fbde3a4 and measured baseline198970161. The
[baseline receipt](../owned-string-reconstruction-baseline-20261019/README.md)
retains unchanged ordinary/original-shadow admission and previous atomic facts
refusal. This component changes only the owned facts helper and reconstruction
Python, plus tests/docs. Original verification, ownership transfer, field/result
limits and acyclic graph authority remain unchanged.

I admit exact mode-zero STRING parameter slots, immutable parameter/literal
expressions and ordered PRINT/PRINTLN statements in the existing consuming graph.
I do not add STRING to mutable Function.locals or admit STORE_LOCAL to a STRING
parameter. Resource layouts/results keep their previous contract. Borrowed
profiles gain no string/print authority. The existing close_file ephemeral field
observation, ASSERT and owner consumption remain intact.

Only executable PUSH_STR indexes enter the private facts table. I use original
index and exact length with hex bytes, retaining distinct equal-byte indexes and
reusing repeated indexes. Both sides enforce at most4096 distinct references,
actual constant-count bounds and the new reconstruction-only1MiB cumulative
payload cap before decoding/emitting bytes. Advisory pooled strings containing
NUL remain untouched. C uses immutable static byte views with exact fwrite length;
Nano uses the tested literal encoder and directly referenced immutable parameters.
Invalid UTF-8 is a target-specific atomic Nano refusal, not a C byte conversion.

`measured-counts.json` records ordinary generated locals3/5/11 and independent
original-shadow executable locals11/5/11/3 through both retained producers. Exact
estimates match each regenerated function. The ordinary input has three functions,
the original-shadow input four; neither is a recovered source shadow declaration.
The consuming renderer adds no invented helper witness. Exact stdout remains two
lines for ordinary execution and five for the independent original-shadow input.

The actual pre-change false-ASSERT probe in `preflush-failure-results.json` shows
original VM/native exit1 with stdout `before`; reconstructed C aborts(-6) with the
same prefix on Darwin. I do not claim a reproduced Darwin loss. Reviewed new-string-
profile-only fflush(stdout) immediately before the existing abort makes this
prefix contract independent of abort buffering behavior. Successful print paths
and old profiles do not change. Final tests require exact original/regenerated
native exit1, reconstructed C SIGABRT with its assertion diagnostic, exact VM
assertion failure, expected prefix and no later output. They explicitly reject
ASan/UBSan/LSan/runtime-error diagnostics and compile with NDEBUG.

`focused.log` retains the first12-method51.475s terminal. All eight intended STRING
methods passed, but a direct imported TestCase exposed four unrelated file-plan
methods to discovery. Three failed at FileNotFoundError launching absent child
bin/nanoc_c; one C allocation test passed. I fixed only the harness import to a
module alias, preserving the existing process-group runner. This is a retained
selection error, not a STRING product defect or a hidden provider rebuild.

`final.log` passes the corrected nine-method class in54.982s with stronger failure
oracles and actual canonical binary/text roundtrips plus advisory-metadata
stripping. `neighbors.log` passes the subsequently added name-independence control
and four old byte goldens in7.191s. That new control renames non-executable function,
layout and field names, while a shared function-name/PUSH_STR index retains its
exact executable bytes. Facts-derived C/Nano output is unchanged by those names.
`defensive-final.log` passes the final exact-int operand guard in0.238s: arg=True
with args=[1] and a matching index1 row cannot exploit Python bool/int equality.
All ten intended methods are qualified across these runs, not a claimed single
full ten-method invocation.

The corpus includes PRINT without newline, empty strings, tabs/CR/LF, quotes,
backslash-zero, unknown-escape bytes, UTF-8 and control01/7f; exact mixed positional
owner/scalar/string forwarding; repeated executable indexes; before/after-print
assertions; actual1MiB acceptance and over-limit atomic refusal; malformed tables,
STRING slots/results and original-verifier corruptions. Invalid UTF-8 original
VM and reconstructed sanitized C execute exact bytes while Nano refuses without
replacing prior output. The megabyte output is not natively compiled merely to
establish its publication bound.

Four existing borrowed-profile output goldens remain exact. The additional
`consuming-parent-golden.json` pins every imported parent reconstruction helper at
97fbde3a4 and compares both C/Nano bytes for the actual verified existing
source_owned_value_graph fixture; both match. Strict GCC16 and Homebrew Clang
facts syntax checks pass. The only facts build compiles the changed main against
retained consistent Homebrew ASan/UBSan objects, whose hashes/command are retained.
No compiler frontend or bootstrap was rebuilt during implementation qualification.

`commands.json` consolidates797 command records with statuses and raw small-stream
bytes; `artifacts.json` hashes3590 files in the retained run directories. Large
streams and binaries remain at the exact paths in `runs.json`; no binary bundle
is copied here. Twenty-four deduplicated representative Nano/assembly sources are
retained, indexed by `retained-files.json`. The test generator and hashes reproduce
large literal boundaries without another megabyte source copy. `hashes.json` pins
final production/tests, facts tool, runtime and all retained providers.

The corpus used NanoVirt04269be4 and retained nanoc_u8_contextd7c3c136, not fresh
installed Stage1/2. Original-shadow output came from the exact existing driver
source built once by pinned C-hosted nanoc_c3ac71b97 against the baseline import
inventory, with ordinary dependency shadows. Native gates use Homebrew Clang
O0/O2 ASan/UBSan and existing host build-budget overrides; guest deadlines remain
180s. `qualification.json` gives exact targets/environment. Fresh aggregate
seed+both-stage and Linux/platform qualification remain separate; full #999 stays
open. No arbitrary STRING fields, results, locals, operations or borrowed mixing
are credited by this component.
