# I retain old aliases when an array local changes identity

I track `task_26691bbc3a968ae68e97f75a848afec4` under owner-array parent430220
and mixed-value parent4be. I start from canonical `8711402be`, after PR861's
paired mutation and complete unchanged owned-pattern acceptance. This is an
acceptance fixture proposal, not a demonstrated runtime defect or new admission.

My original [owner-array contract](NANOISA_OWNED_FLOAT_ARRAY_FIELDS.md) requires
a copied array alias to survive reassignment of another local and explicitly
names overwrites. The existing private owner runtime grows shared arrays;
its mutation supplement stores the same array back into a local. My origins
query checks replacement without executing it. Samples execution replaces an
ordinary record. Those facts do not demonstrate replacement by a distinct array
while both an outside alias and an affine owner field retain the old identity.

## My bounded graph and observations

I use existing public complete admission and the qualified fixture builder. I
introduce no descriptor, instruction, signature, source syntax or runtime change.
A is a flat FLOAT array initially containing 1.5. A mutable local, an independent
local alias and a Bundle field retain A. Bundle also contains the existing Handle
and STRING fields, with exact owner consumption and ordinary string cleanup.

I create B with distinct contents, then execute existing STORE_LOCAL to replace
only the mutable local's A root with B. I observe B through that local and A
through both the independent alias and owner projection. Mutating A through the
owner field must change the outside alias and leave B unchanged. Mutating B must
leave A unchanged. I include a same-identity LOAD/STORE control after the distinct
replacement to check retain-before-release ordering, without claiming a new case
from the already measured same-array replacement alone.

I destructively unpack and consume Bundle/Handle. The independent A alias stays
usable after the shell and transferred field roots are released. B remains
usable too. Appending beyond the initial capacity through the surviving A alias
must preserve A's identity and B's contents. All finite comparisons use exact
FLOAT values and the already qualified typed comparisons. I keep optional-FLOAT,
source binding-reassignment, owner-field assignment and broader profile refusals
unchanged. This runtime fixture does not authorize a new source `set` form.

I use separate normal and explicit-false-ASSERT graphs. The failure graph reaches
the assertion after replacement and independent observations while the shell,
old aliases and B are still live. It must report the existing assertion status
and clean every root. Distinct integer output markers before replacement,
after independent observations and after consumption identify exact externally
visible prefixes; no output from a failed later operation is invented.

## My failure and root boundary

I reuse the existing heap allocation hooks and finite graph accounting. A VM
allocation sweep covers every positive allocation reached in each graph and
must terminate at an unhit/non-memory outcome within the existing bounded ceiling.
Each hit must produce the exact memory result convention and a valid output
prefix, followed by a successful no-fault recovery. Allocation during creation
of B precedes STORE_LOCAL; it cannot publish a replacement on failure. I retain
normal and assertion outcomes separately from injected allocation failures.

I require empty operand/frame/reference activation state and exact external-root
accounting before collection, then baseline object/byte counts before destroy.
I do not mutate reference counts or use a disposal sweep to mask roots. Native
allocation wrappers account for every acquired/freed shell, string and managed
allocation; generated root-count checks before finish/dispose must retain their
existing exact status behavior. A leak status cannot substitute for the expected
success, assertion or allocation status. Native result sentinels remain unchanged
on refusal, and fault sweeps require a terminal unhit outcome plus recovery.

## My qualification sequence

I first send the entire C fixture, native harness and Python driver for review.
No fresh module or emitted program executes before that review. The graph obtains
fresh public nvm_owned_array_admit and nvm_verify success before emission/execution.
I exercise all four synchronous VM APIs, explicit true-switch and computed-goto
builds, both fusion settings, repeated success/assertion calls and allocation
recovery. I compare emitted C bytes between dispatch builds and run native O0/O2.

I qualify fresh Linux and Darwin ordinary/sanitizer configurations with explicit
compiler/SDK identities, leak detection, immutable source/tool/provider inventories,
complete logs and retained artifacts. I reuse only unchanged checked accounting
helpers; original runtime/source fixtures are not edited or narrowed. A source
compiler bootstrap is not required by this fixture-only checkpoint. Any newly
observed defect stops qualification and requires its own recorded correction.

After this criterion is measured and reviewed, the next installed-product gate
is the complete unchanged `tests/test_affine_selfhost.sh`: one frontend parity
method, twenty contract-boundary methods and twelve owned-pattern methods. I
retain all original source-hash guards, PREFIX/shadows, routes, diagnostics and
output-preservation assertions. That future gate requires its own frozen integrated
setup and first-terminal record. I do not equate this supplement with that gate,
whole-product/fixed-point acceptance, or release publication. Parent closure still
requires reconciliation against all original criteria, not just this new fixture.
