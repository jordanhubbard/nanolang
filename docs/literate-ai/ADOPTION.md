# My Literate AI adoption record

I track LITAI-001 in [#1039](https://github.com/jordanhubbard/nanolang/issues/1039).
My checkpoint starts at `37dfd59df4b30ce331f6111a56258e151301d108`, the pushed #1038 correction
stacked above #995 at `b7c20ea901a930d211612ec9d6cc4bdda9d8bcdb`. I preserve those reviews and native gates.

## Inventory and migration choice

I found no canonical project definition, current Component locks or current Literate AI test receipt.
My historical `.mac/project.yaml` is not current authority. My 42 receipt-named files are historical evidence.
My original Git tree contains 89,977 entries; `scripts/litai_source_preservation.py` checks path, mode and
Git object identity against the exact baseline, with five explicitly reviewed metadata edits allowed.
My [preservation report](source-preservation.json) records this initial check.
Its longest-prefix owners account for every original entry and reject equally specific conflicting owners.
This is preservation ownership, not a complete native producer graph or SHA256 source SBOM.

I use `litai project adopt-original` after validated canonical metadata and a clean committed checkpoint,
as documented in Literate AI `docs/user/framework-flow.md` under retained whole projects. This protocol
binds original source in place; it neither quarantines source nor executes builds or models. Original-stage
adoption has no automatic successor in the legacy conversion chain. Retained manifest/profile qualification
and any regenerative cutover remain separate. I do not hand-write the framework's conversion evidence.

I retained a read-only `onboard adopt` inspection. It reported readiness but proposed `npm test`, `npm start`
and `npm pack` from a dependency-only root package.json, plus source quarantine. Those are not my native
compiler lifecycle. I did not apply the plan. LITAI-001.7 records the inspector defect for upstream follow-up.

## Native lifecycle and custody

I preserve GNUmakefile -> Makefile.gnu and the BSD Makefile bridge. My real entry points include `make build`,
`make test`, `make test-quick`, `make test-one-ir-compiler`, bootstrap scripts and `make install PREFIX=...`.
I must run install parity with a fresh PREFIX, not mistake DESTDIR or VSCode package-json for native packaging.
I retain `src/`, `src_nano/`, `modules/`, `std/`, `stdlib/`, `tests/`, `formal/`, `scripts/`, bin/lib/obj and
bootstrap checkpoints. Bootstrap guards bind absolute root and shadow budgets. `make test` temporarily
changes bin/nanoc; its output claim must be serialized against bootstrap/install. I do not copy an output
checkpoint to a new absolute root and call it valid. I never invoke the publication script during adoption.

My eight canonical Components define integration behaviors, public retention constraints and acceptance
obligations. They do not select new implementation languages or pretend to resolve native source units.
No Flavor is invented for metadata validation. Actual native platform/toolchain bindings, mixed C/Nano units,
generated headers, ABI, typed dependencies and output claims remain required before retained execution.
The review workflow is declarative and unconfigured; I have no executable lifecycle binding or provider group.

My new documentation catalog is `docs/literate-ai/`. I keep the large existing docs/evidence tree as retained
original source and the release evidence index; its files are not silently reclassified as newly reviewed
Literate AI documents. Original source retention includes them. I preserve existing agent instructions.

## Evidence and remaining gates

My installed metadata CLI reports Literate AI 1.4.0 at `26289fd2cbcb1ca2fb15bb02119912e8638ec881`,
with publication unverified. I must pin and qualify the execution runtime before native admission.
My cheap baseline passes the binary64 embedding check and 15 existing fixture controls. The default Python
schema import failed for missing PyYAML; I retained that failure and passed the 33 schema controls using the
existing Python 3.13/PyYAML environment. Original selected files, tools and bootstrap sentinels stayed unchanged.
These checks did not run compiler bootstrap, installation or native lifecycle acceptance.

I have no current verifier-owned native test receipt. `literate/verification/current.json` is intentionally
absent until real test policy/actions, case reports and independent acceptance produce one. Missing evidence
must remain visible; metadata, locks or an original adoption record cannot establish native conversion.
Compiler-contract, sanitizer, retirement, exact Linux/Darwin and all original 67 scope rows remain in force,
with LITAI-001 as my 68th headline requirement. I have not authorized release publication or authority cutover.

## My authority review

I reviewed my eight integration contracts, their direct public interface dependencies, acceptance obligations,
original-source skill, workflow/routing, explicit root lineage and preservation limits. I retain original source
as release authority. This review covers metadata consistency; it is not a compiler acceptance or test receipt.

<!-- literate-ai:authority-reviewed sha256:3e49e63ba4ae024561038c41796f42e1aba8695209d2f49526f594c4629d818f -->

## My first metadata checkpoint

I validated the canonical project metadata and all four agent delegates. I resolved and checked all eight
Component locks without a default language Flavor. I initialized explicit root repository lineage through
the supported reparent plan/apply adapter; I did not infer or copy a framework parent repository.
My custody fixture controls exercise real Git content edits, executable-mode changes, deletion and explicitly
reviewed agent metadata changes. My original native implementation and entry points remain unchanged.
My test receipt reports `policy-unconfigured`; I retain that unqualified state.

## My original-source binding, 2026-10-28 UTC

I applied reviewed plan `sha256:81153da7345b39874aede2f6dd75bbba5dfaae30ba9bf6a4afef23b02234f6de`
to clean committed metadata/source checkpoint `dd3af8ec882d7359e5d36a91d692e41a2a8bbb5b`.
The framework wrote only `.literate/original-adoption.json` and conversion-authority metadata.
It reports `admitted: false`, `execution_qualified: false`, stage `original`, `source_moved: false`
and release authority `original-source`. My post-apply check preserves all 89,977 original entries,
and metadata validation remains successful. I did not build, move source or contact a model during apply.

My explicit `project test-receipt require-current` negative control exits 2 with
`project.test_receipt_policy_unconfigured`: no receipt can be accepted or current. I retain this expected
refusal until LITAI-001.3/.4 configure actual native unit/lifecycle and case/report policy. My custody workflow
adds only the real Git fixture and original source preservation gate, independently of the existing native CI.

At my fresh #995 check the integration head is still `b7c20ea901a930d211612ec9d6cc4bdda9d8bcdb`.
Xcode compiler-contract qualification and Memory Sanitizers have failed; Linux compiler-contract and Xcode
retirement jobs are running. #1038's triggered proof/security checks pass, but those do not establish fresh
Linux/Xcode compiler or sanitizer acceptance after integration. I leave these runs and required gates intact.

## My independent review correction

My first checker exempted the five reviewed metadata files too broadly: an isolated Git fixture proved that
deleting an exempted file still reported preservation. I corrected this before treating custody qualification
as complete. Reviewed edits now permit content changes only; original path, Git type/mode and actual regular
file/executable mode must remain intact. New dirty and committed deletion, symlink and mode controls all refuse,
while reviewed content edits pass. My unchanged 89,977-entry source check passes with the corrected guard.
The review also corrected the documented native command to `make test-quick`; I changed no native entry point.
All 67 original scope bullet rows remain byte-identical.

## My integration checkpoint, 2026-11-07

I reconcile my three original metadata-adoption commits with current main `82c0f4f8097bfe41f4fc64f8f887865a43d85fa4`.
PR #995 was squash-merged, so inherited pre-merge ancestry conflicts with later compiler repairs.
I retain current main's entire compiler/runtime/test/build/workflow content and apply only the original
adoption metadata delta. My refreshed custody baseline is the explicit main commit above; my original
37dfd59df baseline report and framework adoption binding remain historical evidence, unchanged.
This refresh does not assert byte identity between the old and new baselines or admit native execution.
I require a new committed-tree custody report and real Git refusal controls at this integration.

My [integration custody report](source-preservation-integration-20261107.json) at `82e958892`
preserves all 89,977 current-main entries with no changed or dirty original paths. Both real Git fixture
methods pass, all eight Component lock checks pass, and all four agent delegates are current.
I reviewed the unchanged eight integration contracts, interfaces, acceptance obligations, original-source
skill and unconfigured workflow/routing; I recorded the current documentation marker through the supported
CLI. Canonical validation passes. I have no native lifecycle receipt or execution admission.

## My combined repair checkpoint

I integrate this metadata after the completed code repairs in PR #1068. My new
custody baseline is `bb60655cd51cbcd1f2737faba4c8905703e3bb7b`, retaining both earlier baseline
reports as history. The metadata merge changes no original compiler/runtime,
Make rule or existing script bytes from that combined baseline. My exact
committed-tree custody report and current authority/lock checks follow below.
Original source remains release authority; native lifecycle admission is still
missing under LITAI-001.3 onward.

[My first combined custody check](combined-custody-first-refusal.json) correctly
refuses an edit to the existing component evidence list. I restore that
baseline file byte-for-byte and record [the added metadata component](combined-adoption-component.json)
in a new artifact. I do not broaden preservation exemptions to obtain acceptance.

My [committed combined custody report](source-preservation-combined-20261107.json)
at `126471404` preserves all 90,002 baseline entries with no changed or dirty
original paths. Both real Git custody methods, all eight lock checks and all
four agent delegates pass. Canonical validation and the supported documentation
authority review pass. These are metadata/custody observations, not native
lifecycle or full 5.1 release acceptance.

## My nested-union code checkpoint

I integrate native storage repair #1061 from code-only commit `4bf010dcc2cb8034e05391aea46bd2428c6e17a8`.
My custody baseline advances to that reviewed native-source checkpoint,
including its local regression evidence in
`docs/evidence/nested-union-native-20261107/README.md`.
I preserve the previous combined baseline and reports as history. This is an
explicit original-source update under my existing authority, not generated
replacement source or retained lifecycle admission. My metadata merge must
preserve every original entry from this new checkpoint.

My [committed nested-union custody report](source-preservation-nested-union-20261107.json)
at `9b0bb858b` preserves all 90,011 baseline entries, with no changed or dirty
original paths. Both real Git custody methods, all eight Component locks, all
four delegates and canonical validation pass. Native retained execution and
fresh hosted compiler qualification remain separate open requirements.


## My long-path and compiler-gate code checkpoint

I integrate reviewed code-only checkpoint `1293dce61adb2661903339e84a0272a227dcb812`
with a real merge that retains my component-review ancestry. It includes
line-bounded assembler import names, literal fixture identifier checks,
long Darwin dependency-record capture, logical shell-path/source-hash bounds
and actual private-union/glibc compiler regression controls. I retain local
native and sanitizer checks, the complete fresh 316-byte-path Stage 1 bootstrap
and measured compiler-profile limits under `docs/evidence/` as original source.

My custody baseline advances only to this reviewed original-source checkpoint.
I keep the nested-union baseline and every earlier report as history. I do not
change preservation exemptions or treat original-source updates as generated
source, a retained native receipt or full 5.1 acceptance. My broader Python/C
profile audit and runtime, exact hosted native gates and GitGuardian #1076
remain open. Auto-merge remains disabled while those release requirements are
unresolved. My committed-tree custody report follows after current checks.

My [repair metadata checks](repair-metadata-checks-20261107.json) record both real Git custody methods, all eight current Component locks, all four delegates and canonical validation after supported authority review. These checks preserve 90,073 reviewed baseline entries and do not qualify native lifecycle execution.

My [committed repair custody report](source-preservation-repairs-20261107.json)
at `3556ed251` preserves all 90,073 reviewed original entries, with no changed
or dirty original paths. I check this metadata with installed Literate AI 1.4.0
at `10e872cfae2465e831901108d50e62c2ff9dfb78`, publication unverified.
Both real Git methods, all eight locks and all four delegates pass. Canonical
validation passes after the supported authority-review update. I keep the
full profile audit/runtime, exact hosted native gates, GitGuardian findings and
retained native lifecycle admission visibly open.


## My compiler profile and binding checkpoint

I prepare code-only `ea5e28e89af9df817d74d2ba9cf5087227f7ee36` as a separate candidate after integration `435dae34c`. I retain every component ancestor, the complete adoption queue and every earlier custody baseline/report. My reviewed original-source update contains preserved file-valued compiler options, actual private managed-runtime IR inputs, writable coverage outputs and explicit Fortify reset, plus shared binding/neighbor recipes and real corpus preparation. I preserve my existing five content-only metadata exemptions without widening them.

My corrected hosted audit passes 1,483 compiles per compiler at `7db768cc6`, with exact identities and outcomes under `docs/evidence/compiler-profile-hosted-20261107/`. My selected binding slice passes eight native tests and 114 ordinary/ASan/UBSan diagnostic compiles per compiler at `55b552dd2`, under `docs/evidence/binding-compile-profiles-20261107/`. I distinguish native sanitizer execution from sanitizer diagnostic compilation. Other C profiles and the serialized default budget remain open; I retain source authority and no native lifecycle receipt.

My [profile metadata checks](profile-metadata-checks-20261107.json) record the current real Git controls, all eight locks, four delegates, supported authority review and canonical validation for this candidate. This report does not replace PR #1068's still-running exact-head native/qualification gates. GitGuardian #1076, main integration, native adoption and full 5.1 release acceptance remain open. I do not move the live PR head or enable auto-merge from this candidate preparation.

My [committed profile custody report](source-preservation-profiles-20261107.json) at `35b4cad936dca765101ade5fd530a143a520e903` preserves all 90,135 reviewed original entries, with no changed or dirty original paths. My metadata checks pass with installed Literate AI 1.4.0 at `10e872cfae2465e831901108d50e62c2ff9dfb78`, publication unverified. This separate candidate preserves original-source authority and has no native lifecycle or release admission.


## My File profiles, policy and qualification-storage checkpoint

I merge reviewed code-only `f0b99facb4b2415761c980f21e69d644326b3825` into my
separate custody candidate with real ancestry. I preserve the complete adoption
queue, every prior baseline/report and the existing five content-only metadata
exemptions. Original source remains release authority.

My source update includes shared actual File hosted/private-VM compile recipes,
the corrected generated-Wasm execution recorder, the opt-in combined profile
budget measurement, additive read-only merge-policy controls, complete-PR change
classification, and isolated qualification temporary storage with verified
successful archives. I retain the complete module lists and native assertions.
The File slice passes eight native tests and 130 diagnostic compiles per compiler;
the Wasm module passes all 14 original tests. My combined 1,517-second budget
measurement times out before GCC and final bootstrap controls finish. The
20-minute budget and broader profile inventory remain unmet.

My storage/classification/policy controls pass 26 local tests. The exact-source
Xcode storage recovery is still pending under #1079. Three Linux qualification
legs pass on the earlier `435dae34c` content tree, with the actual merge checkout
and unchanged source/provider evidence retained under
`docs/evidence/hosted-linux-qualification-20261107/`. Neither those earlier native
results nor these metadata controls establish a native pass for this candidate.
I do not update the still-running PR #1068 head, apply the proposed admin ruleset,
enable auto-merge, manufacture a lifecycle receipt or publish a release.

My [current metadata checks](qualification-metadata-checks-20261107.json) record the real Git,
eight lock, four delegate, supported authority-review and canonical checks. My
committed-tree custody report follows after the metadata checkpoint. My current
check preserves all 90,248 baseline entries, with no changed or dirty original
paths. All eight locks and four delegates pass with installed Literate AI 1.4.0
at `10e872cfae2465e831901108d50e62c2ff9dfb78`, publication unverified.
GitGuardian #1076, exact integration gates and native adoption remain open.

My [committed qualification-code custody report](source-preservation-qualification-20261107.json)
at `2ecd38eab130f4a307d3e1336f269f78e8e01e38` preserves all 90,248 baseline entries with no changed or dirty
original paths. I retain the earlier profile baseline and reports unchanged.
My metadata controls establish source custody, not native lifecycle admission or
a passing exact-candidate compiler/qualification run.

## My original source-contract checkpoint, 2026-11-08

I reconcile parent metadata `7c4c015b4` into the source-contract review branch.
My explicit reviewed original-source baseline is
`043b52d1ade7023b18917737a0d299b6c431ee98`. I preserve all previous reports and
all five existing content-only metadata exemptions. I verify that no original
f0b path, Git type or mode was lost during the source-contract work. The metadata
merge changes none of that reviewed checkpoint's compiler, runtime, Nano sources,
module, Make, test or existing script bytes.

I retain the [first old-baseline refusal](source-contract-first-refusal-20261108.json)
before advancing this explicit baseline. Its changed original content is the
intentional source-contract repair work tracked through #1111, including exact
BSTRING identity, U8 storage, environment contracts and GC correctness. This is
source preservation ownership, not a claim of native acceptance for that code.

My current full source-creation attempts still hit their ten-second supervisor.
The controlled Linux comparison publishes six native compilers with identical
complete shadow selections but does not establish a consistent allocation
speedup or reproduce the full component prelude. Apple comparison and exact
integration remain open. My fresh original corpus remains 214/237 in each
stage, with all 23 failures retained under #1081. I preserve those failures,
original budgets, GitGuardian #1076, current native/platform gates and every
release obligation.

I retain original-source release authority, the historical framework adoption
binding, empty native implementation selections and the absent native lifecycle
receipt. The eight integration Components and four delegates remain metadata;
they do not complete the native producer graph or authorize native cutover.

My [committed source-contract custody report](source-preservation-source-contracts-20261108.json)
at `e07772c724ee08ccbf3cf5ec094f31baa853e2f0` preserves all 92,816 reviewed entries,
with no changed or dirty original paths. My [metadata checks](source-contract-metadata-checks-20261108.json)
retain the real Git controls, all eight current integration locks, four current
delegates, supported authority review and canonical validation. The native receipt
guard still refuses `project.test_receipt_policy_unconfigured`, as required.
I preserve every inherited component commit and all 67 original scope rows.
These facts establish Git custody and initial integration metadata, not native
execution admission or release qualification.

My [merge-tree audit](source-contract-merge-tree-check-20261108.json) checks every
reviewed pre-merge entry against the staged merge tree, permitting only the five
prior content-only metadata exemptions. I retain unchanged path/type/mode checks
and the exact staged tree identity separately from native lifecycle admission.

I freeze my catalog reports before the final documentation-authority review.
I retain post-freeze validation logs outside this catalog, under
`docs/evidence/source-contract-custody-20261108/`, so saving those logs cannot
change the authority inputs they just checked.


## My GC-entry retirement checkpoint, 2026-11-08

I retire private GC-entry tail storage because my fixed Linux and Apple
comparisons do not establish a reliable source-creation benefit. I preserve
all candidate measurements and retain both allocation-overflow corrections
and shutdown registry retirement. My local sanitizer receipt in
`docs/evidence/gc-pointer-entry-20261108/retired/` passes all thirteen GC controls and the existing interpreted BSTRING and
STRING bounds gates; hosted source/platform acceptance remains open.

My reviewed original-source baseline is `91735dabca36439915c02f58848bd1197bb1ad90`.
This is a real code-only descendant of my previous `043b52d1a` baseline,
changing exactly `src/runtime/gc.c` and `tests/test_gc_struct.c` to their
`9baeba459` correctness-checkpoint bytes. I merge it with real ancestry into
my source-contract branch. Its other 92,814 reviewed entries remain unchanged.
I retain my five content-only metadata exceptions without widening them.
This checkpoint contains the complete original source tree; my temporary
sparse working view is not an omitted-input qualification.

I retain custody and post-review logs outside my documentation catalog under
`docs/evidence/gc-pointer-entry-20261108/retired/`. Original-source authority,
empty native implementation selections and the absent native lifecycle
receipt remain unchanged. The original source-creation failures, full corpus,
compiler budget, GitGuardian and release requirements remain open.


## My scalar async compatibility source checkpoint

I retain the original synchronous scalar async contract through my source lexer
and parser. I keep async declaration identity and existing affine/effect
provenance guards. My current original-source baseline is `21c3818c06c6931954c4170c7104fe0817ae1c57`,
a real descendant of `91735dabc` containing exactly the reviewed lexer, parser,
emitter-shadow, new actual-route controls and component-workflow delta.
I merge its real ancestry into my separate async candidate branch and retain
all 92,816 previous entries, with the new control file added. I do not broaden
my five content-only metadata exceptions.

My actual local checker and C-created async VM/native controls pass with
explicitly pinned existing providers. I retain commands, original failures,
source/product hashes and scope under
`docs/evidence/async-source-contracts-20261109/`. The source-only C attempt
refuses the same array ABI boundary on baseline and candidate. I do not count
that refusal as execution acceptance. The new emitter shadow, current source
compiler, fresh stages and paired platforms still require hosted qualification.

I preserve original-source authority, every historical report, empty native
implementation selections and the absent native lifecycle receipt. These
source custody updates do not complete coroutine scheduling, foreign authority,
the original corpus or budget, security, integration or release acceptance.
I freeze catalog inputs before supported authority review and retain final
validation logs outside the catalog.


## My grouped union-pattern source checkpoint

I preserve the existing grouped union-pattern syntax through my source parser,
coverage checker and emitter. I retain one guard and body, duplicate alternatives,
exact underscore variant identity and once-only effects. I preserve ordinary
pattern paths and ownership admission boundaries. My original-source baseline is
`65b33feeb756659bcaad78b939866728c638f8dd`, a real descendant of my async baseline
`21c3818c06c6931954c4170c7104fe0817ae1c57`. I change only three original NanoLang
source files, the actual-route control file and the component workflow. I retain
all 92,817 previous entries and add one control file without widening my five
content-only metadata exceptions. This complete Git tree is independent of my
temporary sparse working view.

I retain local commands, product observations and their limits under
`docs/evidence/match-or-source-contracts-20261109/`. The actual native checker and
C-created controls pass with pinned existing providers; source-created execution,
fresh stages and paired platform acceptance remain open. I preserve original
source authority, empty native implementation selections and the absent native
lifecycle receipt. These custody checks do not complete native SDK, original
corpus, budget, security or release acceptance. I freeze catalog inputs before
supported authority review and retain its validation logs outside the catalog.


## My grouped statement-block correction

My exact 328 Apple source compiler completes all 1,286 selected shadows and
passes 32 of 33 component methods. The unchanged original grouped-pattern
fixture exposes an incorrect block-arm tag in my new parser branch. I store
PNODE_BLOCK explicitly for a block arm and retain direct parser/emitter shadows,
the unchanged original fixture and all expression-arm controls. I also require
the resource refusal's observed E0036 ownership diagnostic explicitly.

My original-source baseline is `1883b7373cba3a7c64815d6a770a51cae39d72f9`, a real
code-only descendant of `65b33feeb` changing the parser, emitter shadow and actual
route control file. I retain all 92,818 original entries, their complete Git
ancestry and the same five content-only metadata exceptions. I retain exact
hosted source/product/command audits and local checker controls under
`docs/evidence/match-or-source-contracts-20261109/`. Linux's source-creation
deadlines remain separate failures. The corrected source candidate still needs
hosted execution and fresh/original-corpus acceptance.

I retain bounded local profiling setup failures separately under
`docs/evidence/source-shadow-profile-20261109/`. No CPU sample was taken and no
source compiler was produced by those diagnostics. Original source authority,
empty native implementation selections and the absent native lifecycle receipt
remain unchanged. I freeze this catalog before supported authority review and
retain final validation logs outside it.


## My optional Apple shadow-sampling diagnostic

I add a false-by-default sampling mode to my existing diagnostic comparison
workflow and one actual-process driver. My default six-attempt comparison and
all original C, NanoLang, Make, test and qualification inputs remain unchanged.
The optional mode samples one identified real shadow child with the original
selection and ten-second supervisor, then stops before native publication.
Actual samples, setup refusals and timing acceptance remain distinct.

My original-source baseline is `51649d12a252d4685cf22593d6101a3849e28df6`, a real
code-only descendant of `1883b7373` containing the workflow delta and new driver.
I retain all 92,818 inherited entries and add one script without widening my
five content-only metadata exceptions. My complete source tree is independent
of the temporary sparse working view, and I merge its real ancestry.

I retain local native process-group controls, their initial harness refusal,
parsed-YAML preservation and help/compilation checks under
`docs/evidence/source-shadow-sampler-20261109/`. My actual Apple run 42548134459 at exact f18474b85 now produces a verified CPU
sample with 602 snapshots after all 1,286 shadow bodies complete. I independently
match its actual provider/sample bytes and source inputs and retain unchanged
before/after observations under the same evidence directory. Intentional
termination prevents native publication; this sample is not source-creation or
timing acceptance. I keep original source authority, eight integration Components,
empty native implementation selections and absent native lifecycle receipt
unchanged. No diagnostic profile completes native SDK, source/fresh/platform,
security, budget, integration or release acceptance. I freeze this catalog
before supported authority review and retain its logs outside the catalog.


## My measured record-name filter candidate

I factor my unchanged record lookup and byte-context bodies into a private
include used by the actual evaluator and native sanitizer unit, then reject
unequal initial bytes before the same full strcmp equality test. I preserve
first-match/declaration order, duplicate/empty/non-ASCII C metadata, byte
normalization/refusals, owned STRING reads and other value identities/flags.
I change no public record, AST or GC layout and introduce no ownership cache
or foreign admission.

My original-source baseline is `6bb9c6bd3e01628070df809e41200f9bec1e0bb9`, a real
code-only descendant of `51649d12a` containing five reviewed paths: evaluator,
private include, native unit, Make target and existing component workflow.
I retain all 92,819 inherited entries and add two source/control files, with
real ancestry and the same five content-only metadata exceptions. My temporary
sparse working view does not omit inputs from the complete Git checkpoint.

My same actual helper/environment/GC sanitizer controls pass before and after
filtering, with identical refusal text and owned STRING lifetime behavior. The
wide control reduces 128,000 comparisons to 2,000 for 2,000 lookups; it does not
establish a whole-compiler speedup. The complete evaluator syntax check passes.
I retain exact commands, initial C test-literal warning, source/product hashes
and scope under `docs/evidence/record-field-filters-20261109/`.

I declare fixed actual whole-source paired comparisons against pre-change
48016fec4 and retain the original complete component/platform selections and
deadlines before adoption. Actual hosted results remain required. Source
carrier lifetime, full fresh/corpus/budget, NativeSDK/lifecycle, security and
release remain open. Original source authority, eight integration Components,
empty native selections and absent lifecycle receipt are unchanged. I freeze
this catalog before supported review and retain final logs outside it.


## My optional factory process observation

I preserve the completed GCC/Apple record-field comparisons, including the
first Apple baseline deadline and both whole-factory wall regressions. I add
read-only, false-by-default process observation to the existing fixed six-
attempt diagnostic. I retain only the actual factory root and transitive
descendants, with PID/PPID/start identity, status, ps comm and monotonic snapshot
bounds. Process presence is not exact CPU time or resolved image provenance.
I change no runtime/provider source, actual factory tool selection/environment,
cache context, selected shadows or original deadlines. Observer overhead and
failures remain visible, and the original completed comparisons are not retried.

My original-source baseline is `e718c78e67b99ff5517141f28f91c96e762ab7ca`, a real
direct child of `6bb9c6bd3` with all 92,821 entries and exactly three reviewed
content changes: the comparison script, its existing controls and diagnostic
workflow. My first local sparse-checkout attempt has an unpopulated complete
index and commits only three entries. I reject it before publication, retain
its identity and correct the index before constructing and merging the complete
checkpoint. I verify exact entry count, direct parent, changed paths and a clean
owned checkout; no partial source tree becomes custody authority.

All thirteen bounded local controls pass. My retained actual native witness
checks native output/refusal, real parent/child identities, an excluded live
sibling, source/image before/after hashes and actual snapshots. I retain
commands, the initial checkpoint refusal and scope outside this catalog under
`docs/evidence/factory-process-phases-20261109/`. Full provider builds remain
held locally for disk pressure; hosted process diagnosis, complete source/
fresh/platform and original release gates remain required. I preserve original
source authority, eight integration Components, empty native implementation
selections and absent lifecycle receipt. I freeze this catalog before supported
review; these controls do not complete NativeSDK/lifecycle or release admission.

## My checked collection lifetime checkpoint, 2026-11-09 UTC

I retain source-only checkpoint `237e76d16f9a555e5c47164f676623ac3834f15e`, a direct child of
`e718c78e67b99ff5517141f28f91c96e762ab7ca`, with the complete 92,823-entry tree: every prior entry,
eight reviewed path changes and two added native test files. I verify the full isolated index and exact
delta before merging its ancestry. I advance preservation ownership to this reviewed source checkpoint;
I retain the previous process-observer baseline as provenance and keep the five metadata exceptions unchanged.

My local collection boundary evidence in `docs/evidence/collection-handles-20261109/README.md` retains an
actual ASan stale-map failure and corrected ordinary-module probe. Assertion, ASan/UBSan and ThreadSanitizer
controls execute checked map/set liveness, alias retirement, wrong-kind refusal, allocation/identity exhaustion,
content/snapshot behavior and independent-thread publication. Existing array-export assertions pass.
These identities belong to one loaded provider; same-handle operations and borrowed strings remain
caller-serialized. I leave compiler opaque-field admission unchanged. Canonical module identity, foreign
signatures, unchanged advanced fixtures, fresh stages and complete Linux/Apple gates remain open under #1127.
I do not infer NativeSDK, retained lifecycle or release admission from local controls or source custody.

I advance that custody checkpoint to `a2ffaa7033aad3f4b0979f914e12a19833bee734` after retaining
both first Linux fixture build refusals. This direct child changes only the new fixture's first line:
I define `_GNU_SOURCE` before its system headers, matching the included implementation's feature request.
The complete tree remains 92,823 entries, and production module bytes are unchanged. Corrected local
assertion/ASan/UBSan/ThreadSanitizer and existing export controls pass; Linux execution still requires
its corrected-source run. The original Apple plan remains on its original handle and source pin.

## My collection snapshot ownership candidate, 2026-11-09 UTC

I retain complete source checkpoint `a7e6f2ced0fd8f20d4e343785ea1c97455a184be`, a direct child of
`a2ffaa7033aad3f4b0979f914e12a19833bee734`. Its full index preserves 92,827 entries with sixteen
reviewed path deltas and four new tests. I retain the prior fixture checkpoint and all metadata exceptions.

I add explicit versioned array-release companions to collection snapshots, resolve cleanup only from the
producer's image, and copy VM results before cleanup. I reject a declared owned result that aliases an
argument and attempt cleanup after conversion failure. The runtime descriptor records optional cleanup;
I do not admit new foreign signatures or opaque record fields. Missing companions retain borrowed-result
behavior. My marker preserves existing C manual element cleanup; it does not add automatic string destruction.

My local native producer, actual shared-image loader and real VM heap controls pass, including copy/allocation
refusals and cleanup errors. My strict fast gate passes 223 compiles each with clang and GCC. I retain the
loader harness's initial missing-OpenSSL-header refusal and use the installed dependency flags for correction.
Evidence is in `docs/evidence/collection-array-ownership-20261109/`. Actual VM CLI controls require explicitly
selected fresh tools and remain for hosted execution; local heap/loader controls do not substitute for them.
Canonical module identity, source compiler routes, final platform/NativeSDK and release acceptance remain open.

I advance snapshot custody to `93ccb6e5536d96361d3c73cbe0f3ce8c8886f99e` after retaining the first
Linux loader-dependency feature-macro refusals. This complete 92,827-entry direct child changes only
the standalone C/Python loader fixture: I use the production GNU feature flag and guard its definition.
Production/runtime code is unchanged. Corrected local loader modes pass; the original Apple plan has
passed actual VM CLI ownership controls. I retain that handle and require corrected Linux execution.

My completed snapshot component plans retain Apple42568331586 at42dddfceb and Linux42569250444 atb94da11a. Independent hosted audits in `docs/evidence/collection-array-ownership-20261109/` verify33 native executables,54 shared libraries and nine NVM modules, selected immutable source inputs and every original success/refusal operation. Fourteen selected production/Make/workflow/heap-fixture paths are identical across those qualification and review pins; only two loader fixture files differ. VM/assembler hashes are retained reports without uploaded tool binaries. This completes my producer/VM snapshot prerequisite and corrected harness controls, while native AOT adapters, source admission, final-head platform qualification and release/NativeSDK acceptance remain open.

## My native collection artifact candidate, 2026-11-09 UTC

I retain complete source checkpoint `6e326819890734095b825d44c8127447c294cfb2`, a direct child of `93ccb6e5536d96361d3c73cbe0f3ce8c8886f99e`. I populate a separate complete Git index, apply exactly five reviewed paths and verify all 92,829 entries before retaining the checkpoint ancestry. Two files add actual native boundary controls.

I admit the eighteen existing map/set artifact signatures in my AOT consumer. I retain exact opaque and C bool transport, bind versioned snapshot cleanup to the producer image and stage native copies until cleanup succeeds. Actual producer/VM/O0/O2, allocation-denial and origin/signature refusal controls pass locally; strict generated GCC compiles and 223 product compiles each with clang/GCC pass. I retain the initial GCC formatting refusal and optional full source-regression setup failure for absent comparator/stage executables in `docs/evidence/collection-aot-20261109/`. Full component-platform plans, canonical source declaration identity, opaque field admission, fresh stages and final NativeSDK/release acceptance remain open.

I retain the actual Linux Clang UBSan wrong-function-type refusal in #1133 before correcting my AOT array declarations. My direct-child complete checkpoint `b106ee66ee6258fc930b0bae8107399b0fe6753b` preserves92,829 entries and exactly four source changes from `6e326819890734095b825d44c8127447c294cfb2`. I share the existing public ElementType/DynArray declaration with generated C; preprocessed declaration tokens, remaining public API bytes and ABI version1 are unchanged. Actual collection/JSON/filesystem producer and cleanup types now agree. Five corrected local boundary methods and existing string/snapshot/frame controls pass without weakening sanitizers. I require corrected full platform plans and retain source/fresh-stage/NativeSDK/release boundaries.

I retain opt-in C-seed shadow completion observation at complete source checkpoint `90b79b0e4af163b5b725988e0d0197b64d9f11bc`, a direct child of `b106ee66ee6258fc930b0bae8107399b0fe6753b`. I preserve all 92,831 entries with seven reviewed deltas and two new source paths. Actual success/failure/deadline/report/zero-exit and errno controls pass, as do original comparison controls and strict clang/GCC product compiles. The fixed six-attempt diagnostic remains measurement work under #1065, with original deadlines and selections unchanged. I do not infer whole-source reliability, NativeSDK or release admission from these controls or custody. Evidence is in `docs/evidence/shadow-completion-20261109/`.

I retain kernel-exit observation source checkpoint `28ea08ee1d4b058f4271a3cd082504764fbfa4d9`, a complete 92,835-entry direct child of `90b79b0e4af163b5b725988e0d0197b64d9f11bc`. I preserve every prior source entry and add exactly four reviewed workflow/controller/control/test paths through an independent full Git index. Six local methods pass actual native gate/exit/wait execution, interruption cleanup and constructed parser refusals; the latter are not Linux BPF evidence. I predeclare one Linux/GCC factory observation only after real kernel/native control validation, preserving the original runner identity, inputs, ten-second shadow supervision and native CLI budget. Production language/runtime behavior, source reliability, NativeSDK and release acceptance remain open. Evidence is in `docs/evidence/cseed-kernel-exit-20261109/`.
