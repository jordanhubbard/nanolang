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

<!-- literate-ai:authority-reviewed sha256:f0c1c7e082054a5e058fc815e7805859322e6a34d7146588de94ad26fdb666a3 -->

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
