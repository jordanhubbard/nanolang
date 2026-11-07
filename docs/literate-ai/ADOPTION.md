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

<!-- literate-ai:authority-reviewed sha256:4186b1a872a14d2348e04b45247f86e75ef3a432c3dc6f0058c8fb1b39846a0f -->

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
