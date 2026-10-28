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
`make test`, `make testquick`, `make test-one-ir-compiler`, bootstrap scripts and `make install PREFIX=...`.
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

<!-- literate-ai:authority-reviewed sha256:8f073bc81c6e6788c8ea7b94e8e9cf53ec75b02cdb9d275bc7bc79b268ad95ad -->

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
