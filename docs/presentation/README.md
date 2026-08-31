# NanoLang developer overview

This is the reproducible authoring package for my developer presentation and technical
narrative: a Google Slides deck and a Google Doc built from my source, specifications,
tests, roadmap, and release evidence.

**3.5 edition.** This package describes my released 3.5 foundation and the 4.0 boundary.

Published presentation: pending publication

Published narrative: pending publication

Start with the [current deliverables](current-deliverables.md), then read
the [deck specification](deck-specification.md),
[narrative specification](narrative-specification.md), [source notes](source-notes.md),
and [QA ledger](qa-ledger.md). The local [regeneration skill](SKILL.md) describes the
update workflow. Its model-facing inputs are the
[deck-authoring prompt](prompts/deck-authoring-prompt.md) and
[image prompts](prompts/image-prompts.md).

The [deck builder](build_deck.py), [narrative builder](build_narrative.py),
[slide rasterizer](render_slides.py), [Google Workspace publisher](publish_google_workspace.py),
and [regeneration launcher](regenerate.sh) derive the PowerPoint and Word outputs
from this reviewed package using `python-pptx` and `python-docx`. Image assets are
retained because they are authored presentation inputs, not generated application source.
`regenerate.sh` prefers the pinned OBJ_DIR toolchain from `make doc-toolchain-bootstrap`
so a vanilla worker with no Codex plugin can rebuild. The prior
[`build_deck.mjs`](build_deck.mjs) path remains a discovered fallback only (see
[source notes](source-notes.md)'s "Codex-independent rebuild" entry). That backend choice
is project-local and is not a dependency of NanoLang's authoring workflow.

## Release Policy source status

The source package now documents writable Free/Pre-release `main`, exact-main RC tags,
release-line lockdown and README Release Engineer authority, per-project strict/loose
patch authority, mandatory `Literate-AI-Release` pull-request classification, and safe
marker-based branch collection. The maintained local PPTX/DOCX and published Google
copies have not been regenerated or published for this source revision; regeneration and
publication remain pending an explicit follow-up.

Visual input: [NanoLang mascot](assets/nanolang-mascot.png), adapted from the user-guide
asset. Mechanism diagrams are generated as native slide shapes.
