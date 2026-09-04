# NanoLang developer overview

This is the reproducible authoring package for my developer presentation and technical
narrative: a Google Slides deck and a Google Doc built from my source, specifications,
tests, roadmap, and release evidence.

**4.0 edition.** This package describes my released 4.0: NanoISA v2, NanoVM v2, what my verifier proves, and what I have not done.

Published presentation: https://docs.google.com/presentation/d/1oWP5WJ7q5XhUF5jB_iLf3qO1mTdtrNt3FqIvYfbH2uM/preview

Published narrative: https://docs.google.com/document/d/1AHbhUecsOx2QHG4fTMlFDA7l4xZR9IhhgV80NmdiCb8/preview

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
from this reviewed package using `python-pptx` and `python-docx`. The only image asset is
the NanoLang mascot from the user guide. Mechanism diagrams use native slide shapes.
Local regeneration is complete. Google publication remains a separate authorized
operation and requires Drive scopes in the token returned by `gcloud auth print-access-token`.

Visual input: [NanoLang mascot](assets/nanolang-mascot.png), adapted from the user-guide
asset. Mechanism diagrams are generated as native slide shapes.
