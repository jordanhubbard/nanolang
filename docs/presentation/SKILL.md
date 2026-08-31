---
name: regenerate-nanolang-developer-overview
description: Regenerate and refresh the NanoLang developer presentation and narrative from current language, compiler, NanoISA, NanoVM, and release evidence.
---

# NanoLang developer overview regeneration

This directory is the durable authoring package for my developer presentation and
technical narrative. It is generated from my current source, specifications,
tests, roadmap, and release evidence.

Before changing or regenerating the deck:

1. Read the repository-root `AGENTS.md`, `docs/PERSONA.md`, and relevant skills completely.
2. Use CodeGraph first when `.codegraph/` exists and code behavior must be located or verified.
3. Use `build_deck.py` (`python-pptx`) to construct the PPTX. Bootstrap the document
   toolchain in an ignored environment if needed. Google Slides publication is a separate,
   explicitly authorized step.
4. Use the installed `imagegen` skill only when an existing image asset must be replaced. Preserve the current palette, composition intent, and negative-space requirements in `prompts/image-prompts.md`.
5. Treat `deck-specification.md` as the presentation's narrative authority and
   `source-notes.md` as my factual-claim ledger. Do not carry claims from another project.

## When to regenerate

Regenerate both members for a major or minor NanoLang release. Patch releases keep the
last published major/minor edition unless the user explicitly authorizes a refresh.

Source changes must update the specifications, factual ledger, builders, prompts, and QA
record together. Do not claim publication until Google read-back succeeds.

## Refresh procedure

1. Revalidate every implementation claim in `source-notes.md` against current specifications and current code. Do not promote roadmap language into implemented claims.
2. Update `deck-specification.md` first. The presentation source is derived from that specification.
3. If a visual must change, regenerate only that asset using its recorded prompt and replace the corresponding file in `assets/`. Keep visuals text-free.
4. Update [build_deck.py](build_deck.py)'s corresponding slide function(s) to match. It ports
   `build_deck.mjs`'s primitive shape vocabulary (`shape`, `text`, `pill`, `line`,
   `arrow`, `footer`, `title`, `wash`, cover-fit image placement) 1:1; add native-shape
   diagrams rather than raster images for anything that shows a real mechanism.
5. Run [regenerate.sh](regenerate.sh). It prefers the pinned OBJ_DIR toolchain from
   [regenerate_python.sh](regenerate_python.sh) and
   `make doc-toolchain-bootstrap` (it will not pip-install on its own), builds the PPTX
   and the narrative `.docx`, rasterizes slides for inspection, then runs
   `scripts/verify_document_pair.py` against both members and writes its report to
   `$OBJ_DIR/nanolang-developer-overview/acceptance.json`. Fix any failing scenario
   before continuing — it checks slide-surface geometry, non-overlapping text-bearing
   frames, non-empty speaker notes on every page, narrative heading levels, no credential
   material, and no unresolved placeholder tokens.
6. Render every slide, inspect a contact sheet and full-size slides, and fix all
   unintended clipping, overflow, overlap, broken connectors, and unreadable type.
   Geometry-escape (elements inside 1280 × 720) is not sufficient: two-line titles
   that paint into subtitles, section tags that collide with the brand pill, and
   bottom captions that collide with the footer all fail this pass. After the
   native Google Slides update, read the published copy as well — font substitution
   can introduce overlaps the local PPTX did not show. Workflow slides must show
   the flow as a picture or diagram; a paragraph that restates the workflow is the
   defect, not a caption. Native-shape diagrams still count as pictures; do not
   replace the specification excerpt or control-model diagrams with decorative
   raster that hides the mechanism.
7. Run the presentation skill’s overflow test when its managed Python environment is
   available and the Codex `presentations` plugin path is in use. For the `python-pptx`
   path, `scripts/verify_document_pair.py` is the mechanical geometry gate: it fails
   elements outside the declared surface **and** overlapping text-bearing frames.
   Geometry-escape alone is not an overflow test. `render_slides.py` also prints an AABB
   overlap report for inspection. Record any fallback in the QA ledger.
8. Import the verified PPTX as native Google Slides. For an authorized existing Slides URL,
   extract its presentation ID, obtain a fresh bearer token with
   `gcloud auth print-access-token`, update that exact resource in place, and verify it by
   reading or exporting it afterward. Never persist the token or include it in logs. Record
   the resulting Slides URL in `README.md`, near the top, alongside the narrative's URL.

## Narrative member refresh procedure

1. Treat `narrative-specification.md` as the narrative's structural authority (heading
   hierarchy, section scope) and `source-notes.md` as its factual-claim ledger — the same
   ledger the presentation member cites. Revalidate every claim before drafting or
   redrafting.
2. Use `build_narrative.py` (`python-docx`) to construct the local `.docx`. This is the
   primary narrative path; it has no LLM calls and no Codex plugin dependency. The Codex
   `documents` plugin remains an optional environment where that plugin happens to be
   installed and funded — it is not a hidden requirement of this package.
3. `./regenerate.sh` builds both members through the portable python-pptx path, writes
   the capability manifest covering both, runs `scripts/verify_document_pair.py`, and
   rasterizes slides with [render_slides.py](render_slides.py) into
   `$OBJ_DIR/nanolang-developer-overview/rendered-slides/` for visual inspection. Fix any
   failing oracle scenario before continuing. Headings must map to Word `Heading 1`
   through `Heading 6` without skips.
4. Inspect the contact sheet and full-size slides. For the narrative, confirm the heading
   tree matches `narrative-specification.md` and that no placeholder or credential
   material ships.
5. Import the verified PPTX as native Google Slides and the verified `.docx` as a native
   Google Doc. For an authorized existing Slides URL, extract its presentation ID, obtain
   a fresh bearer token with `gcloud auth print-access-token`, update that exact resource
   in place, and verify it by reading or exporting it afterward. Create the Google Doc
   when none exists; thereafter update that exact Doc in place unless the user asks for a
   copy. Never persist the token or include it in logs. Record both URLs in `README.md`,
   near the top, as the current major/minor *edition*. Drive `files.export` rejects this
   deck (`exportSizeLimitExceeded`); read-back uses the authenticated
   `https://docs.google.com/presentation/d/{id}/export/pptx` URL.
   [`publish_google_workspace.py`](publish_google_workspace.py) in this package performs
   the authorized update, create, access mapping, and read-back.
6. Both members share one publication-authorization gate. Regenerating one member does
   not authorize publishing either. Asking for this overview to be regenerated and
   published for a named edition authorizes that revision only.

No proprietary presentation or document plugin is required. The local Python toolchain
is the portable authoring path; Google publication uses the authorized `gcloud` token.

## Guardrails

- Do not invent productivity or ROI percentages.
- Keep one cumulative narrative for product, program, engineering, and executive audiences. Do not reintroduce audience-specific sections that repeat the same claims.
- Present the deck as a developer account of language design, compilation, execution,
  evidence, and the next implementation boundary.
- Preserve the distinction between implemented Standard-core behavior and remaining reference-sample or CLI integration work.
- Preserve the source-to-specification boundary: draft extraction is available; effective authority transfers only with trusted, current qualification evidence.
- Generated source is an untrusted candidate until build, test, execution, independent acceptance, and receipt requirements pass.
- Keep the deck visual, but carry mechanism alongside claim. Every headline assertion in
  the main sequence needs a slide, a diagram, or a shown artifact that explains how the
  system produces it. Prefer one precise diagram or real artifact excerpt to a hero image
  with three sentences of consequence.
- Image-led slides carrying only a headline and a paragraph must stay a minority of the
  deck. They punctuate the argument; they cannot be the argument.
- Show the durable authority at least once. A deck asserting that a readable
  specification is the product must display a real specification excerpt, plan, or
  receipt.
- Keep the technical mechanism slides grouped as a labeled sequence so the engineering
  audience can be pointed at them and the decision audience can move past them.
- Author speaker notes for every slide carrying its supporting authority and the limits
  recorded in `source-notes.md`. The deck is forwarded without narration.
- Answer cost, blast radius, failure handling, and the model-egress trust boundary. Those
  are the first questions this audience asks.
- Keep release claims tied to `scripts/release.sh`, `docs/ROADMAP.md`, GitHub checks, and
  actual tags. Never imply that local policy is live forge protection.
