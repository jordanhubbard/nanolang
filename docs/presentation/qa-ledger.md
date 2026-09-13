# NanoLang presentation QA ledger

## 5.0 claims correction checkpoint (local, unpublished)

I regenerated both members from the existing Python builders and preserved
their visual design. I distinguish shadow policy from warnings and exemptions,
tests from proof, abstract reference balance from ownership, and runtime
laboratory evidence from production isolation. I corrected the bounds guard
and #211 status. Both members now share `examples/gcd.nano`.

I extracted the actual DOCX example and compiled and executed it on my C seed,
Stage 2 native compiler and bytecode path. All returned six. Deliberately false
assertions were rejected on every path without replacing previous output.
Artifact regression checks also inspect the PPTX claims and shared example.
The header-dependency gate passes. Mechanical acceptance is not visual acceptance.

I inspected the contact sheet and all sixteen full-size Pillow renders. Text
fits those renders without frame overlaps. This renderer is a layout aid, not
a pixel-identical PowerPoint renderer. Native DOCX/PPTX visual acceptance is
still open: `/opt/homebrew/bin/soffice` is a launcher whose LibreOffice application
is missing. The pinned toolchain also lacks the optional DOCX renderer's
`pdf2image` dependency. The managed dependency loader is unavailable; I used
the repository-prescribed portable authoring path. No external publication
or tag was made. I retain the roadmap item until the remaining review passes.

## 5.0 acceptance-gate repair (local, unpublished)

I found that the portable regeneration script wrote `accepted: true` after
checking a constant count and the narrative's existence. The verifier named
in my authoring skill was missing, and the narrative manifest discarded the
deck path. I now retain both paths, derive the slide count from the PPTX, and
run `scripts/verify_document_pair.py` before reporting mechanical acceptance.

The verifier reads ordered slide relationships, checks the actual count,
flat unrotated shape geometry and text-frame intersections, and requires body
speaker notes on every slide. It checks narrative heading levels and explicit
placeholder/credential patterns in text. Unsupported shapes fail closed.
Pattern checks are not a comprehensive secret audit, and geometric checks do
not establish text fit, appearance, factual accuracy or visual acceptance.

The retained local pair passes these mechanical checks with 16 slides and
34 headings. Mutation tests reject count/path/package failures, off-surface
and overlapping frames, unsupported transforms, empty notes, invalid heading
levels, placeholder and credential patterns, and invalid slide relationships.
The CLI replaces stale success reports with failure. Real regeneration builds
both members in isolated outputs; an induced narrative-authoring failure leaves
acceptance false instead of retaining success from the preceding invocation.

I have not regenerated or visually accepted a corrected edition in this
checkpoint. The managed presentation/document dependency loader is unavailable.
I bootstrapped the repository's pinned, ignored authoring environment as the
portable fallback and ran the real builders in the integration test. Shadow claims, the
narrative example and the stale #211 note remain the next repair. I have not
changed external Google artifacts or their publication records.

Ten document-pair tests pass. The wider `make test-release-gates` run exposed
CONTRIBUTING staleness, now tracked separately on the roadmap. I do not bypass
that failure or treat this mechanical gate as release readiness.

## 4.5 developer edition

- Rebuilt the deck and narrative from NanoLang source, `docs/PERSONA.md`,
  `docs/NANOISA.md`, `docs/NSI.md`, `docs/NSI_FABRIC.md`, `docs/NSI_EFFECTS.md`,
  `docs/NANO_EMACS.md`, `docs/ROADMAP.md`, `docs/RELEASE_4.5.md`,
  `docs/NANOISA_MEASUREMENTS.md`, and current tests on the 4.1–4.5
  public-release branch.
- Expanded to 16 slides. Slide 1 names 4.5 as a language and a secure
  runtime. Slide 14 is the five-layer runtime plus journal. Slide 15 names
  what 4.1–4.5 shipped and refused. Slide 7 still shows the 4.0 verifier
  failure; that lesson did not expire.
- Last public GitHub Release remains `v4.0.0` until `v4.5.0`. Google
  publication uses the existing file ids and happens after that tag.
- Published in place after `v4.5.0` (`ef32c833`). Read-back matched the
  local build: 16 slides, 16 note pages, 34 narrative headings. Anyone
  with the link can read; search discovery stays off. Anonymous `/preview`
  returned HTTP 200 with no sign-in wall.
  - Deck: https://docs.google.com/presentation/d/1oWP5WJ7q5XhUF5jB_iLf3qO1mTdtrNt3FqIvYfbH2uM/preview
  - Narrative: https://docs.google.com/document/d/1AHbhUecsOx2QHG4fTMlFDA7l4xZR9IhhgV80NmdiCb8/preview
- I do not claim a Forth Standard System, GNU Emacs, a kernel, or that the
  system is internationalized. The trap journal is a tested library, not a
  hook on every `vm.c` trap.

## 4.4 developer edition

- Rebuilt the deck and narrative from NanoLang source, `docs/PERSONA.md`,
  `docs/NANOISA.md`, `docs/NSI.md`, `docs/NSI_FABRIC.md`, `docs/NANO_EMACS.md`,
  `docs/ROADMAP.md`, `docs/RELEASE_4.4.md`, `docs/NANOISA_MEASUREMENTS.md`,
  and current tests on the 4.4 product branch.
- Kept 15 slides. Slide 1 and 14 name 4.4. Slide 7 still shows the 4.0
  verifier failure; that lesson did not expire.
- Local PowerPoint and Word artifacts regenerated. Google publication uses
  the existing file ids and happens after the `v4.4.0` tag.
- I do not claim a Forth Standard System, GNU Emacs, a kernel, or that the
  system is internationalized.

## 4.0 developer edition

- Rebuilt the deck and narrative from NanoLang source, `docs/PERSONA.md`,
  `docs/NANOISA.md`, `docs/ROADMAP.md`, `docs/RELEASE_4.0.md`,
  `docs/NANOISA_MEASUREMENTS.md`, and current tests at the `v4.0.0` tag.
- Revalidated every count in `source-notes.md` by running the suites: 2,632
  NanoISA, 621 NanoVM, 93 verifier, 63 NanoVirt.
- Built 15 slides and rendered all 15 with the Pillow-based renderer. Contact
  sheet and full-size inspection completed. No text-frame overlaps detected.
- Fixed a defect in `build_deck.py` that had been present since the 3.5
  edition: `title()` drew the headline in ink regardless of the slide's
  background, so the headline was invisible on every dark slide. `slide()` now
  records the fill it painted and `title()` picks a colour that contrasts,
  which puts the choice in one place rather than at fourteen call sites.
- Added `make doc-toolchain-bootstrap`. `regenerate_python.sh` refuses to
  pip-install on its own and directed the reader to that target, which did not
  exist, so the documented regeneration path could not be followed.
- Published both artifacts and verified them by read-back. Exported the
  uploads back out of Google and compared: 15 slides and 15 note pages for the
  deck, 29 headings for the narrative, all matching the local build.
  - Deck: https://docs.google.com/presentation/d/1oWP5WJ7q5XhUF5jB_iLf3qO1mTdtrNt3FqIvYfbH2uM/preview
  - Narrative: https://docs.google.com/document/d/1AHbhUecsOx2QHG4fTMlFDA7l4xZR9IhhgV80NmdiCb8/preview
- The blocker recorded for the 3.5 edition -- that the gcloud token lacked
  Drive scopes -- was stale. The token carries `auth/drive`; publication had
  been failing for two other reasons, both now fixed in
  `publish_google_workspace.py`.
- `_ensure_org_reader` granted read to one hardcoded corporate domain. A domain
  permission can only name a domain the authenticated principal belongs to, so
  from a personal account it fails after the files are already created --
  leaving orphans in Drive. Replaced with `_ensure_link_reader`, granting
  `type: anyone, role: reader` with `allowFileDiscovery` false.
- Sharing was applied to the narrative only; the deck was left private under a
  `slides_access` literal that merely asserted `owner-authenticated`. The deck
  is the artifact the release announcement cites, so a published link would
  have resolved to a permission wall. Both files are now shared through the
  same call, and the recorded access is what Drive returned rather than a
  constant.
- The file names were hardcoded `(3.5 edition)`. Drive bakes the name in at
  creation, so publishing 4.0 would have produced correctly-built files under
  the previous release's name.
- Verified anonymously, without credentials, that the recorded links open. The
  `/edit` and `/view` forms present a sign-in prompt to an anonymous reader
  even on a world-readable file; `/preview` does not. The recorded links are
  the `/preview` form for that reason.

## 3.5 developer edition

- Rebuilt the deck and narrative from NanoLang source, `docs/PERSONA.md`,
  `docs/NANOISA.md`, `docs/ROADMAP.md`, `docs/RELEASE_3.5.md`, and current tests.
- Replaced copied project imagery with the NanoLang mascot from
  `userguide/Nanolang_Mascot.png`.
- Built 12 slides and rendered all 12 slides with the Pillow-based renderer.
- Contact sheet and full-size render inspection completed.
- No text-frame overlaps detected.
- Google publication is pending because the authenticated gcloud token lacks
  the Drive scopes required to create or update files.
