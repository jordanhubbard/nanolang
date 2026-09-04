# NanoLang presentation QA ledger

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
