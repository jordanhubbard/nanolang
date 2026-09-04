# NanoLang presentation QA ledger

## 4.0 developer edition

- Rebuilt the deck and narrative from NanoLang source, `docs/PERSONA.md`,
  `docs/NANOISA.md`, `docs/ROADMAP.md`, `docs/RELEASE_4.0.md`,
  `docs/NANOISA_MEASUREMENTS.md`, and current tests at the `v4.0.0` tag.
- Revalidated every count in `source-notes.md` by running the suites: 2,632
  NanoISA, 621 NanoVM, 93 verifier, 63 NanoVirt.
- Built 14 slides and rendered all 14 with the Pillow-based renderer. Contact
  sheet and full-size inspection completed. No text-frame overlaps detected.
- Fixed a defect in `build_deck.py` that had been present since the 3.5
  edition: `title()` drew the headline in ink regardless of the slide's
  background, so the headline was invisible on every dark slide. `slide()` now
  records the fill it painted and `title()` picks a colour that contrasts,
  which puts the choice in one place rather than at fourteen call sites.
- Added `make doc-toolchain-bootstrap`. `regenerate_python.sh` refuses to
  pip-install on its own and directed the reader to that target, which did not
  exist, so the documented regeneration path could not be followed.
- Google publication remains pending; no URL is recorded because no upload and
  read-back has succeeded.

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
