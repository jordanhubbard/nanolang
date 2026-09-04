---
name: releasing-presentation
description: >-
  Bring the developer deck and narrative current before a NanoLang release is
  tagged, and publish them afterwards. Use when make release-docs-check reports
  the presentation stale or disagreeing with its specification.
---

# Release gate: deck and narrative

Owned by [`../SKILL.md`](../SKILL.md). Authoring and regeneration live in
[`docs/presentation/SKILL.md`](../../../docs/presentation/SKILL.md) -- this
file is the release gate, and does not repeat what that one says.

```sh
make release-docs-check          # or: python3 skills/releasing/presentation/check_presentation.py
```

## What it asks

**Staleness.** Did the sources the deck is built from -- the release document,
the roadmap, the measurements, the instruction set, the verifier -- change
while `docs/presentation/` did not? Same check every other artifact gets, from
`../lib/staleness.py`.

**Agreement with its own specification.** `deck-specification.md` enumerates
the slide sequence one per line and `build_deck.py` implements it, so the two
counts must match. They silently did not: `main()` printed `built 12 slides`
through the entire 14-slide 4.0 edition, and the manifest hardcoded a third
number beside it. A constant that describes a list is a claim, and claims get
checked.

The specification is the authority. When the counts disagree, either implement
the missing slide or amend the sequence -- do not let the two drift.

## Rebuilding, before the tag

```sh
make presentation
```

This rebuilds the deck and narrative, renders every slide, builds a contact
sheet, and fails on text-frame overlaps. Look at the rendered slides. The
renderer catches overlapping frames; it does not catch a headline that is
invisible against its own background, which is what happened to every dark
slide of the 3.5 edition until 4.0.

## Publishing, after the tag

Deliberately not automated, and deliberately last. It writes to a personal
Google Drive and is outward-facing.

```sh
python3 docs/presentation/publish_google_workspace.py \
    --slides-id <existing> --doc-id <existing>
```

**Always pass the existing IDs.** Creating new files orphans the URLs already
in circulation -- release notes, posts, `current-deliverables.md`. Drive also
fixes a file's name at creation, so a new file inherits whatever name the
script hardcodes.

Verify before recording anything:

1. Read-back. The publisher exports the uploads back out of Google and compares
   slide, note and heading counts against the local build. They must match.
2. Anonymous access, with no credentials. `/edit` and `/view` prompt a signed-out
   reader to log in *even on a world-readable file*; `/preview` does not.
   Record the `/preview` form. A link that works while you are signed in tells
   you nothing about what anyone else sees.

Then record the URLs in `docs/presentation/current-deliverables.md`,
`qa-ledger.md`, and `README.md`.
