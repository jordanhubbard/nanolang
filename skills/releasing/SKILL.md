---
name: releasing
description: >-
  Cut a NanoLang release. Defines the ordered gates, and indexes the child
  skills that own each one. Use before running make release, make release-minor,
  or make release-major, and whenever a release is being prepared by hand.
---

# Releasing

I require an LLM to release me, so my release process is written to be read
rather than only executed. This file is the rule set and the index. Each gate
that needs judgment has its own `SKILL.md` in a subdirectory, and each of those
has a Python helper that does the mechanical part.

## The rule that motivates all of this

**My documentation is updated as the last step before tagging, not the first
step of the next release.**

Before 4.0 there was no documentation step in `scripts/release.sh` at all. It
went: prerequisites, version, changelog, `package.json`, `make test`, tag. The
consequences were not hypothetical:

- `CONTRIBUTING.md` went the entire 4.0 development span without a single
  commit while the verifier, the optimization policy, and the fuzzing surfaces
  all changed underneath it. Contributors were being held to the 3.5 rules.
- The presentation deck was rebuilt and published *after* the tag, by hand.
- `build_deck.py` printed `built 12 slides` throughout the 14-slide edition,
  and the manifest hardcoded a third number.

A release that ships correct code with prose describing the previous release
has shipped a lie about itself. That is the same failure as a verifier that
reports success for what it did not check, and it gets the same treatment: the
check runs, and it is not allowed to be silent.

## Order

Gates run in this order. Each one must pass, or be acknowledged with a reason,
before the next begins.

| # | Gate | Owner | Command |
|---|---|---|---|
| 1 | Release-scoped issues and PRs are closed or moved | [`release-readiness`](../release-readiness/SKILL.md) | `gh` review, by hand |
| 2 | The tree builds clean and every test passes | this file | `make clean && make && make test` |
| 3 | User-facing prose is current | [`documentation/`](documentation/SKILL.md) | `make release-docs-check` |
| 4 | Deck and narrative are current and self-consistent | [`presentation/`](presentation/SKILL.md) | `make release-docs-check` |
| 5 | Changelog, version bump, tag, GitHub release | `scripts/release.sh` | `make release[-minor\|-major]` |
| 6 | Deck and narrative published to Google | [`presentation/`](presentation/SKILL.md) | explicit, human-initiated |

Gates 3 and 4 run together as `make release-docs-check`, and `scripts/release.sh`
invokes it immediately before it tags. Gate 6 is deliberately after the tag and
deliberately manual: it writes to a personal Google Drive and is outward-facing,
so it is not something an automated release run should do unattended.

## Acknowledging rather than fixing

A gate may be released past, but not silently. Set `RELEASE_DOCS_ACK` to a
*reason*:

```sh
RELEASE_DOCS_ACK="user guide covers no surface that changed" make release
```

A boolean is rejected on purpose. `RELEASE_DOCS_ACK=1` tells a later reader
nothing, and the entire point of the gate is that silence about a document
stops being available. Whatever reason is given belongs in the release notes.

## How this is factored

The child skills own *what* to check. The shared library owns *how*, because
every artifact asks the same question and four copies of it would drift:

```
skills/releasing/
  SKILL.md                          this file: rules, order, index
  lib/
    repo.py                         git, the release boundary, path matching
    staleness.py                    "its subject moved and it did not"
    report.py                       one finding format, one exit convention
  documentation/
    SKILL.md                        README, CONTRIBUTING, user guide, roadmap
    check_documentation.py          a manifest, not an algorithm
  presentation/
    SKILL.md                        deck and narrative
    check_presentation.py           staleness, plus spec-vs-builder agreement
```

Adding an artifact is adding a `Watch` to a manifest. If you find yourself
writing a second implementation of "did this change since the last tag", it
belongs in `lib/`.

The gates are tested in `tests/test_release_gates.py`, which replays the
3.5 → 4.0 release and requires the documentation gate to catch
`CONTRIBUTING.md`. A gate nobody has watched fail is not evidence of anything.

## Before tagging, by hand

The helpers report staleness; they cannot write prose. When a gate names a
document, open it and ask what this release changed about what it claims:

- `README.md` — do I still describe what I am and how to run me?
- `CONTRIBUTING.md` — are the rules and the gate commands the ones now enforced?
- `userguide/` — does any lesson teach a surface that moved?
- `docs/ROADMAP.md` — is anything still listed as pending that shipped?
- `docs/presentation/` — see [`presentation/SKILL.md`](presentation/SKILL.md).

Then re-run `make release-docs-check` and proceed to gate 5.
