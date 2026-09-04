---
name: releasing-documentation
description: >-
  Bring README, CONTRIBUTING, the user guide and the roadmap current before a
  NanoLang release is tagged. Use when make release-docs-check names one of
  them, or when preparing a release by hand.
---

# Release gate: user-facing prose

Owned by [`../SKILL.md`](../SKILL.md), which defines the order this runs in.

```sh
make release-docs-check          # or: python3 skills/releasing/documentation/check_documentation.py
```

## What it asks

One question per document: **did what it describes change since the last tag,
while it did not?** If both moved, or neither did, there is nothing to report.
The check is in `../lib/staleness.py`; `check_documentation.py` is only the
manifest of which document watches which paths.

It cannot tell you whether the prose is *correct* -- only that its subject
moved and nobody looked. Judgment is the part that stays here.

## When a document is named

Open it and ask what this release changed about what it claims.

**`README.md`** — I state what I am and how to run me. A new release document,
a changed instruction set, or a changed CLI entry point can each make it wrong.
Do not expand it into API documentation; that is what `--help` and the user
guide are for.

**`CONTRIBUTING.md`** — the rules a contributor is held to, and the commands
that enforce them. Check both halves. In 4.0 the rules were stale in substance
(no mention of the verifier, fuzzing, or the measurement policy) while the
commands were fine, which is the easier half to miss. If a gate exists in
`Makefile.gnu` and is not named here, a contributor learns it by failing it.

**`userguide/`** — lessons about the surface language. A parser, type checker,
or lexer change can strand one. Every example must still compile and pass its
shadow tests.

**`docs/ROADMAP.md`** — anything the release document says shipped must not
still be listed as pending.

## Adding an artifact

Add a `Watch` to `WATCHES` in `check_documentation.py`:

```python
Watch(
    document="path/to/doc.md",
    describes=["src/thing.c", "docs/POLICY_*.md"],
    why="one sentence a releaser can act on at 2am",
)
```

A pattern without a wildcard matches by prefix, so `src/nanoisa` covers
everything beneath it. Write `why` for the person the gate interrupts: it is
read at the least patient moment of the release.

Do not add a checking *algorithm* here. If the question is new, it belongs in
`../lib/`.
