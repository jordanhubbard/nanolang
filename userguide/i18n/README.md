# User-guide translations

English under `userguide/` is the canonical source. These directories are
machine-generated drafts until a human review is recorded.

| Directory | Language | BCP 47 |
| --- | --- | --- |
| `zh/` | Mandarin Chinese | `zh-Hans` |
| `hi/` | Hindi | `hi` |
| `es/` | Spanish | `es` |
| `ar/` | Modern Standard Arabic | `ar` |
| `fr/` | French | `fr` |

## Source format

Each translated page starts with YAML front matter:

```text
---
title: translated navigation title
machine_generated: true
reviewed: false
lang: zh
---
```

Then Markdown. Keep code fences, `<!--nl-snippet ...-->` markers, identifiers,
commands, and `.md` links byte-identical to English. Translate explanation and
interface prose only.

Generated reference pages (`generated/*.md`) have no drafts here. The builder
falls back to English with a banner.

## Translation memory

`memory.json` stores SHA-256 of the English sources listed in `nav.txt`
(except generated pages). When English changes and the hash does not match, the
HTML build inserts a stale-translation banner. Update the hash after you refresh
the draft.

## How to correct a translation

1. Open an issue or pull request against the file under `userguide/i18n/<lang>/`.
2. Keep fences and identifiers unchanged.
3. Set `reviewed: true` only after a named human reviewer accepts the page.
4. Credit the reviewer in the pull request; leave `machine_generated: true`
   until the draft is replaced, not merely patched.

Reviewed pages stay distinguished from machine drafts by `reviewed: false`
until that field changes. I do not treat a machine draft as a human translation.
