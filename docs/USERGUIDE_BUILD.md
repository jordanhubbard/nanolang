# My User Guide Build

I publish the Markdown sources selected by `userguide/nav.txt`. That manifest is
the navigation order and the complete list of public pages. Old files under
`userguide/` remain historical input until they are removed; they are not
published merely because they exist.

## Build and Check

```bash
make userguide-check
make userguide-html
make -C userguide serve
```

I write the site to `build/userguide/html/`. The build starts from an empty
output directory, generates reference pages, renders Markdown, then validates
every local link and fragment for English plus Simplified Chinese, Hindi,
Spanish, Arabic, and French editions. Non-English HTML lives under
`build/userguide/html/<lang>/`. Each page has `lang`, `dir` (Arabic is `rtl`),
`hreflang` alternates, a canonical link, and a language switcher that keeps the
current page. Generated reference pages stay English with a fallback banner.

Default: `python3 scripts/build_userguide.py --check`. Restrict with
`--locales en` during debugging; CI builds all six.

## Translations

English under `userguide/` is canonical. Drafts live in `userguide/i18n/`
with YAML front matter, preserved code fences, and `memory.json` source hashes.
Stale drafts get a banner rather than silent publication. Contributor workflow:
[userguide/i18n/README.md](../userguide/i18n/README.md). Machine drafts are not
human translations; `reviewed: false` until a named reviewer accepts a page.

Snippet checks walk published English Markdown only. They skip `userguide/i18n/`.

## Accessibility

The generated pages provide semantic landmarks, a language attribute, skip
navigation, keyboard focus, stable heading anchors, responsive navigation,
overflow-safe tables, a CJK/Devanagari/Arabic font fallback stack, and `dir=rtl`
on the Arabic edition. `tests/test_build_userguide.py` checks those properties
in HTML/CSS. On 2026-09-05 I rendered desktop screenshots of English, Simplified
Chinese, Hindi, and Spanish, a mobile screenshot of Chinese and Hindi, and an
Arabic print raster (Chrome `--screenshot` of `dir=rtl` pages stayed black).
Hans, Devanagari, Latin, and Arabic glyphs rendered; the language switcher
stays on the current page; Arabic `dir=rtl` reverses the switcher. Code fences
on RTL pages are `direction:ltr` with `unicode-bidi:isolate` so braces stay
source order. WCAG conformance still requires an accessibility audit; I do not
claim one has occurred.

The published guide has no in-page search. `nano-docs` searches `userguide/`
Markdown separately.

## Generated Reference

`scripts/build_userguide.py` generates these pages during each build:

| Page | Source |
| --- | --- |
| Builtins | `docs/STDLIB.md`, checked against `src/builtins_registry.c` |
| Examples | Every `.nano` file under `examples/` and its metadata header |
| Modules | `modules/`, `module.json`, `module.manifest.json`, and declarations |
| Compiler CLI | `bin/nanoc_c --help` |

Generated Markdown lives under `build/userguide/generated/`; it is a build
artifact, not a second hand-maintained reference.

## Snippets

Executable examples use an `nl-snippet` marker immediately before a NanoLang
fence. `make userguide-check` compiles marked snippets and runs those with
`run:true`. Unmarked blocks may be excerpts or conceptual fragments. They are
not claimed as independently runnable programs.

## Publication

`.github/workflows/userguide_pages.yml` builds the same site on relevant pull
requests. Pushes to `main` additionally upload and deploy the validated Pages
artifact. The workflow does not edit tracked API Markdown while publishing.
